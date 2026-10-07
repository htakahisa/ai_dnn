"""Rule layer using only public state and teammates' own observations.

This object never reads game.chars, replay frames, analytics, or real_character.
Enemy axis counts are recent sightings, not an omniscient setup count.
"""
from __future__ import annotations
from collections import Counter
import math
import numpy as np
from grid_paths import distance_map
from grid_lines import line_cells
from gc_v1.gc_facing import facing_towards
from .config import active_flags
from .geometry import (CARDINAL, angle_at, axis_for, los, nearest_floor,
                       projectile_path, recon_aim, route, valid, watch_cells)


def pos(char):
    return tuple(map(int, char.pos))


def ability_charges(char):
    return sum(int(getattr(char, key+"_charges", 0))
               for key in ("smoke", "flash", "recon", "ramp", "dance", "ash"))


class AttackerTactics:
    def __init__(self, config, stage="entry"):
        from party_presets import get_preset
        self.config, self.stage = config, stage
        self.roster_profiles = {}
        for name, profile in config["profiles"].items():
            if profile.get("preset"):
                preset = get_preset(profile["preset"])
                self.roster_profiles[tuple(sorted(map(str, preset.players)))] = name
        # Validate the stage even before the first live decision.
        active_flags(config, stage, {})
        self.reset_round()

    def reset_round(self):
        self.opponent = "unknown"
        self.sightings = {}
        self.own_positions = {}
        self.inventories = {}
        self.last_cast = {}
        self.last_recon_request = {}
        self.recon_sequences = Counter()
        self.scout_assignments = {}
        self.scout_supports = {}
        self.surveyed_axes = set()
        self.pending_surveys = []
        self.recon_ready_tick = 0
        self.utility_used = False
        self.plant_attempt = None
        self.spike = None
        self.hold_assignments = {}
        self.assignment_key = None
        self.selected_site = None
        self.target_plant_pos = None
        self.site_selection_active = False
        self.site_reason = "not_selected"
        self.observed_defenders_by_axis = dict.fromkeys(("A", "Mid", "B"), 0)
        self.max_observed_defenders_by_axis = dict.fromkeys(("A", "Mid", "B"), 0)
        self.events = []

    def observe(self, char, state):
        self.tick = int(state.get("battle_tick", 0))
        grid = state["grid"]
        self.own_positions[str(char.name)] = (pos(char), self.tick)
        roster = state.get("enemy_roster", ())
        names = tuple(sorted(str(e.get("base_name") or e.get("name")) for e in roster))
        if names in self.roster_profiles:
            self.opponent = self.roster_profiles[names]
        if self.config["profiles"][self.opponent].get("complete_site_survey") and not self.scout_assignments:
            seekers=sorted(str(c.name) for c in state.get("chars",())
                           if c.team==char.team and getattr(c,"ability_name",None)=="RECON")
            self.scout_assignments={name:index for index,name in enumerate(seekers)}
            supports=sorted(str(c.name) for c in state.get("chars",()) if c.team==char.team
                            and getattr(c,"ability_name",None)!="RECON" and not getattr(c,"has_spike",False))
            self.scout_supports=dict(zip(seekers,supports))
        for other in state.get("chars", ()):
            name = str(other.name)
            if other.team == char.team:
                charges = ability_charges(other)
                previous = self.inventories.get(name, charges)
                if charges < previous:
                    if getattr(other, "ability_name", None) in ("RECON", "FLASH"):
                        self.utility_used = True
                    request = self.last_recon_request.pop(name, None)
                    if request and getattr(other, "ability_name", None) == "RECON":
                        self.recon_ready_tick = max(self.recon_ready_tick,request["ready_tick"])
                        if request["covers_anchor"]:
                            self.pending_surveys.append(request)
                        self.events.append(dict(tick=self.tick, type="recon_cast", **request))
                self.inventories[name] = charges
                continue
            if (not other.is_alive or not getattr(other, "position_known", False)
                    or not valid(grid, pos(other))):
                continue
            self.sightings[name] = (axis_for(pos(other), grid.shape[1]), self.tick)
        ttl = self.config["entry"]["sighting_ttl"]
        self.sightings = {name: entry for name, entry in self.sightings.items()
                          if self.tick-entry[1] <= ttl}
        counts = Counter(axis for axis, _ in self.sightings.values())
        self.observed_defenders_by_axis = {axis: counts[axis] for axis in ("A", "Mid", "B")}
        for axis,count in self.observed_defenders_by_axis.items():
            self.max_observed_defenders_by_axis[axis] = max(self.max_observed_defenders_by_axis[axis],count)
        for survey in self.pending_surveys:
            if self.tick >= survey["ready_tick"]:
                self.surveyed_axes.add(survey["axis"])
        self.pending_surveys = [s for s in self.pending_surveys if self.tick < s["ready_tick"]]
        if self.opponent in ("unknown", "observed_mid_heavy") and counts["Mid"] >= 2:
            self.opponent = "observed_mid_heavy"
        profile = self.config["profiles"][self.opponent]
        return profile, active_flags(self.config, self.stage, profile)

    def allies(self, char, state):
        return sorted((c for c in state.get("chars", ()) if c.team == char.team and c.is_alive),
                      key=lambda c: str(c.name))

    def ally_position(self, char):
        saved = self.own_positions.get(str(char.name))
        return saved[0] if saved and self.tick-saved[1] <= 1 else pos(char)

    def enemies(self, char, state):
        return [c for c in state.get("chars", ()) if c.team != char.team and c.is_alive
                and getattr(c, "position_known", False) and valid(state["grid"], pos(c))]

    def occupied(self, char, state):
        return {self.ally_position(c) if c.team == char.team else pos(c)
                for c in state.get("chars", ()) if str(c.name) != str(char.name) and c.is_alive
                and (c.team == char.team or getattr(c, "position_known", False))
                and valid(state["grid"], pos(c))}

    def preplant_route(self, char, state, goals, reserved=()):
        """Avoid long detours around a teammate who will move this tick."""
        enemies = {pos(c) for c in self.enemies(char,state)}|set(reserved)
        direct = route(state["grid"],pos(char),goals,enemies)
        free = route(state["grid"],pos(char),goals,self.occupied(char,state)|set(reserved))
        if len(free)>1 and len(free)<=len(direct)+self.config["entry"]["maximum_detour"]:
            return free
        # The normal IQ adapter and engine reject an occupied next step. Wait
        # for it to clear rather than taking a lengthy route through Mid.
        return direct

    def scout_axis(self, char, allies):
        seekers = [c for c in allies if getattr(c,"ability_name",None)=="RECON"]
        index = self.scout_assignments.get(str(char.name))
        if index is None:
            index = next(i for i,c in enumerate(seekers) if str(c.name)==str(char.name))
        return "A" if index%2==0 else ("Mid" if self.recon_sequences[str(char.name)]==0 else "B")

    def recon_config(self):
        cfg=dict(self.config["recon"])
        for key,value in self.config["profiles"][self.opponent].get("recon",{}).items():
            cfg[key]={**cfg[key],**value} if isinstance(value,dict) else value
        return cfg

    def scout_waypoint(self, char, state, allies):
        axis = self.scout_axis(char,allies)
        group = "secondary_waypoints" if self.recon_sequences[str(char.name)] else "waypoints"
        return nearest_floor(state["grid"],self.recon_config()[group][axis])

    def wait_carrier(self, char, state, target, allies):
        for scout in allies:
            if getattr(scout,"ability_name",None)!="RECON" or getattr(scout,"recon_charges",0)<=0:
                continue
            waypoint = self.scout_waypoint(scout,state,allies)
            scout_path = route(state["grid"],self.ally_position(scout),[waypoint])
            if pos(char) not in scout_path:
                continue
            choices = [(pos(char)[0]+dr,pos(char)[1]+dc) for dr,dc in CARDINAL
                if valid(state["grid"],(pos(char)[0]+dr,pos(char)[1]+dc))
                and (pos(char)[0]+dr,pos(char)[1]+dc) not in scout_path
                and (pos(char)[0]+dr,pos(char)[1]+dc) not in self.occupied(char,state)]
            if choices:
                return self.move_action(char,min(choices),target)
        return self.move_action(char,pos(char),target)

    def escort_goals(self, state, leader_point, target):
        ahead = set(route(state["grid"],leader_point,[target])[1:3])
        return [(leader_point[0]+dr,leader_point[1]+dc) for dr,dc in CARDINAL
                if valid(state["grid"],(leader_point[0]+dr,leader_point[1]+dc))
                and (leader_point[0]+dr,leader_point[1]+dc) not in ahead]

    def clear_carrier_route(self, char, state, holder, target, allies):
        """Let the front of a narrow queue clear before the carrier follows."""
        path=route(state["grid"],self.ally_position(holder),[target],
                   {pos(c) for c in self.enemies(char,state)})
        point=pos(char)
        if point not in path[1:]:
            return None
        occupied=self.occupied(char,state)
        def supported(destination):
            return any(str(c.name)!=str(char.name) and math.dist(self.ally_position(c),destination)
                       <=self.config["entry"]["trade_radius"] for c in allies)
        side_steps=[(point[0]+dr,point[1]+dc) for dr,dc in CARDINAL
                    if valid(state["grid"],(point[0]+dr,point[1]+dc))
                    and (point[0]+dr,point[1]+dc) not in occupied
                    and (point[0]+dr,point[1]+dc) not in path
                    and supported((point[0]+dr,point[1]+dc))]
        if side_steps:
            return self.move_action(char,min(side_steps,key=lambda p:(math.dist(p,self.ally_position(holder)),p)),target)
        index=path.index(point)
        if index+1<len(path) and path[index+1] not in occupied and supported(path[index+1]):
            return self.move_action(char,path[index+1],target)
        return self.move_action(char,point,target)

    @staticmethod
    def move_action(char, destination, look=None):
        destination = tuple(destination)
        if destination != pos(char):
            # Explicit limit prevents awakened two-step movement leaving the leash.
            return list(destination), "MOVE", {"move_step_limit": 1}
        facing = facing_towards(destination, look) if look is not None else None
        return (list(destination), {"facing": facing}) if facing else (list(destination), "MOVE")

    def remember_result(self, char, result):
        if isinstance(result, tuple) and len(result) > 1 and result[1] == "PLANT":
            self.plant_attempt = (pos(char), self.tick)

    def planted_position(self, state):
        if self.spike is None:
            # Planting teammate knows its own exact position. Share that action
            # memory, rather than bypassing IQ perception to read engine state.
            if self.plant_attempt and self.tick-self.plant_attempt[1] <= 1:
                self.spike = self.plant_attempt[0]
            elif state.get("planted_pos") is not None:
                point = tuple(map(int, state["planted_pos"]))
                self.spike = point if valid(state["grid"], point) else nearest_floor(state["grid"], point)
        return self.spike

    def _hold_cells(self, grid, spike, smoke):
        cfg = self.config["hold"]
        watch = watch_cells(grid, spike)
        cells = [tuple(map(int, p)) for p in np.argwhere(grid != 1)
                 if math.dist(p, spike) <= cfg["radius"]
                 and any(los(grid, tuple(p), q, smoke) for q in watch)]
        # If an opponent smokes the entire spike, close to the adjacent cells
        # where the engine explicitly allows shooting through smoke.
        return cells or watch

    def _assign_hold(self, char, state, spike):
        grid, cfg = state["grid"], self.config["hold"]
        allies = self.allies(char, state)
        key = spike, tuple(str(c.name) for c in allies), frozenset(state.get("smoke_cells", ()))
        if self.assignment_key == key:
            return
        self.assignment_key = key
        cells = self._hold_cells(grid, spike, state.get("smoke_cells", ()))
        watch = watch_cells(grid, spike)
        chosen = {}
        # Assign nearest players first so distant players cannot displace a
        # teammate already holding a legal crossfire position.
        for ally in sorted(allies, key=lambda c: (math.dist(self.ally_position(c), spike), str(c.name))):
            distances = distance_map(grid, self.ally_position(ally))
            remaining = [p for p in cells if p not in chosen.values() and distances[p] >= 0]
            tradeable = [p for p in remaining if any(
                math.dist(p, q) <= cfg["trade_radius"] for q in chosen.values())]
            if tradeable:
                remaining = tradeable
            if not remaining:
                continue
            def score(point):
                coverage = sum(los(grid, point, q, state.get("smoke_cells", ()), chosen.values()) for q in watch)
                crossfire = sum(angle_at(spike, point, q) >= cfg["crossfire_angle"] for q in chosen.values())
                trade = not chosen or any(math.dist(point, q) <= cfg["trade_radius"] for q in chosen.values())
                return (cfg["trade_weight"]*trade + cfg["coverage_weight"]*coverage
                        + cfg["crossfire_weight"]*bool(crossfire)
                        - cfg["travel_weight"]*distances[point]
                        + (math.dist(point, spike) >= cfg["minimum_radius"]), -math.dist(point, spike), point)
            chosen[str(ally.name)] = max(remaining, key=score)
        self.hold_assignments = chosen

    def _flash(self, char, state, targets, target_cells=None):
        if (state.get("defender_setup_active") or getattr(char, "ability_name", None) != "FLASH"
                or getattr(char, "flash_charges", 0) <= 0):
            return None
        from game_core import FLASH_SPEED_CELLS_PER_TICK, FLASH_MAX_FLIGHT_TICKS
        grid, smoke = state["grid"], state.get("smoke_cells", ())
        cells = [pos(c) for c in targets] if target_cells is None else list(target_cells)
        aims = set(cells)
        aims.update((pos(char)[0]+dr, pos(char)[1]+dc)
                    for dr in (-1,0,1) for dc in (-1,0,1) if dr or dc)
        candidates = []
        for aim in sorted(aims):
            if not valid(grid, aim):
                continue
            path = projectile_path(grid, pos(char), aim)
            if len(path) <= 1:
                continue
            index = min(len(path)-1, FLASH_SPEED_CELLS_PER_TICK*FLASH_MAX_FLIGHT_TICKS)
            end = path[index]
            hit = sum(los(grid, cell, end, smoke) for cell in cells)
            if hit:
                candidates.append((hit, -index, aim))
        if not candidates:
            return None
        aim = max(candidates)[2]
        return list(char.pos), {"ability": "FLASH", "target": aim}

    def _smoke(self, char, state, spike, targets):
        if getattr(char, "ability_name", None) != "SMOKE" or getattr(char, "smoke_charges", 0) <= 0:
            return None
        grid = state["grid"]
        watch = watch_cells(grid, spike)
        # Do not block ANY currently assigned hold-to-defuse ray, including
        # cells next to the spike where defusing is also legal.
        protected = {p for hold in self.hold_assignments.values() for q in watch
                     if los(grid, hold, q) for p in line_cells(hold,q)}
        for target in sorted(targets, key=lambda c: (math.dist(pos(c), spike), str(c.name)), reverse=True):
            center = pos(target)
            cells = {(center[0]+dr,center[1]+dc) for dr in (-1,0,1) for dc in (-1,0,1)}
            if (not cells.intersection(protected) and not cells.intersection(watch)
                    and math.dist(center, spike) > self.config["hold"]["defuse_radius"]):
                return list(char.pos), {"ability": "SMOKE", "target": center}
        return None

    def _heal(self, char, state):
        if (state.get("defender_setup_active") or getattr(char,"plant_timer",0)>0
                or getattr(char,"ability_name",None)!="DANCE" or getattr(char,"dance_charges",0)<=0):
            return None
        from game_core import DANCE_MAX_HP
        wounded = []
        for ally in self.allies(char,state):
            if str(ally.name)==str(char.name):
                continue
            cap = min(DANCE_MAX_HP,getattr(ally,"max_hp",DANCE_MAX_HP)) if getattr(ally,"contract_max_hp_lost",0)>0 else DANCE_MAX_HP
            missing = cap-getattr(ally,"hp",cap)
            if missing>=self.config["support"]["dance_min_missing_hp"]:
                wounded.append((missing,str(ally.name)))
        if wounded:
            return list(char.pos),{"ability":"DANCE","target_name":max(wounded)[1]}
        return None

    def postplant(self, char, state):
        heal = self._heal(char,state)
        if heal:
            return heal
        spike = self.planted_position(state)
        if spike is None:
            return None
        grid, cfg = state["grid"], self.config["hold"]
        self._assign_hold(char, state, spike)
        enemies = self.enemies(char, state)
        taps = {name for name, (timer, _) in (state.get("defender_defuse_info") or {}).items() if timer > 0}
        urgent = [c for c in enemies if str(c.name) in taps or math.dist(pos(c), spike) <= cfg["defuse_radius"]]
        responders = sorted(self.allies(char, state),
            key=lambda c: (math.dist(self.ally_position(c), spike), str(c.name)))[:cfg["responders"]]
        responding = bool(taps or urgent) and any(str(c.name) == str(char.name) for c in responders)
        occupied = self.occupied(char, state)
        visible = [c for c in enemies if los(grid, pos(char), pos(c),
            () if getattr(c, "reveal_remaining", 0) > 0 or getattr(char, "sees_through_smoke", False)
            else state.get("smoke_cells", ()), occupied)]
        if taps:
            # Audible defusing constrains the real position to the public
            # defuse cells. IQ-noisy enemy coordinates can lie outside that
            # area and must not turn our guns away from the actual defuser.
            watch = watch_cells(grid,spike)
            flash = self._flash(char,state,[],target_cells=watch)
            if flash:
                return flash
            if responding:
                goals = [p for p in watch if p!=spike and p not in occupied]
                if pos(char) in goals:
                    return self.move_action(char,pos(char),spike)
                path = route(grid,pos(char),goals,occupied,spike,cfg["leash_radius"])
                return self.move_action(char,path[min(1,len(path)-1)],spike)
            if math.dist(pos(char),spike)<=cfg["radius"]:
                return self.move_action(char,pos(char),spike)
        if urgent:
            flash = self._flash(char, state, urgent)
            if flash:
                return flash
        if responding and not any(str(c.name) in taps or c in urgent for c in visible):
            # Route to public defuse cells, never to an unseen defuser coordinate.
            watch = watch_cells(grid, spike)
            goals = [p for p in watch if p not in occupied]
            if urgent:
                goals = [p for p in self._hold_cells(grid, spike, state.get("smoke_cells", ()))
                         if any(los(grid, p, pos(c), state.get("smoke_cells", ()), occupied) for c in urgent)] or goals
            path = route(grid, pos(char), goals, occupied, spike, cfg["leash_radius"])
            step = path[min(1,len(path)-1)]
            return self.move_action(char, step, pos(urgent[0]) if urgent else spike)
        # Hold only on a legal position; do not remain dueling far from the spike.
        if visible and math.dist(pos(char), spike) <= cfg["radius"]:
            target = min(visible, key=lambda c: (str(c.name) not in taps,
                math.dist(pos(c), spike), math.dist(pos(c), pos(char)), str(c.name)))
            return self.move_action(char, pos(char), pos(target))
        smoke = self._smoke(char, state, spike, enemies)
        if smoke:
            return smoke
        goal = self.hold_assignments.get(str(char.name), spike)
        path = route(grid, pos(char), [goal], occupied, spike, cfg["leash_radius"])
        if len(path) == 1 and pos(char) != goal:
            alternatives = [p for p in self._hold_cells(grid, spike, state.get("smoke_cells", ()))
                            if p not in set(self.hold_assignments.values()) - {goal}]
            path = route(grid, pos(char), alternatives, occupied, spike, cfg["leash_radius"])
        return self.move_action(char, path[min(1,len(path)-1)], spike)

    def recon(self, char, state, scouting=False, discipline=False):
        cfg = self.recon_config()
        name = str(char.name)
        if (state.get("defender_setup_active")
                or getattr(char, "ability_name", None) != "RECON" or getattr(char, "recon_charges", 0) <= 0
                or self.tick > cfg["deadline"] or getattr(char, "plant_timer", 0) > 0):
            return None
        axis = self.scout_axis(char,self.allies(char,state))
        waypoint = self.scout_waypoint(char,state,self.allies(char,state))
        if (scouting and math.dist(pos(char),waypoint) > cfg["waypoint_radius"]
                and self.tick < cfg["scout_cast_tick"]):
            if discipline and not any(
                    str(c.name) != name and math.dist(self.ally_position(c),pos(char))
                    <= self.config["entry"]["trade_radius"] for c in self.allies(char,state)):
                return self.move_action(char,pos(char),waypoint)
            path = self.preplant_route(char,state,[waypoint])
            return self.move_action(char,path[min(1,len(path)-1)],waypoint)
        if self.tick-self.last_cast.get(name, -cfg["gap_ticks"]) < cfg["gap_ticks"]:
            return self.move_action(char,pos(char)) if scouting else None
        reference = cfg["secondary_aims" if self.recon_sequences[name] else "aims"][axis]
        aim = recon_aim(state["grid"],pos(char),axis,reference)
        if aim is None:
            return None
        self.last_cast[name] = self.tick
        self.recon_sequences[name] += 1
        from game_core import RECON_SPEED_CELLS_PER_TICK
        path = projectile_path(state["grid"],pos(char),aim)
        impact = path[-1]
        self.last_recon_request[name] = dict(player=name, axis=axis, origin=pos(char), target=aim,
            impact=impact, covers_anchor=max(abs(impact[i]-reference[i]) for i in (0,1)) <= 4,
            ready_tick=self.tick+math.ceil((len(path)-1)/RECON_SPEED_CELLS_PER_TICK))
        return list(char.pos), {"ability":"RECON", "target":aim}

    def choose_site(self, profile):
        counts = self.observed_defenders_by_axis
        if profile.get("preferred_site"):
            self.selected_site = profile["preferred_site"]
            self.site_reason = "profile_preference"
        elif profile.get("confirm_defenders") is not None:
            choices = [axis for axis in ("A","B") if counts[axis] == profile["confirm_defenders"]
                       and self.max_observed_defenders_by_axis[axis] <= profile["confirm_defenders"]]
            if choices:
                self.selected_site = min(choices, key=lambda s:(counts[s], s))
                self.site_reason = "observed_two_defenders"
            elif self.selected_site and max(counts[self.selected_site],self.max_observed_defenders_by_axis[self.selected_site]) > profile["confirm_defenders"]:
                self.selected_site = None
                self.site_reason = "observation_invalidated"
            elif self.selected_site is None and profile.get("complete_site_survey"):
                self.site_reason = "awaiting_two_defenders"
            # No unconfirmed rush after a timeout: retain the information gate.
        elif any(counts.values()) or len(self.surveyed_axes) >= 2:
            limit = profile.get("avoid_defenders", float("inf"))
            choices = [s for s in ("A","B") if max(counts[s],self.max_observed_defenders_by_axis[s]) < limit
                       and (not profile.get("avoid_defenders") or counts[s] > 0 or s in self.surveyed_axes)]
            if choices:
                self.selected_site = min(choices, key=lambda s:(counts[s], s))
                self.site_reason = "recent_sighting_lower_bound"
            elif profile.get("avoid_defenders"):
                self.selected_site = None
                self.site_reason = "no_observed_safe_site"
        elif not profile.get("avoid_defenders"):
            self.selected_site = "A"
            self.site_reason = "unconfirmed_default"
        return self.selected_site

    def preplant(self, char, state, profile, flags):
        self.site_selection_active = True
        allies = self.allies(char,state)
        holder = next((c for c in allies if getattr(c,"has_spike",False)),None)
        if holder is None:
            return None  # Existing uninterrupted dropped-spike recovery.
        heal = self._heal(char,state)
        if heal:
            return heal
        grid = state["grid"]
        awaiting_recon = (profile.get("confirm_defenders") is not None and flags["early_recon"]
            and ((self.tick <= self.config["recon"]["deadline"] and any(
                getattr(c,"ability_name",None)=="RECON" and getattr(c,"recon_charges",0)>0 for c in allies))
                or self.tick < self.recon_ready_tick))
        if awaiting_recon:
            self.selected_site,self.site_reason = None,"awaiting_recon_completion"
            site = None
        else:
            site = self.choose_site(profile)
        side = site or "A"
        plantable = [tuple(map(int,p)) for p in np.argwhere(grid == 2)
                     if (p[1] < grid.shape[1]/2) == (side == "A")]
        if not plantable:
            return None
        holder_pos = self.ally_position(holder)
        distances = distance_map(grid, holder_pos)
        reachable = [p for p in plantable if distances[p] >= 0]
        if not reachable:
            return None
        anchor = self.config["plant_targets"][side]
        target = min(reachable,key=lambda p:(math.dist(p,anchor),distances[p],p))
        self.target_plant_pos = target if site else None
        if (str(char.name) == str(holder.name) and site and pos(char) in plantable
                and (pos(char)==target or getattr(char,"plant_timer",0)>0)):
            cfg = self.config["entry"]
            if flags["entry_discipline"] and not getattr(char,"plant_timer",0):
                supports = sum(str(c.name) != str(char.name)
                    and math.dist(self.ally_position(c),pos(char)) <= cfg["plant_support_radius"]
                    for c in allies)
                if supports < min(cfg["minimum_plant_support"],len(allies)-1):
                    return self.wait_carrier(char,state,target,allies)
            return list(char.pos), "PLANT"
        enemies = self.enemies(char,state)
        in_los = [c for c in enemies if los(grid,pos(char),pos(c),state.get("smoke_cells",()),self.occupied(char,state))]
        if flags["early_recon"] and profile.get("complete_site_survey"):
            cast=self.recon(char,state,scouting=True,
                discipline=flags["entry_discipline"] and not profile.get("allow_solo_bait",False))
            if cast:
                return cast
        if in_los:
            if flags["early_recon"]:
                cast = self.recon(char,state)
                if cast:
                    return cast
            flash_targets = [c for c in in_los if math.dist(pos(c),pos(char))
                             <= self.config["entry"]["flash_before_contact_radius"]]
            flash = self._flash(char,state,flash_targets) if flags["entry_discipline"] else None
            if flash:
                return flash
            nearest = min(in_los,key=lambda c:(math.dist(pos(char),pos(c)),str(c.name)))
            return self.move_action(char,pos(char),pos(nearest))
        if flags["early_recon"]:
            cast = self.recon(char,state,scouting=True,
                discipline=flags["entry_discipline"] and not profile.get("allow_solo_bait",False))
            if cast:
                return cast
        # Pair an escort with each seeker while they scout their own side.
        scouts = [c for c in allies if getattr(c,"ability_name",None) == "RECON"
                  and getattr(c,"recon_charges",0) > 0
                  and self.tick <= self.config["recon"]["scout_cast_tick"]]
        supports = [c for c in allies if c not in scouts and str(c.name) != str(holder.name)]
        if profile.get("complete_site_survey"):
            pairs=[(scout,next((c for c in supports if str(c.name)==self.scout_supports.get(str(scout.name))),None))
                   for scout in scouts]
        else:
            pairs=list(zip(scouts,supports))
        for scout, support in pairs:
            if support is None:
                continue
            if str(support.name) == str(char.name):
                point = self.ally_position(scout)
                waypoint = self.scout_waypoint(scout,state,allies)
                goals = self.escort_goals(state,point,waypoint)
                path = self.preplant_route(char,state,goals)
                return self.move_action(char,path[min(1,len(path)-1)],point)
        cfg = self.config["entry"]
        if str(char.name) == str(holder.name):
            path = self.preplant_route(char,state,[target])
            if site is None:
                # Advance only to staging while recon has not confirmed a site.
                if len(path)-1 <= cfg["staging_distance"]:
                    return self.wait_carrier(char,state,target,allies)
            if flags["entry_discipline"] and not profile.get("allow_solo_bait",False):
                nearby = sum(math.dist(self.ally_position(c),pos(char)) <= cfg["trade_radius"]
                             for c in allies if str(c.name) != str(char.name))
                if nearby < min(cfg["minimum_support"],len(allies)-1):
                    return self.wait_carrier(char,state,target,allies)
                if (cfg["require_utility"] and not self.utility_used
                        and len(path)-1 <= cfg["staging_distance"]):
                    return self.wait_carrier(char,state,target,allies)
            return self.move_action(char,path[min(1,len(path)-1)],target)
        # Escort the carrier in distinct adjacent cells; never leave a solo bait.
        if site and profile.get("carrier_route_priority"):
            clearing=self.clear_carrier_route(char,state,holder,target,allies)
            if clearing is not None:
                return clearing
        goals = self.escort_goals(state,holder_pos,target)
        if site and profile.get("carrier_route_priority"):
            reserved=route(grid,holder_pos,[target])[1:4]
            path=self.preplant_route(char,state,goals,reserved)
        else:
            path = self.preplant_route(char,state,goals)
        return self.move_action(char,path[min(1,len(path)-1)],target)

    def discipline(self, char, state, profile, result):
        """Entry flag also works when the independent profile flag is disabled."""
        if profile.get("allow_solo_bait") or state.get("is_planted"):
            return result
        allies = self.allies(char,state)
        cfg = self.config["entry"]
        enemies = self.enemies(char,state)
        contacts = [c for c in enemies if math.dist(pos(c),pos(char)) <= cfg["contact_radius"]]
        has_support = any(str(c.name) != str(char.name)
                          and math.dist(self.ally_position(c),pos(char)) <= cfg["trade_radius"] for c in allies)
        # A combat hold still faces the enemy; it does not make an unsupported
        # forward move into first contact.
        if contacts and not has_support and len(allies) > 1:
            return self.move_action(char,pos(char),pos(contacts[0]))
        if contacts and cfg["require_utility"] and not self.utility_used:
            flash = self._flash(char,state,contacts)
            return flash or self.move_action(char,pos(char),pos(contacts[0]))
        return result

    def snapshot(self):
        return dict(schema_version=1, opponent=self.opponent,
            observed_defenders_by_axis=dict(self.observed_defenders_by_axis),
            max_observed_defenders_by_axis=dict(self.max_observed_defenders_by_axis),
            selected_site=self.selected_site, site_reason=self.site_reason,
            site_selection_active=self.site_selection_active,
            target_plant_pos=list(self.target_plant_pos) if self.target_plant_pos is not None else None,
            surveyed_axes=sorted(self.surveyed_axes),
            hold_assignments={name:list(point) for name,point in self.hold_assignments.items()},
            spike=list(self.spike) if self.spike is not None else None,
            utility_used=self.utility_used, events=list(self.events))
