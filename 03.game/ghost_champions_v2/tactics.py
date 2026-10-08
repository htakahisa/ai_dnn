"""Rule layer using only public state and teammates' own observations.

This object never reads game.chars, replay frames, analytics, or real_character.
Enemy axis counts are recent sightings, not an omniscient setup count.
"""
from __future__ import annotations
from collections import Counter
import math
import random
import numpy as np
from grid_paths import distance_map
from grid_lines import line_cells
from gc_v1.gc_facing import facing_towards
from .config import active_flags
from .hazards import PublicHazards
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
        # Labels are observation metadata; they never select tactical settings.
        self.roster_labels = {}
        for name in ("Touyama Gaming", "Omoko Gaming", "Furina Classic",
                     "Fnatic2023", "Gorigons", "SUPES"):
            preset = get_preset(name)
            self.roster_labels[tuple(sorted(map(str, preset.players)))] = preset.short_name
        # Validate the stage even before the first live decision.
        active_flags(config, stage)
        self.reset_round()

    def reset_round(self):
        self.hazards = PublicHazards()
        self.hazard_escape = {}
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
        self.plant_commit = None
        self.spike = None
        self.hold_assignments = {}
        self.assignment_key = None
        self.selected_site = None
        self.initial_site = None
        self.opening_strategy = None
        self.split_players = set()
        self.split_progress = {}
        self.opening_roles_assigned = False
        self.target_plant_pos = None
        self.site_selection_active = False
        self.site_reason = "not_selected"
        self.observed_defenders_by_axis = dict.fromkeys(("A", "Mid", "B"), 0)
        self.max_observed_defenders_by_axis = dict.fromkeys(("A", "Mid", "B"), 0)
        self.events = []

    def observe(self, char, state):
        self.hazards.update(state)
        self.tick = int(state.get("battle_tick", 0))
        grid = state["grid"]
        self.own_positions[str(char.name)] = (pos(char), self.tick)
        roster = state.get("enemy_roster", ())
        names = tuple(sorted(str(e.get("base_name") or e.get("name")) for e in roster))
        if names in self.roster_labels:
            self.opponent = self.roster_labels[names]
        if self.config["team_tactics"]["complete_site_survey"] and not self.scout_assignments:
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
        return self.config["team_tactics"], active_flags(self.config, self.stage)

    def dodge(self, char, state):
        self.hazard_escape.pop(str(char.name),None)
        if state.get("defender_setup_active"):
            return None
        # A flash alone should not cancel an almost finished plant. Damaging
        # area warnings can interrupt it; the ordinary plant commitment stays.
        if getattr(char,"plant_timer",0)>0 and max(self.hazards.channels(pos(char))[:2])<.8:
            return None
        destination=self.hazards.escape(char,self.occupied(char,state))
        if destination is None:
            return None
        self.hazard_escape[str(char.name)]=self.tick
        return self.move_action(char,destination)

    def avoid_entry(self, char, state, result):
        destination=result[0] if isinstance(result,tuple) else result
        current=pos(char)
        if (not char.is_alive or state.get("defender_setup_active") or not self.hazards.effects
                or tuple(destination)==current or self.hazards.risk(destination)<.35
                or self.hazards.risk(destination)<=self.hazards.risk(current)):
            return result
        occupied=self.occupied(char,state)
        choices=[current,*[(current[0]+dr,current[1]+dc) for dr,dc in CARDINAL]]
        choices=[p for p in choices if valid(state["grid"],p) and p not in occupied]
        safe=min(choices,key=lambda p:(self.hazards.risk(p),math.dist(p,destination)))
        self.hazard_escape[str(char.name)]=self.tick
        return self.move_action(char,safe)

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
        if self.opening_strategy in ("RUSH","SPLIT"):
            if (self.opening_strategy=="SPLIT" and str(char.name) in self.split_players
                    and self.recon_sequences[str(char.name)]==0):
                return "Mid"
            return self.selected_site or self.initial_site or "A"
        seekers = [c for c in allies if getattr(c,"ability_name",None)=="RECON"]
        index = self.scout_assignments.get(str(char.name))
        if index is None:
            index = next(i for i,c in enumerate(seekers) if str(c.name)==str(char.name))
        return "A" if index%2==0 else ("Mid" if self.recon_sequences[str(char.name)]==0 else "B")

    def recon_config(self):
        return self.config["recon"]

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
        self.hazards.note_action(char,result,self.tick)
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
            cap = min(DANCE_MAX_HP,getattr(ally,"max_hp",DANCE_MAX_HP)) if (getattr(ally,"contract_max_hp_lost",0)>0 or getattr(ally,"fate_max_hp_lost",0)>0) else DANCE_MAX_HP
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

    def choose_opening(self):
        if self.opening_strategy is None:
            weights=self.config["opening"]["weights"]
            self.opening_strategy=random.choices(tuple(weights),weights=tuple(weights.values()),k=1)[0]
        return self.opening_strategy

    def assign_opening_roles(self, holder, allies):
        if self.opening_roles_assigned:
            return
        self.opening_roles_assigned=True
        if self.opening_strategy!="SPLIT":
            return
        seekers=[c for c in allies if getattr(c,"ability_name",None)=="RECON" and c is not holder]
        if not seekers:
            return
        flank=seekers[-1]
        escort=next((c for c in allies if str(c.name)==self.scout_supports.get(str(flank.name))
                     and c is not holder),None)
        if escort is None:
            escort=next((c for c in allies if c is not holder and c not in seekers),None)
        # A split requires a pair; never send an unsupported solo player.
        if escort is not None:
            self.split_players={str(flank.name),str(escort.name)}

    def split_move(self, char, state, target):
        name=str(char.name)
        if name not in self.split_players:
            return None
        allies=self.allies(char,state)
        partners=[c for c in allies if str(c.name) in self.split_players and str(c.name)!=name]
        if not partners:
            return None  # Rejoin the carrier if the flank partner is lost.
        points=[nearest_floor(state["grid"],self.config["opening"]["split_waypoint"]),
                nearest_floor(state["grid"],self.config["opening"]["split_entries"][self.selected_site])]
        progress=self.split_progress.get(name,0)
        while progress<len(points) and pos(char)==points[progress]:
            progress+=1
        self.split_progress[name]=progress
        if progress>=len(points):
            return None
        if not any(math.dist(self.ally_position(c),pos(char))<=self.config["entry"]["trade_radius"]
                   for c in partners):
            partner=min(partners,key=lambda c:math.dist(self.ally_position(c),pos(char)))
            # Catch up to a partner already ahead; otherwise wait for them.
            own_path=route(state["grid"],pos(char),[points[progress]])
            partner_point=self.ally_position(partner)
            if partner_point not in own_path[1:]:
                return self.move_action(char,pos(char),partner_point)
        path=self.preplant_route(char,state,[points[progress]])
        # An escort must yield the single waypoint cell to its partner.
        if len(path)>1 and path[1] in self.occupied(char,state):
            return self.wait_carrier(char,state,target,partners)
        return self.move_action(char,path[min(1,len(path)-1)],target)

    def choose_site(self, available=("A", "B")):
        """Share one initial draw per round, then compare observed sites."""
        if not available:
            return None
        self.choose_opening()
        if self.initial_site is None:
            self.initial_site = random.choice(available)
        if self.selected_site not in available:
            self.selected_site = self.initial_site if self.initial_site in available else available[0]
            self.site_reason = "initial_random" if len(available)>1 else "only_reachable_site"
        counts = self.observed_defenders_by_axis
        # An unseen site is unknown, rather than an observed empty site.
        if len(available)>1 and all(counts[s]>0 or s in self.surveyed_axes for s in available):
            least = min(counts[s] for s in available)
            if counts[self.selected_site] > least:
                self.selected_site = next(s for s in available if counts[s]==least)
            self.site_reason = "observed_lower_count" if len({counts[s] for s in available})>1 else "equal_keep"
        return self.selected_site

    def objective_action(self, char, state, holder, cells, distances):
        """Commit to a legal plant using public danger, cover and remaining time."""
        from game_core import PLANT_REQUIRED_TICKS
        holder_pos=self.ally_position(holder)
        cfg=self.config["entry"]
        name=str(holder.name)
        if self.plant_commit and self.plant_commit["holder"]!=name:
            self.plant_commit=None
        reason=None
        plant_target=holder_pos
        if getattr(holder,"plant_timer",0)>0 and holder_pos in cells:
            reason="continue_plant"
        elif holder_pos in cells:
            threats=[c for c in self.enemies(holder,state) if los(
                state["grid"],holder_pos,pos(c),state.get("smoke_cells",()),self.occupied(holder,state))]
            allies=self.allies(holder,state)
            enemy_alive=sum(c.is_alive for c in state.get("chars",()) if c.team!=holder.team)
            support=sum(str(c.name)!=name and math.dist(self.ally_position(c),holder_pos)
                        <=cfg["plant_support_radius"] for c in allies)
            if not threats:
                reason="plant_no_observed_threat"
                # With ample time, allow only a short, unobstructed adjustment
                # to the existing hold position. Never wait there for the team.
                if len(allies)<=enemy_alive:
                    side="A" if holder_pos[1]<state["grid"].shape[1]/2 else "B"
                    local=[p for p in cells if (p[1]<state["grid"].shape[1]/2)==(side=="A")]
                    preferred=min(local,key=lambda p:(math.dist(p,self.config["plant_targets"][side]),distances[p],p))
                    path=route(state["grid"],holder_pos,[preferred],self.occupied(holder,state))
                    if (1<len(path)<=cfg["plant_reposition_ticks"]+1
                            and all(p in cells for p in path)
                            and state.get("round_timer",float("inf"))>
                                len(path)-1+PLANT_REQUIRED_TICKS+cfg["plant_deadline_margin"]):
                        plant_target=preferred
                        reason="plant_nearby_position"
            elif len(allies)>enemy_alive and support>=cfg["minimum_plant_support"]:
                reason="plant_advantage_with_cover"
        nearest=min(cells,key=lambda p:(distances[p],p))
        if reason:
            self.plant_commit=dict(holder=name,target=plant_target,reason=reason)
        elif (state.get("round_timer",float("inf"))
              <=distances[nearest]+PLANT_REQUIRED_TICKS+cfg["plant_deadline_margin"]):
            self.plant_commit=dict(holder=name,target=nearest,reason="plant_deadline")
        if self.plant_commit is None:
            return None
        target=self.plant_commit["target"]
        if target not in cells:
            target=self.plant_commit["target"]=nearest
        if self.plant_commit["reason"]=="plant_nearby_position" and holder_pos in cells:
            adjustment=route(state["grid"],holder_pos,[target],self.occupied(holder,state))
            if target!=holder_pos and (len(adjustment)==1
                    or len(adjustment)>cfg["plant_reposition_ticks"]+1
                    or not all(p in cells for p in adjustment)):
                target=self.plant_commit["target"]=holder_pos
                self.plant_commit["reason"]="plant_position_blocked"
        # Once committed, a new sighting cannot send the carrier across the map.
        self.selected_site="A" if target[1]<state["grid"].shape[1]/2 else "B"
        self.target_plant_pos=target
        self.site_reason=self.plant_commit["reason"]
        if str(char.name)!=name:
            return None
        if pos(char) in cells and (self.plant_commit["reason"]!="plant_nearby_position" or pos(char)==target):
            self.plant_commit["target"]=pos(char)
            self.target_plant_pos=pos(char)
            return list(char.pos),"PLANT"
        path=self.preplant_route(char,state,[target])
        return self.move_action(char,path[min(1,len(path)-1)],target)

    def preplant(self, char, state, settings, flags):
        if state.get("defender_setup_active"):
            return None
        self.site_selection_active = True
        allies = self.allies(char,state)
        holder = next((c for c in allies if getattr(c,"has_spike",False)),None)
        if holder is None:
            self.plant_commit=None
            return None  # Existing uninterrupted dropped-spike recovery.
        grid = state["grid"]
        holder_pos = self.ally_position(holder)
        distances = distance_map(grid, holder_pos)
        cells = [tuple(map(int,p)) for p in np.argwhere(grid == 2) if distances[tuple(p)]>=0]
        available = tuple(s for s in ("A","B") if any(
            (p[1]<grid.shape[1]/2)==(s=="A") for p in cells))
        if not available:
            return None
        objective=self.objective_action(char,state,holder,cells,distances)
        if objective is not None:
            return objective
        heal = self._heal(char,state)
        if heal:
            return heal
        committed=self.plant_commit is not None
        if committed:
            site=self.selected_site
        else:
            site = self.choose_site(available)
        self.choose_opening()
        self.assign_opening_roles(holder,allies)
        side = site
        plantable = [p for p in cells if (p[1]<grid.shape[1]/2)==(side=="A")]
        anchor = self.config["plant_targets"][side]
        target = (self.plant_commit["target"] if committed else
                  min(plantable,key=lambda p:(math.dist(p,anchor),distances[p],p)))
        self.target_plant_pos = target
        enemies = self.enemies(char,state)
        in_los = [c for c in enemies if los(grid,pos(char),pos(c),state.get("smoke_cells",()),self.occupied(char,state))]
        scouting=self.opening_strategy=="DEFAULT" and not committed
        if flags["early_recon"] and scouting and settings["complete_site_survey"]:
            cast=self.recon(char,state,scouting=True,
                discipline=flags["entry_discipline"])
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
        near_entry=len(route(grid,pos(char),[target]))-1<=self.config["opening"]["utility_distance"]
        split_ready=self.split_progress.get(str(char.name),0)>0
        if flags["early_recon"] and (scouting or near_entry or split_ready):
            cast = self.recon(char,state,scouting=scouting,
                discipline=flags["entry_discipline"])
            if cast:
                return cast
        if self.opening_strategy=="SPLIT" and not committed:
            flank_move=self.split_move(char,state,target)
            if flank_move is not None:
                return flank_move
        # Pair an escort with each seeker while they scout their own side.
        scouts = [c for c in allies if scouting and getattr(c,"ability_name",None) == "RECON"
                  and getattr(c,"recon_charges",0) > 0
                  and self.tick <= self.config["recon"]["scout_cast_tick"]]
        supports = [c for c in allies if c not in scouts and str(c.name) != str(holder.name)]
        if settings["complete_site_survey"]:
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
            if flags["entry_discipline"]:
                nearby = sum(math.dist(self.ally_position(c),pos(char)) <= cfg["trade_radius"]
                             for c in allies if str(c.name) != str(char.name))
                if nearby < min(cfg["minimum_support"],len(allies)-1):
                    return self.wait_carrier(char,state,target,allies)
                if (cfg["require_utility"] and not self.utility_used
                        and len(path)-1 <= cfg["staging_distance"]):
                    return self.wait_carrier(char,state,target,allies)
            return self.move_action(char,path[min(1,len(path)-1)],target)
        # Escort the carrier in distinct adjacent cells; never leave a solo bait.
        if settings["carrier_route_priority"]:
            clearing=self.clear_carrier_route(char,state,holder,target,allies)
            if clearing is not None:
                return clearing
        goals = self.escort_goals(state,holder_pos,target)
        if settings["carrier_route_priority"]:
            reserved=route(grid,holder_pos,[target])[1:4]
            path=self.preplant_route(char,state,goals,reserved)
        else:
            path = self.preplant_route(char,state,goals)
        return self.move_action(char,path[min(1,len(path)-1)],target)

    def discipline(self, char, state, result):
        """Apply the same entry discipline with or without site selection."""
        if state.get("is_planted"):
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
        return dict(schema_version=2, opponent=self.opponent,
            public_effects=[dict(kind=e.kind,phase=e.phase,position=e.position,cells=e.cells,
                                 direction=e.direction) for e in self.hazards.effects],
            hazard_escape=sorted(name for name,tick in self.hazard_escape.items() if tick==getattr(self,"tick",0)),
            opening_strategy=self.opening_strategy, split_players=sorted(self.split_players),
            split_progress=dict(self.split_progress),
            plant_commit=({**self.plant_commit,"target":list(self.plant_commit["target"])}
                          if self.plant_commit else None),
            observed_defenders_by_axis=dict(self.observed_defenders_by_axis),
            max_observed_defenders_by_axis=dict(self.max_observed_defenders_by_axis),
            initial_site=self.initial_site, selected_site=self.selected_site, site_reason=self.site_reason,
            site_selection_active=self.site_selection_active,
            target_plant_pos=list(self.target_plant_pos) if self.target_plant_pos is not None else None,
            surveyed_axes=sorted(self.surveyed_axes),
            hold_assignments={name:list(point) for name,point in self.hold_assignments.items()},
            spike=list(self.spike) if self.spike is not None else None,
            utility_used=self.utility_used, events=list(self.events))
