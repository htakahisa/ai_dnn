"""Fixed public-information executor and real-engine analysis data collection."""
import contextlib
from collections import deque
import os

import numpy as np

from frc_v1.perception import FrcPerceptionBuilder
from frc_v1.actions import build_masks, KINDS
from toruAI_v4.tv4_observer import facing
from toruAI_v4.tv4_scenario import OPPONENTS
from toruAI_v4.tv4_learn_attacker_analysis import (
    AttackerEncoder, Route, ENTRY_MODES, candidate_routes, branch_sequence, shortest_path,
)
from toruAI_v4.tv4_train_defender_analysis import legacy_root, seed_all, relocate_debug_logs
from toruAI_v4.tv4_attacker_entry_utility import predicted_entry_utility, projectile_impact, flash_entry_step, SupportUseTracker
from toruAI_v4.tv4_attacker_route_planner import AdaptiveAttackPlanner
from toruAI_v4.tv4_attacker_combat_audit import CombatAudit
from toruAI_v4.tv4_attacker_combat import AttackerCombatCoach, effective_utility, plant_ready

COMBAT_DIAGNOSTICS = True  # Diagnostic labels are never used by the action controller.
COMBAT_POLICY_ENABLED = True


class AnalysisCollectorController:
    """Data executor, not the future learned attacker policy.

    Routes are provisional and reassessed from public observations. Labels
    describe the selected route followed by adaptive continuation. Arrived
    escorts clear the route and move deeper into the site without returning to
    its entrance. Spike recovery is an explicit exception to forward travel.
    """
    handles_team_perception = True

    def __init__(self, scenario, model, rng, exploration=1.):
        self.scenario, self.model, self.rng = scenario, model, rng
        self.exploration = exploration
        self.combat_enabled = COMBAT_POLICY_ENABLED
        self.sensor = FrcPerceptionBuilder("A")
        self.encoder = AttackerEncoder(scenario)
        self.previous_rounds = []
        self.game = None
        self.reset_round()

    def set_game(self, game):
        self.game = getattr(game, "_real", game)

    def reset_round(self):
        self.sensor.reset()
        self.encoder.reset()
        self.route = self.mode = self.snapshot = None
        self.cache = None
        self.actions, self.cursors, self.connectors = {}, {}, {}
        self.frames, self.events = [], []
        self.retriever = None
        self.parked = set()
        self.site_depth = {}
        self.entry_wait_until = 0
        self.support_uses = SupportUseTracker()
        self.entry_flash_impact = None
        self.route_planner = AdaptiveAttackPlanner()
        self.attack_plan = None
        self.combat_coach = AttackerCombatCoach(self.scenario)

    def _select(self, snapshot, observation):
        self.route_planner.enable_flanks = self.combat_enabled
        plan = self.route_planner.update(snapshot, observation, self.encoder, self.model, self.route, self.mode,
                                         rng=self.rng, exploration=self.exploration)
        self.attack_plan = plan
        if plan is None:
            self.route = self.mode = None
            return
        if plan.changed:
            self.route, self.mode = plan.route, plan.mode
            self.cursors, self.connectors, self.site_depth = {}, {}, {}
            self.parked.clear()
            self.events.append({"tick": snapshot.tick, "type": "select", "site": self.route.site,
                                "route": self.route.key, "branches": self.route.branches, "mode": self.mode,
                                "phase": plan.phase, "reason": plan.reason})

    def _remaining(self, holder):
        index = self.cursors.get(holder.slot, 0) if holder else 0
        if holder and holder.position in self.route.cells[index:]:
            index = self.route.cells.index(holder.position, index)
            self.cursors[holder.slot] = index
        return Route(self.route.site, self.route.cells[index:],
                     branch_sequence(self.scenario, self.route.cells[index:]))

    def _utility(self, ally, snapshot, masks):
        if not masks.kind[ally.slot, KINDS.index("ABILITY")]:
            return None
        if self.mode != "supported" and not (self.combat_enabled and self.combat_coach.contacts(snapshot, ally, ally.position)):
            return None
        if any(e.kind == ally.ability_name and e.phase in ("flight", "active") for e in snapshot.effects):
            return None
        sightings = [s.position for s in snapshot.sightings]
        targets = []
        if ally.ability_name == "RECON" and not sightings:
            if self.combat_enabled:
                return None  # Unseen entry recon uses the belief/impact planner, not the route endpoint.
            targets = [self.route.cells[-1]]
            if abs(ally.position[0] - targets[0][0]) + abs(ally.position[1] - targets[0][1]) > 10:
                return None
        elif ally.ability_name == "SMOKE" and len(sightings) >= (1 if self.combat_enabled else 2):
            targets = sightings
        elif ally.ability_name in ("FLASH", "ASH") and sightings:
            targets = sorted(sightings, key=lambda p: abs(ally.position[0] - p[0]) + abs(ally.position[1] - p[1]))
            if ally.ability_name == "FLASH" and max(abs(ally.position[0] - targets[0][0]),
                                                       abs(ally.position[1] - targets[0][1])) > 5:
                return None
        elif ally.ability_name == "DANCE":
            hurt = [a for a in snapshot.allies if a.alive and masks.target[ally.slot, 0, a.slot]]
            if hurt:
                target = min(hurt, key=lambda a: a.hp)
                return {"ability": "DANCE", "target_name": target.name}
        width = self.scenario.grid.shape[1]
        target = next((p for p in targets if masks.target[ally.slot, 0, p[0] * width + p[1]]
                       and (not self.combat_enabled or effective_utility(self.scenario, snapshot, ally, {"ability": ally.ability_name, "target": p}))
                       and self.support_uses.available(ally.ability_name, snapshot, p)), None)
        return {"ability": ally.ability_name, "target": target} if target else None

    def _site_clearance_step(self, origin, blocked):
        """Clear the approach, then push inward so the entrance stays usable.

        Depth uses only static geometry. Each move increases depth, excluding
        every route cell, so an escort cannot park on the carrier's route or
        oscillate back into it. Occupancy and reservations are public allies.
        """
        if not self.site_depth:
            cells = set(self.scenario.sites[self.route.site])
            terminal = self.route.cells[-1]
            queue = deque([terminal])
            self.site_depth = {terminal: 0}
            while queue:
                current = queue.popleft()
                for nxt in self.scenario.neighbors(current):
                    if nxt in cells and nxt not in self.site_depth:
                        self.site_depth[nxt] = self.site_depth[current] + 1
                        queue.append(nxt)
        depth = self.site_depth.get(origin, -1)
        route_cells = set(self.route.cells)
        candidates = [p for p in self.scenario.neighbors(origin)
                      if p not in blocked and p not in route_cells
                      and self.site_depth.get(p, -1) > depth]
        return min(candidates, key=lambda p: (-self.site_depth[p], p)) if candidates else origin

    def prepare_team_tick(self):
        snapshot = self.sensor.build(self.game)
        if self.cache == snapshot.key:
            return
        self.cache, self.snapshot = snapshot.key, snapshot
        self.actions = {}
        if snapshot.phase == "setup" or snapshot.is_planted:
            return  # No postplant guard is trained or executed.
        observation = self.encoder.observe(snapshot, self.previous_rounds)
        self._select(snapshot, observation)
        if self.route is None:
            return
        holder = next((a for a in snapshot.allies if a.alive and a.has_spike), None)
        remaining = self._remaining(holder)
        belief = self.model.analyze(self.encoder, snapshot, observation, [remaining])
        self.frames.append({"tick": snapshot.tick, "observation": observation,
            "route_features": self.encoder.route_features(snapshot, remaining, self.mode, belief["placement"]),
            "decision_id": self.route_planner.decision_id, "site": self.route.site,
            "phase": self.attack_plan.phase,
            "alive": sum(a.alive for a in snapshot.allies)})
        if snapshot.spike_dropped is not None and holder is None:
            alive = [a for a in snapshot.allies if a.alive]
            current = next((a for a in alive if a.slot == self.retriever), None)
            if current is None and alive:
                current = min(alive, key=lambda a: (len(shortest_path(self.scenario, a.position,
                                                  (snapshot.spike_dropped,))) or 9999, a.slot))
            self.retriever = current.slot if current else None
        else:
            self.retriever = None
        masks = build_masks(snapshot) if self.mode == "supported" or self.combat_enabled else None
        # Select one support cast; other players keep making forward progress.
        # Never use true defender coordinates: analyze() receives the public DTO.
        entry_cast = None
        if masks and self.mode == "supported" and snapshot.tick >= self.entry_wait_until:
            priority = {"FLASH": 0, "SMOKE": 1, "RECON": 2, "ASH": 3}
            for ally in sorted(snapshot.allies, key=lambda a: (a.has_spike, priority.get(a.ability_name, 4), a.slot)):
                if not ally.alive:
                    continue
                if ally.has_spike and self.scenario.grid[ally.position] == 2:
                    continue
                plan = predicted_entry_utility(self.scenario, snapshot, ally, belief, remaining, masks)
                if plan and self.support_uses.available(ally.ability_name, snapshot, plan[0]["target"]):
                    utility, delay = plan
                    entry_cast = ally.slot, utility
                    self.entry_wait_until = snapshot.tick + delay if utility["ability"] == "FLASH" else snapshot.tick
                    self.entry_flash_impact = projectile_impact(self.scenario, ally.position, utility["target"], "FLASH")[0] if utility["ability"] == "FLASH" else None
                    self.support_uses.record(snapshot, utility)
                    self.events.append({"tick": snapshot.tick, "type": "predicted_entry_utility",
                                        "slot": ally.slot, "from": ally.position, **utility,
                                        "flash_impact": self.entry_flash_impact, "wait_until": self.entry_wait_until})
                    break
        occupied = {a.position for a in snapshot.allies if a.alive}
        reserved = set()
        ordered = sorted((a for a in snapshot.allies if a.alive),
                         key=lambda a: (not a.has_spike, a.slot != self.retriever, a.slot))
        for ally in ordered:
            origin = ally.position
            blocked = (occupied - {origin}) | reserved
            if ally.has_spike and self.scenario.grid[origin] == 2 and (
                    not self.combat_enabled or self.scenario.site_of(origin) == self.route.site):
                advice = self.combat_coach.advise(snapshot, ally, origin, self.route.cells[-1], self.encoder.history.tracks)
                if not self.combat_enabled or plant_ready(ally, advice):
                    self.actions[ally.name] = (list(origin), "PLANT")
                    reserved.add(origin)
                    continue
            if entry_cast and ally.slot == entry_cast[0]:
                utility = dict(entry_cast[1])
                if self.combat_enabled:
                    utility["facing"] = ally.facing if ally.forced_facing else facing(origin, utility["target"])
                self.actions[ally.name] = (list(origin), utility)
                reserved.add(origin)
                continue
            scout_goal = self.attack_plan.scout_goals.get(ally.slot)
            if scout_goal is None and not ally.has_spike and ally.slot != self.retriever and origin in self.scenario.sites[self.route.site]:
                destination = self._site_clearance_step(origin, blocked)
                # Traffic clearance takes priority over utility at the entrance.
                if destination != origin or origin in self.route.cells:
                    if destination != origin:
                        self.parked.add(ally.slot)
                        self.events.append({"tick": snapshot.tick, "type": "site_clearance",
                                            "slot": ally.slot, "from": origin, "to": destination})
                    look = snapshot.sightings[0].position if snapshot.sightings else self.route.cells[-1]
                    if self.combat_enabled:
                        advice = self.combat_coach.advise(snapshot, ally, destination, look, self.encoder.history.tracks)
                        if advice.position not in blocked and not (advice.position == origin and origin in self.route.cells):
                            destination = advice.position
                        self.actions[ally.name] = (list(destination), {"facing": advice.facing})
                        reserved.add(destination)
                        continue
                    self.actions[ally.name] = (list(destination), {"facing": facing(origin, look)})
                    reserved.add(destination)
                    continue
                # Already safely off the route; do not reconnect to its end.
                self.parked.add(ally.slot)
            utility = self._utility(ally, snapshot, masks) if masks else None
            if utility:
                self.support_uses.record(snapshot, utility)
                if self.combat_enabled and "target" in utility:
                    utility["facing"] = ally.facing if ally.forced_facing else facing(origin, utility["target"])
                self.actions[ally.name] = (list(origin), utility)
                reserved.add(origin)
                continue
            path = self.route.cells
            cursor = self.cursors.get(ally.slot, 0)
            if origin in path[cursor:]:
                cursor = path.index(origin, cursor)
                self.cursors[ally.slot] = cursor
                self.connectors.pop(ally.slot, None)
            destination = origin
            if ally.slot == self.retriever:
                self.parked.discard(ally.slot)
                recovery = shortest_path(self.scenario, origin, (snapshot.spike_dropped,), blocked)
                destination = recovery[1] if len(recovery) > 1 else origin
                # A recovered carrier may reconnect to the current forward suffix.
                self.connectors.pop(ally.slot, None)
            elif scout_goal is not None and not ally.has_spike:
                site_block = set()
                if ally.slot in self.route_planner.flank_entries and self.scenario.grid[scout_goal] != 2:
                    site_block = set(self.scenario.sites["L"]) | set(self.scenario.sites["R"])
                scouting = shortest_path(self.scenario, origin, (scout_goal,), blocked | site_block)
                destination = scouting[1] if len(scouting) > 1 else origin
            elif ally.slot in self.parked and not ally.has_spike:
                destination = origin
            elif origin not in path[cursor:]:
                connector = self.connectors.get(ally.slot)
                if connector is None or origin not in connector:
                    connector = shortest_path(self.scenario, origin, path[cursor:], blocked)
                    self.connectors[ally.slot] = connector
                if origin in connector:
                    i = connector.index(origin)
                    if i + 1 < len(connector) and connector[i + 1] not in blocked:
                        destination = connector[i + 1]
            elif cursor + 1 < len(path) and path[cursor + 1] not in blocked:
                destination = path[cursor + 1]
            look = snapshot.sightings[0].position if snapshot.sightings else scout_goal or path[min(cursor + 4, len(path) - 1)]
            combat_facing = None
            if self.combat_enabled:
                hint = self.combat_coach.preaim_hint(snapshot, ally, destination, scout_goal or path[-1], belief)
                advice = self.combat_coach.advise(snapshot, ally, destination, hint, self.encoder.history.tracks)
                if advice.position not in blocked:
                    destination = advice.position
                combat_facing = advice.facing
                if advice.reason != "advance":
                    self.events.append({"tick": snapshot.tick, "type": "combat", "slot": ally.slot,
                        "reason": advice.reason, "contacts": advice.contacts, "supporters": advice.supporters})
            if snapshot.tick < self.entry_wait_until and self.entry_flash_impact is not None:
                normal_destination = destination
                destination = flash_entry_step(self.scenario, origin, destination, self.entry_flash_impact,
                                               snapshot.smoke_cells, blocked)
                if destination != normal_destination:
                    self.events.append({"tick": snapshot.tick, "type": "entry_cover_wait" if destination == origin else "entry_cover_move",
                                        "slot": ally.slot, "from": origin, "to": destination})
            self.actions[ally.name] = (list(destination), {"facing": combat_facing or facing(origin if not self.combat_enabled else destination, look)})
            reserved.add(destination)

    def decide_move(self, char, state):
        self.prepare_team_tick()
        name = str(getattr(char, "base_name", char.name))
        return self.actions.get(name, (list(char.pos), {"facing": char.facing}))


def play_block(opponent, scenario, model, config, seed, exploration, *, preset_name):
    """Twelve rounds preserve defender history. Hidden coordinates are labels only."""
    from simulation_runtime import cpu_inference
    results, samples, history = [], [], []
    with legacy_root(), open(os.devnull, "w", encoding="utf-8") as raw, \
            contextlib.redirect_stdout(raw), contextlib.redirect_stderr(raw), cpu_inference(enabled=True):
        from controllers import DefaultDefenderController
        from team_ai import DualRoleTeamAI
        from party_presets import get_preset
        from run_game import VisualFPSBattle, _build_team_ai
        seed_all(seed)
        attacker = get_preset(preset_name)
        defender = get_preset(OPPONENTS[opponent][1])
        if set(attacker.players) & set(defender.players):
            raise ValueError("Attacker and defender rosters must have distinct player names")
        controller = AnalysisCollectorController(scenario, model, np.random.default_rng(seed), exploration)
        controller.combat_enabled = config.get("combat_policy_enabled", COMBAT_POLICY_ENABLED)
        team = DualRoleTeamAI("Toru v4 analysis collector", lambda: controller, DefaultDefenderController)
        game = VisualFPSBattle(scenario.maze, team, _build_team_ai(OPPONENTS[opponent][0], device="cpu"), headless=True,
            attacker_roster=list(attacker.players), defender_roster=list(defender.players),
            spike_holder_name=attacker.spike_holder, defender_spike_holder_name=defender.spike_holder,
            attacker_igl_name=attacker.igl, defender_igl_name=defender.igl,
            attacker_team_name=attacker.name, defender_team_name=defender.name, disable_side_swap=True)
        game.stop_after_round, game.analytics_tracker = True, None
        game._record_replay_frame = lambda: None
        relocate_debug_logs(game.defender_controller, None)
        for round_number in range(1, 13):
            audit = CombatAudit() if config.get("combat_diagnostics", COMBAT_DIAGNOSTICS) else None
            controller.previous_rounds = list(history)
            before_wins = game.attacker_wins
            own = [c for c in game.chars if c.team == "A"]
            enemies = [c for c in game.chars if c.team == "D"]
            initial_alive = sum(c.is_alive for c in own)
            initial_team_hp = float(sum(c.max_hp for c in own if c.is_alive))
            initial_enemy_alive = sum(c.is_alive for c in enemies)
            hp = np.asarray([max(0., c.hp) if c.is_alive else 0. for c in own])
            charges = np.asarray([getattr(c, c.ability_name.lower() + "_charges", 0) for c in own])
            initial_abilities = int(charges.sum())
            damage = uses = 0.
            terminal = None
            steps = 0
            while not game.round_over and not game.match_over:
                if not game.defender_setup_phase.active and not game.is_planted:
                    controller.prepare_team_tick()
                    if controller.frames and "placement" not in controller.frames[-1]:
                        frame = controller.frames[-1]
                        # Read after public features/actions are frozen; never fed back.
                        frame["placement"] = np.asarray([int(scenario.region[tuple(c.pos)]) if c.is_alive
                            else len(scenario.names) for c in game.chars if c.team == "D"], dtype=np.int64)
                        frame["damage"], frame["uses"] = damage, uses
                    if audit and controller.snapshot:
                        audit.before(controller.snapshot, controller.actions, controller.encoder.history.tracks)
                game.step_tick()
                if audit:
                    audit.after(game)
                steps += 1
                now_hp = np.asarray([max(0., c.hp) if c.is_alive else 0. for c in own])
                now_charges = np.asarray([getattr(c, c.ability_name.lower() + "_charges", 0) for c in own])
                damage += float(np.maximum(0., hp - now_hp).sum())
                uses += float(np.maximum(0., charges - now_charges).sum())
                hp, charges = now_hp, now_charges
                if terminal is None and (game.is_planted or game.round_over):
                    terminal = {"planted": bool(game.is_planted), "tick": int(game.battle_tick),
                        "alive": sum(c.is_alive for c in own), "damage": damage, "uses": uses,
                        "remaining_hp": float(sum(max(0., c.hp) for c in own if c.is_alive)),
                        "initial_team_hp": initial_team_hp,
                        "enemy_alive": sum(c.is_alive for c in enemies),
                        "initial_alive": initial_alive, "initial_enemy_alive": initial_enemy_alive,
                        "initial_abilities": initial_abilities,
                        "remaining_abilities_alive": int(sum(now_charges[i] for i, c in enumerate(own) if c.is_alive)),
                        "site": scenario.site_of(game.planted_pos) if game.is_planted else None}
                if steps > config["max_round_steps"]:
                    raise RuntimeError(f"{opponent} round {round_number}: step limit exceeded; no fabricated labels")
            if terminal is None or game.current_round != round_number or not game.round_over:
                raise RuntimeError("Twelve-round collector block ended unexpectedly")
            frames = controller.frames
            if frames:
                targets = [[float(terminal["planted"]), min(1., (terminal["damage"] - f["damage"]) / 500),
                    (f["alive"] - terminal["alive"]) / 5, min(1., (terminal["uses"] - f["uses"]) / 10),
                    (terminal["tick"] - f["tick"]) / 100 if terminal["planted"] else 0.] for f in frames]
                samples.append({"observations": np.stack([f["observation"] for f in frames]).astype(np.float16),
                    "routes": np.stack([f["route_features"] for f in frames]).astype(np.float16),
                    "outcomes": np.asarray(targets, dtype=np.float32),
                    "decision_ids": np.asarray([f["decision_id"] for f in frames], dtype=np.int32),
                    "selected_sites": np.asarray([f["site"] == "R" for f in frames], dtype=np.int8),
                    "placements": np.stack([f["placement"] for f in frames]), "round": round_number})
            result = {"round": round_number, "preset": preset_name, **terminal, "route": controller.route.key if controller.route else None,
                      "branches": controller.route.branches if controller.route else (), "mode": controller.mode,
                      "entry_utility": [e for e in controller.events if e["type"] == "predicted_entry_utility"],
                      "entry_cover": [e for e in controller.events if e["type"] in ("entry_cover_wait", "entry_cover_move")],
                      "route_decisions": controller.route_planner.events,
                      "combat_diagnostics": audit.report() if audit else None,
                      "target_site": controller.route.site if controller.route else None,
                      "reason": "planted" if terminal["planted"] else "attacker_eliminated" if not terminal["alive"]
                                else "timeout" if game.round_timer <= 0 else "defender_eliminated"}
            results.append(result)
            history.append({"site": terminal["site"], "winner": "A" if game.attacker_wins > before_wins else "D"})
            if round_number < 12:
                game.current_round += 1
                game.init_round()
    return results, samples
