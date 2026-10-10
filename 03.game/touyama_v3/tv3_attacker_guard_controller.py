"""Plant-to-guard handoff preserving public memory at the end of the plant tick."""
import numpy as np
import torch

from frc_v1.actions import build_masks, to_game_action, MOVE_STEPS
from frc_v1.perception import FrcPerceptionBuilder
from grid_paths import distance_map
from touyama_v3.tv3_observer import FeatureHistory
from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController
from touyama_v3.tv3_learn_attacker_guard import (
    GuardEncoder, guard_positions, MOVEMENTS, guard_defuse_cells,
    guard_fire_line, guard_smoke_pressure,
)
from touyama_v3.tv3_observer import facing
from frc_v1 import FACING


class TouyamaV3AttackerGuardController:
    handles_team_perception = True

    def __init__(self, scenario, policy, *, sensor=None, history=None, seed=42):
        self.scenario, self.policy = scenario, policy
        self.sensor = sensor if sensor is not None else FrcPerceptionBuilder("A")
        self.history = history if history is not None else FeatureHistory(scenario)
        self.encoder = GuardEncoder(scenario)
        self.rng = np.random.default_rng(seed)
        self.training = False
        self.epsilon = self.teacher_probability = 0.
        self._recorded_round = None
        self.game = self.snapshot = self.cache = None
        self.inputs, self.actions, self.plans, self.decisions = {}, {}, {}, {}
        self.goals = {}
        self.initial_charges = {}
        self.initial_alive = self.initial_enemy_alive = 0
        self.start_tick = self.start_timer = None
        self.ramp_attempted = set()

    def set_game(self, game):
        self.game = getattr(game, "_real", game)

    def initialize(self, snapshot):
        if not snapshot.is_planted or snapshot.spike_planted is None:
            raise ValueError("Guard handoff requires a real plant")
        self.start_tick, self.start_timer = snapshot.tick, snapshot.detonate_timer
        self.initial_charges = {a.name: a.charges for a in snapshot.allies}
        self.initial_alive = sum(a.alive for a in snapshot.allies)
        self.initial_enemy_alive = sum(e.alive for e in snapshot.enemies)
        self.goals = guard_positions(self.scenario, snapshot)
        self.history.encode(snapshot, [])
        self.snapshot = snapshot

    def prepare_team_tick(self):
        snapshot = self.sensor.build(self.game)
        if self.cache == snapshot.key:
            return
        if self.start_tick is None:
            self.initialize(snapshot)
        self.history.encode(snapshot, [])
        self.snapshot, self.cache = snapshot, snapshot.key
        self.inputs, self.actions, self.plans = {}, {}, {}
        masks = build_masks(snapshot)
        reserved = set()
        assigned_close = set()
        names = {a.slot: a.name for a in snapshot.allies}
        plant_distances = distance_map(self.scenario.grid, snapshot.spike_planted)
        for ally in sorted((a for a in snapshot.allies if a.alive), key=lambda a: (a.position, a.ability_name, a.slot)):
            if ally.slot not in self.goals:
                self.goals.update(guard_positions(self.scenario, snapshot))
            goal = self.goals[ally.slot]
            smoke_pressure = guard_smoke_pressure(self.scenario, snapshot, ally)
            if ally.blind > 0 and not (smoke_pressure and snapshot.defuse_notified):
                live = {e.enemy_id for e in snapshot.enemies if e.alive}
                threats = [p for i, (p, tick, _) in self.history.tracks.items()
                           if i in live and snapshot.tick - tick <= 15]
                if not threats:
                    from touyama_v3.tv3_learn_attacker_guard import guard_approaches
                    threats = guard_approaches(self.scenario, snapshot.spike_planted)
                occupied = {a.position for a in snapshot.allies if a.alive and a.slot != ally.slot}
                covers = [(p, d) for p, d in self.scenario.local(ally.position, 4)
                          if p not in occupied and plant_distances[p] <= 10]
                if covers:
                    goal = min(covers, key=lambda row: (sum(self.scenario.clear(row[0], t) for t in threats),
                                                      row[1], row[0]))[0]
            # Public defuse notification gives a plant location, never a hidden defuser cell.
            if snapshot.defuse_notified and ally.blind == 0:
                nearby = [p for p, d in self.scenario.local(snapshot.spike_planted, 3)
                          if d > 0 and self.scenario.clear(p, snapshot.spike_planted)]
                if nearby:
                    goal = min(nearby, key=lambda p: (distance_map(self.scenario.grid, p)[ally.position], p))
            # Pre-position when smoke covers the spike; do not wait for a six-tick defuse.
            if smoke_pressure and (ally.blind == 0 or snapshot.defuse_notified):
                zone = guard_defuse_cells(self.scenario, snapshot.spike_planted)
                occupied = {a.position for a in snapshot.allies if a.alive and a.slot != ally.slot}
                close = [p for p in zone if p not in occupied and p not in assigned_close
                         and distance_map(self.scenario.grid, p)[ally.position] >= 0]
                if close:
                    goal = min(close, key=lambda p: (
                        p != snapshot.spike_planted,
                        -sum(guard_fire_line(self.scenario, p, t, snapshot.smoke_cells) for t in zone),
                        distance_map(self.scenario.grid, p)[ally.position], p))
                    assigned_close.add(goal)
            inputs = self.encoder.encode(snapshot, ally, goal, self.history.tracks, masks,
                self.start_tick, self.start_timer, self.initial_alive, self.initial_enemy_alive,
                self.initial_charges.get(ally.name, 0), self.ramp_attempted)
            for i in np.flatnonzero(inputs.mask[:40]):
                kind = MOVEMENTS[i // 8]
                if kind == "STAY":
                    continue
                dr, dc = MOVE_STEPS[kind]
                p = ally.position[0] + dr, ally.position[1] + dc
                if p in reserved:
                    inputs.mask[i] = False
                elif plant_distances[ally.position] <= 10 and plant_distances[p] > 10:
                    inputs.mask[i] = False  # No unrelated map excursion while guarding.
            # Supply pressure and a demonstration, preserving every legal learned alternative.
            if smoke_pressure and snapshot.defuse_notified:
                inputs.defuse_pressure = True
                old = inputs.distances[ally.position]
                moves = []
                for i in np.flatnonzero(inputs.mask[:40]):
                    dr, dc = MOVE_STEPS.get(MOVEMENTS[i // 8], (0, 0))
                    p = ally.position[0] + dr, ally.position[1] + dc
                    if 0 <= inputs.distances[p] < old:
                        aim = ally.facing if ally.forced_facing else facing(p, snapshot.spike_planted)
                        if FACING[i % 8] == aim:
                            moves.append(i)
                if moves:
                    inputs.teacher = int(moves[0])
                elif old == 0:
                    directions = {facing(ally.position, p) for p in guard_defuse_cells(self.scenario, snapshot.spike_planted)
                                  if p != ally.position and guard_fire_line(self.scenario, ally.position, p, snapshot.smoke_cells)}
                    shots = [i for i in np.flatnonzero(inputs.mask[:8]) if FACING[i] in directions]
                    if shots:
                        inputs.teacher = int(shots[snapshot.tick % len(shots)])
            if not inputs.mask[inputs.teacher]:
                candidates = np.flatnonzero(inputs.mask[:40])
                def rank(i):
                    dr, dc = MOVE_STEPS.get(MOVEMENTS[i // 8], (0, 0))
                    p = ally.position[0] + dr, ally.position[1] + dc
                    distance = inputs.distances[p]
                    return distance if distance >= 0 else 9999, i
                inputs.teacher = int(min(candidates, key=rank))
            if self.policy is None or (self.training and self.rng.random() < self.teacher_probability):
                chosen = inputs.teacher
            elif self.training and self.rng.random() < self.epsilon:
                chosen = int(self.rng.choice(np.flatnonzero(inputs.mask)))
            else:
                with torch.no_grad():
                    q = self.policy(torch.tensor(inputs.observation).unsqueeze(0))[0].numpy()
                chosen = int(np.where(inputs.mask, q, -np.inf).argmax())
            action = inputs.actions[chosen]
            self.inputs[ally.name] = inputs
            self.plans[ally.name] = (chosen, inputs, ally)
            self.actions[ally.name] = to_game_action(snapshot, ally.slot, action, names)
            dr, dc = MOVE_STEPS.get(action.kind, (0, 0))
            reserved.add((ally.position[0] + dr, ally.position[1] + dc))

    def decide_move(self, char, state):
        self.prepare_team_tick()
        name = str(getattr(char, "base_name", char.name))
        if name in self.plans:
            self.decisions[name] = self.plans[name]
            chosen, inputs, ally = self.plans[name]
            if ally.ability_name == "RAMP" and inputs.actions[chosen].kind == "ABILITY":
                self.ramp_attempted.add(ally.position)
        return self.actions.get(name, (list(char.pos), {"facing": char.facing}))


class TouyamaV3AttackerPlantGuardController:
    """Guard activates only after mark_plant_boundary() at a completed engine tick."""
    handles_team_perception = True

    def __init__(self, scenario, analysis, plant_policy, guard_policy=None, *, seed=42):
        self.scenario = scenario
        self.plant = TouyamaV3AttackerPlantController(scenario, analysis, plant_policy, seed=seed)
        self.guard_policy, self.guard = guard_policy, None
        self._recorded_round = None
        self.game = None
        self.seed = seed
        self.training = False
        self.epsilon = self.teacher_probability = 0.

    def set_game(self, game):
        self.game = getattr(game, "_real", game)
        self.plant.set_game(self.game)
        if self.guard is not None:
            self.guard.set_game(self.game)

    @classmethod
    def from_best(cls, opponent, *, best_dir=None, plant_dir=None, analysis_dir=None, scenario=None):
        from pathlib import Path
        from touyama_v3.tv3_scenario import Scenario
        from touyama_v3.tv3_guard_runtime import load_sources
        from touyama_v3.tv3_learn_attacker_guard import load_guard
        scenario = scenario or Scenario()
        best_dir = Path(best_dir) if best_dir else Path(__file__).resolve().parent / "data" / "best"
        source = load_sources(scenario, (opponent,), plant_dir or best_dir, analysis_dir or best_dir)[opponent]
        guard, _ = load_guard(best_dir / opponent / "attacker_guard_best.pt", scenario, opponent, source["hashes"])
        return cls(scenario, source["analysis"], source["plant"], guard)

    def reset_round(self):
        self.plant.reset_round()
        self.guard = None

    def record_opponent_round_end(self):
        key = self.game.current_round
        if getattr(self, "_recorded_round", None) == key:
            return
        self._recorded_round = key
        site = self.scenario.site_of(self.game.planted_pos) if self.game.is_planted else None
        defender_won = self.game.is_defused or (not self.game.is_planted and (
            self.game.round_timer <= 0 or not any(c.is_alive for c in self.game.chars if c.team == "A")))
        self.plant.previous_rounds.append({"site": site, "winner": "D" if defender_won else "A"})

    def mark_plant_boundary(self):
        if self.guard is not None:
            return self.guard
        if not self.game.is_planted:
            raise ValueError("Cannot activate guard before a real plant")
        self.guard = TouyamaV3AttackerGuardController(self.scenario, self.guard_policy,
            sensor=self.plant.sensor, history=self.plant.analysis_encoder.history, seed=self.seed + 31)
        self.guard.set_game(self.game)
        self.guard.training = self.training
        self.guard.epsilon, self.guard.teacher_probability = self.epsilon, self.teacher_probability
        self.guard.initialize(self.guard.sensor.build(self.game))
        return self.guard

    def prepare_team_tick(self):
        if self.game.is_planted:
            if self.guard is None:
                before_tick = self.plant.cache[2] if self.plant.cache is not None else None
                if before_tick is None or self.game.battle_tick > before_tick:
                    self.mark_plant_boundary()
            if self.guard is not None:
                self.guard.prepare_team_tick()
        else:
            self.plant.prepare_team_tick()

    def decide_move(self, char, state):
        if not self.game.is_planted:
            return self.plant.decide_move(char, state)
        if self.guard is None:
            self.prepare_team_tick()
            if self.guard is None:
                return list(char.pos), {"facing": char.facing}  # Hold remaining actions in the first plant tick.
        return self.guard.decide_move(char, state)
