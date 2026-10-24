"""Postplant shared-player policy and public-only guard inputs."""
from dataclasses import dataclass
import json

import numpy as np
import torch
from torch import nn

from frc_v1.actions import FrcAction, KINDS, MOVE_STEPS, target_required, validate_action
from frc_v1 import FACING
from grid_paths import distance_map
from grid_lines import line_cells
from touyama_v3.tv3_observer import facing
from touyama_v3.tv3_learn_public_hazards import hazard_schema
from touyama_v3.tv3_defender_policy import PolicyEncoder, OBS_DIM as BASE_OBS_DIM, ACTION_DIM, MOVEMENTS
from touyama_v3.tv3_learn_attacker_plant import learn_plant

GUARD_VERSION = 3
OBS_DIM = BASE_OBS_DIM + 38
DISABLED_DEFUSE_ACTION = ACTION_DIM - 2
GAMMA = .98
GUARD_MAX_DISTANCE = 8


def guard_fire_line(scenario, origin, target, smoke_cells):
    """Match the engine's public smoke rule, including adjacent shots."""
    cells = line_cells(origin, target)
    return scenario.clear(origin, target) and (len(cells) <= 2 or not set(cells).intersection(smoke_cells))


def guard_defuse_cells(scenario, plant):
    return tuple(p for p, _ in scenario.local(plant, 2)
                 if max(abs(p[0] - plant[0]), abs(p[1] - plant[1])) <= 1)


def guard_smoke_pressure(scenario, snapshot, ally):
    zone = guard_defuse_cells(scenario, snapshot.spike_planted)
    # Seeing one nearby enemy does not mean the whole smoked defuse zone is
    # covered. Keep closing during a public defuse notification, including
    # when an adjacent enemy becomes visible before reaching the spike.
    if not snapshot.defuse_notified and any(
            s.position in zone and guard_fire_line(scenario, ally.position, s.position, snapshot.smoke_cells)
            for s in snapshot.sightings):
        return False
    return bool(set(zone).intersection(snapshot.smoke_cells)) or (bool(snapshot.smoke_cells) and any(
        scenario.clear(ally.position, p)
        and not guard_fire_line(scenario, ally.position, p, snapshot.smoke_cells)
        for p in zone))


class GuardDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, 256), nn.ReLU(), nn.Linear(256, 128),
                                 nn.ReLU(), nn.Linear(128, ACTION_DIM))

    def forward(self, observation):
        return self.net(observation)


def guard_schema(scenario):
    return json.loads(json.dumps({"version": GUARD_VERSION, "board": scenario.grid.tolist(),
        "branches": scenario.branches, "entries": scenario.retake_entries,
        "obs_dim": OBS_DIM, "action_dim": ACTION_DIM, "gamma": GAMMA,
        "max_guard_distance": GUARD_MAX_DISTANCE, "sensor": "frc_public_team_v1",
        "scope": "postplant_guard", "site_choice": "actual_plant_no_balancing",
        "tactics": "learned_smoke_defuse_crossfire_stop_shoot_v2", "scenario": scenario.signature,
        "utility_strategy": "all_legal_delay_types_v1", "public_hazards": hazard_schema()}))


def crossfire_score(scenario, position, teammates, targets):
    """Public map geometry: wide angles around the same threatened point."""
    if not targets:
        return 0.
    score = 0.
    for target in targets:
        if position == target or not scenario.clear(position, target):
            continue
        own = np.asarray(position, float) - np.asarray(target, float)
        best = 0.
        for other in teammates:
            if other == target or not scenario.clear(other, target):
                continue
            vector = np.asarray(other, float) - np.asarray(target, float)
            cosine = float(own @ vector / (np.linalg.norm(own) * np.linalg.norm(vector)))
            best = max(best, (1 - cosine) / 2)  # Opposite firing angles score highest.
        score += best
    return score / len(targets)


def guard_approaches(scenario, plant):
    side = scenario.site_of(plant)
    candidates = [p for cells in scenario.retake_entries[side].values() for p in cells]
    if not candidates:
        candidates = [p for p, d in scenario.local(plant, 6) if 3 <= d <= 6]
    field = distance_map(scenario.grid, plant)
    ordered = sorted(set(candidates), key=lambda p: (field[p] if field[p] >= 0 else 9999, p))
    return tuple(ordered[:3])


def guard_positions(scenario, snapshot):
    """Covered map geometry around the planted spike, not enemy ground truth."""
    plant = snapshot.spike_planted
    targets = tuple(s.position for s in snapshot.sightings) or guard_approaches(scenario, plant)
    choices = []
    for p, steps in scenario.local(plant, GUARD_MAX_DISTANCE):
        if steps < 2:
            continue
        peek = scenario.clear(p, plant) or any(scenario.clear(n, plant) for n in scenario.neighbors(p))
        cover = any(not scenario.clear(n, plant) for n in scenario.neighbors(p))
        choices.append(((not peek, not cover, abs(steps - 4), p), p))
    ordered = [p for _, p in sorted(choices)]
    goals = {}
    used = set()
    allies = sorted((a for a in snapshot.allies if a.alive), key=lambda a: (a.position, a.ability_name, a.slot))
    for ally in allies:
        choices = [p for p in ordered if p not in used]
        if not choices:
            goals[ally.slot] = ally.position
            continue
        # Spread the five players while allowing the nearest covered posts.
        ranked = choices[:30]
        goal = min(ranked, key=lambda p: (-crossfire_score(scenario, p, used, targets),
                                        sum(max(0, 3 - abs(p[0] - q[0]) - abs(p[1] - q[1])) for q in used),
                                        distance_map(scenario.grid, p)[ally.position], ordered.index(p)))
        goals[ally.slot] = goal
        used.add(goal)
    return goals


@dataclass
class GuardInputs:
    observation: np.ndarray
    mask: np.ndarray
    actions: tuple
    teacher: int
    distances: np.ndarray
    goal: tuple
    threats: tuple
    teammates: tuple
    blind: bool
    defuse_pressure: bool = False


class GuardEncoder:
    def __init__(self, scenario):
        self.scenario = scenario
        self.base = PolicyEncoder(scenario)

    def encode(self, snapshot, ally, goal, tracks, masks, start_tick, start_timer, initial_alive,
               initial_enemy_alive, initial_charges, ramp_attempted=()):
        if snapshot.side != "A" or not snapshot.is_planted or snapshot.spike_planted is None:
            raise ValueError("Guard needs an attacker public snapshot with a real plant")
        side = self.scenario.site_of(snapshot.spike_planted)
        probability = np.asarray([float(side == "L"), float(side == "R")], np.float32)
        base = self.base.encode(snapshot, ally, goal, probability, tracks, masks)
        live = {e.enemy_id for e in snapshot.enemies if e.alive}
        known = [(pos, tick) for i, (pos, tick, _) in tracks.items() if i in live and snapshot.tick - tick <= 15]
        known.sort(key=lambda row: (abs(row[0][0] - ally.position[0]) + abs(row[0][1] - ally.position[1]), row[0]))
        approaches = guard_approaches(self.scenario, snapshot.spike_planted)
        current = [s.position for s in snapshot.sightings]
        threats = tuple(current or [p for p, _ in known])
        teammates = tuple(a.position for a in snapshot.allies if a.alive and a.slot != ally.slot)
        look = (current[0] if current else known[0][0] if known else
                approaches[ally.slot % len(approaches)] if approaches else snapshot.spike_planted)
        if snapshot.defuse_notified and not current:
            look = snapshot.spike_planted
        direction = ally.facing if ally.forced_facing else facing(ally.position, look)
        actions, legal = list(base.actions), base.mask.copy()
        legal[DISABLED_DEFUSE_ACTION] = False  # Attacker never defuses or plants again.
        coordinates = [p for p, _ in known[:3]] + [None] * max(0, 3 - len(known))
        coordinates += [approaches[i] if i < len(approaches) else None for i in range(2)]
        coordinates += [goal, snapshot.spike_planted, ally.position]
        allies = sorted(snapshot.allies, key=lambda a: (a.position, a.ability_name))
        for kind_index, kind in enumerate(("ABILITY", "ULTIMATE")):
            for index, pos in enumerate(coordinates):
                if kind == "ABILITY" and ally.ability_name == "DANCE":
                    recipient = allies[index] if index < 5 else None
                    action = FrcAction(kind, direction, ally_slot=recipient.slot if recipient else None)
                else:
                    no_target = kind == "ULTIMATE" and not target_required(ally.slot, kind, masks)
                    action = FrcAction(kind, direction, None if no_target else pos)
                action_index = 40 + kind_index * 8 + index
                actions[action_index] = action
                try:
                    validate_action(snapshot, masks, ally.slot, action)
                    legal[action_index] = True
                except (ValueError, TypeError, IndexError):
                    legal[action_index] = False
                if kind == "ABILITY" and ally.ability_name == "RAMP" and ally.position in ramp_attempted:
                    legal[action_index] = False  # Do not repeat a trap request at the same remembered own position.
        h, w = self.scenario.grid.shape
        plant = snapshot.spike_planted
        extra = [plant[0] / h, plant[1] / w, float(snapshot.defuse_notified), snapshot.detonate_timer / 55,
                 (snapshot.tick - start_tick) / max(1., start_timer), initial_alive / 5, initial_enemy_alive / 5]
        for i in range(3):
            p = approaches[i] if i < len(approaches) else (0, 0)
            extra += [p[0] / h, p[1] / w]
        extra += [float(self.scenario.clear(goal, plant)), len(current) / 5, initial_charges / 10,
                  ally.charges / max(1, initial_charges), max(0, distance_map(self.scenario.grid, plant)[goal]) / 20,
                  float(ally.position in snapshot.smoke_cells), float(ally.reveal > 0)]
        geometry_targets = threats or approaches
        for kind in MOVEMENTS:
            dr, dc = MOVE_STEPS.get(kind, (0, 0))
            p = ally.position[0] + dr, ally.position[1] + dc
            valid = 0 <= p[0] < h and 0 <= p[1] < w and self.scenario.grid[p] != 1
            extra += [sum(self.scenario.clear(p, t) for t in threats) / 5 if valid else 0.,
                      crossfire_score(self.scenario, p, teammates, geometry_targets) if valid else 0.,
                      float(valid and self.scenario.clear(p, plant))]
        extra += [min(1., ally.blind / 10), float(ally.movement_disabled > 0),
                  float(any(e.kind == "FLASH" for e in snapshot.effects))]
        observation = np.concatenate((base.observation, np.asarray(extra, np.float32)))
        if len(observation) != OBS_DIM or not np.isfinite(observation).all():
            raise ValueError("Invalid guard observation")
        movement = np.flatnonzero(legal[:40])
        def rank(i):
            dr, dc = MOVE_STEPS.get(MOVEMENTS[i // 8], (0, 0))
            p = ally.position[0] + dr, ally.position[1] + dc
            distance = base.distances[p]
            return distance if distance >= 0 else 9999, int(FACING[i % 8] != direction), i
        teacher = int(min(movement, key=rank))
        from touyama_v3.tv3_attacker_combat import AttackerCombatCoach
        selected = actions[teacher]
        dr, dc = MOVE_STEPS.get(selected.kind, (0, 0))
        advice = AttackerCombatCoach(self.scenario).advise(snapshot, ally,
            (ally.position[0] + dr, ally.position[1] + dc), goal, tracks)
        preferred = [i for i in movement if
                     (ally.position[0] + MOVE_STEPS.get(actions[i].kind, (0, 0))[0],
                      ally.position[1] + MOVE_STEPS.get(actions[i].kind, (0, 0))[1]) == advice.position
                     and actions[i].facing == advice.facing]
        if preferred:
            teacher = int(preferred[0])
        active = {e.kind for e in snapshot.effects if e.phase in ("active", "flight")}
        choices = []
        if ally.blind > 0 and ally.ability_name == "FLASH":
            # An active flash may belong to the enemy; it must not block a legal counterflash.
            choices = [40, 41, 42, 43, 44]
        elif ally.ability_name == "FLASH" and "FLASH" not in active:
            choices = [40, 41, 42] if current else [46] if snapshot.defuse_notified else []
        elif ally.ability_name == "SMOKE" and (current or known or snapshot.defuse_notified):
            # Block approach sightlines, rather than teaching smoke on top of a defusing enemy.
            plant = snapshot.spike_planted
            smoke = set(snapshot.smoke_cells)
            choices = [40 + i for i in (3, 4, 0, 1, 2) if coordinates[i] is not None
                       and coordinates[i] not in smoke
                       and max(abs(coordinates[i][0] - plant[0]), abs(coordinates[i][1] - plant[1])) > 1]
        elif ally.ability_name == "ASH" and "ASH" not in active:
            choices = [46, 40, 41, 42] if snapshot.defuse_notified else [40, 41, 42] if known else []
        elif ally.ability_name == "RECON" and "RECON" not in active and not current:
            if snapshot.defuse_notified or snapshot.tick - start_tick >= 6:
                choices = [43, 44, 40, 41, 42]
        elif ally.ability_name == "RAMP" and ally.position not in ramp_attempted:
            near_entry = any(distance_map(self.scenario.grid, p)[ally.position] in (0, 1, 2) for p in approaches)
            near_plant = distance_map(self.scenario.grid, snapshot.spike_planted)[ally.position] <= 3
            if near_entry or near_plant:
                choices = [47]  # Native RAMP places a trap at the caster's current tile.
        elif ally.ability_name == "DANCE":
            choices = [40 + i for i, a in enumerate(allies) if a.alive and a.hp < .8 * a.max_hp]
        # Reaching the smoked spike takes priority over repeated support casts.
        if not ((ally.blind == 0 or snapshot.defuse_notified) and guard_smoke_pressure(self.scenario, snapshot, ally)
                and base.distances[ally.position] > 0):
            teacher = next((i for i in choices if legal[i]), teacher)
        return GuardInputs(observation, legal, tuple(actions), teacher, base.distances, goal,
                           threats, teammates, bool(ally.blind > 0))


def learn_guard(model, target, optimizer, replay, rng, updates, batch_size, demonstration_weight=.05):
    return learn_plant(model, target, optimizer, replay, rng, updates, batch_size, demonstration_weight)


def load_guard(path, scenario, opponent=None, source_hashes=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["schema"] != guard_schema(scenario):
        raise ValueError("Guard schema mismatch")
    if opponent is not None and state["opponent"] != opponent:
        raise ValueError("Guard opponent mismatch")
    if source_hashes is not None and state["source_hashes"] != source_hashes:
        raise ValueError("Guard uses different plant/analysis best models")
    model = GuardDQN()
    model.load_state_dict(state["model"])
    model.eval()
    return model, state
