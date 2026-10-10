"""Opponent-specific shared-player plant DQN, driven by frozen route analysis."""
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import io
import json
import math

import numpy as np
import torch
from torch import nn
from grid_lines import WALL_LOS_VERSION

from frc_v1 import FACING
from frc_v1.actions import FrcAction, KINDS, MOVE_STEPS, validate_action, target_required
from touyama_v3.tv3_defender_policy import PolicyEncoder, OBS_DIM as BASE_OBS_DIM, ACTION_DIM, MOVEMENTS
from touyama_v3.tv3_observer import facing
from touyama_v3.tv3_learn_public_hazards import hazard_schema
from touyama_v3.tv3_learn_attacker_analysis import load_analysis
from touyama_v3.tv3_checkpoint import load_checkpoint
from touyama_v3.tv3_attacker_entry_utility import predicted_entry_utility, utility_schema, pending_flash_impact, flash_entry_step
from touyama_v3.tv3_attacker_route_planner import planner_schema
from touyama_v3.tv3_attacker_combat import AttackerCombatCoach, effective_utility, combat_schema, COMBAT_FEATURES, plant_ready, LOW_HP_FRACTION
from touyama_v3.tv3_attacker_entry_coordination import ENTRY_FEATURES, coordination_schema, entry_features
from touyama_v3.tv3_attacker_site_entry import (
    ENTRY_FEATURES as SITE_ENTRY_FEATURES, site_entry_schema, traffic_distances, site_entry_features,
    teacher_distances,
)

from touyama_v3.tv3_learn_attacker_fire_support import (
    FIRE_SUPPORT_FEATURES, fire_support_schema, fire_support_features,
    public_threats, defensive_advice, smoke_screen, PublicPlantCombatCoach,
)

PLANT_VERSION = 10
PLANT_ACTION = ACTION_DIM - 2
ORB_ACTION = ACTION_DIM - 1
REGIONS = 14
LEGACY_OBS_DIM = BASE_OBS_DIM + 5 * (REGIONS + 1) + 24 + COMBAT_FEATURES + ENTRY_FEATURES + SITE_ENTRY_FEATURES + 2
OBS_DIM = LEGACY_OBS_DIM + FIRE_SUPPORT_FEATURES
CARRIER_FEATURE_INDEX = BASE_OBS_DIM + 5 * (REGIONS + 1)
GAMMA = .98
ROUTE_CORRIDOR_CELLS = 2
ROUTE_RETREAT_CELLS = 3
MAX_EDGE_TRAVERSALS = 2
EDGE_TRAVERSAL_WINDOW = 8


class PlantDQN(nn.Module):
    def __init__(self, execution_version=PLANT_VERSION):
        super().__init__()
        if execution_version not in (9, 10):
            raise ValueError("Unsupported plant execution version")
        self.execution_version = execution_version
        dimension = LEGACY_OBS_DIM if execution_version == 9 else OBS_DIM
        self.net = nn.Sequential(nn.Linear(dimension, 256), nn.ReLU(), nn.Linear(256, 128),
                                 nn.ReLU(), nn.Linear(128, ACTION_DIM))

    def forward(self, observation):
        return self.net(observation)


def policy_schema(scenario, execution_version=PLANT_VERSION):
    if execution_version not in (9, 10):
        raise ValueError("Unsupported plant execution version")
    schema = {"version": execution_version, "board": scenario.grid.tolist(), "scenario": scenario.signature, "branches": scenario.branches,
        "obs_dim": LEGACY_OBS_DIM if execution_version == 9 else OBS_DIM, "action_dim": ACTION_DIM, "gamma": GAMMA,
        "route_corridor": ROUTE_CORRIDOR_CELLS, "retreat": ROUTE_RETREAT_CELLS,
        "edge_limit": MAX_EDGE_TRAVERSALS, "edge_window": EDGE_TRAVERSAL_WINDOW, "planner": planner_schema(),
        "sensor": "frc_public_team_v1", "scope": "preplant_only",
        "site_choice": "public_online_analysis_no_site_balancing", "execution": "bounded_rally_mission_plant_v9" if execution_version == 9 else "public_target_support_plant_v10",
        "action_selection": "learned_no_forced_forward_or_plant",
        "combat": combat_schema(),
        "entry_utility": utility_schema(), "entry_coordination": coordination_schema(),
        "site_entry": site_entry_schema(),
        "plant_legal_progress_features": 2, "wall_los": WALL_LOS_VERSION, "public_hazards": hazard_schema()}
    if execution_version == 10:
        schema["fire_support"] = fire_support_schema()
    return schema


def load_frozen_analysis(directory, opponent, scenario):
    path = Path(directory) / opponent / "attacker_analysis_best.pt"
    payload = path.read_bytes()
    try:
        model, _, state = load_analysis(io.BytesIO(payload), scenario, opponent)
    except ValueError as error:
        raise ValueError(f"Frozen attacker analysis is incompatible: opponent={opponent}, file={path}. "
                         f"{error}") from error
    model.requires_grad_(False)
    return model, hashlib.sha256(payload).hexdigest(), state


def load_plant(path, scenario, opponent=None, analysis_hash=None):
    state = load_checkpoint(path)
    version = state["schema"].get("version")
    if version not in (9, 10) or json.dumps(state["schema"], sort_keys=True) != json.dumps(policy_schema(scenario, version), sort_keys=True):
        raise ValueError("Plant model map/action/schema mismatch")
    if opponent is not None and state["opponent"] != opponent:
        raise ValueError("Plant model opponent mismatch")
    if analysis_hash is not None and state["analysis_hash"] != analysis_hash:
        raise ValueError("Frozen analysis changed since plant training")
    model = PlantDQN(version)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


@dataclass
class PlantInputs:
    observation: np.ndarray
    mask: np.ndarray
    actions: tuple
    teacher: int
    distances: np.ndarray
    goal: tuple
    combat: object = None
    traffic: object = None


def predicted_targets(scenario, snapshot, ally, analysis):
    """Map landmarks weighted by regional beliefs, never hidden enemy cells."""
    seen = {s.position for s in snapshot.sightings}
    visible = set(snapshot.visible_cells)
    ranked = []
    for region in scenario.names:
        mass = sum(row[region] for row in analysis["placement"].values())
        cells = scenario.branches[region]
        candidates = [p for p in cells if p not in visible or p in seen]
        if mass < .35 or not candidates:
            continue
        pos = min(candidates, key=lambda p: (not scenario.clear(ally.position, p),
                        abs(p[0] - ally.position[0]) + abs(p[1] - ally.position[1]), p))
        distance = abs(pos[0] - ally.position[0]) + abs(pos[1] - ally.position[1])
        ranked.append((mass / (1 + distance / 12), pos, mass))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [(p, mass) for _, p, mass in ranked[:3]]


class PlantEncoder:
    def __init__(self, scenario, execution_version=PLANT_VERSION):
        if execution_version not in (9, 10):
            raise ValueError("Unsupported plant execution version")
        self.execution_version = execution_version
        if len(scenario.names) != REGIONS:
            raise ValueError("Plant encoder needs the current 14-region map")
        self.scenario = scenario
        self.base = PolicyEncoder(scenario)
        self.combat_coach = (AttackerCombatCoach if execution_version == 9 else PublicPlantCombatCoach)(scenario)

    def encode(self, snapshot, ally, goal, route, analysis, tracks, masks, cursor, route_mode, coordination=None):
        site = np.asarray([float(route.site == "L"), float(route.site == "R")], np.float32)
        base = self.base.encode(snapshot, ally, goal, site, tracks, masks)
        traffic = traffic_distances(snapshot, ally, goal)
        routes = teacher_distances(base.distances, traffic, ally.position)
        targets = predicted_targets(self.scenario, snapshot, ally, analysis)
        entry_plan = predicted_entry_utility(self.scenario, snapshot, ally, analysis, route, masks)
        seen = self.combat_coach.contacts(snapshot, ally, ally.position)
        direction = ally.facing if ally.forced_facing else facing(ally.position,
            min(seen, key=lambda p: max(abs(p[0]-ally.position[0]), abs(p[1]-ally.position[1]))) if seen else targets[0][0] if targets else goal)
        movement = np.flatnonzero(base.mask[:40])
        def initial_score(i):
            dr, dc = MOVE_STEPS.get(base.actions[i].kind, (0, 0))
            p = ally.position[0]+dr, ally.position[1]+dc
            d = routes[p]
            return d if d >= 0 else 9999, int(base.actions[i].facing != direction), i
        initial = int(min(movement, key=initial_score))
        dr, dc = MOVE_STEPS.get(base.actions[initial].kind, (0, 0))
        destination = ally.position[0]+dr, ally.position[1]+dc
        hint = self.combat_coach.preaim_hint(snapshot, ally, destination, goal, analysis)
        advice = self.combat_coach.advise(snapshot, ally, destination, hint, tracks)
        # Plant training needs examples of winning a supported 3v2/5v3 fight,
        # not an unconditional retreat merely because two contacts are visible.
        if (self.execution_version == 9 and advice.reason == 'cover_retreat' and advice.contacts >= 2
                and advice.supporters >= advice.contacts and not ally.blind
                and ally.hp > ally.max_hp * LOW_HP_FRACTION):
            advice = replace(advice, position=ally.position, reason='stop_shoot')
        utility_snapshot = snapshot
        screen = None
        if self.execution_version == 10:
            advice = defensive_advice(self.scenario, snapshot, ally, advice, tracks, masks)
            utility_snapshot = replace(snapshot, sightings=public_threats(snapshot, tracks))
            if ally.ability_name == 'SMOKE':
                screen = smoke_screen(self.scenario, snapshot, ally, utility_snapshot.sightings,
                                      masks.target[ally.slot, 0].reshape(self.scenario.grid.shape))
        direction = advice.facing
        actions, legal = list(base.actions), base.mask.copy()
        # Replace defender-specific terminal actions with attacker PLANT.
        actions[PLANT_ACTION] = FrcAction("PLANT", direction)
        legal[PLANT_ACTION] = (masks.kind[ally.slot, KINDS.index("PLANT")]
                              and self.scenario.site_of(ally.position) == route.site)
        actions[ORB_ACTION] = FrcAction("COLLECT_ORB", direction)
        known = sorted({*seen, *(p for p, tick, _ in tracks.values() if snapshot.tick-tick <= 15)},
                       key=lambda p: (not self.scenario.clear(ally.position, p),
                                      max(abs(p[0]-ally.position[0]), abs(p[1]-ally.position[1])), p))
        coordinates = (known[:3] + [None] * max(0, 3 - len(known)))
        coordinates += [targets[i][0] if i < len(targets) else None for i in range(2)]
        if entry_plan:
            coordinates[3] = entry_plan[0]["target"]
        if self.execution_version == 10:
            known = sorted({s.position for s in utility_snapshot.sightings},
                           key=lambda p: (max(abs(p[0]-ally.position[0]), abs(p[1]-ally.position[1])), p))
            coordinates[:3] = known[:3] + [None] * max(0, 3-len(known))
            if screen is not None:
                coordinates[3] = screen
                entry_plan = None
        coordinates += [route.cells[-1], snapshot.spike_dropped, ally.position]
        allies = sorted(snapshot.allies, key=lambda a: (a.position, a.ability_name))
        for kind_index, kind in enumerate(("ABILITY", "ULTIMATE")):
            for index, pos in enumerate(coordinates):
                action_index = 40 + kind_index * 8 + index
                if kind == "ABILITY" and ally.ability_name == "DANCE":
                    recipient = allies[index] if index < 5 else None
                    action = FrcAction(kind, direction, ally_slot=recipient.slot if recipient else None)
                else:
                    no_target = kind == "ULTIMATE" and not target_required(ally.slot, kind, masks)
                    action = FrcAction(kind, direction, None if no_target else pos)
                actions[action_index] = action
                try:
                    validate_action(snapshot, masks, ally.slot, action)
                    cast = {"ability": ally.ability_name, "target": pos}
                    predicted_cast = entry_plan is not None and pos == entry_plan[0]["target"]
                    legal[action_index] = (not (kind == "ABILITY" and ally.ability_name in ("FLASH", "RECON", "SMOKE", "ASH"))
                                           or predicted_cast or effective_utility(self.scenario, utility_snapshot, ally, cast))
                except (ValueError, TypeError, IndexError):
                    legal[action_index] = False
        extra = [float(ally.has_spike), ally.plant_progress / 4, float(route.site == "R"),
                 cursor / max(1, len(route.cells) - 1), len(route.cells) / 100,
                 float(route_mode == "supported"), analysis["placement_entropy"] / math.log(REGIONS + 1),
                 float(snapshot.spike_dropped is not None), goal[0] / 26, goal[1] / 44]
        extra += [float(a.has_spike) for a in allies]  # Match the base encoder's geometry order.
        for i in range(3):
            p, mass = targets[i] if i < len(targets) else ((0, 0), 0.)
            extra += [p[0] / 26, p[1] / 44, mass / 5]
        beliefs = [analysis["placement"][i][n] for i in range(5) for n in (*self.scenario.names, "dead")]
        impact = pending_flash_impact(self.scenario, snapshot, ally)
        support = entry_features(self.scenario, snapshot, ally, destination, impact, coordination)
        observation = np.concatenate((base.observation, np.asarray(
            beliefs + extra + advice.features(ally, self.scenario.grid.shape) + support
            + (fire_support_features(self.scenario, snapshot, ally, tracks).tolist()
               if self.execution_version == 10 else [])
            + site_entry_features(snapshot, ally, goal, traffic, coordination).tolist()
            + [float(legal[PLANT_ACTION]), 0.], np.float32)))
        if len(observation) != (LEGACY_OBS_DIM if self.execution_version == 9 else OBS_DIM) or not np.isfinite(observation).all():
            raise ValueError("Invalid plant observation")
        join_wait = (coordination and coordination['wait'] and ally.slot in coordination['main']
                     and ally.slot in coordination['ready'] and advice.reason == 'advance')
        flash_wait = impact is not None and advice.reason == 'advance'
        if legal[PLANT_ACTION] and plant_ready(ally, advice) and not join_wait and not flash_wait:
            teacher = PLANT_ACTION
        else:
            movement = np.flatnonzero(legal[:40])
            def score(index):
                dr, dc = MOVE_STEPS.get(MOVEMENTS[index // 8], (0, 0))
                p = ally.position[0] + dr, ally.position[1] + dc
                distance = routes[p]
                return (distance if distance >= 0 else 9999,
                        int(FACING[index % 8] != direction), index)
            teacher = int(min(movement, key=score))
            for i in movement:
                dr, dc = MOVE_STEPS.get(actions[i].kind, (0, 0))
                if (ally.position[0]+dr, ally.position[1]+dc) == advice.position and actions[i].facing == advice.facing:
                    teacher = int(i)
                    break
            active = {e.kind for e in snapshot.effects if e.phase in ("active", "flight")}
            threat = bool(advice.contacts) or bool(targets and targets[0][1] >= .8 and
                       max(abs(targets[0][0][0] - ally.position[0]), abs(targets[0][0][1] - ally.position[1])) <= 8)
            choices = []
            if entry_plan or screen is not None:
                choices = [43]
            elif ally.ability_name in ("FLASH", "SMOKE", "ASH") and threat and ally.ability_name not in active:
                choices = [40 + i for i, p in enumerate(coordinates[:5]) if p is not None]
            elif ally.ability_name == "RECON" and not seen and threat and "RECON" not in active:
                choices = [43, 44, 45]
            elif ally.ability_name == "DANCE":
                choices = [40 + i for i, a in enumerate(allies) if a.alive and a.hp < .7 * a.max_hp]
            if self.execution_version == 10 and advice.reason == 'cover_retreat':
                choices = []  # Teach the legal escape instead of a stationary cast in the open.
            teacher = next((i for i in choices if legal[i]), teacher)
            # Only pause before leaving cover; an exposed player should not be
            # taught to freeze. Recon flights never require an entry pause.
            if impact is not None and advice.reason == "advance":
                movement_index = int(min(movement, key=score))
                dr, dc = MOVE_STEPS.get(actions[movement_index].kind, (0, 0))
                destination = (ally.position[0] + dr, ally.position[1] + dc)
                blocked = {a.position for a in snapshot.allies if a.alive and a.slot != ally.slot}
                step = flash_entry_step(self.scenario, ally.position, destination, impact, snapshot.smoke_cells, blocked)
                for i in movement:
                    dr, dc = MOVE_STEPS.get(actions[i].kind, (0, 0))
                    if (ally.position[0] + dr, ally.position[1] + dc) == step and actions[i].facing == direction:
                        teacher = int(i)
                        break
            if (coordination and coordination['wait'] and ally.slot in coordination['main']
                    and ally.slot in coordination['ready'] and advice.reason == 'advance'
                    and actions[teacher].kind != 'ABILITY'):
                stays = [i for i in movement if actions[i].kind == 'STAY' and actions[i].facing == direction]
                if stays:
                    teacher = int(stays[0])
        return PlantInputs(observation, legal, tuple(actions), teacher, base.distances, goal, advice, traffic)


def demonstration_kind_loss(values, teachers, masks):
    """Teach move/stay/utility/plant independently of eight facing variants."""
    legal = values.masked_fill(~masks, -1e9)
    groups = torch.stack([legal[:,i:i+8].max(1).values for i in range(0,56,8)]
                         + [legal[:,56],legal[:,57]],dim=1)
    kinds = torch.where(teachers < 56, teachers//8, teachers-49)
    return nn.functional.cross_entropy(groups,kinds)


def sample_plant_batch(replay, rng, batch_size, carrier_fraction=0., carriers=None):
    if carriers is None:
        carriers = [row for row in replay if len(row[0]) > CARRIER_FEATURE_INDEX
                    and row[0][CARRIER_FEATURE_INDEX] > .5] if carrier_fraction else []
    count = int(batch_size*carrier_fraction) if carriers else 0
    return ([carriers[int(rng.integers(len(carriers)))] for _ in range(count)]
            + [replay[int(rng.integers(len(replay)))] for _ in range(batch_size-count)])


def learn_plant(model, target, optimizer, replay, rng, updates, batch_size, demonstration_weight=.05,
                demonstration_kind_weight=0., carrier_sample_fraction=0.):
    if len(replay) < batch_size:
        return None
    replay = list(replay)  # Avoid repeated linear-time random indexing of a deque.
    losses = []
    carriers = [row for row in replay if row[0][CARRIER_FEATURE_INDEX] > .5] if carrier_sample_fraction else []
    model.train()
    for _ in range(updates):
        batch = sample_plant_batch(replay,rng,batch_size,carrier_sample_fraction,carriers)
        obs = torch.tensor(np.stack([t[0] for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_obs = torch.tensor(np.stack([t[3] for t in batch]), dtype=torch.float32)
        next_masks = torch.tensor(np.stack([t[4] for t in batch]), dtype=torch.bool)
        done = torch.tensor([t[5] for t in batch], dtype=torch.float32)
        teachers = torch.tensor([t[6] for t in batch], dtype=torch.long)
        masks = torch.tensor(np.stack([t[7] for t in batch]), dtype=torch.bool)
        with torch.no_grad():
            chosen = model(next_obs).masked_fill(~next_masks, -1e9).argmax(1)
            expected = rewards + GAMMA * (1 - done) * target(next_obs).gather(1, chosen[:, None]).squeeze(1)
        values = model(obs)
        loss = nn.functional.smooth_l1_loss(values.gather(1, actions[:, None]).squeeze(1), expected)
        loss += demonstration_weight * nn.functional.cross_entropy(values.masked_fill(~masks, -1e9), teachers)
        if demonstration_kind_weight:
            loss += demonstration_kind_weight * demonstration_kind_loss(values,teachers,masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        with torch.no_grad():
            for dst, src in zip(target.parameters(), model.parameters()):
                dst.lerp_(src, .02)
        losses.append(float(loss.detach()))
    model.eval()
    return float(np.mean(losses))
