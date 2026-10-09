"""Opponent-specific shared-player plant DQN, driven by frozen route analysis."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import io
import json
import math

import numpy as np
import torch
from torch import nn

from frc_v1 import FACING
from frc_v1.actions import FrcAction, KINDS, MOVE_STEPS, validate_action, target_required
from toruAI_v4.tv4_defender_policy import PolicyEncoder, OBS_DIM as BASE_OBS_DIM, ACTION_DIM, MOVEMENTS
from toruAI_v4.tv4_observer import facing
from toruAI_v4.tv4_learn_attacker_analysis import load_analysis
from toruAI_v4.tv4_attacker_entry_utility import predicted_entry_utility, utility_schema, pending_flash_impact, flash_entry_step
from toruAI_v4.tv4_attacker_route_planner import planner_schema
from toruAI_v4.tv4_attacker_combat import AttackerCombatCoach, effective_utility, combat_schema, COMBAT_FEATURES, plant_ready

PLANT_VERSION = 4
PLANT_ACTION = ACTION_DIM - 2
ORB_ACTION = ACTION_DIM - 1
REGIONS = 14
OBS_DIM = BASE_OBS_DIM + 5 * (REGIONS + 1) + 24 + COMBAT_FEATURES
GAMMA = .98
ROUTE_CORRIDOR_CELLS = 2
ROUTE_RETREAT_CELLS = 3
MAX_EDGE_TRAVERSALS = 2


class PlantDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, 256), nn.ReLU(), nn.Linear(256, 128),
                                 nn.ReLU(), nn.Linear(128, ACTION_DIM))

    def forward(self, observation):
        return self.net(observation)


def policy_schema(scenario):
    return {"version": PLANT_VERSION, "board": scenario.grid.tolist(), "branches": scenario.branches,
        "obs_dim": OBS_DIM, "action_dim": ACTION_DIM, "gamma": GAMMA,
        "route_corridor": ROUTE_CORRIDOR_CELLS, "retreat": ROUTE_RETREAT_CELLS,
        "edge_limit": MAX_EDGE_TRAVERSALS, "planner": planner_schema(),
        "sensor": "frc_public_team_v1", "scope": "preplant_only",
        "site_choice": "public_online_analysis_no_site_balancing", "execution": "adaptive_combat_plant_v5",
        "action_selection": "learned_no_forced_forward_or_plant",
        "combat": combat_schema(),
        "entry_utility": utility_schema()}


def load_frozen_analysis(directory, opponent, scenario):
    path = Path(directory) / opponent / "attacker_analysis_best.pt"
    payload = path.read_bytes()
    model, _, state = load_analysis(io.BytesIO(payload), scenario, opponent)
    model.requires_grad_(False)
    return model, hashlib.sha256(payload).hexdigest(), state


def load_plant(path, scenario, opponent=None, analysis_hash=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if json.dumps(state["schema"], sort_keys=True) != json.dumps(policy_schema(scenario), sort_keys=True):
        raise ValueError("Plant model map/action/schema mismatch")
    if opponent is not None and state["opponent"] != opponent:
        raise ValueError("Plant model opponent mismatch")
    if analysis_hash is not None and state["analysis_hash"] != analysis_hash:
        raise ValueError("Frozen analysis changed since plant training")
    model = PlantDQN()
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
    def __init__(self, scenario):
        if len(scenario.names) != REGIONS:
            raise ValueError("Plant encoder needs the current 14-region map")
        self.scenario = scenario
        self.base = PolicyEncoder(scenario)
        self.combat_coach = AttackerCombatCoach(scenario)

    def encode(self, snapshot, ally, goal, route, analysis, tracks, masks, cursor, route_mode):
        site = np.asarray([float(route.site == "L"), float(route.site == "R")], np.float32)
        base = self.base.encode(snapshot, ally, goal, site, tracks, masks)
        targets = predicted_targets(self.scenario, snapshot, ally, analysis)
        entry_plan = predicted_entry_utility(self.scenario, snapshot, ally, analysis, route, masks)
        seen = self.combat_coach.contacts(snapshot, ally, ally.position)
        direction = ally.facing if ally.forced_facing else facing(ally.position,
            min(seen, key=lambda p: max(abs(p[0]-ally.position[0]), abs(p[1]-ally.position[1]))) if seen else targets[0][0] if targets else goal)
        movement = np.flatnonzero(base.mask[:40])
        def initial_score(i):
            dr, dc = MOVE_STEPS.get(base.actions[i].kind, (0, 0))
            p = ally.position[0]+dr, ally.position[1]+dc
            d = base.distances[p]
            return d if d >= 0 else 9999, int(base.actions[i].facing != direction), i
        initial = int(min(movement, key=initial_score))
        dr, dc = MOVE_STEPS.get(base.actions[initial].kind, (0, 0))
        destination = ally.position[0]+dr, ally.position[1]+dc
        hint = self.combat_coach.preaim_hint(snapshot, ally, destination, goal, analysis)
        advice = self.combat_coach.advise(snapshot, ally, destination, hint, tracks)
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
                                           or predicted_cast or effective_utility(self.scenario, snapshot, ally, cast))
                except (ValueError, TypeError, IndexError):
                    legal[action_index] = False
        extra = [float(ally.has_spike), ally.plant_progress / 4, float(route.site == "R"),
                 cursor / max(1, len(route.cells) - 1), len(route.cells) / 100,
                 float(route_mode == "supported"), analysis["placement_entropy"] / math.log(REGIONS + 1),
                 float(snapshot.spike_dropped is not None), goal[0] / 26, goal[1] / 44]
        extra += [float(a.has_spike) for a in snapshot.allies]
        for i in range(3):
            p, mass = targets[i] if i < len(targets) else ((0, 0), 0.)
            extra += [p[0] / 26, p[1] / 44, mass / 5]
        beliefs = [analysis["placement"][i][n] for i in range(5) for n in (*self.scenario.names, "dead")]
        observation = np.concatenate((base.observation, np.asarray(beliefs + extra + advice.features(ally, self.scenario.grid.shape), np.float32)))
        if len(observation) != OBS_DIM or not np.isfinite(observation).all():
            raise ValueError("Invalid plant observation")
        if legal[PLANT_ACTION] and plant_ready(ally, advice):
            teacher = PLANT_ACTION
        else:
            movement = np.flatnonzero(legal[:40])
            def score(index):
                dr, dc = MOVE_STEPS.get(MOVEMENTS[index // 8], (0, 0))
                p = ally.position[0] + dr, ally.position[1] + dc
                distance = base.distances[p]
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
            if entry_plan and route_mode == "supported":
                choices = [43]
            elif ally.ability_name in ("FLASH", "SMOKE", "ASH") and threat and ally.ability_name not in active:
                choices = [40 + i for i, p in enumerate(coordinates[:5]) if p is not None]
            elif ally.ability_name == "RECON" and not seen and threat and "RECON" not in active:
                choices = [43, 44, 45]
            elif ally.ability_name == "DANCE":
                choices = [40 + i for i, a in enumerate(allies) if a.alive and a.hp < .7 * a.max_hp]
            teacher = next((i for i in choices if legal[i]), teacher)
            # Only pause before leaving cover; an exposed player should not be
            # taught to freeze. Recon flights never require an entry pause.
            impact = pending_flash_impact(self.scenario, snapshot, ally)
            if impact is not None and route_mode == "supported" and advice.reason == "advance":
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
        return PlantInputs(observation, legal, tuple(actions), teacher, base.distances, goal, advice)


def learn_plant(model, target, optimizer, replay, rng, updates, batch_size, demonstration_weight=.05):
    if len(replay) < batch_size:
        return None
    losses = []
    model.train()
    for _ in range(updates):
        batch = [replay[int(rng.integers(len(replay)))] for _ in range(batch_size)]
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
