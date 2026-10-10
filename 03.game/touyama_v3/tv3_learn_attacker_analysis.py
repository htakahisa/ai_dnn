"""Public-observation attacker analysis; no plant/guard policy is learned here."""
from dataclasses import dataclass
from collections import deque
import hashlib
import json

import numpy as np
import torch
from torch import nn

from game_core import PLANT_REQUIRED_TICKS
from touyama_v3.tv3_observer import FeatureHistory
from touyama_v3.tv3_attacker_entry_utility import utility_schema
from touyama_v3.tv3_attacker_route_planner import planner_schema
from touyama_v3.tv3_attacker_combat import combat_schema

ANALYSIS_VERSION = 3  # Online observations, belief-aware routes, adaptive continuation labels.
ROUTE_PRESSURE_WEIGHT = .15
ENTRY_PRESSURE_WEIGHT = .35
ENTRY_MODES = ("preserve", "supported")
MAX_ROUTES_PER_SITE = 6
TIME_MARGIN_TICKS = 5


def shortest_path(scenario, start, goals, blocked=()):
    goals, blocked = set(goals), set(blocked)
    queue, parent = deque([start]), {start: None}
    while queue:
        pos = queue.popleft()
        if pos in goals:
            path = []
            while pos is not None:
                path.append(pos)
                pos = parent[pos]
            return tuple(reversed(path))
        for nxt in scenario.neighbors(pos):
            if nxt not in parent and nxt not in blocked:
                parent[nxt] = pos
                queue.append(nxt)
    return ()


@dataclass(frozen=True)
class Route:
    site: str
    cells: tuple
    branches: tuple

    @property
    def key(self):
        return hashlib.sha256(repr((self.site, self.cells)).encode()).hexdigest()[:16]


def branch_sequence(scenario, cells):
    result = []
    for cell in cells:
        name = scenario.names[int(scenario.region[cell])]
        if not result or result[-1] != name:
            result.append(name)
    return tuple(result)


def candidate_routes(scenario, origin, remaining_ticks, limit=MAX_ROUTES_PER_SITE):
    """Bounded diverse simple cell paths via branch waypoints, including detours.

    Region letters annotate geometry, never create fictitious connections.
    Waypoint search is deliberately bounded, not an exhaustive path optimizer.
    """
    routes = []
    for side in ("L", "R"):
        goals = scenario.sites[side]
        other_sites = set(scenario.sites["R" if side == "L" else "L"])
        direct = shortest_path(scenario, origin, goals, other_sites)
        proposals = [direct]
        for cells in scenario.branches.values():
            waypoint = cells[len(cells) // 2]
            prefix = shortest_path(scenario, origin, (waypoint,), other_sites | set(goals))
            if not prefix:
                continue
            suffix = shortest_path(scenario, waypoint, goals, other_sites | set(prefix[:-1]))
            if suffix:
                proposals.append(prefix + suffix[1:])
        unique = {}
        for path in sorted(proposals, key=lambda p: (len(p), p)):
            if not path or len(path) != len(set(path)):
                continue
            if len(path) - 1 + PLANT_REQUIRED_TICKS + TIME_MARGIN_TICKS > remaining_ticks:
                continue
            sequence = branch_sequence(scenario, path)
            unique.setdefault(sequence, Route(side, path, sequence))
        routes.extend(list(unique.values())[:limit])
    return routes


class AttackerEncoder:
    def __init__(self, scenario):
        self.scenario = scenario
        self.history = FeatureHistory(scenario)
        self.fields = self.history.fields + [f"ally_{i}_{name}" for i in range(5)
            for name in ("charges", "spike", "plant_progress", "movement_disabled")]
        self.fields += [f"observed_fraction_{name}" for name in scenario.names]
        self.route_fields = ["site_R", "supported", "steps", "time_slack", "visible_fraction",
                             "known_exposure", "information_age"]
        self.route_fields += [f"route_{n}_{field}" for n in scenario.names
                              for field in ("fraction", "first", "last")]
        self.route_fields += ["belief_route_pressure", "belief_entry_pressure", "observed_empty_fraction", "information_gain"]
        self.region_cells = {n: tuple(map(tuple, np.argwhere((scenario.region == i) & (scenario.grid != 1))))
                             for i, n in enumerate(scenario.names)}
        self.visibility_cache = {}
        self.observed_cells = set()

    def reset(self):
        self.history.reset()
        self.observed_cells = set()

    def observe(self, snapshot, rounds):
        if snapshot.side != "A":
            raise ValueError("Attacker analysis requires an A-side public snapshot")
        base = self.history.encode(snapshot, rounds)
        self.observed_cells.update(snapshot.visible_cells)
        extra = [v for a in snapshot.allies for v in
                 (a.charges / 10, float(a.has_spike), a.plant_progress / PLANT_REQUIRED_TICKS,
                  float(a.movement_disabled > 0))]
        extra += [sum(p in self.observed_cells for p in self.region_cells[n]) / len(self.region_cells[n])
                  for n in self.scenario.names]
        return np.concatenate((base, np.asarray(extra, dtype=np.float32)))

    def route_features(self, snapshot, route, mode, placement=None):
        if mode not in ENTRY_MODES:
            raise ValueError(f"Unknown entry mode: {mode}")
        cells, visible = route.cells, set(snapshot.visible_cells)
        alive = {e.enemy_id for e in snapshot.enemies if e.alive}
        tracks = [(pos, tick) for i, (pos, tick, _) in self.history.tracks.items() if i in alive]
        exposure = sum(self.scenario.clear(p, pos) * max(0., 1 - (snapshot.tick - tick) / 20)
                       for p in cells for pos, tick in tracks)
        ages = [min(1., (snapshot.tick - tick) / 100) for _, tick in tracks]
        values = [float(route.site == "R"), float(mode == "supported"), (len(cells) - 1) / 100,
                  (snapshot.round_timer - len(cells) + 1 - PLANT_REQUIRED_TICKS) / 100,
                  sum(p in visible for p in cells) / len(cells), exposure / (5 * len(cells)),
                  float(np.mean(ages)) if ages else 1.]
        regions = [int(self.scenario.region[p]) for p in cells]
        for i in range(len(self.scenario.names)):
            indexes = [j for j, region in enumerate(regions) if region == i]
            values += [len(indexes) / len(cells), (indexes[0] + 1) / len(cells) if indexes else 0.,
                       (indexes[-1] + 1) / len(cells) if indexes else 0.]
        rows = placement.values() if placement is not None else [
            {name: float(enemy.alive) / len(self.scenario.names) for name in self.scenario.names}
            for enemy in snapshot.enemies]
        rows = list(rows)
        route_regions = {self.scenario.names[int(self.scenario.region[p])] for p in cells}
        entry_regions = {self.scenario.names[int(self.scenario.region[p])] for p in cells[-6:]}
        values += [sum(row.get(n, 0.) for row in rows for n in route_regions) / 5,
                   sum(row.get(n, 0.) for row in rows for n in entry_regions) / 5]
        prefix = cells[:8]
        values += [sum(p in visible and p not in {s.position for s in snapshot.sightings} for p in prefix) / len(prefix)]
        probe = cells[min(4, len(cells)-1)]
        if probe not in self.visibility_cache:
            self.visibility_cache[probe] = tuple(p for region in self.region_cells.values() for p in region
                if max(abs(p[0]-probe[0]), abs(p[1]-probe[1])) <= 6 and self.scenario.clear(probe, p))
        potential = self.visibility_cache[probe]
        values += [sum(p not in self.observed_cells for p in potential) / max(1, len(potential))]
        return np.asarray(values, dtype=np.float32)

    def schema(self):
        return {"version": ANALYSIS_VERSION, "board": self.scenario.grid.tolist(), "scenario": self.scenario.signature,
                "branches": self.scenario.branches, "fields": self.fields,
                "route_fields": self.route_fields, "modes": list(ENTRY_MODES),
                "max_routes_per_site": MAX_ROUTES_PER_SITE, "margin": TIME_MARGIN_TICKS,
                "executor": "public_online_combat_attack_v7", "entry_utility": utility_schema(), "combat": combat_schema(),
                "planner": planner_schema(), "risk_weights": [ROUTE_PRESSURE_WEIGHT, ENTRY_PRESSURE_WEIGHT],
                "sensor": "frc_public_team_v1"}


class AttackerAnalysisModel(nn.Module):
    def __init__(self, observation_size, route_size, regions):
        super().__init__()
        self.regions = regions
        self.encoder = nn.Sequential(nn.Linear(observation_size, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.placement = nn.Linear(64, 5 * (regions + 1))
        self.outcome = nn.Sequential(nn.Linear(64 + route_size, 64), nn.ReLU(), nn.Linear(64, 5))
        self.trained_rounds = 0

    def forward(self, observation, route):
        hidden = self.encoder(observation)
        return self.placement(hidden).reshape(-1, 5, self.regions + 1), self.outcome(torch.cat((hidden, route), -1))

    @torch.no_grad()
    def analyze(self, encoder, snapshot, observation, candidates=None):
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        placement_logits = self.placement(self.encoder(observation_tensor)).reshape(5, self.regions + 1)
        probabilities = np.column_stack((placement_logits[:, :self.regions].softmax(-1).numpy(), np.zeros(5)))
        sightings = {s.enemy_id: s.position for s in snapshot.sightings}
        for enemy in snapshot.enemies:
            if not enemy.alive or enemy.enemy_id in sightings:
                probabilities[enemy.enemy_id] = 0.
                region = self.regions if not enemy.alive else int(encoder.scenario.region[sightings[enemy.enemy_id]])
                probabilities[enemy.enemy_id, region] = 1.
        names = (*encoder.scenario.names, "dead")
        belief = {"placement": {i: dict(zip(names, map(float, row))) for i, row in enumerate(probabilities)},
                  "placement_entropy": float(-(probabilities * np.log(probabilities + 1e-8)).sum(-1).mean()),
                  "trained_rounds": self.trained_rounds}
        holder = next((a for a in snapshot.allies if a.alive and a.has_spike), None)
        origin = holder.position if holder else snapshot.spike_dropped
        if candidates is None:
            candidates = candidate_routes(encoder.scenario, origin, snapshot.round_timer) if origin else []
        if not candidates:
            return {"candidates": [], **belief}
        pairs = [(r, m) for r in candidates for m in ENTRY_MODES]
        x = torch.as_tensor(np.stack([observation] * len(pairs)), dtype=torch.float32)
        route_inputs = [encoder.route_features(snapshot, r, m, belief["placement"]) for r, m in pairs]
        z = torch.as_tensor(np.stack(route_inputs), dtype=torch.float32)
        _, logits = self(x, z)
        outcomes = logits.sigmoid().numpy()
        results = []
        for (route, mode), outcome, route_input in zip(pairs, outcomes, route_inputs):
            success, damage, losses, charges, ticks = map(float, outcome)
            results.append({"route": route, "mode": mode, "plant_probability": success,
                "expected_hp_damage": damage * 500, "expected_losses": losses * 5,
                "expected_ability_uses": charges * 10, "expected_plant_ticks_if_success": ticks * 100,
                "minimum_ticks": len(route.cells) - 1 + PLANT_REQUIRED_TICKS,
                "route_pressure": float(route_input[-4]), "entry_pressure": float(route_input[-3]),
                "observed_empty_fraction": float(route_input[-2]), "information_gain": float(route_input[-1]),
                "score": success - .25 * losses - .15 * damage - .05 * charges - .05 * ticks
                         - ROUTE_PRESSURE_WEIGHT * float(route_input[-4]) - ENTRY_PRESSURE_WEIGHT * float(route_input[-3])})
        results.sort(key=lambda r: r["score"], reverse=True)
        return {"candidates": results, **belief}


def optimize(model, optimizer, replay, rng, updates, batch_size):
    if not replay:
        return None
    model.train()
    losses = []
    for _ in range(updates):
        rounds = [replay[int(rng.integers(len(replay)))] for _ in range(batch_size)]
        samples = [(r, int(rng.integers(len(r["observations"])))) for r in rounds]
        x = torch.tensor(np.stack([r["observations"][i] for r, i in samples]), dtype=torch.float32)
        z = torch.tensor(np.stack([r["routes"][i] for r, i in samples]), dtype=torch.float32)
        y = torch.tensor(np.stack([r["outcomes"][i] for r, i in samples]), dtype=torch.float32)
        placement_y = torch.tensor(np.stack([r["placements"][i] for r, i in samples]), dtype=torch.long)
        placement, raw = model(x, z)
        success_loss = nn.functional.binary_cross_entropy_with_logits(raw[:, 0], y[:, 0])
        metric_loss = nn.functional.smooth_l1_loss(raw[:, 1:4].sigmoid(), y[:, 1:4])
        planted = y[:, 0] > .5
        time_loss = nn.functional.smooth_l1_loss(raw[planted, 4].sigmoid(), y[planted, 4]) if planted.any() else raw.sum() * 0
        loss = success_loss + metric_loss + time_loss + .25 * nn.functional.cross_entropy(
            placement.flatten(0, 1), placement_y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return float(np.mean(losses))


def load_analysis(path, scenario, opponent=None):
    encoder = AttackerEncoder(scenario)
    state = torch.load(path, map_location="cpu", weights_only=True)
    # JSON normalizes tuples consistently when a schema was serialized elsewhere.
    if json.dumps(state["schema"], sort_keys=True) != json.dumps(encoder.schema(), sort_keys=True):
        raise ValueError("Attacker analysis schema does not match the current map/executor/LOS. "
                         "After the wall-corner LOS fix, retrain attacker analysis with RESUME_TRAINING=False before plant.")
    if opponent is not None and state["opponent"] != opponent:
        raise ValueError("Attacker analysis opponent mismatch")
    model = AttackerAnalysisModel(len(encoder.fields), len(encoder.route_fields), len(scenario.names))
    model.load_state_dict(state["model"])
    model.trained_rounds = state["trained_rounds"]
    model.eval()
    return model, encoder, state
