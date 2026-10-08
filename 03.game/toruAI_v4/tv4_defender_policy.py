"""Name-free, shared per-player policy inputs and legal action candidates."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from frc_v1 import FACING
from frc_v1.actions import FrcAction, KINDS, MOVE_STEPS, build_masks, validate_action
from grid_paths import distance_map
from toruAI_v4.tv4_observer import facing

POLICY_VERSION = 2
ABILITIES = ("FLASH", "SMOKE", "RECON", "HUNT", "RAMP", "DANCE", "ASH")
ULTIMATES = ("TUNNEL", "ESCAPE", "MONITOR", "RAID", "NEON", "SERENADE", "BALEMOON")
MOVEMENTS = ("STAY", "N", "E", "S", "W")
TARGETS = 8
ACTION_DIM = 40 + 2 * TARGETS + 2  # 5 moves x 8 directions; utility, ult, defuse, orb.
OBS_DIM = 505
DEFUSE_ACTION, ORB_ACTION = ACTION_DIM - 2, ACTION_DIM - 1
GAMMA = .98


class DefenderDQN(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, 128),
                                 nn.ReLU(), nn.Linear(128, ACTION_DIM))

    def forward(self, x):
        return self.net(x)


@dataclass
class Inputs:
    observation: np.ndarray
    mask: np.ndarray
    actions: tuple
    goal: tuple
    distances: np.ndarray
    teacher: int


def staging_positions(scenario):
    """User-specified a/b rally areas; no fixed ability positions."""
    return {side: tuple(p for cells in scenario.rally_points[side].values() for p in cells)
            for side in ("L", "R")}


def legacy_staging_positions(scenario):
    """Map-only covered cells 4..8 walking ticks outside each plant zone."""
    result = {}
    for side in ("L", "R"):
        distances = scenario.site_dist[side]
        cells = [tuple(map(int, p)) for p in np.argwhere((distances >= 4) & (distances <= 8))
                 if scenario.grid[tuple(p)] != 1 and scenario.setup[tuple(p)] == 0]
        cells.sort(key=lambda p: (sum(scenario.clear(p, s) for s in scenario.sites[side]),
                                  distances[p], scenario.spawn_dist[p], p))
        selected = []
        for p in cells:
            if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) >= 2 for q in selected):
                selected.append(p)
            if len(selected) == 5:
                break
        if len(selected) != 5:
            raise ValueError(f"Five staging positions are required for site {side}")
        result[side] = tuple(selected)
    return result


def assign_goals(snapshot, scenario, probability, staging):
    """Assignments depend on positions; names only identify action recipients."""
    own = sorted((a for a in snapshot.allies if a.alive), key=lambda a: (a.position, a.ability_name))
    if snapshot.is_planted:
        plant = snapshot.spike_planted
        if plant is None:
            raise ValueError("Planted spike has no public location")
        cells = [(plant[0] + dr, plant[1] + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                 if 0 <= plant[0] + dr < scenario.grid.shape[0] and 0 <= plant[1] + dc < scenario.grid.shape[1]
                 and scenario.grid[plant[0] + dr, plant[1] + dc] != 1]
    elif snapshot.phase != "setup" and max(probability) >= .65:
        cells = list(staging[("L", "R")[int(np.argmax(probability))]])
    else:
        cells = [p.watch for p in scenario.posts]
    result = {}
    for ally in own:
        if not cells:
            cells = [snapshot.spike_planted] if snapshot.is_planted else [p.watch for p in scenario.posts]
        goal = min(cells, key=lambda p: (distance_map(scenario.grid, p)[ally.position], p))
        cells.remove(goal)
        result[ally.slot] = goal
    return result


class PolicyEncoder:
    """No player/team IDs; other players are pooled or ordered by geometry."""
    def __init__(self, scenario):
        self.scenario = scenario
        self.dim = None

    def encode(self, snapshot, ally, goal, probabilities, tracks, masks=None):
        masks = build_masks(snapshot) if masks is None else masks
        board = np.asarray(snapshot.grid)
        h, w = board.shape
        origin = ally.position
        route_board = board
        if snapshot.phase == "setup":
            route_board = board.copy()
            allowed = np.zeros(board.shape, bool)
            for pos in snapshot.setup_cells:
                allowed[pos] = True
            route_board[~allowed] = 1
        distances = distance_map(route_board, goal)
        visible, smoke = set(snapshot.visible_cells), set(snapshot.smoke_cells)
        allies = sorted(snapshot.allies, key=lambda a: (a.position, a.ability_name))
        occupied = {a.position for a in allies if a.alive}
        alive_ids = {e.enemy_id for e in snapshot.enemies if e.alive}
        current = {s.enemy_id: s.position for s in snapshot.sightings}
        known = [(pos, tick, velocity, enemy_id in current) for enemy_id, (pos, tick, velocity) in tracks.items()
                 if enemy_id in alive_ids and snapshot.tick - tick <= 15]
        known.sort(key=lambda e: (abs(e[0][0] - origin[0]) + abs(e[0][1] - origin[1]), e[0]))
        target = known[0][0] if known else goal
        direction = ally.facing if ally.forced_facing else facing(origin, target)
        features = [ally.hp / 100, ally.max_hp / 100, ally.charges / 10, ally.points / 10, ally.cost / 10,
                    ally.accuracy, ally.hs_rate, ally.dodge, ally.reaction / 200, ally.effective_iq / 200,
                    float(ally.alive), float(ally.blind > 0), float(ally.reveal > 0),
                    ally.defuse_progress / 6, float(ally.forced_facing), float(ally.movement_disabled > 0),
                    origin[0] / (h - 1), origin[1] / (w - 1)]
        features += [float(ally.ability_name == a) for a in ABILITIES]
        features += [float(ally.ultimate_name == a) for a in ULTIMATES]
        features += [float(ally.facing == f) for f in FACING]
        alive_allies = [a for a in allies if a.alive]
        pool = np.asarray([[a.hp / 100, a.position[0] / (h - 1), a.position[1] / (w - 1), a.charges / 10,
                            a.accuracy, a.hs_rate, a.dodge, a.effective_iq / 200] for a in alive_allies], np.float32)
        features += np.concatenate((pool.mean(0), pool.min(0), pool.max(0))).tolist() if len(pool) else [0.] * 24
        features += [sum(a.ability_name == kind for a in alive_allies) / 5 for kind in ABILITIES]
        features += [sum(a.charges for a in alive_allies if a.ability_name == kind) / 10 for kind in ABILITIES]
        for other in allies:
            features += [other.position[0] / (h - 1), other.position[1] / (w - 1), other.hp / 100,
                         float(other.alive), other.charges / 10]
            features += [float(other.ability_name == kind) for kind in ABILITIES]
            features += [float(other.facing == f) for f in FACING]
        features += [*probabilities, max(probabilities), snapshot.round_number / 12, snapshot.tick / 200,
                    snapshot.round_timer / 100, snapshot.detonate_timer / 55, float(snapshot.is_planted),
                    len(alive_allies) / 5, len(alive_ids) / 5, goal[0] / (h - 1), goal[1] / (w - 1),
                    max(0, distances[origin]) / 100,
                    (goal[0] - origin[0]) / h, (goal[1] - origin[1]) / w,
                    float(snapshot.spike_dropped is not None), float(snapshot.phase == "setup")]
        for i in range(5):
            if i < len(known):
                pos, tick, velocity, seen = known[i]
                features += [1., float(seen), pos[0] / (h - 1), pos[1] / (w - 1),
                             (pos[0] - origin[0]) / h, (pos[1] - origin[1]) / w,
                             min(1., (snapshot.tick - tick) / 15), *np.clip(velocity, -1, 1)]
            else:
                features += [0.] * 9
        known_positions = {e[0] for e in known}
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                p = origin[0] + dr, origin[1] + dc
                valid = 0 <= p[0] < h and 0 <= p[1] < w
                features += [float(not valid or board[p] == 1), float(p in occupied),
                             float(p in visible), float(p in known_positions), float(p in smoke)]
        for kind in MOVEMENTS:
            dr, dc = MOVE_STEPS.get(kind, (0, 0))
            p = origin[0] + dr, origin[1] + dc
            valid = 0 <= p[0] < h and 0 <= p[1] < w and board[p] != 1
            features += [float(valid), float(p in occupied and p != origin),
                         float(np.clip(distances[origin] - distances[p], -1, 1)) if valid else 0.,
                         sum(self.scenario.clear(p, e[0]) for e in known) / 5 if valid else 0.]
        actions, mask = [], []
        for kind in MOVEMENTS:
            for f in FACING:
                actions.append(FrcAction(kind, f))
                mask.append(bool(masks.kind[ally.slot, KINDS.index(kind)] and masks.facing[ally.slot, FACING.index(f)]))
        coordinates = [e[0] for e in known[:5]] + [None] * max(0, 5 - len(known))
        coordinates += [goal, snapshot.spike_planted or snapshot.spike_dropped, origin]
        for kind in ("ABILITY", "ULTIMATE"):
            for index, pos in enumerate(coordinates):
                if kind == "ABILITY" and ally.ability_name == "DANCE":
                    recipient = allies[index] if index < len(allies) else None
                    action = FrcAction(kind, direction, ally_slot=recipient.slot if recipient else None)
                else:
                    ultimate_no_target = kind == "ULTIMATE" and ally.ultimate_name in ("RAID", "MONITOR", "TUNNEL")
                    action = FrcAction(kind, direction, None if ultimate_no_target else pos)
                try:
                    validate_action(snapshot, masks, ally.slot, action)
                    legal = pos is not None or (kind == "ABILITY" and ally.ability_name == "DANCE")
                except (ValueError, TypeError, IndexError):
                    legal = False
                actions.append(action)
                mask.append(legal)
        for kind in ("DEFUSE", "COLLECT_ORB"):
            actions.append(FrcAction(kind, direction))
            mask.append(bool(masks.kind[ally.slot, KINDS.index(kind)]))
        mask = np.asarray(mask, bool)
        observation = np.asarray(features, np.float32)
        if self.dim is None:
            self.dim = len(observation)
        if len(actions) != ACTION_DIM or len(observation) != OBS_DIM or len(observation) != self.dim or not np.isfinite(observation).all() or not mask.any():
            raise ValueError("Invalid generic defender inputs")
        # Demonstration policy is only used when explicitly requested in training.
        if mask[DEFUSE_ACTION]:
            teacher = DEFUSE_ACTION
        else:
            candidates = np.flatnonzero(mask[:40])
            def score(action):
                kind = MOVEMENTS[action // 8]
                dr, dc = MOVE_STEPS.get(kind, (0, 0))
                p = origin[0] + dr, origin[1] + dc
                aim = 0 if FACING[action % 8] == direction else 1
                risk = (.5 if snapshot.is_planted else 5) * sum(self.scenario.clear(p, e[0]) for e in known)
                return distances[p] + risk, aim, action
            teacher = int(min(candidates, key=score))
            # Early training demonstrations include utility timing. Inference
            # still chooses solely from the trained Q values and legal mask.
            active_effects = {e.kind for e in snapshot.effects if e.phase in ("flight", "active")}
            can_spend = snapshot.is_planted or ally.charges > 1
            if known and can_spend and ally.ability_name == "FLASH" and "FLASH" not in active_effects and mask[40]:
                teacher = 40
            elif len(known) >= 2 and can_spend and ally.ability_name == "SMOKE" and "SMOKE" not in active_effects and mask[40]:
                teacher = 40
            elif not current and can_spend and max(probabilities) >= .65 and ally.ability_name == "RECON" and "RECON" not in active_effects and mask[45]:
                teacher = 45  # Public prediction goal, never a hidden enemy position.
            elif ally.ability_name == "DANCE" and can_spend:
                healing = [i for i in range(5) if mask[40 + i] and allies[i].hp < .6 * allies[i].max_hp]
                if healing:
                    teacher = 40 + min(healing, key=lambda i: allies[i].hp)
        return Inputs(observation, mask, tuple(actions), goal, distances, teacher)


def learn_dqn(model, target, optimizer, replay, rng, updates=100, batch_size=64):
    if len(replay) < batch_size:
        return None
    losses = []
    for _ in range(updates):
        batch = [replay[int(rng.integers(len(replay)))] for _ in range(batch_size)]
        obs = torch.tensor(np.stack([t[0] for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_obs = torch.tensor(np.stack([t[3] for t in batch]), dtype=torch.float32)
        masks = torch.tensor(np.stack([t[4] for t in batch]), dtype=torch.bool)
        done = torch.tensor([t[5] for t in batch], dtype=torch.float32)
        with torch.no_grad():
            selected = model(next_obs).masked_fill(~masks, -1e9).argmax(1)
            future = target(next_obs).gather(1, selected[:, None]).squeeze(1)
            expected = rewards + GAMMA * (1 - done) * future
        values = model(obs).gather(1, actions[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(values, expected)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        with torch.no_grad():
            for dest, source in zip(target.parameters(), model.parameters()):
                dest.lerp_(source, .02)
        losses.append(float(loss.detach()))
    return float(np.mean(losses))
