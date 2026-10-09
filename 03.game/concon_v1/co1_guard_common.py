"""Guard observations and learned movement, facing and utility actions."""

import math

import numpy as np
import torch
from torch import nn

from game_core import FACING_DIRECTIONS, FACING_VECTORS
from grid_lines import line_cells
from concon_v1.co1_attacker_common import bfs_distance_map, GORIGONS

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
ABILITIES = ("SMOKE", "FLASH", "RECON")
TARGET_COUNT = 3  # planted spike, assigned facing point, known enemy
LEGACY_ACTION_DIM = (len(MOVES) + len(ABILITIES) * TARGET_COUNT) * 8
ULTIMATE_ACTION = LEGACY_ACTION_DIM
PRE_COUNTER_ACTION_DIM = LEGACY_ACTION_DIM + TARGET_COUNT * 8
SELF_SMOKE_ACTION = PRE_COUNTER_ACTION_DIM
ACTION_DIM = SELF_SMOKE_ACTION + 8
WAIT_ACTION = 4 * 8
MAP_CHANNELS = 6
LEGACY_FEATURE_DIM = 34 + 5 * 15 + 5 * 16
STATUS_FIELDS = ("blind_remaining", "reveal_remaining", "electric_remaining",
                 "life_contract_remaining", "movement_disabled_remaining")
FEATURE_DIM = LEGACY_FEATURE_DIM + len(STATUS_FIELDS) + 5 + 2
MEMORY_TICKS = 12


def observation_dim(scenario):
    return MAP_CHANNELS * scenario.grid.size + FEATURE_DIM


def compatible_observation_dims(scenario):
    return (observation_dim(scenario), MAP_CHANNELS * scenario.grid.size + LEGACY_FEATURE_DIM)


def threat_exposure(position, threats, grid, smoke, revealed=False):
    """Conservative firing lanes from IQ-disclosed positions and recent memory."""
    count = 0
    for source, through_smoke in threats:
        cells = line_cells(position, source)
        if any(grid[cell] == 1 for cell in cells):
            continue
        if (len(cells) > 2 and not revealed and not through_smoke
                and any(cell in smoke for cell in cells)):
            continue
        count += 1
    return float(count)


def facing_onehot(direction):
    return [float(direction == item) for item in FACING_DIRECTIONS]


def aim_alignment(source, target, direction):
    dr, dc = target[0] - source[0], target[1] - source[1]
    length = math.hypot(dr, dc)
    if not length:
        return 1.0
    vx, vy = FACING_VECTORS[direction]
    return (dc * vx + dr * vy) / length


def clear_shot(char, enemy, chars, grid, smoke, game=None):
    """Geometry from perceived positions only, with the engine's smoke rules."""
    start, end = tuple(char.pos), tuple(enemy.pos)
    height, width = grid.shape
    if not all(0 <= r < height and 0 <= c < width for r, c in (start, end)):
        return False
    cells = line_cells(start, end)
    if any(grid[cell] == 1 for cell in cells):
        return False
    ignore_smoke = (getattr(char, "sees_through_smoke", False)
                    or getattr(enemy, "reveal_remaining", 0) > 0)
    if len(cells) > 2 and not ignore_smoke and any(cell in smoke for cell in cells):
        return False
    occupied = {tuple(other.pos) for other in chars
                if other.name not in (char.name, enemy.name)
                and getattr(other, "is_alive", True)
                and getattr(other, "position_known", True)}
    return not occupied.intersection(cells[1:-1])


class GuardDQN(nn.Module):
    def __init__(self, scenario, navigation=False):
        super().__init__()
        self.height, self.width = scenario.grid.shape
        self.map_size = MAP_CHANNELS * scenario.grid.size
        self.encoder = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 6)), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(32 * 4 * 6 + FEATURE_DIM, 256),
                                  nn.ReLU(), nn.Linear(256, ACTION_DIM))
        # A learned tabular head for the fixed map's quiet positioning task.
        # Combat retains the perception-dependent network. No route search or
        # move/facing override is performed when this model is used at runtime.
        self.navigation = navigation
        if navigation:
            self.navigation_values = nn.Embedding(5 * self.height * self.width * 16, 40)
            nn.init.zeros_(self.navigation_values.weight)
            self.navigation_yield_values = nn.Embedding(5 * 16, 40)
            nn.init.zeros_(self.navigation_yield_values.weight)
            self.register_buffer("navigation_other_posts", torch.tensor([
                [scenario.positions[other][0] * self.width + scenario.positions[other][1]
                 for other in "abcde" if other != letter] for letter in "abcde"], dtype=torch.long))

    def forward(self, observations):
        maps = observations[:, :self.map_size].reshape(-1, MAP_CHANNELS, self.height, self.width)
        features = observations[:, self.map_size:]
        values = self.head(torch.cat((self.encoder(maps), features), dim=1))
        if self.navigation:
            rows = (features[:, 0] * self.height).round().long().clamp(0, self.height - 1)
            cols = (features[:, 1] * self.width).round().long().clamp(0, self.width - 1)
            slots = features[:, 29:34].argmax(1)
            other_posts = self.navigation_other_posts[slots]
            occupied = ((maps[:, 3] - maps[:, 2]).flatten(1).gather(1, other_posts) > 0)
            occupancy = (occupied.long() * torch.tensor([1, 2, 4, 8], device=observations.device)).sum(1)
            indices = (slots * self.height * self.width + rows * self.width + cols) * 16 + occupancy
            quiet = ((features[:, 12] == 0) & (features[:, 16] == 0)
                     & (features[:, -1] == 0))
            navigation_moves = self.navigation_values(indices)
            ally_features = features[:, 34:109].reshape(-1, 5, 15)
            self_indices = ((ally_features[:, :, 1:3] - features[:, None, :2]).square().sum(2)
                            + (1 - ally_features[:, :, 0]) * 100).argmin(1)
            ally_rows = (ally_features[:, :, 1] * self.height).round().long()
            ally_cols = (ally_features[:, :, 2] * self.width).round().long()
            higher = ((torch.arange(5, device=observations.device)[None, :] < self_indices[:, None])
                      & (ally_features[:, :, 0] > 0))
            higher_neighbors = torch.zeros(len(values), dtype=torch.long, device=observations.device)
            for bit, (dr, dc) in enumerate(MOVES[:4]):
                neighbor = higher & (ally_rows == rows[:, None] + dr) & (ally_cols == cols[:, None] + dc)
                higher_neighbors += neighbor.any(1).long() * (1 << bit)
            preferred = navigation_moves.reshape(-1, 5, 8).amax(2).argmax(1)
            navigation_moves = navigation_moves + self.navigation_yield_values(preferred * 16 + higher_neighbors)
            navigation_values = torch.cat((navigation_moves, values[:, 40:]), dim=1)
            values = torch.where(quiet[:, None], navigation_values, values)
        return values


def load_guard_weights(model, state, *, strict=True):
    """Expand old checkpoints without changing any existing learned outputs."""
    state = dict(state)
    previous = state["head.0.weight"]
    if previous.shape[1] == model.head[0].in_features - (FEATURE_DIM - LEGACY_FEATURE_DIM):
        state["head.0.weight"] = torch.cat((previous, previous.new_zeros(
            previous.shape[0], FEATURE_DIM - LEGACY_FEATURE_DIM)), dim=1)
    if state["head.2.bias"].shape[0] == LEGACY_ACTION_DIM:
        for key in ("head.2.weight", "head.2.bias"):
            previous = state[key]
            # Until trained, ultimates start with the corresponding WAIT values.
            state[key] = torch.cat((previous, previous[32:40].repeat(
                (TARGET_COUNT,) + (1,) * (previous.ndim - 1))), dim=0)
    if state["head.2.bias"].shape[0] == PRE_COUNTER_ACTION_DIM:
        for key in ("head.2.weight", "head.2.bias"):
            previous = state[key]
            state[key] = torch.cat((previous, previous[32:40]), dim=0)
    return model.load_state_dict(state, strict=strict)


def build_inputs(controller, char, state, *, counter_features=True):
    scenario = controller.scenario
    grid = np.asarray(state["grid"])
    height, width = grid.shape
    tick = int(state.get("battle_tick", 0))
    chars = state.get("chars", [])
    slot = controller.assignments[char.name]
    goal, aim = scenario.positions[slot], scenario.facing_points[slot]
    spike = state.get("planted_pos")
    spike = tuple(map(int, spike)) if spike is not None else goal
    position = tuple(map(int, char.pos))
    smoke = set(map(tuple, state.get("smoke_cells", ())))
    tap_info = state.get("defender_defuse_info") or {}
    tap_progress = max((float(value[0]) / max(1, float(value[1]))
                        for value in tap_info.values()), default=0.0)
    tap_remaining = min((max(0.0, float(required) - float(progress))
                         for progress, required in tap_info.values() if progress > 0), default=10.0)
    statuses = [max(0.0, float(getattr(char, field, 0))) for field in STATUS_FIELDS]
    impaired = any(statuses[:4])  # Smoke alone does not call for evasion.
    known = [other for other in chars if other.team != char.team
             and getattr(other, "is_alive", True) and getattr(other, "position_known", True)
             and 0 <= other.pos[0] < height and 0 <= other.pos[1] < width]
    # Store only IQ-filtered disclosures; never read hidden live enemy positions.
    for enemy in known:
        controller.sightings[enemy.name] = (tuple(enemy.pos), tick)
    for name, (_, seen_tick) in list(controller.sightings.items()):
        if tick < seen_tick or tick - seen_tick >= MEMORY_TICKS:
            del controller.sightings[name]
    fireable = [enemy for enemy in known if clear_shot(char, enemy, chars, grid, smoke)]
    fireable.sort(key=lambda enemy: (
        -float(tap_info.get(enemy.name, (0, 1))[0]),
        max(abs(enemy.pos[0] - position[0]), abs(enemy.pos[1] - position[1])), enemy.name))
    target = fireable[0] if fireable else None
    known.sort(key=lambda enemy: (-float(tap_info.get(enemy.name, (0, 1))[0]),
                                 math.dist(position, enemy.pos), enemy.name))
    remembered = sorted(controller.sightings.values(),
                        key=lambda item: (-item[1], math.dist(position, item[0])))
    targets = (spike, aim, tuple(known[0].pos) if known else
               remembered[0][0] if counter_features and remembered else None)
    smoke_vision = {enemy.name: bool(getattr(enemy, "sees_through_smoke", False)) for enemy in known}
    dead = {other.name for other in chars if not getattr(other, "is_alive", True)}
    threats = [(pos, smoke_vision.get(name, False))
               for name, (pos, _) in controller.sightings.items() if name not in dead]
    exposures = {}
    for dr, dc in MOVES:
        cell = (position[0] + dr, position[1] + dc)
        r, c = cell
        exposures[cell] = (threat_exposure(cell, threats, grid, smoke, statuses[1] > 0 or statuses[2] > 0)
                           if 0 <= r < height and 0 <= c < width and grid[r, c] != 1 else float(len(threats)))
    stopped = controller.stationary_ticks(char, tick)
    distance_goal = int(bfs_distance_map(grid, goal)[position])
    spike_in_bounds = 0 <= spike[0] < height and 0 <= spike[1] < width
    distance_spike = int(bfs_distance_map(grid, spike)[position]) if spike_in_bounds else -1
    spike_line = line_cells(position, spike) if spike_in_bounds else []
    spike_clear = bool(spike_line and all(grid[cell] != 1 for cell in spike_line)
                       and (len(spike_line) <= 2 or not any(cell in smoke for cell in spike_line)))

    maps = np.zeros((MAP_CHANNELS, height, width), dtype=np.float32)
    maps[0] = np.where(grid == 1, 1.0, np.where(grid == 2, 0.5, 0.0))
    for r, c in smoke:
        if 0 <= r < height and 0 <= c < width:
            maps[1, r, c] = 1
    maps[2, position[0], position[1]] = 1
    maps[5, goal[0], goal[1]] = 1
    for other in chars:
        if other.team == char.team and getattr(other, "is_alive", True):
            r, c = map(int, other.pos)
            if 0 <= r < height and 0 <= c < width:
                maps[3, r, c] = 1
    for enemy_pos, seen_tick in controller.sightings.values():
        r, c = map(int, enemy_pos)
        if 0 <= r < height and 0 <= c < width:
            maps[4, r, c] = 1 - (tick - seen_tick) / MEMORY_TICKS

    def normalized(pos):
        return [pos[0] / height, pos[1] / width]

    features = (normalized(position) + normalized(goal) + normalized(aim) + normalized(spike)
                + [distance_goal / 100, distance_spike / 100,
                   char.hp / max(1, char.max_hp), float(state.get("detonate_timer", 0)) / 50,
                   float(tap_progress > 0), tap_progress, min(stopped, 4) / 4,
                   float(getattr(char, "moved_this_tick", False)),
                   float(target is not None), float(spike_clear)]
                + facing_onehot(char.facing)
                + [min(2, getattr(char, ability.lower() + "_charges", 0)) / 2
                   for ability in ABILITIES]
                + [float(slot == letter) for letter in "abcde"])
    allies = {other.name: other for other in chars if other.team == char.team}
    for name in GORIGONS.players:
        ally = allies.get(name)
        if ally is None or not ally.is_alive:
            features.extend([0.0] * 15)
        else:
            features.extend([1.0] + normalized(ally.pos) + [ally.hp / max(1, ally.max_hp)]
                            + facing_onehot(ally.facing)
                            + [min(2, getattr(ally, ability.lower() + "_charges", 0)) / 2
                               for ability in ABILITIES])
    enemies = sorted([other for other in chars if other.team != char.team], key=lambda e: e.name)[:5]
    for index in range(5):
        enemy = enemies[index] if index < len(enemies) else None
        memory = controller.sightings.get(enemy.name) if enemy is not None else None
        disclosed = enemy is not None and enemy in known
        if enemy is None:
            features.extend([0.0] * 16)
            continue
        pos = tuple(enemy.pos) if disclosed else memory[0] if memory else None
        features.extend([float(pos is not None), float(enemy.is_alive)]
                        + (normalized(pos) if pos is not None else [0.0, 0.0])
                        + [enemy.hp / max(1, enemy.max_hp) if disclosed else 0.0]
                        + (facing_onehot(enemy.facing) if disclosed else [0.0] * 8)
                        + [float(getattr(enemy, "reveal_remaining", 0) > 0) if disclosed else 0.0,
                           float(tap_info.get(enemy.name, (0, 1))[0]) > 0,
                           (tick - memory[1]) / MEMORY_TICKS if memory else 1.0])
    features.extend([min(value, 10) / 10 for value in statuses]
                    + [exposures[(position[0] + dr, position[1] + dc)] / 5 for dr, dc in MOVES]
                    + [min(tap_remaining, 10) / 10, float(impaired)])
    observation = np.concatenate((maps.ravel(), np.asarray(features, dtype=np.float32)))
    if len(observation) != observation_dim(scenario):
        raise RuntimeError("guard feature dimension changed")
    mask = np.zeros(ACTION_DIM, dtype=bool)
    occupied = {tuple(other.pos) for other in chars if other.name != char.name
                and getattr(other, "is_alive", True) and getattr(other, "position_known", True)}
    for move_index, (dr, dc) in enumerate(MOVES):
        destination = (position[0] + dr, position[1] + dc)
        r, c = destination
        if (0 <= r < height and 0 <= c < width and grid[r, c] != 1
                and (move_index == 4 or destination not in occupied)):
            if getattr(char, "facing_forced_this_tick", False):
                # Incoming-fire response owns this tick's facing. Do not train
                # a turn which the engine will replace with the locked facing.
                mask[move_index * 8 + FACING_DIRECTIONS.index(char.facing)] = True
            else:
                mask[move_index * 8:(move_index + 1) * 8] = True
    for ability_index, ability in enumerate(ABILITIES):
        if getattr(char, "ability_name", ability) != ability:
            continue
        if getattr(char, ability.lower() + "_charges", 0) <= 0:
            continue
        for target_index, destination in enumerate(targets):
            if destination is None:
                continue
            r, c = destination
            if not (0 <= r < height and 0 <= c < width) or grid[r, c] == 1:
                continue
            game = getattr(controller, "game", None)
            if ability != "SMOKE" and (destination == position or (
                    game is not None and len(game._projectile_path(position, destination)) <= 1)):
                continue
            start = (len(MOVES) + ability_index * TARGET_COUNT + target_index) * 8
            # The engine applies explicit facing to movement/turn actions;
            # ability ticks retain the current facing. Do not offer fictitious turns.
            mask[start + FACING_DIRECTIONS.index(char.facing)] = True
    ultimate_actions = {}
    if getattr(char, "ability_name", "") == "SMOKE" and getattr(char, "smoke_charges", 0) > 0:
        mask[SELF_SMOKE_ACTION + FACING_DIRECTIONS.index(char.facing)] = True
    name = str(getattr(char, "ultimate_name", "")).upper()
    cost = getattr(char, "ultimate_cost", 0)
    game = getattr(controller, "game", None)
    ready = cost > 0 and getattr(char, "ultimate_points", 0) >= cost and char.is_alive
    movement_blocked = (game is not None and name in ("RAID", "ESCAPE")
                        and game._ramp_blocks_movement(char))
    portal_active = (game is not None and name == "ESCAPE" and any(
        portal.get("owner") == char.name for portal in getattr(game, "escape_portals", [])))
    if ready and not movement_blocked and not portal_active:
        for target_index, destination in enumerate(targets):
            if name in ("NEON", "ESCAPE"):
                if destination is None:
                    continue
                r, c = destination
                if not (0 <= r < height and 0 <= c < width) or grid[r, c] == 1:
                    continue
                if name == "ESCAPE" and (destination == position or destination in occupied):
                    continue
            elif target_index != 0 or name not in ("RAID", "MONITOR", "TUNNEL", "BALEMOON"):
                continue
            for facing_index, facing in enumerate(FACING_DIRECTIONS):
                if getattr(char, "facing_forced_this_tick", False) and facing != char.facing:
                    continue
                if name == "RAID":
                    vx, vy = FACING_VECTORS[facing]
                    next_cell = (position[0] + int(round(vy)), position[1] + int(round(vx)))
                    r, c = next_cell
                    if not (0 <= r < height and 0 <= c < width) or grid[r, c] == 1 or next_cell in occupied:
                        continue
                elif name not in ("RAID", "TUNNEL") and facing != char.facing:
                    continue
                action = ULTIMATE_ACTION + target_index * 8 + facing_index
                payload = {"ultimate": name, "facing": facing}
                if name in ("NEON", "ESCAPE"):
                    payload["target"] = destination
                mask[action] = True
                ultimate_actions[action] = payload
    # Keep movement and waiting available alongside casts so the model learns
    # when to spend resources instead of exhausting them on phase entry.
    context = {
        "position": position, "goal": goal, "aim": aim, "spike": spike,
        "distance_goal": distance_goal, "distance_spike": distance_spike,
        "tap": tap_progress > 0, "fireable": target is not None,
        "target": tuple(target.pos) if target is not None else None,
        "targets": targets, "stopped": stopped, "tick": tick,
        "can_move": any(0 <= position[0] + dr < height and 0 <= position[1] + dc < width
                        and grid[position[0] + dr, position[1] + dc] != 1
                        and (position[0] + dr, position[1] + dc) not in occupied
                        for dr, dc in MOVES[:4]),
        "ultimate_actions": ultimate_actions,
        "impaired": impaired, "blind": statuses[0] > 0,
        "revealed": statuses[1] > 0 or statuses[2] > 0,
        "exposures": exposures, "tap_remaining": tap_remaining,
    }
    if not counter_features:
        # Defender retake shares the original guard encoder. Keep its saved
        # observations/action indices unchanged when extending attacker guard.
        observation = observation[:MAP_CHANNELS * scenario.grid.size + LEGACY_FEATURE_DIM]
        mask = mask[:PRE_COUNTER_ACTION_DIM]
    return observation, mask, context


def decode_action(action, position, targets, ultimate_actions=None):
    if int(action) >= SELF_SMOKE_ACTION:
        return list(position), {"ability": "SMOKE", "target": position,
                                "facing": FACING_DIRECTIONS[int(action) % 8]}
    if int(action) >= ULTIMATE_ACTION:
        return list(position), ultimate_actions[int(action)]
    operation, facing_index = divmod(int(action), 8)
    payload = {"facing": FACING_DIRECTIONS[facing_index]}
    if operation < len(MOVES):
        dr, dc = MOVES[operation]
        return [position[0] + dr, position[1] + dc], payload
    ability_index, target_index = divmod(operation - len(MOVES), TARGET_COUNT)
    payload.update(ability=ABILITIES[ability_index], target=targets[target_index])
    return list(position), payload
