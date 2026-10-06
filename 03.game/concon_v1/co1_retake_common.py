"""Retake perception, legal actions and coordination reward context.

Movement, facing, casts and defusing are chosen by the network. Coordination
geometry supplies observations and training rewards, never movement overrides.
"""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from game_core import FACING_DIRECTIONS, DEFUSE_REQUIRED_TICKS
from grid_lines import line_cells
from concon_v1.co1_retake_navigation import assembly_step_allowed
from concon_v1.co1_attacker_common import bfs_distance_map, GORIGONS
from concon_v1.co1_guard_common import (
    build_inputs as guard_inputs, clear_shot, aim_alignment, MOVES, ABILITIES,
    FEATURE_DIM, ACTION_DIM as GUARD_ACTION_DIM, ULTIMATE_ACTION, decode_action as guard_decode,
)

DEFUSE_ACTION = GUARD_ACTION_DIM
ACTION_DIM = DEFUSE_ACTION + 1
EXTRA_FEATURES = 21
MAP_CHANNELS = 12
GAMMA = .99


def observation_dim(scenario):
    return MAP_CHANNELS * scenario.grid.size + FEATURE_DIM + EXTRA_FEATURES


class RetakeDQN(nn.Module):
    def __init__(self, scenario, foundation=False):
        super().__init__()
        self.height, self.width = scenario.grid.shape
        self.map_size = MAP_CHANNELS * scenario.grid.size
        self.encoder = nn.Sequential(nn.Conv2d(MAP_CHANNELS, 16, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((4, 6)), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(768 + FEATURE_DIM + EXTRA_FEATURES, 256), nn.ReLU(),
                                  nn.Linear(256, ACTION_DIM))
        self.foundation = foundation
        if foundation:
            from concon_v1.co1_retake_foundation import navigation_cells
            cells = navigation_cells(scenario)
            lookup = torch.full((self.height * self.width,), -1, dtype=torch.long)
            for index, (r, c) in enumerate(cells):
                lookup[r * self.width + c] = index
            self.register_buffer("foundation_plant_lookup", lookup)
            # Six learned operations: four moves, wait, defuse. Facing retains
            # the perception-dependent network, as do combat and utility.
            self.foundation_values = nn.Embedding(len(cells) * self.height * self.width, 6)
            nn.init.zeros_(self.foundation_values.weight)
            self.register_buffer("foundation_trained", torch.zeros(self.foundation_values.num_embeddings, dtype=torch.bool))

    def forward(self, observation):
        maps = observation[:, :self.map_size].reshape(-1, MAP_CHANNELS, self.height, self.width)
        features = observation[:, self.map_size:]
        values = self.head(torch.cat((self.encoder(maps), features), dim=1))
        if self.foundation:
            quiet, basic = self.foundation_inputs(observation)
            basic = basic * quiet[:, None]
            addition = torch.cat((basic[:, :5].repeat_interleave(8, dim=1),
                                  torch.zeros((len(values), DEFUSE_ACTION - 40), device=values.device), basic[:, 5:]), dim=1)
            values = values + addition
        return values

    def foundation_inputs(self, observation):
        """Use learned basic values for quiet assembly, movement and defusing."""
        if not self.foundation:
            return (torch.zeros(len(observation), dtype=torch.bool, device=observation.device),
                    torch.zeros((len(observation), 6), device=observation.device))
        features = observation[:, self.map_size:]
        row = (features[:, 0] * self.height).round().long().clamp(0, self.height - 1)
        col = (features[:, 1] * self.width).round().long().clamp(0, self.width - 1)
        spike_row = (features[:, 2] * self.height).round().long().clamp(0, self.height - 1)
        spike_col = (features[:, 3] * self.width).round().long().clamp(0, self.width - 1)
        plants = self.foundation_plant_lookup[spike_row * self.width + spike_col]
        indices = plants.clamp(min=0) * self.height * self.width + row * self.width + col
        # Enemy memory elsewhere does not mean this actor is firing. Feature
        # 16 is the existing clear-shot target flag from production perception.
        quiet = ((features[:, 16] == 0)
                 & (plants >= 0) & self.foundation_trained[indices])
        return quiet, self.foundation_values(indices)


def foundation_preservation_loss(model, observations, values, masks, margin=.5):
    """Retain basic operation rankings without anchoring absolute TD values.

    Only legal moves/wait/defuse outside combat participate, including assembly.
    Facing preferences and utility/ultimate values are left to battle learning.
    """
    active, teacher = model.foundation_inputs(observations)
    legal = torch.cat((masks[:, :40].reshape(-1, 5, 8).any(2), masks[:, DEFUSE_ACTION:DEFUSE_ACTION + 1]), 1)
    active = active & legal.any(1)
    if not active.any():
        return values.sum() * 0.
    scores = torch.cat((values[:, :40].masked_fill(~masks[:, :40], -torch.inf)
                        .reshape(-1, 5, 8).amax(2), values[:, DEFUSE_ACTION:DEFUSE_ACTION + 1]), 1)[active]
    teacher, legal = teacher.detach()[active], legal[active]
    maximum = teacher.masked_fill(~legal, -torch.inf).amax(1, keepdim=True)
    preferred = legal & (teacher >= maximum - 1e-4)
    best = scores.masked_fill(~preferred, -torch.inf).amax(1, keepdim=True)
    alternatives = legal & ~preferred
    gap = margin * (maximum - teacher).clamp(0., 1.)
    # Zero illegal operations before reductions to avoid inf/nan gradients.
    penalties = torch.relu(scores.masked_fill(~alternatives, 0.) - best + gap)
    return penalties.masked_fill(~alternatives, 0.).sum() / alternatives.sum().clamp(min=1)


def wall_clear(grid, source, target):
    return all(grid[cell] != 1 for cell in line_cells(source, target))


def coordination(char, state, controller=None):
    grid, spike = np.asarray(state["grid"]), tuple(state["planted_pos"])
    position = tuple(char.pos)
    chars = state.get("chars", [])
    smoke = set(map(tuple, state.get("smoke_cells", ())))
    known = [other for other in chars if other.team != char.team and other.is_alive
             and getattr(other, "position_known", True) and 0 <= other.pos[0] < grid.shape[0]
             and 0 <= other.pos[1] < grid.shape[1]]
    allies = [other for other in chars if other.team == char.team and other.is_alive]
    distances = bfs_distance_map(grid, spike)
    remaining = float(state.get("detonate_timer", 0))
    if controller is None:
        from concon_v1.co1_retake_coordination import RetakeAssembly
        from concon_v1.co1_retake_scenarios import get_scenario, plant_site
        planner = RetakeAssembly(get_scenario(plant_site(spike, grid)))
    else:
        planner = controller.assembly
    plan = planner.update(allies, spike, remaining, int(state.get("battle_tick", 0)))
    waiting = plan["waiting"]
    # Safe means no firing line from disclosed enemies. Hidden enemies are
    # not consulted, including when finding a staging position.
    def safe(point):
        proxy = SimpleNamespace(name=char.name, pos=point, sees_through_smoke=False)
        return not any(clear_shot(proxy, enemy, chars, grid, smoke) for enemy in known)
    staging = plan["assigned"].get(char.name, position)
    # Preserve the assigned entrance during release, without requiring a stop
    # at A. Ignore a detour when only direct defusing can still meet the clock.
    via = planner.routes.get(staging)
    use_entrance = (via is not None and (char.name, staging) not in planner.reached and via[position] > 1
                    and via[position] + distances[staging] + DEFUSE_REQUIRED_TICKS + 3 < remaining)
    goal = staging if waiting or (len(allies) > 1 and use_entrance) else spike
    fireable = [enemy for enemy in known if clear_shot(char, enemy, chars, grid, smoke)]
    neutralized = bool(fireable) and all(getattr(enemy, "blind_remaining", 0) > 0
                                      or getattr(enemy, "stun_remaining", 0) > 0 for enemy in fireable)
    return dict(goal=goal, staging=staging, waiting=waiting, safe=safe(position),
                urgent=plan["urgent"], remaining=remaining,
                distances=distances, eligible=len(plan["eligible"]), near=plan["near"], allies=allies,
                safe_at=safe, known=known, neutralized=neutralized,
                smoke_defuse=spike in smoke and position in smoke)


def build_inputs(controller, char, state):
    scenario, position = controller.scenario, tuple(char.pos)
    tactical = coordination(char, state, controller)
    ability = str(getattr(char, "ability_name", ""))
    points = scenario.points.get(ability, ())
    distances = {point: bfs_distance_map(scenario.grid, point) for point in points}
    limit = controller.ability_distances.get(ability, 0)
    valid = [point for point in points if 0 <= distances[point][position] <= limit
             and (ability == "SMOKE" or (point != position and wall_clear(scenario.grid, position, point)))]
    aim = min(valid or points, key=lambda p: (distances[p][position], p)) if points else tactical["goal"]
    utility_target = aim if valid else None
    ordered_targets = sorted(valid, key=lambda p: (distances[p][position], p))
    proxy_scenario = SimpleNamespace(grid=scenario.grid, positions={slot: tactical["goal"] for slot in "abcde"},
                                     facing_points={slot: aim for slot in "abcde"})
    proxy = SimpleNamespace(scenario=proxy_scenario, assignments=controller.assignments,
                            sightings=controller.sightings, stationary_ticks=controller.stationary_ticks,
                            game=getattr(controller, "game", None))
    base, guard_mask, context = guard_inputs(proxy, char, state)
    mask = np.zeros(ACTION_DIM, dtype=bool)
    # Recreate legal movement: guard's mandatory ability priority is unsuitable
    # for defusing or waiting for teammates.
    occupied = {tuple(other.pos) for other in state.get("chars", []) if other.name != char.name
                and other.is_alive and getattr(other, "position_known", True)}
    for index, (dr, dc) in enumerate(MOVES):
        point = position[0] + dr, position[1] + dc
        r, c = point
        if 0 <= r < scenario.grid.shape[0] and 0 <= c < scenario.grid.shape[1] and scenario.grid[point] != 1:
            if tactical["waiting"] and not assembly_step_allowed(controller.assembly.front, position, point):
                continue
            if index == 4 or point not in occupied:
                if getattr(char, "facing_forced_this_tick", False):
                    mask[index * 8 + FACING_DIRECTIONS.index(char.facing)] = True
                else:
                    mask[index * 8:(index + 1) * 8] = True
    targets = (context["spike"], utility_target, context["targets"][2])
    ability_targets = (context["spike"] if ability == "SMOKE" else None,
                       ordered_targets[0] if ordered_targets else None,
                       ordered_targets[1] if len(ordered_targets) > 1 else None)
    for index, name in enumerate(ABILITIES):
        if ability != name or getattr(char, name.lower() + "_charges", 0) <= 0:
            continue
        for target_index in (0, 1, 2):
            target = ability_targets[target_index]
            if target is None or (target_index == 0 and name != "SMOKE"):
                continue
            distance = bfs_distance_map(scenario.grid, target)[position]
            if 0 <= distance <= controller.ability_distances[name] and (name == "SMOKE" or wall_clear(scenario.grid, position, target)):
                mask[(5 + index * 3 + target_index) * 8 + FACING_DIRECTIONS.index(char.facing)] = True
    # Existing engine-specific ultimate legality is retained. Targeted ults
    # use the disclosed enemy slot only; direction-based ults require disclosure.
    ultimates = {}
    if tactical["known"]:
        name = str(getattr(char, "ultimate_name", "")).upper()
        for action, payload in context["ultimate_actions"].items():
            if name in ("NEON", "ESCAPE") and (action - ULTIMATE_ACTION) // 8 != 2:
                continue
            mask[action] = guard_mask[action]
            ultimates[action] = payload
    spike = context["spike"]
    active_defuser = state.get("active_defuser_name")
    # The production state exposes per-actor progress, rather than the lock.
    tap_info = state.get("defender_defuse_info") or {}
    other_defusing = any(name != char.name and value[0] > 0 for name, value in tap_info.items())
    mask[DEFUSE_ACTION] = (max(abs(position[0] - spike[0]), abs(position[1] - spike[1])) <= 1
                           and active_defuser in (None, char.name) and not other_defusing)
    # Assembly forbids fixed casts, including smoke on the planted spike.
    if tactical["waiting"]:
        mask[40:ULTIMATE_ACTION] = False
    marker_maps = np.zeros((6, *scenario.grid.shape), dtype=np.float32)
    for index, name in enumerate(ABILITIES):
        for point in scenario.points[name]:
            marker_maps[index][point] = 1
    marker_maps[3][spike] = 1
    for point in scenario.rally_points:
        marker_maps[4][point] = 1
    for point in controller.assembly.assigned.values():
        marker_maps[5][point] = 1
    size = 6 * scenario.grid.size
    extras = [float(tactical["waiting"]), float(tactical["safe"]), float(tactical["urgent"]),
              tactical["eligible"] / 5, tactical["near"] / 5, tactical["remaining"] / 50]
    extras += [float(tactical["safe_at"]((position[0] + dr, position[1] + dc)))
               if mask[index * 8:(index + 1) * 8].any() else 0 for index, (dr, dc) in enumerate(MOVES)]
    allies = {ally.name: ally for ally in tactical["allies"]}
    extras += [tactical["distances"][tuple(allies[name].pos)] / 100 if name in allies else -1
               for name in GORIGONS.players]
    cost = getattr(char, "ultimate_cost", 0)
    extras += [float(cost > 0 and getattr(char, "ultimate_points", 0) >= cost),
               getattr(char, "ultimate_points", 0) / max(1, cost),
               float(tap_info.get(char.name, (0, 1))[0]) / DEFUSE_REQUIRED_TICKS,
               float(tactical["smoke_defuse"]), float(tactical["neutralized"])]
    observation = np.concatenate((base[:size], marker_maps.ravel(), base[size:], np.asarray(extras, dtype=np.float32)))
    context.update(tactical, targets=targets, ultimate_actions=ultimates,
                   ability_targets=ability_targets,
                   grid=scenario.grid,
                   reward_distances=(controller.assembly.routes[tactical["goal"]] if tactical["waiting"]
                                     and tactical["goal"] in controller.assembly.routes
                                     else bfs_distance_map(scenario.grid, tactical["goal"])))
    tick = int(state.get("battle_tick", 0))
    if context["target"] is not None:
        controller.combat_history[char.name] = context["target"], tick
    previous = controller.combat_history.get(char.name)
    context["post_kill_target"] = (previous[0] if previous is not None and 0 <= tick - previous[1] <= 3
                                   and wall_clear(scenario.grid, position, previous[0]) else None)
    memories = list(controller.sightings.values())
    context["memory_target"] = (min(memories, key=lambda item: abs(position[0] - item[0][0])
                                   + abs(position[1] - item[0][1]))[0] if memories else None)
    # Shared sightings guide a peek from each possible destination. Without a
    # disclosed threat, a travel-facing reward supplies a default while leaving
    # all facings learnable (including likely enemy positions learned in battle).
    context["facing_targets"] = {}
    context["facing_threats"] = set()
    context["facing_locked"] = getattr(char, "facing_forced_this_tick", False)
    context["facing_before"] = char.facing
    threats = [tuple(enemy.pos) for enemy in tactical["known"]]
    threats += [point for point, _ in memories if point not in threats]
    for index, (dr, dc) in enumerate(MOVES):
        destination = position[0] + dr, position[1] + dc
        if not mask[index * 8:(index + 1) * 8].any():
            continue
        proxy_actor = SimpleNamespace(name=char.name, pos=destination,
                                      sees_through_smoke=getattr(char, "sees_through_smoke", False))
        visible = [point for point in threats if clear_shot(proxy_actor,
                   SimpleNamespace(name="remembered", pos=point, reveal_remaining=0),
                   [ally for ally in tactical["allies"]], scenario.grid,
                   set(map(tuple, state.get("smoke_cells", ())))) and point != destination]
        target = min(visible, key=lambda p: (abs(p[0] - destination[0]) + abs(p[1] - destination[1]), p)) if visible else (
            (destination[0] + dr, destination[1] + dc) if index < 4 else None)
        if target is None:
            # At an assembly stop, watch the next walkable step into the site,
            # not a utility marker behind a wall.
            routes = tactical["distances"]
            next_steps = [(destination[0] + rr, destination[1] + cc) for rr, cc in MOVES[:4]
                          if 0 <= destination[0] + rr < scenario.grid.shape[0]
                          and 0 <= destination[1] + cc < scenario.grid.shape[1]
                          and 0 <= routes[destination[0] + rr, destination[1] + cc] < routes[destination]]
            target = min(next_steps, key=lambda p: (routes[p], p)) if next_steps else None
        if target is not None:
            context["facing_targets"][destination] = target
        if visible:
            context["facing_threats"].add(destination)
        if visible and not context["facing_locked"]:
            allowed = [aim_alignment(destination, target, face) >= .7 for face in FACING_DIRECTIONS]
            mask[index * 8:(index + 1) * 8] &= allowed
    return observation, mask, context


def decode_action(action, char, context):
    if action == DEFUSE_ACTION:
        return list(char.pos), "DEFUSE"
    return guard_decode(action, tuple(char.pos), context["ability_targets"], context["ultimate_actions"])


def decision_reward(action, context, position, facing):
    moved = position != context["position"]
    reward = -.015
    distances = context["reward_distances"]
    before, after = distances[context["position"]], distances[position]
    if before >= 0 and after >= 0:
        reward += .08 * (before - after)
    facing_target = context.get("facing_targets", {}).get(position)
    if facing_target is not None and not context["fireable"] and not context.get("facing_locked"):
        weight = .12 if position in context.get("facing_threats", ()) else .04
        alignment = aim_alignment(position, facing_target, facing)
        if not moved:
            alignment -= aim_alignment(position, facing_target, context["facing_before"])
        reward += weight * alignment
    if context["fireable"]:
        aligned = aim_alignment(position, context["target"], facing) >= .7
        reward += .10 if aligned and not moved else -.15 if moved and not context["neutralized"] else 0
    elif context.get("post_kill_target") is not None and context["remaining"] > 12:
        reward += .08 if not moved and aim_alignment(position, context["post_kill_target"], facing) >= .7 else -.08
    if context["waiting"] and context["safe"] and before == 0:
        reward += .10 if not moved else -.10
    if action == DEFUSE_ACTION:
        reward += .3 if not context["fireable"] or context["smoke_defuse"] else -.15
    if 40 <= action < ULTIMATE_ACTION:
        reward += .04
    return reward
