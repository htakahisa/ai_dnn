"""Training-only search shaping from IQ-filtered geometry and real outcomes."""

from collections import deque
from math import hypot
from types import SimpleNamespace

from concon_v1.co1_attacker_common import bfs_distance_map
from concon_v1.co1_guard_common import aim_alignment, clear_shot
from concon_v1.co1_defender_search_common import MOVES, SUPPORT_DISTANCE, _at

GAMMA = .99
ROUND_REWARD = 3.
DEATH_PENALTY = .6
REWARD_VERSION = 7
STATIONARY_FIRE_REWARD = .20
COMBAT_MOVE_PENALTY = .25
BREAK_SHOT_PENALTY = .20


def facing_target(context, position):
    """Reward aim only along a clear ray from the actual action destination.

    Check every disclosed enemy so a peek can acquire a different target from
    the remembered/support target. A last-known or killed position is useful
    for facing only while its ray is still clear.
    """
    probe = _at(context["actor"], position)
    enemies = sorted(context["disclosed"], key=lambda enemy: (
        tuple(enemy.pos) != context["target"],
        hypot(enemy.pos[0] - position[0], enemy.pos[1] - position[1])))
    for enemy in enemies:
        if clear_shot(probe, enemy, context["chars"], context["grid"], context["smoke"]):
            return tuple(enemy.pos)
    if context["target"] is not None:
        remembered = SimpleNamespace(name="__search_memory__", pos=context["target"],
                                     reveal_remaining=0)
        if clear_shot(probe, remembered, context["chars"], context["grid"], context["smoke"]):
            return context["target"]
    return None


def firing_angle(context, cell, enemy):
    """Favor a different firing angle from allies already covering this enemy."""
    dr, dc = cell[0] - enemy.pos[0], cell[1] - enemy.pos[1]
    length = hypot(dr, dc)
    angles = []
    for ally in context["chars"]:
        if (ally.team != context["actor"].team or not ally.is_alive
                or ally.name == context["actor"].name
                or not clear_shot(ally, enemy, context["chars"], context["grid"], context["smoke"])):
            continue
        ar, ac = ally.pos[0] - enemy.pos[0], ally.pos[1] - enemy.pos[1]
        angles.append(1 - (dr * ar + dc * ac) / max(1., length * hypot(ar, ac)))
    return min(angles, default=0.)


def search_score(end_reason, defender_alive, attacker_alive):
    if end_reason in ("attacker_eliminated", "timeout"):
        return 1.
    if end_reason == "planted":
        # Equal numbers are neutral; only the retake headcount advantage
        # determines the plant score. Extra survivors cannot offset a deficit.
        return .5 + (defender_alive - attacker_alive) / 10
    return 0.


def support_goal(context, grid, limit=SUPPORT_DISTANCE):
    """Nearest firing cell within the support limit, used only in training.

    Team-disclosed enemies are considered even before an ally starts shooting.
    Distant actors receive post-holding/return rewards instead of chase rewards.
    """
    enemies = context.get("disclosed", context["engaged"])
    if not enemies or context["fireable"] or context.get("post_kill_hold", False):
        return None
    context = {**context, "grid": grid}
    position = context["position"]
    occupied = {tuple(other.pos) for other in context["chars"] if other.is_alive
                and other.name != context["actor"].name
                and (other.team == context["actor"].team or getattr(other, "position_known", False))}
    queue, visited = deque([(position, 0)]), {position}
    candidates = []
    nearest = None
    while queue:
        cell, distance = queue.popleft()
        if nearest is not None and distance > nearest:
            break
        covered = [enemy for enemy in enemies if cell not in occupied and clear_shot(_at(context["actor"], cell), enemy,
                context["chars"], grid, context["smoke"])]
        if covered:
            nearest = distance
            candidates.append((max(firing_angle(context, cell, enemy) for enemy in covered), cell, distance))
            continue
        if distance == limit:
            continue
        for dr, dc in MOVES[:4]:
            next_cell = cell[0] + dr, cell[1] + dc
            r, c = next_cell
            if (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != 1
                    and next_cell not in occupied and next_cell not in visited):
                visited.add(next_cell)
                queue.append((next_cell, distance + 1))
    if candidates:
        _, cell, distance = max(candidates, key=lambda candidate: candidate[0])
        return cell, distance
    return None


def prepare_reward_context(context, grid, support_distance=SUPPORT_DISTANCE):
    context = dict(context)
    context["grid"] = grid
    if context.get("post_kill_hold", False) and facing_target(context, context["position"]) is None:
        context["post_kill_hold"] = False
    assist = support_goal(context, grid, support_distance)
    context["support_goal"] = assist[0] if assist is not None else None
    context["support_distance"] = assist[1] if assist is not None else None
    if assist is not None:
        enemy = next(enemy for enemy in context["disclosed"] if clear_shot(
            _at(context["actor"], assist[0]), enemy, context["chars"], grid, context["smoke"]))
        context["target"] = tuple(enemy.pos)
    goal = context["support_goal"] or context["goal"]
    context["reward_distances"] = bfs_distance_map(grid, goal)
    return context


def decision_reward(action, context, position, facing):
    moved = tuple(position) != context["position"]
    reward = -.004
    if action >= 40:
        ability = context["ability_payloads"][action]["ability"]
        # Availability is never an automatic-cast rule. In particular a smoke
        # must earn its use through survival/outcomes despite the reserve cost.
        return reward + {"SMOKE": -.45, "FLASH": -.015, "RECON": -.025}[ability]
    operation = action // 8
    target = facing_target(context, position)
    if context["fireable"]:
        alignment = aim_alignment(position, target, facing) if target is not None else 0.
        reward += .04 * alignment
        # Losing every current enemy's firing ray is retreat, even if another
        # remembered position would still yield a facing reward.
        maintains_shot = any(clear_shot(_at(context["actor"], position), enemy,
            context["chars"], context["grid"], context["smoke"]) for enemy in context["disclosed"])
        if moved and not maintains_shot:
            return reward - COMBAT_MOVE_PENALTY - BREAK_SHOT_PENALTY
        if context["neutralized"]:
            # Exploit a blind opponent while avoiding the adjacent-distance
            # exception to the engine's blind accuracy penalty.
            separation = max(abs(position[0] - context["target"][0]), abs(position[1] - context["target"][1]))
            before = max(abs(context["position"][0] - context["target"][0]),
                         abs(context["position"][1] - context["target"][1]))
            advances = moved and maintains_shot and 1 < separation < before
            reward += .025 if advances else -COMBAT_MOVE_PENALTY if moved else 0.
            if separation <= 1:
                reward -= .06
        else:
            reward += STATIONARY_FIRE_REWARD if not moved and alignment >= .7 else -COMBAT_MOVE_PENALTY if moved else 0.
        return reward
    if context.get("post_kill_hold", False) and target is not None:
        alignment = aim_alignment(position, target, facing)
        return reward + .04 * alignment + (STATIONARY_FIRE_REWARD if not moved and alignment >= .7
                                          else -COMBAT_MOVE_PENALTY if moved else 0.)
    distances = context["reward_distances"]
    before, after = int(distances[context["position"]]), int(distances[tuple(position)])
    if before >= 0 and after >= 0:
        reward += .025 * (before - after)
        # A facing reward must not make stepping away from the return/support
        # goal profitable. Otherwise two opposite moves can form a reward loop.
        if moved and after >= before:
            reward -= .15
    if context["support_goal"] is not None:
        reward += .025 if context["lines"][operation] else 0.
        if not moved:
            reward -= .04
    elif context["position"] == context["goal"]:
        reward += -.06 if moved else .025
    elif not moved:
        reward -= .02
    if target is not None:
        reward += .035 * aim_alignment(position, target, facing)
    elif context["target"] is None:
        reward += .02 * aim_alignment(position, context["aim"], facing)
    return reward
