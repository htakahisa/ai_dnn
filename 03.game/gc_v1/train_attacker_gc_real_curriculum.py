"""Learn Carry/Escort/Guard together in real rounds against progressively stronger foes.

Each phase keeps its own network, optimizer and replay. Production weights are
never overwritten. All three best checkpoints come from the same evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from train_attacker_carry_gc_real import (
    ROOT,
    Pending,
    RealCarrySession,
    expanded_state,
    optimize,
    runtime,
)
import learning_attacker_escort_gc as escort_runtime
import learning_attacker_guard_gc as guard_runtime
from gc_facing import FACING_DIRS, facing_towards, nearest_alive_enemy_facing
from game_core import SHOOTING_SITE_DIGREE
from carry_route_priority import current_carry_route_priority
from positioning_gc import REGISTERED_PLANT_CELLS, team_plant_target
from navigation_intent_gc import (
    _distance_map,
    navigation_intent,
    own_macro,
    can_engage,
    carrier_screening_status,
    carrier_screen_commitment_features,
    FORMATION_MIN_FINAL_DISTANCE,
    FORMATION_MAX_FINAL_DISTANCE,
    SCREEN_COMMITMENT_MAX_FINAL_DISTANCE,
    ENTRY_SYNC_STAGING_DISTANCE,
    ENTRY_SYNC_MAX_FORMATION_DISTANCE,
    ENTRY_SYNC_MIN_ROUND_TIME,
    ENTRY_SYNC_MAX_IDLE_FEATURE,
    designated_route_blocking,
    FAKE_WAIT_SUPPORT_DIM,
)
from ultimate_tactics_gc import ATTACKER_ORB_APPROACH_RADIUS, tactical_ultimate_window

PHASES = ("carry", "escort", "guard")
VERSIONS = {"carry": 13, "escort": 14, "guard": 5}
FACING_PHASES = ("carry", "escort")
TRAINING_REVISION = "independent_confident_facing_v21"
FACING_PARAMETER_PREFIXES = (
    "facing_head.",
    "facing_feature.",
    "facing_output.",
)
MOVEMENT_ACTION_ROWS = {
    # Carry v5+ has explicit STAY/ABILITY/PLANT/ULT actions.  Displacement and
    # orb collection are trainable; ability, plant, ultimate, and facing tensors
    # remain bit-for-bit fixed during movement-only training.
    "carry": (0, 2, 4, 6, 8, runtime.COLLECT_ORB_ACTION_INDEX),
    "escort": (
        escort_runtime.ACTION_UP,
        escort_runtime.ACTION_DOWN,
        escort_runtime.ACTION_LEFT,
        escort_runtime.ACTION_RIGHT,
        escort_runtime.ACTION_STAY,
        escort_runtime.ACTION_COLLECT_ORB,
    ),
}
RUNTIME_DATA_FILES = tuple(
    ROOT / name
    for name in ("character_stats.py", "player_combos.py", "awakening_events.py")
)

CARRY_ROUTE_PRIORITY = current_carry_route_priority()

# Stage-2 navigation shaping. Waiting for a synchronized entry or fighting is
# still valid; an unforced Carry stall becomes increasingly costly late in the
# round, and timing out is materially worse than an ordinary losing tick.
CARRY_QUIET_STALL_PENALTY = 0.08
LATE_ROUND_THRESHOLD = 30
LATE_QUIET_STALL_EXTRA_PENALTY = 0.12
TIMEOUT_PENALTY = 20.0
# A gunfight-contact reward is granted only on the tick a shooter newly gains
# a valid shot line.  It uses the identical angle and firing cone as the
# engine, but affects training only; live inference remains policy-driven.
AIM_CONTACT_REWARD = 1.20
AIM_CONTACT_MISS_PENALTY = 0.30
ORB_COLLECTION_REWARD = 3.00
ORB_COLLECTION_PROGRESS_REWARD = 0.50
ORB_APPROACH_STEP_REWARD = 0.35
ORB_INVALID_ACTION_PENALTY = 0.15
ORB_ELIGIBLE_MISS_PENALTY = 2.00
ORB_INTERRUPTED_COLLECTION_PENALTY = 1.00
ORB_ROUND_MISS_PENALTY = 3.00
FAKE_WAIT_SUPPORT_RADIUS = 4
FAKE_WAIT_SUPPORT_DESIRED_RADIUS = 2


def expand_policy_state(checkpoint, obs_dim, action_dim=None):
    state = (
        expanded_state(checkpoint)
        if "model_state_dict" in checkpoint
        else dict(checkpoint)
    )
    state = dict(state)
    weights = state["feature.0.weight"]
    if weights.shape[1] > obs_dim:
        raise ValueError(
            "source observation dimension is larger than the requested model"
        )
    if weights.shape[1] < obs_dim:
        extended = weights.new_zeros((weights.shape[0], obs_dim))
        extended[:, : weights.shape[1]] = weights
        state["feature.0.weight"] = extended
    facing_weights = state.get("facing_feature.0.weight")
    if facing_weights is not None:
        if facing_weights.shape[1] > obs_dim:
            raise ValueError(
                "source facing observation dimension is larger than requested"
            )
        if facing_weights.shape[1] < obs_dim:
            extended = facing_weights.new_zeros((facing_weights.shape[0], obs_dim))
            extended[:, : facing_weights.shape[1]] = facing_weights
            state["facing_feature.0.weight"] = extended
    if action_dim is not None:
        out_weight = state["advantage_head.2.weight"]
        out_bias = state["advantage_head.2.bias"]
        old_action_dim = out_weight.shape[0]
        if out_weight.shape[0] > action_dim:
            raise ValueError(
                "source action dimension is larger than the requested model"
            )
        if out_weight.shape[0] < action_dim:
            extended_weight = out_weight.new_zeros((action_dim, out_weight.shape[1]))
            extended_bias = out_bias.new_empty(action_dim)
            extended_weight[: out_weight.shape[0]] = out_weight
            extended_bias[: out_bias.shape[0]] = out_bias
            extended_bias[out_bias.shape[0] :] = torch.min(out_bias) - 0.5
            state["advantage_head.2.weight"] = extended_weight
            state["advantage_head.2.bias"] = extended_bias
        facing_weight = state.get("facing_head.weight")
        if facing_weight is not None:
            hidden = state["feature.2.bias"].shape[0]
            if facing_weight.shape[1] != hidden + action_dim:
                if facing_weight.shape[1] != hidden + old_action_dim:
                    raise ValueError("source facing-head action input is inconsistent")
                extended_facing = facing_weight.new_zeros(
                    (facing_weight.shape[0], hidden + action_dim)
                )
                extended_facing[:, :hidden] = facing_weight[:, :hidden]
                extended_facing[:, hidden : hidden + old_action_dim] = facing_weight[
                    :, hidden:
                ]
                state["facing_head.weight"] = extended_facing
        facing_output_weight = state.get("facing_output.weight")
        if facing_output_weight is not None:
            facing_hidden = state["facing_feature.2.bias"].shape[0]
            if facing_output_weight.shape[1] != facing_hidden + action_dim:
                if facing_output_weight.shape[1] != facing_hidden + old_action_dim:
                    raise ValueError(
                        "source independent facing action input is inconsistent"
                    )
                extended_facing = facing_output_weight.new_zeros(
                    (facing_output_weight.shape[0], facing_hidden + action_dim)
                )
                extended_facing[:, :facing_hidden] = facing_output_weight[
                    :, :facing_hidden
                ]
                extended_facing[:, facing_hidden : facing_hidden + old_action_dim] = (
                    facing_output_weight[:, facing_hidden:]
                )
                state["facing_output.weight"] = extended_facing
    return state


def restrict_policy_to_input_columns(policy, columns):
    """Freeze deployed behavior except first-layer weights for new features."""
    columns = tuple(int(column) for column in columns)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    weight = policy.feature[0].weight
    weight.requires_grad_(True)
    gradient_mask = torch.zeros_like(weight)
    gradient_mask[:, list(columns)] = 1
    weight.register_hook(lambda gradient: gradient * gradient_mask)


def restrict_policy_updates(policy, input_columns=(), action_rows=()):
    """Allow only selected new observation columns and advantage output rows."""
    input_columns = tuple(map(int, input_columns))
    action_rows = tuple(map(int, action_rows))
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    # A newly introduced facing head must remain trainable even when a
    # curriculum run intentionally protects the deployed movement policy.
    if hasattr(policy, "facing_parameters"):
        for parameter in policy.facing_parameters():
            parameter.requires_grad_(True)
    if input_columns:
        feature_weight = policy.feature[0].weight
        feature_weight.requires_grad_(True)
        feature_mask = torch.zeros_like(feature_weight)
        feature_mask[:, list(input_columns)] = 1
        feature_weight.register_hook(lambda gradient: gradient * feature_mask)
    if action_rows:
        action_weight = policy.advantage_head[-1].weight
        action_bias = policy.advantage_head[-1].bias
        action_weight.requires_grad_(True)
        action_bias.requires_grad_(True)
        weight_mask = torch.zeros_like(action_weight)
        bias_mask = torch.zeros_like(action_bias)
        weight_mask[list(action_rows)] = 1
        bias_mask[list(action_rows)] = 1
        action_weight.register_hook(lambda gradient: gradient * weight_mask)
        action_bias.register_hook(lambda gradient: gradient * bias_mask)


def restrict_policy_to_facing_head(policy):
    """Freeze every movement parameter and train only the factorized facing head."""
    if not hasattr(policy, "facing_parameters"):
        raise ValueError("facing-only training requires a factorized facing head")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.facing_parameters():
        parameter.requires_grad_(True)


def restrict_policy_to_movement_rows(
    policy, rows, input_columns=(), train_facing=False,
):
    """Train displacement rows and, optionally, independent facing weights."""
    rows = tuple(map(int, rows))
    input_columns = tuple(map(int, input_columns))
    if not rows:
        raise ValueError("movement-only training requires at least one action row")
    output = policy.advantage_head[-1]
    if min(rows) < 0 or max(rows) >= output.out_features:
        raise ValueError(f"movement rows are outside action space: {rows}")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    if input_columns:
        feature_weight = policy.feature[0].weight
        if min(input_columns) < 0 or max(input_columns) >= feature_weight.shape[1]:
            raise ValueError(
                f"input columns are outside observation space: {input_columns}"
            )
        feature_weight.requires_grad_(True)
        feature_mask = torch.zeros_like(feature_weight)
        feature_mask[:, list(input_columns)] = 1
        feature_weight.register_hook(lambda gradient: gradient * feature_mask)
    output.weight.requires_grad_(True)
    output.bias.requires_grad_(True)
    weight_mask = torch.zeros_like(output.weight)
    bias_mask = torch.zeros_like(output.bias)
    weight_mask[list(rows)] = 1
    bias_mask[list(rows)] = 1
    output.weight.register_hook(lambda gradient: gradient * weight_mask)
    output.bias.register_hook(lambda gradient: gradient * bias_mask)
    if train_facing:
        if not hasattr(policy, "facing_parameters"):
            raise ValueError("joint movement/facing training requires a facing head")
        for parameter in policy.facing_parameters():
            parameter.requires_grad_(True)


def reset_movement_rows(policy, rows):
    """Reinitialize displacement rows while preserving every other tensor."""
    rows = tuple(map(int, rows))
    output = policy.advantage_head[-1]
    with torch.no_grad():
        fresh_weight = torch.empty(
            (len(rows), output.in_features),
            dtype=output.weight.dtype,
            device=output.weight.device,
        )
        torch.nn.init.kaiming_uniform_(fresh_weight, a=math.sqrt(5))
        output.weight[list(rows)] = fresh_weight
        if output.bias is not None:
            bound = 1 / math.sqrt(output.in_features)
            output.bias[list(rows)].uniform_(-bound, bound)


def movement_only_transitions(rows, allowed_actions):
    """Exclude ability/objective transitions from movement-row TD updates."""
    allowed = set(map(int, allowed_actions))
    return [(name, row) for name, row in rows if int(row[1]) in allowed]


def movement_only_demonstrations(rows, allowed_actions):
    """Keep only navigation labels that directly supervise displacement."""
    allowed = set(map(int, allowed_actions))
    return [row for row in rows if int(row[1]) in allowed]


def is_facing_parameter(name):
    return str(name).startswith(FACING_PARAMETER_PREFIXES)


def runtime_data_fingerprint():
    """Fingerprint mutable roster data so one run cannot mix data revisions."""
    missing = [str(path) for path in RUNTIME_DATA_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"runtime data files are missing: {missing}")
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in RUNTIME_DATA_FILES
    }


class ProgressTracker:
    """Reward new navigation progress once; repeated A-B-A loops cannot farm it."""

    def __init__(self):
        self.best = {}
        self.idle = {}

    def step(self, name, goal, before, after, engaged, required=True):
        if goal is None or before < 0 or after < 0:
            return 0.0
        key = (name, tuple(goal))
        best = self.best.get(key, before)
        improvement = max(0, best - after)
        self.best[key] = min(best, before, after)
        idle = (
            0 if improvement or engaged or not required else self.idle.get(name, 0) + 1
        )
        self.idle[name] = idle
        retreat = max(0, after - before) if not engaged and required else 0
        # Navigation bonuses must not outweigh the combat movement penalty.
        # During a real duel the round outcome, damage and kills teach tactics.
        advance = 0.0 if engaged else 0.20 * improvement
        return advance - 0.20 * retreat - (0.10 if idle >= 8 and after > 1 else 0.0)


def combat_reward(phase, action, engaged):
    if not engaged:
        return 0.0
    ultimate_action = {
        "carry": runtime.ULTIMATE_ACTION_INDEX,
        "escort": escort_runtime.ACTION_ULTIMATE,
        "guard": guard_runtime.ULTIMATE_ACTION_INDEX,
    }[phase]
    stationary = action == ultimate_action or (
        action in (escort_runtime.ACTION_STAY, escort_runtime.ACTION_ABILITY)
        if phase == "escort"
        else action in (0, 1)
    )
    if phase == "carry" and not stationary:
        return -0.02
    # Less than the time cost: stopping can improve combat, but waiting itself
    # never becomes an unlimited positive reward.
    return 0.02 if stationary else -0.12


def combat_contact_aim(game, chars):
    """Return potential shot contacts and each shooter's post-move aim angle.

    This intentionally mirrors the engine's pre-facing target filtering.  A
    target outside the firing cone is retained here so a newly exposed enemy
    can reward good pre-aim instead of disappearing from the signal.
    """
    contacts = {}
    for shooter in chars:
        if (
            not getattr(shooter, "is_alive", True)
            or getattr(shooter, "plant_timer", 0) > 0
            or getattr(shooter, "defuse_timer", 0) > 0
            or getattr(shooter, "collecting_orb_this_tick", False)
        ):
            continue
        candidates = [
            target
            for target in chars
            if target.team != shooter.team
            and getattr(target, "is_alive", True)
            and game.check_line_of_sight(shooter, target)
            and game.check_shot_line_of_sight(shooter, target)
        ]
        if not candidates:
            continue
        target = min(
            candidates,
            key=lambda item: (
                max(
                    abs(int(item.pos[0]) - int(shooter.pos[0])),
                    abs(int(item.pos[1]) - int(shooter.pos[1])),
                ),
                item.hp,
                item.name,
            ),
        )
        contacts[(shooter.name, target.name)] = float(
            game._facing_angle_diff(shooter, target)
        )
    return contacts


def n_step_transitions(rows, gamma, horizon, allowed_start_actions=None):
    """Back up rewards within one actor/phase segment, never across terminal."""
    allowed = (
        None if allowed_start_actions is None else set(map(int, allowed_start_actions))
    )
    by_actor = {}
    for name, row in rows:
        by_actor.setdefault(name, []).append(row)
    for actor_rows in by_actor.values():
        for start, first in enumerate(actor_rows):
            if allowed is not None and int(first[1]) not in allowed:
                continue
            reward, ticks = 0.0, 0
            for row in actor_rows[start : start + horizon]:
                reward += gamma**ticks * row[2]
                ticks += row[6]
                last = row
                if row[5]:
                    break
            yield first[0], first[1], reward, last[3], last[4], last[5], ticks


def navigation_teacher_action(phase, obs, mask, final_approach=False):
    """Training-only demonstrations from observable navigation inputs.

    Quiet approaches follow a legal decreasing-distance neighbor; the policy
    remains responsible for fights and abilities. Evaluation never calls this.
    """
    if phase == "guard":
        return None
    old_dim = (
        runtime.TACTICAL_OBS_DIM
        if phase == "carry"
        else escort_runtime.TACTICAL_OBS_DIM
    )
    if (
        len(obs) < old_dim + 18
        or obs[old_dim - 1]
        or obs[13 if phase == "carry" else 24]
    ):
        return None
    if phase == "carry" and mask[runtime.PLANT_ACTION_INDEX] and obs[25] == 0:
        return runtime.PLANT_ACTION_INDEX
    stay = 0 if phase == "carry" else escort_runtime.ACTION_STAY
    # Intermediate waypoints allow synchronization within one cell. The final
    # registered plant cell requires exact arrival, not a permanent one-cell hold.
    at_waypoint = obs[old_dim + 17]
    if at_waypoint and not (phase == "carry" and final_approach):
        return stay if mask[stay] else None
    actions = (2, 4, 6, 8) if phase == "carry" else (0, 1, 2, 3)
    candidates = [
        (float(obs[old_dim + i]), a) for i, a in enumerate(actions) if mask[a]
    ]
    if candidates:
        gain, action = max(candidates, key=lambda pair: pair[0])
        if gain > 0:
            return action
    return None  # Let exploration learn detours when all short steps are blocked.


def nearby_orb_assignment(view, chars, available_orbs, cache):
    """Choose one nearby, ult-hungry non-carrier using walkable-path distance."""
    grid = getattr(view, "grid", None)
    candidates = []
    for cell in available_orbs:
        orb = tuple(map(int, cell))
        distances = _distance_map(view, orb, cache) if grid is not None else None
        for unit in chars:
            if (
                not getattr(unit, "is_alive", True)
                or getattr(unit, "team", None) != "A"
                or getattr(unit, "has_spike", False)
                or getattr(unit, "ultimate_cost", 0) <= 0
                or getattr(unit, "ultimate_points", 0)
                >= getattr(unit, "ultimate_cost", 0)
            ):
                continue
            pos = tuple(map(int, unit.pos))
            distance = (
                int(distances[pos]) if distances is not None
                else abs(pos[0] - orb[0]) + abs(pos[1] - orb[1])
            )
            if 0 <= distance <= ATTACKER_ORB_APPROACH_RADIUS:
                candidates.append((distance, str(unit.name), orb, unit, distances))
    if not candidates:
        return None
    distance, _name, orb, unit, distances = min(candidates, key=lambda row: row[:3])
    return unit, orb, distances, distance


def observable_teacher_action(
    phase, obs, mask, controller, context, view=None, orb_only=False,
):
    """Label policy-visited states using only the acting character's IQ view.

    Labels can be trained without executing them. No production controller
    imports or calls this helper, and evaluation collects no demonstrations.
    """
    if context is None:
        return None
    char, state = context
    view = view if view is not None else controller.game
    # Designate the nearest non-carrier attacker as the orb collector.  This
    # gives the new action a supervised path to the orb; the TD reward then
    # decides whether the detour is worthwhile in the real round.
    if phase in ("carry", "escort"):
        raw_orbs = state.get("available_orbs", ())
        if not raw_orbs:
            raw_orbs = getattr(view, "available_orbs", ())
        if not raw_orbs:
            raw_orbs = getattr(getattr(controller, "game", None), "available_orbs", ())
        if not raw_orbs:
            grid = getattr(view, "grid", None)
            if grid is not None:
                raw_orbs = list(zip(*np.where(np.asarray(grid) == 5)))
        if raw_orbs:
            real_game = getattr(view, "real_game", view)
            if getattr(controller, "_orb_teacher_game", None) is not real_game:
                controller._orb_teacher_game = real_game
                controller._orb_teacher_maps = {}
            assignment = nearby_orb_assignment(
                view, state.get("chars", ()), raw_orbs,
                controller._orb_teacher_maps,
            )
            # Perception wrappers may create an equivalent proxy object for
            # the acting character, so identity is not reliable here.
            if assignment is not None and assignment[0].name == getattr(char, "name", None):
                _collector, orb, distances, current_dist = assignment
                orb_action = (
                    runtime.COLLECT_ORB_ACTION_INDEX
                    if phase == "carry"
                    else escort_runtime.ACTION_COLLECT_ORB
                )
                if tuple(map(int, char.pos)) == orb and orb_action < len(mask) and mask[orb_action]:
                    return orb_action
                move_actions = (
                    ((2, (-1, 0)), (4, (1, 0)), (6, (0, -1)), (8, (0, 1)))
                    if phase == "carry" else
                    ((0, (-1, 0)), (1, (1, 0)), (2, (0, -1)), (3, (0, 1)))
                )
                improving = []
                for action, (dr, dc) in move_actions:
                    if action >= len(mask) or not mask[action]:
                        continue
                    nxt = (int(char.pos[0]) + dr, int(char.pos[1]) + dc)
                    next_dist = (
                        int(distances[nxt]) if distances is not None
                        else abs(nxt[0] - orb[0]) + abs(nxt[1] - orb[1])
                    )
                    if 0 <= next_dist < current_dist:
                        improving.append((next_dist, action))
                if improving:
                    return min(improving)[1]
    if orb_only:
        return None
    ultimate_action = {
        "carry": runtime.ULTIMATE_ACTION_INDEX,
        "escort": escort_runtime.ACTION_ULTIMATE,
        "guard": guard_runtime.ULTIMATE_ACTION_INDEX,
    }[phase]
    if (
        ultimate_action < len(mask)
        and mask[ultimate_action]
        and ultimate_teacher_needed(phase, char, state, view, obs)
    ):
        return ultimate_action
    if can_engage(view, char, state.get("chars", [])):
        if phase == "carry":
            final_approach = navigation_intent(view, char)[0] == team_plant_target(view)
            advance = navigation_teacher_action(
                phase, obs, mask, final_approach=final_approach
            )
            if advance not in (None, 0):
                return advance
        stay = escort_runtime.ACTION_STAY if phase == "escort" else 0
        return stay if mask[stay] else None
    if phase == "guard":
        return None
    fake_wait_action = fake_wait_support_teacher_action(
        phase, char, state, view, controller, mask
    )
    if fake_wait_action is not None:
        return fake_wait_action
    if phase == "escort" and len(obs) >= escort_runtime.SCREEN_COMMITMENT_OBS_DIM:
        commitment = obs[
            escort_runtime.ENTRY_SUPPORT_OBS_DIM : escort_runtime.SCREEN_COMMITMENT_OBS_DIM
        ]
        if commitment[4] > 0:
            moves = [
                (float(commitment[action]), action)
                for action in range(4)
                if commitment[action] > 0 and mask[action]
            ]
            if moves:
                return max(moves, key=lambda pair: pair[0])[1]
        if commitment[5] > 0:
            return (
                escort_runtime.ACTION_STAY if mask[escort_runtime.ACTION_STAY] else None
            )
    escort_blocks_route = bool(
        phase == "escort"
        and len(obs) >= escort_runtime.CLEARANCE_OBS_DIM
        and obs[escort_runtime.FORMATION_OBS_DIM]
    )
    if (
        phase == "escort"
        and len(obs) >= escort_runtime.DIRECTIONAL_CLEARANCE_OBS_DIM
        and escort_blocks_route
    ):
        clearance = obs[
            escort_runtime.CLEARANCE_OBS_DIM : escort_runtime.DIRECTIONAL_CLEARANCE_OBS_DIM
        ]
        moves = [
            (float(obs[escort_runtime.TACTICAL_OBS_DIM + action]), action)
            for action in range(4)
            if clearance[action] > 0 and mask[action]
        ]
        if moves:
            return max(moves, key=lambda pair: pair[0])[1]
    if phase == "escort" and len(obs) >= escort_runtime.ENTRY_SUPPORT_OBS_DIM:
        safe_advance = obs[
            escort_runtime.DIRECTIONAL_CLEARANCE_OBS_DIM : escort_runtime.ENTRY_SUPPORT_OBS_DIM
        ]
        moves = [
            (float(obs[escort_runtime.TACTICAL_OBS_DIM + action]), action)
            for action in range(4)
            if safe_advance[action] > 0 and mask[action]
        ]
        if moves:
            return max(moves, key=lambda pair: pair[0])[1]
    if (
        phase == "escort"
        and len(obs) >= escort_runtime.NAVIGATION_OBS_DIM
        and (obs[40] or escort_blocks_route)
    ):
        # Reaching an escort waypoint does not justify blocking the holder's
        # next cell. A legal step clears it; navigation gain breaks ties.
        moves = [
            (float(obs[escort_runtime.TACTICAL_OBS_DIM + i]), i)
            for i in range(4)
            if mask[i]
        ]
        if moves:
            return max(moves, key=lambda pair: pair[0])[1]
        return None
    chars = state.get("chars", [])
    carrier = next(
        (
            c
            for c in chars
            if c.team == char.team
            and getattr(c, "is_alive", True)
            and getattr(c, "has_spike", False)
        ),
        None,
    )
    if carrier is not None:
        cache = controller.__dict__.setdefault("_intent_distance_cache", {})
        screen = carrier_screening_status(view, carrier, chars, cache)
        final_distance = screen["final_distance"]
        if (
            FORMATION_MIN_FINAL_DISTANCE
            < final_distance
            <= SCREEN_COMMITMENT_MAX_FINAL_DISTANCE
        ):
            designated = screen["designated"]
            if (
                phase == "escort"
                and designated is not None
                and designated.name == char.name
            ):
                if screen["screen_ready"]:
                    stay = escort_runtime.ACTION_STAY
                    return stay if mask[stay] else None
                pos = tuple(map(int, char.pos))
                distance = screen["formation_target_distance"]
                actions = []
                for action, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                    nxt = (pos[0] + dr, pos[1] + dc)
                    if mask[action] and int(distance[nxt]) >= 0:
                        actions.append((int(distance[nxt]), action))
                if actions:
                    return min(actions)[1]
    final = phase == "carry" and navigation_intent(view, char)[0] == team_plant_target(
        view
    )
    return navigation_teacher_action(phase, obs, mask, final_approach=final)


def fake_wait_support_teacher_action(phase, char, state, view, controller, mask):
    """Training-only label: keep a fake's waiting guard near the Spike holder."""
    if phase != "escort":
        return None
    carrier = next(
        (
            ally
            for ally in state.get("chars", ())
            if ally.team == char.team
            and ally.name != char.name
            and getattr(ally, "is_alive", True)
            and getattr(ally, "has_spike", False)
        ),
        None,
    )
    if carrier is None:
        return None
    if (
        navigation_intent(view, carrier)[2] != "FAKE_WAIT"
        or navigation_intent(view, char)[2] != "FAKE_WAIT"
    ):
        return None
    distance = controller._get_carry_dist_map(view.grid, tuple(map(int, carrier.pos)))
    pos = tuple(map(int, char.pos))
    current = float(distance[pos])
    if not np.isfinite(current):
        return None
    if current <= FAKE_WAIT_SUPPORT_DESIRED_RADIUS:
        # The ordinary waypoint teacher can send a newly arrived bodyguard
        # straight back toward its separate staging slot. Keep the support
        # band instead; step aside if directly adjacent to the holder.
        if current <= 1:
            goal = navigation_intent(view, carrier)[0]
            route = (
                controller._get_goal_dist_map(view.grid, goal)
                if goal is not None and hasattr(controller, "_get_goal_dist_map")
                else None
            )
            carrier_route = float(route[tuple(map(int, carrier.pos))]) if route is not None else -1
            sidesteps = []
            for action, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                nxt = (pos[0] + dr, pos[1] + dc)
                if not mask[action]:
                    continue
                support_distance = float(distance[nxt])
                if not current < support_distance <= FAKE_WAIT_SUPPORT_RADIUS:
                    continue
                route_distance = float(route[nxt]) if route is not None else -1
                blocks_route = (
                    carrier_route >= 0 and 0 <= route_distance < carrier_route
                )
                sidesteps.append((blocks_route, support_distance, action))
            if sidesteps:
                return min(sidesteps)[2]
        return escort_runtime.ACTION_STAY if mask[escort_runtime.ACTION_STAY] else None
    choices = []
    for action, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        nxt = (pos[0] + dr, pos[1] + dc)
        if mask[action] and float(distance[nxt]) < current:
            choices.append((float(distance[nxt]), action))
    return min(choices)[1] if choices else None


def ultimate_context_from_observation(phase, obs):
    start = {
        "carry": runtime.ENTRY_SYNC_OBS_DIM,
        "escort": escort_runtime.SCREEN_COMMITMENT_OBS_DIM,
        "guard": guard_runtime.OBS_DIM,
    }[phase]
    if obs is None or len(obs) < start + 4:
        return None
    return obs[start : start + 4]


def ultimate_teacher_needed(phase, char, state, view, obs=None):
    """Observable tactical windows used only to label learned ultimate actions."""
    learned_context = ultimate_context_from_observation(phase, obs)
    if learned_context is not None:
        return tactical_ultimate_window(char, learned_context)
    chars = state.get("chars", [])
    if getattr(char, "ultimate_points", 0) < getattr(char, "ultimate_cost", 1):
        return False
    if phase == "guard":
        active_defuse = any(
            timer > 0
            for timer, _required in (state.get("defender_defuse_info") or {}).values()
        )
        return active_defuse or can_engage(view, char, chars)
    carrier = next(
        (
            c
            for c in chars
            if c.team == char.team
            and getattr(c, "is_alive", True)
            and getattr(c, "has_spike", False)
        ),
        None,
    )
    if carrier is None:
        return False
    cache = getattr(view, "_gc_ultimate_teacher_cache", None)
    if cache is None:
        cache = {}
        setattr(view, "_gc_ultimate_teacher_cache", cache)
    screen = carrier_screening_status(view, carrier, chars, cache)
    if phase == "carry":
        return carrier.name == char.name and 0 <= screen["final_distance"] <= 10
    designated = screen.get("designated")
    return bool(
        designated is not None
        and designated.name == char.name
        and 0 <= screen["final_distance"] <= 16
        and not screen["screen_ready"]
    )


def carrier_entry_sync_needed(screen, state, obs=None):
    """Whether Carry may pause at the entry boundary for at most two ticks."""
    designated = screen.get("designated")
    final_distance = int(screen.get("final_distance", -1))
    formation_distance = int(screen.get("designated_distance", -1))
    round_time = int(state.get("round_timer", 0))
    idle_feature = (
        float(obs[runtime.TACTICAL_OBS_DIM + 14])
        if obs is not None and len(obs) >= runtime.NAVIGATION_OBS_DIM
        else 0.0
    )
    return bool(
        designated is not None
        and final_distance == ENTRY_SYNC_STAGING_DISTANCE
        and not screen.get("screen_ready", False)
        and 0 <= formation_distance <= ENTRY_SYNC_MAX_FORMATION_DISTANCE
        and round_time >= ENTRY_SYNC_MIN_ROUND_TIME
        and idle_feature < ENTRY_SYNC_MAX_IDLE_FEATURE
    )


def optimize_demonstrations(
    policy, optimizer, samples, weight, batch_size=64,
    focus_samples=None, focus_fraction=0.5, focus_with_replacement=False,
):
    if not samples or weight <= 0:
        return None
    count = min(batch_size, len(samples))
    focus_count = int(count * focus_fraction) if focus_samples else 0
    if not focus_with_replacement:
        focus_count = min(len(focus_samples) if focus_samples else 0, focus_count)
    batch = (
        random.choices(list(focus_samples), k=focus_count)
        if focus_count and focus_with_replacement
        else random.sample(list(focus_samples), focus_count)
        if focus_count else []
    )
    batch += random.sample(list(samples), count - focus_count)
    obs, action, mask = zip(*batch)
    q = policy(torch.from_numpy(np.array(obs)))
    valid = torch.from_numpy(np.array(mask))
    indices = torch.tensor(action)
    competitor = q.masked_fill(~valid, -torch.inf)
    margin = torch.full_like(q, 0.8)
    margin.scatter_(1, indices[:, None], 0.0)
    loss = (
        weight
        * (
            torch.max(competitor + margin, dim=1).values
            - q.gather(1, indices[:, None]).squeeze(1)
        ).mean()
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach())


def optimize_orb_collection_head(
    policy, optimizer, samples, orb_action, batch_size=64,
):
    """Raise only COLLECT's advantage against legal alternatives.

    A separate optimizer gives the rare collection row enough learning rate
    without changing navigation, ability, ultimate, or facing parameters.
    """
    if not samples:
        return None
    batch = random.sample(list(samples), min(batch_size, len(samples)))
    obs, actions, masks = zip(*batch)
    if any(int(action) != orb_action for action in actions):
        raise ValueError("orb collection replay contains a non-collection label")
    output = policy.advantage_head[-1]
    q = policy(torch.from_numpy(np.asarray(obs, dtype=np.float32)))
    valid = torch.from_numpy(np.asarray(masks, dtype=bool))
    competitor = q.masked_fill(~valid, -torch.inf)
    margin = torch.full_like(q, 0.8)
    margin[:, orb_action] = 0.0
    loss = (
        torch.max(competitor + margin, dim=1).values - q[:, orb_action]
    ).mean()
    if float(loss.detach()) <= 1e-6:
        return 0.0
    weight_grad, bias_grad = torch.autograd.grad(
        loss, (output.weight, output.bias)
    )
    optimizer.zero_grad()
    output.weight.grad = torch.zeros_like(output.weight)
    output.bias.grad = torch.zeros_like(output.bias)
    output.weight.grad[orb_action] = weight_grad[orb_action]
    output.bias.grad[orb_action] = bias_grad[orb_action]
    torch.nn.utils.clip_grad_norm_((output.weight, output.bias), 5.0)
    optimizer.step()
    return float(loss.detach())


def optimize_facing(policy, optimizer, samples, weight, batch_size=64):
    """Train the phase-local facing head without expanding base actions."""
    if not samples or weight <= 0 or not hasattr(policy, "facing_values"):
        return None
    batch = random.sample(list(samples), min(batch_size, len(samples)))
    normalized = [(*sample, 1.0) if len(sample) == 3 else sample for sample in batch]
    obs, actions, targets, confidences = zip(*normalized)
    obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32))
    action_tensor = torch.tensor(actions, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    logits = policy.facing_values(obs_tensor, action_tensor)
    confidence_tensor = torch.tensor(confidences, dtype=torch.float32)
    losses = torch.nn.functional.cross_entropy(logits, target_tensor, reduction="none")
    loss = (
        weight
        * (losses * confidence_tensor).sum()
        / confidence_tensor.sum().clamp_min(1e-6)
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach())


def optimize_ultimate_classification(
    policy, optimizer, positives, negatives, ultimate_action, weight, batch_size=64
):
    """Teach both tactical casts and explicit saves with a balanced margin."""
    if weight <= 0 or (not positives and not negatives):
        return None
    if positives and negatives:
        per_class = max(1, batch_size // 2)
        positive_batch = random.sample(list(positives), min(per_class, len(positives)))
        negative_batch = random.sample(list(negatives), min(per_class, len(negatives)))
    else:
        positive_batch = random.sample(list(positives), min(batch_size, len(positives)))
        negative_batch = random.sample(list(negatives), min(batch_size, len(negatives)))
    batch = [(obs, mask, 1.0) for obs, mask in positive_batch]
    batch.extend((obs, mask, 0.0) for obs, mask in negative_batch)
    random.shuffle(batch)
    obs, mask, labels = zip(*batch)
    q = policy(torch.from_numpy(np.asarray(obs, dtype=np.float32)))
    valid = torch.from_numpy(np.asarray(mask, dtype=bool))
    other_valid = valid.clone()
    other_valid[:, ultimate_action] = False
    competitor = q.masked_fill(~other_valid, -torch.inf).max(dim=1).values
    margin = q[:, ultimate_action] - competitor
    label_tensor = torch.tensor(labels, dtype=q.dtype)
    # Positive windows put ULT above the best executable alternative; negative
    # windows put it below. Equal class sampling prevents plentiful save labels
    # from erasing rare tactical casts.
    loss = (
        weight
        * torch.where(
            label_tensor > 0.5,
            torch.relu(0.8 - margin),
            torch.relu(0.8 + margin),
        ).mean()
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach())


def selection_score(metrics, max_no_entry, max_timeout):
    # Among failed checkpoints, improve the actual entry/timeout failures first.
    # A passing checkpoint still prioritizes robust wins over cosmetic position.
    no_entry = metrics.get("worst_carry_no_entry_rate", metrics["carry_no_entry_rate"])
    timeout = metrics.get("worst_timeout_rate", metrics["timeout_rate"])
    violation = max(0, no_entry - max_no_entry) + max(0, timeout - max_timeout)
    carrier_death = metrics.get(
        "worst_no_entry_carrier_death_rate",
        metrics.get("no_entry_carrier_death_rate", 1.0),
    )
    worst_registered = metrics.get(
        "worst_registered_plant_rate", metrics.get("registered_plant_rate", 0.0)
    )
    return (
        int(violation == 0),
        -violation,
        -carrier_death,
        worst_registered,
        metrics["worst_round_win_rate"],
        metrics["round_win_rate"],
        -metrics["carry_spawn_tick_rate"],
    )


def carry_movement_selection_score(metrics, max_no_entry, max_timeout):
    """Select Carry displacement without hiding entry failures behind wins."""
    no_entry = metrics.get("worst_carry_no_entry_rate", metrics["carry_no_entry_rate"])
    timeout = metrics.get("worst_timeout_rate", metrics["timeout_rate"])
    violation = max(0.0, no_entry - max_no_entry) + max(0.0, timeout - max_timeout)
    return (
        int(violation <= 1e-12),
        -violation,
        -no_entry,
        -timeout,
        metrics.get("worst_registered_plant_rate", 0.0),
        metrics.get("worst_round_win_rate", 0.0),
        metrics.get("round_win_rate", 0.0),
        -metrics.get("carry_spawn_tick_rate", 1.0),
        -metrics.get("carry_reversal_tick_rate", 1.0),
        -metrics.get("carry_quiet_stall_tick_rate", 1.0),
    )


def escort_support_selection_score(metrics, max_no_entry, max_timeout):
    """Prefer preventing carrier deaths, while rejecting timeout regressions."""
    no_entry = metrics.get("worst_carry_no_entry_rate", metrics["carry_no_entry_rate"])
    timeout = metrics.get("worst_timeout_rate", metrics["timeout_rate"])
    carrier_death = metrics.get(
        "worst_carrier_preentry_death_rate",
        metrics.get("carrier_preentry_death_rate", no_entry),
    )
    mean_carrier_death = metrics.get("carrier_preentry_death_rate", carrier_death)
    violation = max(0.0, no_entry - max_no_entry) + max(0.0, timeout - max_timeout)
    return (
        int(violation <= 1e-12),
        -carrier_death,
        -mean_carrier_death,
        -violation,
        -no_entry,
        -timeout,
        metrics.get("worst_registered_plant_rate", 0.0),
        metrics.get("worst_round_win_rate", 0.0),
        metrics.get("round_win_rate", 0.0),
        metrics.get("first_threat_screen_ready_rate", 0.0),
        metrics.get("screened_approach_rate", 0.0),
    )


def joint_movement_selection_score(
    metrics, baseline_metrics, max_no_entry, max_timeout,
    max_quiet_stall_increase, max_carrier_death_increase, max_plant_regression,
):
    """Select coordinated Carry/Escort movement without trading away planting."""
    no_entry = metrics.get("worst_carry_no_entry_rate", metrics["carry_no_entry_rate"])
    timeout = metrics.get("worst_timeout_rate", metrics["timeout_rate"])
    violation = max(0.0, no_entry - max_no_entry) + max(0.0, timeout - max_timeout)
    unsafe = joint_movement_guardrail_violated(
        metrics,
        baseline_metrics,
        max_timeout,
        max_quiet_stall_increase,
        max_carrier_death_increase,
        max_plant_regression,
    )
    return (
        int(not unsafe),
        int(violation <= 1e-12),
        int(metrics.get("orb_collections", 0) > 0),
        metrics.get("orb_collections", 0) / max(1, metrics.get("episodes", 1)),
        -violation,
        metrics.get("worst_registered_plant_rate", 0.0),
        -metrics.get("worst_carrier_preentry_death_rate", 1.0),
        -metrics.get("carrier_preentry_death_rate", 1.0),
        metrics.get("worst_round_win_rate", 0.0),
        metrics.get("round_win_rate", 0.0),
        -metrics.get("carry_quiet_stall_tick_rate", 1.0),
    )


def carry_movement_guardrail_violated(
    metrics,
    baseline_metrics,
    max_timeout,
    max_quiet_stall_increase,
):
    """Detect sustained deployment regressions without changing inference behavior."""
    timeout = metrics.get("worst_timeout_rate", metrics["timeout_rate"])
    quiet_stall = metrics.get("carry_quiet_stall_tick_rate", 0.0)
    baseline_quiet_stall = baseline_metrics.get("carry_quiet_stall_tick_rate", 0.0)
    return (
        timeout > max_timeout
        or quiet_stall > baseline_quiet_stall + max_quiet_stall_increase
    )


def escort_support_guardrail_violated(
    metrics,
    baseline_metrics,
    max_timeout,
    max_carrier_death_increase,
    max_plant_regression,
    max_no_entry_increase=0.03,
):
    timeout = metrics.get("worst_timeout_rate", metrics["timeout_rate"])
    baseline_timeout = baseline_metrics.get(
        "worst_timeout_rate", baseline_metrics["timeout_rate"]
    )
    death = metrics.get(
        "worst_carrier_preentry_death_rate",
        metrics.get("carrier_preentry_death_rate", 1.0),
    )
    baseline_death = baseline_metrics.get(
        "worst_carrier_preentry_death_rate",
        baseline_metrics.get("carrier_preentry_death_rate", 1.0),
    )
    plant = metrics.get(
        "worst_registered_plant_rate", metrics.get("registered_plant_rate", 0.0)
    )
    baseline_plant = baseline_metrics.get(
        "worst_registered_plant_rate",
        baseline_metrics.get("registered_plant_rate", 0.0),
    )
    no_entry = metrics.get(
        "worst_carry_no_entry_rate", metrics.get("carry_no_entry_rate", 1.0)
    )
    baseline_no_entry = baseline_metrics.get(
        "worst_carry_no_entry_rate", baseline_metrics.get("carry_no_entry_rate", 1.0)
    )
    return (
        timeout > max(max_timeout, baseline_timeout + 0.02)
        or death > baseline_death + max_carrier_death_increase
        or plant < baseline_plant - max_plant_regression
        or no_entry > baseline_no_entry + max_no_entry_increase
    )


def joint_movement_guardrail_violated(
    metrics,
    baseline_metrics,
    max_timeout,
    max_quiet_stall_increase,
    max_carrier_death_increase,
    max_plant_regression,
):
    return carry_movement_guardrail_violated(
        metrics,
        baseline_metrics,
        max_timeout,
        max_quiet_stall_increase,
    ) or escort_support_guardrail_violated(
        metrics,
        baseline_metrics,
        max_timeout,
        max_carrier_death_increase,
        max_plant_regression,
    )


def scheduled_learning_rate(base_lr, episode, decay_start, decay_episodes, final_scale):
    """Linear decay after a stable warm phase; episode is schedule-relative."""
    if episode <= decay_start:
        return float(base_lr)
    progress = min(1.0, (episode - decay_start) / max(1, decay_episodes))
    scale = 1.0 - progress * (1.0 - final_scale)
    return float(base_lr * scale)


def scheduled_orb_teacher_probability(episode, bootstrap_episodes, final_probability):
    """Fade forced orb actions so missed opportunities enter the replay buffer."""
    if bootstrap_episodes <= 0:
        return float(final_probability)
    remaining = max(0.0, 1.0 - (episode - 1) / bootstrap_episodes)
    return float(final_probability + (0.80 - final_probability) * remaining)


def phase_facing_selection_score(
    phase, metrics, baseline, max_no_entry, max_timeout, regression_tolerance=0.05
):
    """Select each phase head independently without accepting tactical collapse."""
    facing = metrics.get("facing_confident_match_rate_by_phase", {}).get(phase, 0.0)
    facing = metrics.get("facing_weighted_match_rate_by_phase", {}).get(phase, facing)
    timeout_limit = max(max_timeout, baseline.get("worst_timeout_rate", max_timeout))
    timeout_violation = max(0.0, metrics.get("worst_timeout_rate", 1.0) - timeout_limit)
    if phase == "carry":
        no_entry_limit = min(
            1.0,
            baseline.get("worst_carry_no_entry_rate", max_no_entry)
            + regression_tolerance,
        )
        behavior_violation = max(
            0.0,
            metrics.get("worst_carry_no_entry_rate", 1.0) - no_entry_limit,
        )
        tie_break = -metrics.get("worst_carry_no_entry_rate", 1.0)
    elif phase == "escort":
        plant_floor = max(
            0.0,
            baseline.get("registered_plant_rate", 0.0) - regression_tolerance,
        )
        formation_floor = max(
            0.0,
            baseline.get("formation_ready_approach_rate", 0.0) - regression_tolerance,
        )
        behavior_violation = max(
            0.0, plant_floor - metrics.get("registered_plant_rate", 0.0)
        ) + max(
            0.0,
            formation_floor - metrics.get("formation_ready_approach_rate", 0.0),
        )
        tie_break = metrics.get("registered_plant_rate", 0.0)
    else:
        raise ValueError(f"phase-specific facing selection is unsupported: {phase}")
    violation = timeout_violation + behavior_violation
    return (
        int(violation == 0.0),
        -violation,
        facing,
        metrics.get("worst_round_win_rate", 0.0),
        tie_break,
    )


def phase_facing_ab_score(phase, metrics, baseline, regression_tolerance=0.05):
    """Score one changed facing head while every other policy stays fixed.

    Movement quality is not an objective here.  The candidate only has to avoid
    regressing the unchanged baseline beyond the tolerance; facing accuracy is
    then maximized among safe candidates.
    """
    facing = metrics.get("facing_weighted_match_rate_by_phase", {}).get(phase, 0.0)
    violation = max(
        0.0,
        metrics.get("worst_timeout_rate", 1.0)
        - baseline.get("worst_timeout_rate", 1.0)
        - regression_tolerance,
    )
    violation += max(
        0.0,
        baseline.get("worst_round_win_rate", 0.0)
        - metrics.get("worst_round_win_rate", 0.0)
        - regression_tolerance,
    )
    if phase == "carry":
        violation += max(
            0.0,
            metrics.get("worst_carry_no_entry_rate", 1.0)
            - baseline.get("worst_carry_no_entry_rate", 1.0)
            - regression_tolerance,
        )
        tie_break = -metrics.get("worst_carry_no_entry_rate", 1.0)
    elif phase == "escort":
        violation += max(
            0.0,
            baseline.get("registered_plant_rate", 0.0)
            - metrics.get("registered_plant_rate", 0.0)
            - regression_tolerance,
        )
        violation += max(
            0.0,
            baseline.get("formation_ready_approach_rate", 0.0)
            - metrics.get("formation_ready_approach_rate", 0.0)
            - regression_tolerance,
        )
        tie_break = metrics.get("registered_plant_rate", 0.0)
    else:
        raise ValueError(f"phase-specific facing selection is unsupported: {phase}")
    return (
        int(violation <= 1e-12),
        -violation,
        facing,
        metrics.get("worst_round_win_rate", 0.0),
        tie_break,
    )


def entry_quality_passed(metrics, max_no_entry, max_timeout):
    return (
        metrics.get("worst_carry_no_entry_rate", metrics["carry_no_entry_rate"])
        <= max_no_entry
        and metrics.get("worst_timeout_rate", metrics["timeout_rate"]) <= max_timeout
    )


def screen_state_at_contact(status, previously_ready=False, alive_allies=None):
    """Classify why the designated Escort was not screening at contact."""
    if status is None or status.get("route_goal") is None:
        return "no_route"
    final_distance = status.get("final_distance", -1)
    if final_distance > SCREEN_COMMITMENT_MAX_FINAL_DISTANCE:
        return "before_commitment_window"
    if final_distance <= FORMATION_MIN_FINAL_DISTANCE:
        return "after_commitment_window"
    if not status.get("formation_candidates"):
        if alive_allies == 0:
            return "no_alive_escort"
        return "no_same_route_candidate"
    if status.get("designated") is None:
        return "no_reachable_designated"
    if status.get("screen_ready"):
        return "ready"
    if previously_ready:
        return "lost_after_ready"
    if status.get("designated_distance", -1) > 1:
        return "designated_lagging"
    return "insufficient_lead"


def screen_macro_context(game, carrier, alive_names=None, positions=None):
    """Diagnostic only: summarize the current observable Macro assignments."""
    macro = own_macro(game)
    env = getattr(macro, "env", None)
    assignments = getattr(env, "assignment", {})

    def role(name):
        assignment = assignments.get(name, ())
        return assignment[2] if len(assignment) > 2 else "MAIN"

    allies = [
        c
        for c in game.chars
        if c.team == "A"
        and c.name != carrier.name
        and (c.name in alive_names if alive_names is not None else c.is_alive)
    ]

    def distance(ally):
        ally_pos = positions.get(ally.name, ally.pos) if positions else ally.pos
        carrier_pos = (
            positions.get(carrier.name, carrier.pos) if positions else carrier.pos
        )
        return max(
            abs(int(ally_pos[0]) - int(carrier_pos[0])),
            abs(int(ally_pos[1]) - int(carrier_pos[1])),
        )

    same_role = [ally for ally in allies if role(ally.name) == role(carrier.name)]
    return {
        "strategy": str(getattr(env, "current_strategy", "") or ""),
        "carrier_role": role(carrier.name),
        "alive_ally_roles": [role(c.name) for c in allies],
        "nearest_ally_distance": min(map(distance, allies), default=None),
        "nearest_same_role_distance": min(map(distance, same_role), default=None),
    }


def entered_site(row):
    # Macro may retarget while the carrier plants elsewhere. A completed plant
    # always proves entry, even if proximity to the current plan was not seen.
    return bool(row["planted"] or row.get("carry_entry_tick") is not None)


def opponent_stats(episode, curriculum_episodes, start, final):
    fraction = min(1.0, max(0.0, (episode - 1) / max(1, curriculum_episodes - 1)))
    if curriculum_episodes == 0:
        fraction = 1.0
    return tuple(a + fraction * (b - a) for a, b in zip(start, final))


def phase_of(char, planted, holder):
    if planted:
        return "guard"
    if holder is None:
        return None  # Retrieve remains the deployed policy.
    return "carry" if char.name == holder.name else "escort"


def observable_facing_target(
    phase, char, state, controller, goal=None, action=None, team_sighting=None
):
    """Return ``(direction, confidence, source)`` for facing supervision.

    During training, the closest alive enemy is a privileged teacher target,
    including while it is behind a wall or smoke.  The policy still receives
    only its normal observation, so inference remains network-driven.
    """
    chars = state.get("chars", ())
    nearest_enemy_facing = nearest_alive_enemy_facing(char, chars)
    if nearest_enemy_facing is not None:
        return FACING_DIRS.index(nearest_enemy_facing), 1.0, "nearest_enemy"

    sighting = team_sighting or getattr(controller, "_sighting", None)
    if (
        sighting is not None
        and sighting.get("pos") is not None
        and tuple(sighting["pos"]) != tuple(char.pos)
    ):
        facing = facing_towards(tuple(char.pos), tuple(sighting["pos"]))
        age = max(0, int(sighting.get("tick_ago", 0)))
        confidence = max(0.35, 0.80 * (1.0 - age / 25.0))
        return FACING_DIRS.index(facing), confidence, "team_sighting"

    if phase == "escort":
        carrier = next(
            (
                other
                for other in chars
                if getattr(other, "is_alive", True)
                and other.team == char.team
                and getattr(other, "has_spike", False)
            ),
            None,
        )
        if carrier is not None and tuple(carrier.pos) != tuple(char.pos):
            # Screeners watch away from the protected carrier.  This is only a
            # training label; inference remains entirely network-driven.
            dr = int(char.pos[0]) - int(carrier.pos[0])
            dc = int(char.pos[1]) - int(carrier.pos[1])
            outward = (int(char.pos[0]) + dr, int(char.pos[1]) + dc)
            facing = facing_towards(tuple(char.pos), outward)
            return FACING_DIRS.index(facing), 0.55, "escort_outward"

    movement = None
    if action is not None:
        if phase == "carry" and 0 <= int(action) < runtime.PLANT_ACTION_INDEX:
            movement = runtime.MOVES[int(action) // 2]
        elif phase == "escort" and 0 <= int(action) < 4:
            movement = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(action)]
    if movement is not None and movement != (0, 0):
        target = (int(char.pos[0]) + movement[0], int(char.pos[1]) + movement[1])
        facing = facing_towards(tuple(char.pos), target)
        return FACING_DIRS.index(facing), 0.35, "movement"

    facing = getattr(char, "facing", "N")
    return (
        FACING_DIRS.index(facing) if facing in FACING_DIRS else 0,
        0.15,
        "hold",
    )


def observable_facing_teacher(phase, char, state, controller, goal=None):
    """Backward-compatible direction-only wrapper used by focused tests."""
    return observable_facing_target(phase, char, state, controller, goal)[0]


class CurriculumSession(RealCarrySession):
    def __init__(self, sources, policies, gamma):
        super().__init__(sources["carry"], policies["carry"], lambda *a: None, gamma)
        self.policies = policies
        inner = self.game.attacker_controller.inner_controller
        if inner.macro_controller is None:
            raise RuntimeError("Macro must be loaded for tactical-intent training")
        inner.escort = escort_runtime.LearningAttackerEscortGCController(
            str(sources["escort"]), device="cpu", greedy=True
        )
        # Strict load here: the Guard runtime's legacy fallback must not silently
        # turn a bad training source into a randomly initialized network.
        with contextlib.redirect_stdout(io.StringIO()):
            inner.guard = guard_runtime.LearningAttackerGuardGCController(
                str(sources["guard"])
            )
        self.controllers = {
            "carry": inner.carry,
            "escort": inner.escort,
            "guard": inner.guard,
        }
        self.actor = None
        self.decision_context = None
        self.action_goals = {}
        self.teacher_probability = 0.0
        self.orb_visible_ticks = 0
        self.orb_teacher_actions = 0
        self.orb_teacher_forced_actions = 0
        self.orb_eligible_decisions = 0
        self.orb_eligible_misses = 0
        self.orb_greedy_choices = 0
        self.orb_q_margin_sum = 0.0
        self.orb_q_margin_count = 0
        self.orb_eligible_pending = {}
        self.orb_eligible_actor_names = set()
        self.orb_teacher_probability = 0.0
        self.orb_teacher_round_enabled = False
        self.collect_demonstrations = False
        self.frozen_phases = set()
        self.frozen_facing_phases = set()
        self.demonstrations = {phase: [] for phase in PHASES}
        self.orb_demonstrations = {phase: [] for phase in PHASES}
        self.orb_approach_demonstrations = {phase: [] for phase in PHASES}
        self.orb_collection_demonstrations = {phase: [] for phase in PHASES}
        self.ultimate_examples = {phase: [] for phase in PHASES}
        self.facing_examples = {phase: [] for phase in FACING_PHASES}
        self.facing_decisions = {phase: 0 for phase in FACING_PHASES}
        self.facing_teacher_matches = {phase: 0 for phase in FACING_PHASES}
        self.facing_confident_decisions = {phase: 0 for phase in FACING_PHASES}
        self.facing_confident_matches = {phase: 0 for phase in FACING_PHASES}
        self.facing_confidence_mass = {phase: 0.0 for phase in FACING_PHASES}
        self.facing_weighted_matches = {phase: 0.0 for phase in FACING_PHASES}
        self.facing_context_counts = {phase: {} for phase in FACING_PHASES}
        self.facing_context_matches = {phase: {} for phase in FACING_PHASES}
        for phase, controller in self.controllers.items():
            controller.positioning_version = VERSIONS[phase]
            if phase == "guard":
                controller.model = policies[phase]
            else:
                controller.policy_net = policies[phase]
            if phase in FACING_PHASES:
                controller.facing_head_enabled = True
                controller._select_facing = self.facing_selector(phase)
            if hasattr(controller, "set_game"):
                controller.set_game(self.game)
            controller._select_action = self.selector(phase)
            original = controller.decide_move

            def decide(char, state, original=original):
                self.actor = char.name
                self.decision_context = (char, state)
                try:
                    return original(char, state)
                finally:
                    self.actor = None
                    self.decision_context = None

            controller.decide_move = decide
        self.transitions = {phase: [] for phase in PHASES}
        self.arrived = set()
        self.progress = ProgressTracker()

    def selector(self, phase):
        def choose(obs, mask):
            context = getattr(self, "decision_context", None)
            key = (phase, self.actor)
            previous = self.pending.pop(key, None)
            if previous is not None:
                self.flush(key, previous, obs, mask, False)
            valid = np.flatnonzero(mask)
            view = None
            if getattr(self, "collect_demonstrations", False) and phase == "guard":
                # Guard consumes state directly and has no set_game method.
                # The wrapper holds this decision's existing perceived view.
                view = self.game.attacker_controller.inner_controller.game
            orb_teacher_action = None
            teacher = None
            if getattr(self, "collect_demonstrations", False):
                orb_teacher_action = observable_teacher_action(
                    phase,
                    obs,
                    mask,
                    self.controllers[phase],
                    getattr(self, "decision_context", None),
                    view=view,
                    orb_only=True,
                )
                teacher = (
                    orb_teacher_action
                    if orb_teacher_action is not None
                    else observable_teacher_action(
                        phase,
                        obs,
                        mask,
                        self.controllers[phase],
                        getattr(self, "decision_context", None),
                        view=view,
                    )
                )
            orb_action = {
                "carry": runtime.COLLECT_ORB_ACTION_INDEX,
                "escort": escort_runtime.ACTION_COLLECT_ORB,
                "guard": guard_runtime.COLLECT_ORB_ACTION_INDEX,
            }[phase]
            orb_eligible = bool(
                phase in ("carry", "escort")
                and orb_action < len(mask)
                and mask[orb_action]
            )
            if orb_eligible and self.collect_demonstrations:
                # Every executable on-orb state is a positive example, even
                # when another ally was the designated approach collector.
                orb_teacher_action = teacher = orb_action
            if phase in ("carry", "escort"):
                state_orbs = context[1].get("available_orbs") if context is not None else ()
                if not state_orbs:
                    state_orbs = getattr(view, "available_orbs", ())
                if not state_orbs:
                    grid = getattr(view, "grid", None)
                    if grid is not None:
                        state_orbs = list(zip(*np.where(np.asarray(grid) == 5)))
                if state_orbs:
                    self.orb_visible_ticks += 1
                if teacher in {
                    runtime.COLLECT_ORB_ACTION_INDEX,
                    escort_runtime.ACTION_COLLECT_ORB,
                }:
                    self.orb_teacher_actions += 1
            if teacher is not None:
                sample = (obs.copy(), teacher, mask.copy())
                self.demonstrations[phase].append(sample)
                if orb_teacher_action is not None:
                    # Keep every teacher step toward the orb (not merely the
                    # final COLLECT action) in a dedicated replay pool.
                    self.orb_demonstrations[phase].append(sample)
                    if teacher == orb_action:
                        self.orb_collection_demonstrations[phase].append(sample)
                    else:
                        self.orb_approach_demonstrations[phase].append(sample)
            orb_teacher = orb_teacher_action is not None
            if (
                getattr(self, "collect_demonstrations", False)
                and getattr(self, "decision_context", None) is not None
            ):
                ultimate_action = {
                    "carry": runtime.ULTIMATE_ACTION_INDEX,
                    "escort": escort_runtime.ACTION_ULTIMATE,
                    "guard": guard_runtime.ULTIMATE_ACTION_INDEX,
                }[phase]
                if ultimate_action < len(mask) and mask[ultimate_action]:
                    char, state = self.decision_context
                    tactical = ultimate_teacher_needed(
                        phase, char, state, view or self.controllers[phase].game, obs
                    )
                    self.__dict__.setdefault(
                        "ultimate_examples", {item: [] for item in PHASES}
                    )[phase].append((obs.copy(), mask.copy(), bool(tactical)))
            if (
                orb_teacher
                and phase not in getattr(self, "frozen_phases", ())
                and getattr(self, "orb_teacher_round_enabled", False)
            ):
                action = teacher
                self.orb_teacher_forced_actions = getattr(
                    self, "orb_teacher_forced_actions", 0
                ) + 1
            elif (
                teacher is not None
                and not orb_teacher
                and phase not in getattr(self, "frozen_phases", ())
                and random.random() < self.teacher_probability
            ):
                action = teacher
            elif (
                phase not in getattr(self, "frozen_phases", ())
                and random.random() < self.epsilon
            ):
                action = (
                    int(random.choice(valid))
                    if len(valid)
                    else (4 if phase == "escort" else 0)
                )
            else:
                with torch.no_grad():
                    q = (
                        self.policies[phase](torch.from_numpy(obs).unsqueeze(0))
                        .squeeze(0)
                        .numpy()
                    )
                action = int(np.argmax(np.where(mask, q, -1e9)))
            self.__dict__.setdefault("orb_eligible_pending", {})[key] = orb_eligible
            if orb_eligible:
                if context is not None:
                    self.__dict__.setdefault("orb_eligible_actor_names", set()).add(
                        str(context[0].name)
                    )
                self.orb_eligible_decisions += 1
                self.orb_eligible_misses += int(action != orb_action)
                with torch.no_grad():
                    greedy_q = (
                        self.policies[phase](torch.from_numpy(obs).unsqueeze(0))
                        .squeeze(0)
                        .numpy()
                    )
                rival_mask = np.asarray(mask, dtype=bool).copy()
                rival_mask[orb_action] = False
                if rival_mask.any():
                    margin = float(greedy_q[orb_action] - np.max(greedy_q[rival_mask]))
                    self.orb_q_margin_sum += margin
                    self.orb_q_margin_count += 1
                self.orb_greedy_choices += int(
                    int(np.argmax(np.where(mask, greedy_q, -1e9))) == orb_action
                )
            self.pending[key] = Pending(obs.copy(), action)
            if getattr(self, "decision_context", None) is not None and phase != "guard":
                char, state = self.decision_context
                self.action_goals[key] = navigation_intent(
                    self.controllers[phase].game, char
                )[0]
            return action

        return choose

    def facing_selector(self, phase):
        def choose(obs, action):
            if not getattr(self.controllers[phase], "facing_head_enabled", False):
                return None
            key = (phase, self.actor)
            context = getattr(self, "decision_context", None)
            teacher = confidence = source = None
            if context is not None:
                char, state = context
                # The IQ wrapper may blur or omit enemies in ``state``.  That
                # remains the policy input, but the training-only label must
                # use the real positions to guarantee the truly nearest enemy.
                teacher_char = char
                teacher_state = state
                real_chars = getattr(getattr(self, "game", None), "chars", None)
                if real_chars:
                    real_char = next(
                        (
                            item
                            for item in real_chars
                            if getattr(item, "name", None) == getattr(char, "name", None)
                        ),
                        None,
                    )
                    if real_char is not None:
                        teacher_char = real_char
                        teacher_state = dict(state, chars=real_chars)
                teacher, confidence, source = observable_facing_target(
                    phase,
                    teacher_char,
                    teacher_state,
                    self.controllers[phase],
                    self.action_goals.get(key),
                    action=action,
                    team_sighting=getattr(
                        self.controllers.get("carry"), "_sighting", None
                    ),
                )
                if getattr(self, "collect_demonstrations", False):
                    self.facing_examples[phase].append(
                        (obs.copy(), int(action), int(teacher), float(confidence))
                    )
            if teacher is not None and getattr(self, "collect_demonstrations", False):
                # Training actors always execute the privileged pre-aim label.
                # Evaluation/inference still uses the learned facing head.
                index = teacher
            elif (
                phase not in getattr(self, "frozen_facing_phases", ())
                and random.random() < self.epsilon
            ):
                index = random.randrange(len(FACING_DIRS))
            else:
                with torch.no_grad():
                    tensor = torch.from_numpy(obs).unsqueeze(0)
                    values = (
                        self.policies[phase]
                        .facing_values(tensor, torch.tensor([action]))
                        .squeeze(0)
                    )
                index = int(values.argmax().item())
            if teacher is not None:
                self.facing_decisions[phase] += 1
                self.facing_teacher_matches[phase] += int(index == teacher)
                if confidence >= 0.5:
                    self.facing_confident_decisions[phase] += 1
                    self.facing_confident_matches[phase] += int(index == teacher)
                self.facing_confidence_mass[phase] += float(confidence)
                self.facing_weighted_matches[phase] += float(confidence) * int(
                    index == teacher
                )
                counts = self.facing_context_counts[phase]
                matches = self.facing_context_matches[phase]
                counts[source] = counts.get(source, 0) + 1
                matches[source] = matches.get(source, 0) + int(index == teacher)
            return FACING_DIRS[index]

        return choose

    def flush(self, key, pending, obs=None, mask=None, terminal=True):
        phase, name = key
        if obs is None:
            obs = np.zeros_like(pending.obs)
            mask = np.ones(
                self.policies[phase].advantage_head[-1].out_features, dtype=bool
            )
        self.transitions[phase].append(
            (
                name,
                (
                    pending.obs,
                    pending.action,
                    pending.reward,
                    obs.copy(),
                    mask.copy(),
                    terminal,
                    pending.ticks,
                ),
            )
        )

    def add_reward(self, key, reward):
        pending = self.pending[key]
        pending.reward += self.gamma**pending.ticks * reward
        pending.ticks += 1

    def play(
        self, seed, stats, epsilon=0.0, training=True,
        teacher_probability=0.0, orb_teacher_probability=None,
    ):
        self.collect_demonstrations = bool(training)
        self.epsilon = epsilon
        self.teacher_probability = teacher_probability if training else 0.0
        self.orb_teacher_probability = (
            (teacher_probability if orb_teacher_probability is None else orb_teacher_probability)
            if training else 0.0
        )
        # A full orb pickup needs five uninterrupted COLLECT ticks. Choose
        # teacher-guided *rounds*, not independently sampled teacher ticks.
        self.orb_teacher_round_enabled = bool(
            training and random.random() < self.orb_teacher_probability
        )
        self.orb_visible_ticks = 0
        self.orb_teacher_actions = 0
        self.orb_teacher_forced_actions = 0
        self.orb_eligible_decisions = 0
        self.orb_eligible_misses = 0
        self.orb_greedy_choices = 0
        self.orb_q_margin_sum = 0.0
        self.orb_q_margin_count = 0
        self.orb_eligible_pending = {}
        self.orb_eligible_actor_names = set()
        self.demonstrations = {phase: [] for phase in PHASES}
        self.orb_demonstrations = {phase: [] for phase in PHASES}
        self.orb_approach_demonstrations = {phase: [] for phase in PHASES}
        self.orb_collection_demonstrations = {phase: [] for phase in PHASES}
        self.ultimate_examples = {phase: [] for phase in PHASES}
        self.facing_examples = {phase: [] for phase in FACING_PHASES}
        self.facing_decisions = {phase: 0 for phase in FACING_PHASES}
        self.facing_teacher_matches = {phase: 0 for phase in FACING_PHASES}
        self.facing_confident_decisions = {phase: 0 for phase in FACING_PHASES}
        self.facing_confident_matches = {phase: 0 for phase in FACING_PHASES}
        self.facing_confidence_mass = {phase: 0.0 for phase in FACING_PHASES}
        self.facing_weighted_matches = {phase: 0.0 for phase in FACING_PHASES}
        self.facing_context_counts = {phase: {} for phase in FACING_PHASES}
        self.facing_context_matches = {phase: {} for phase in FACING_PHASES}
        self.action_goals.clear()
        self.transitions = {phase: [] for phase in PHASES}
        self.arrived.clear()
        self.progress = ProgressTracker()
        # Training and evaluation vary score/round/fatigue and both use the
        # normal deployed Macro plan, including its own information limits.
        self.reset(seed, augment=True, randomize_target=False)
        game = self.game
        # Single-round curriculum episodes do not naturally inherit match
        # points. Recreate a round-number-dependent ready/partial distribution
        # so learned ultimate actions receive both available and masked states.
        current_round = int(getattr(game, "current_round", 1))
        ready_probability = min(0.60, 0.10 + 0.50 * max(0, current_round - 1) / 23.0)
        for char in game.chars:
            if char.team != "A":
                continue
            if random.random() < ready_probability:
                char.ultimate_points = char.ultimate_cost
            else:
                char.ultimate_points = random.randrange(max(1, char.ultimate_cost))
        for char in game.chars:
            if char.team == "D":
                char.accuracy, char.hs_rate, char.dodge_rate = (
                    x / 100.0 for x in stats
                )
                # Condition-change events use these bases. Initial effective
                # combat stats equal the requested curriculum values exactly.
                multiplier = max(0.01, 1.0 + char.condition_modifier)
                char.base_accuracy_before_condition = char.accuracy / multiplier
                char.base_hs_rate_before_condition = char.hs_rate / multiplier
        initial_wins = game.attacker_wins
        guard_ticks = guard_arrivals = combat_ticks = combat_stops = 0
        combat_start_events = 0
        combat_start_angle_total = combat_start_alignment_total = 0.0
        combat_start_reward_total = 0.0
        active_combat_contacts = set()
        carry_ticks = carry_spawn_ticks = 0
        entry_tick = None
        plant_tick = None
        carry_reversals = carry_quiet_stalls = 0
        carry_positions = {}
        carrier_preentry_death = False
        carrier_preentry_death_engaged = False
        carrier_preentry_death_support_near = None
        carrier_preentry_death_support_ahead = None
        first_carrier_threat_tick = None
        first_carrier_threat_support_near = None
        first_carrier_threat_support_ahead = None
        carrier_preentry_death_screen_ready = False
        first_carrier_threat_screen_ready = False
        carrier_preentry_death_screen_state = None
        first_carrier_threat_screen_state = None
        carrier_preentry_death_final_distance = None
        first_carrier_threat_final_distance = None
        carrier_preentry_death_alive_allies = None
        first_carrier_threat_alive_allies = None
        carrier_preentry_death_macro_context = None
        first_carrier_threat_macro_context = None
        screen_ready_seen = False
        screening_opportunities = screened_opportunities = 0
        formation_opportunities = formation_ready_opportunities = 0
        entry_formation_opportunities = entry_formation_ready_opportunities = 0
        entry_sync_opportunities = entry_sync_holds = 0
        designated_route_block_ticks = 0
        designated_route_clear_opportunities = designated_route_clear_moves = 0
        screen_guidance_opportunities = screen_guidance_follows = 0
        ultimate_ready_ticks = ultimate_uses = tactical_ultimate_uses = 0
        orb_actions = orb_collections = 0
        orb_approach_opportunities = orb_approach_steps = 0
        orb_reward_maps = {}
        orb_opportunity_names = set()
        preentry_ultimate_uses = 0
        ultimate_uses_by_phase = {phase: 0 for phase in PHASES}
        tactical_ultimate_uses_by_phase = {phase: 0 for phase in PHASES}
        source_positions = {c.name: tuple(c.pos) for c in game.chars if c.team == "A"}

        def support_counts(carrier, carrier_pos, goal, positions):
            allies = [
                c
                for c in game.chars
                if c.team == "A" and c.name != carrier.name and c.is_alive
            ]
            near = sum(
                max(
                    abs(positions[c.name][0] - carrier_pos[0]),
                    abs(positions[c.name][1] - carrier_pos[1]),
                )
                <= 4
                for c in allies
            )
            ahead = 0
            if goal is not None:
                distance = self.distances(goal)
                carrier_distance = int(distance[carrier_pos])
                if carrier_distance >= 0:
                    ahead = sum(
                        0 <= int(distance[positions[c.name]]) < carrier_distance
                        and max(
                            abs(positions[c.name][0] - carrier_pos[0]),
                            abs(positions[c.name][1] - carrier_pos[1]),
                        )
                        <= 8
                        for c in allies
                    )
            return near, ahead

        while not game.round_over:
            first_carrier_threat_this_tick = False
            planted_before = game.is_planted
            holder = next(
                (c for c in game.chars if c.team == "A" and c.is_alive and c.has_spike),
                None,
            )
            holder_pos = tuple(holder.pos) if holder else None
            pre_screen_status = (
                carrier_screening_status(game, holder, game.chars, self.maps)
                if holder is not None
                else None
            )
            pre_alive_allies = (
                sum(
                    c.team == "A" and c.is_alive and c.name != holder.name
                    for c in game.chars
                )
                if holder is not None
                else 0
            )
            screen_ready_seen |= bool(
                pre_screen_status and pre_screen_status.get("screen_ready")
            )
            pre_route_blocking = bool(
                pre_screen_status is not None
                and designated_route_blocking(pre_screen_status, holder)
            )
            designated_route_clear_opportunities += int(pre_route_blocking)
            carry_pending = (
                self.pending.get(("carry", holder.name)) if holder is not None else None
            )
            pre_sync_needed = bool(
                pre_screen_status is not None
                and carrier_entry_sync_needed(
                    pre_screen_status,
                    {"round_timer": game.round_timer},
                    carry_pending.obs if carry_pending is not None else None,
                )
            )
            before = {
                c.name: (tuple(c.pos), c.hp, c.plant_timer, c.kills, c.is_alive)
                for c in game.chars
            }
            orb_assignment = (
                nearby_orb_assignment(
                    game, game.chars, getattr(game, "available_orbs", ()),
                    orb_reward_maps,
                ) if not planted_before else None
            )
            if orb_assignment is not None:
                orb_opportunity_names.add(str(orb_assignment[0].name))
            ultimate_before = {
                c.name: int(getattr(c, "ultimate_points", 0))
                for c in game.chars
                if c.team == "A"
            }
            ultimate_ready_ticks += sum(
                getattr(c, "ultimate_points", 0) >= getattr(c, "ultimate_cost", 1)
                for c in game.chars
                if c.team == "A" and c.is_alive
            )
            engaged = {
                c.name: can_engage(game, c, game.chars)
                for c in game.chars
                if c.team == "A" and c.is_alive
            }
            pre_screen_guidance = None
            if (
                pre_screen_status is not None
                and pre_screen_status.get("designated") is not None
            ):
                screener = pre_screen_status["designated"]
                commitment = carrier_screen_commitment_features(
                    game, screener, game.chars, self.maps
                )
                preferred = np.flatnonzero(commitment[:4] > 0)
                if (
                    commitment[4] > 0
                    and len(preferred)
                    and not engaged.get(screener.name, False)
                ):
                    pre_screen_guidance = (screener.name, int(preferred[0]))
                    screen_guidance_opportunities += 1
            goals = {}
            for phase, name in self.pending:
                if phase == "guard":
                    goal = self.controllers["guard"]._assigned_guard_positions.get(name)
                    if goal is not None:
                        goals[name] = tuple(goal)
            game._build_occupancy_counts()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    for char in game._move_order():
                        if char.is_alive:
                            game.move_character(char)
                    # Movement and independent facing are now both settled.
                    # Measure aim before battle resolution can force a dead
                    # unit's facing or remove its target.
                    current_contacts = combat_contact_aim(game, game.chars)
                    attacker_names = {
                        unit.name for unit in game.chars if unit.team == "A"
                    }
                    combat_start_aim = {
                        shooter: angle
                        for (shooter, target), angle in current_contacts.items()
                        if (shooter, target) not in active_combat_contacts
                        and shooter in attacker_names
                    }
                    active_combat_contacts = set(current_contacts)
                    game.process_battle()
                    game._advance_combo_announcement()
            finally:
                game._clear_occupancy_counts()
            new_plant = game.is_planted and not planted_before
            if (
                holder is not None
                and holder.is_alive
                and pre_sync_needed
                and not engaged.get(holder.name, False)
            ):
                entry_sync_opportunities += 1
                entry_sync_holds += int(tuple(holder.pos) == holder_pos)
            # The normal production wrapper syncs Macro from the same IQ view
            # used by inference. Reward that tick's selected waypoint, without
            # running a separate omniscient Macro decision in the trainer.
            if new_plant:
                plant_tick = game.battle_tick
                if entry_tick is None:
                    entry_tick = game.battle_tick
            target = team_plant_target(game)
            screen_status = None
            screening_names = set()
            screen_ahead_names = set()
            if holder is not None and holder.is_alive:
                screen_status = carrier_screening_status(
                    game, holder, game.chars, self.maps
                )
                screening_names = {
                    c.name for c in screen_status["formation_candidates"]
                }
                screen_ahead_names = {c.name for c in screen_status["ahead"]}
                if 8 < screen_status["final_distance"] <= 24 and screening_names:
                    screening_opportunities += 1
                    screened_opportunities += int(bool(screen_ahead_names))
                if (
                    FORMATION_MIN_FINAL_DISTANCE
                    < screen_status["final_distance"]
                    <= FORMATION_MAX_FINAL_DISTANCE
                    and screening_names
                ):
                    formation_opportunities += 1
                    formation_ready_opportunities += int(screen_status["screen_ready"])
                    designated_route_block_ticks += int(
                        designated_route_blocking(screen_status, holder)
                    )
                    if screen_status["final_distance"] <= 8:
                        entry_formation_opportunities += 1
                        entry_formation_ready_opportunities += int(
                            screen_status["screen_ready"]
                        )
            if holder is not None and target is not None and entry_tick is None:
                if 0 <= self.distances(target)[tuple(holder.pos)] <= 8:
                    entry_tick = game.battle_tick
            if holder is not None and entry_tick is None:
                route_goal = self.action_goals.get(("carry", holder.name), target)
                carrier_pos = before[holder.name][0]
                near, ahead = support_counts(
                    holder,
                    carrier_pos,
                    route_goal,
                    {name: values[0] for name, values in before.items()},
                )
                took_damage = holder.hp < before[holder.name][1]
                if first_carrier_threat_tick is None and (
                    engaged.get(holder.name, False) or took_damage
                ):
                    first_carrier_threat_tick = game.battle_tick
                    first_carrier_threat_this_tick = True
                    first_carrier_threat_support_near = near
                    first_carrier_threat_support_ahead = ahead
                    first_carrier_threat_screen_ready = bool(
                        pre_screen_status and pre_screen_status["screen_ready"]
                    )
                    first_carrier_threat_screen_state = screen_state_at_contact(
                        pre_screen_status, screen_ready_seen, pre_alive_allies
                    )
                    first_carrier_threat_alive_allies = pre_alive_allies
                    first_carrier_threat_macro_context = screen_macro_context(
                        game,
                        holder,
                        {name for name, values in before.items() if values[4]},
                        {name: values[0] for name, values in before.items()},
                    )
                    first_carrier_threat_final_distance = (
                        pre_screen_status.get("final_distance")
                        if pre_screen_status is not None
                        else None
                    )
                if before[holder.name][4] and not holder.is_alive:
                    carrier_preentry_death = True
                    carrier_preentry_death_engaged = bool(
                        engaged.get(holder.name, False)
                    )
                    carrier_preentry_death_support_near = near
                    carrier_preentry_death_support_ahead = ahead
                    carrier_preentry_death_screen_ready = bool(
                        pre_screen_status and pre_screen_status["screen_ready"]
                    )
                    carrier_preentry_death_screen_state = screen_state_at_contact(
                        pre_screen_status, screen_ready_seen, pre_alive_allies
                    )
                    carrier_preentry_death_alive_allies = pre_alive_allies
                    carrier_preentry_death_macro_context = screen_macro_context(
                        game,
                        holder,
                        {name for name, values in before.items() if values[4]},
                        {name: values[0] for name, values in before.items()},
                    )
                    carrier_preentry_death_final_distance = (
                        pre_screen_status.get("final_distance")
                        if pre_screen_status is not None
                        else None
                    )
            screen_ready_seen |= bool(
                screen_status and screen_status.get("screen_ready")
            )
            # New Guard assignments are created on their first real decision.
            for name, goal in self.controllers[
                "guard"
            ]._assigned_guard_positions.items():
                goals.setdefault(name, tuple(goal))
            chars = {c.name: c for c in game.chars}
            next_holder = next(
                (c for c in game.chars if c.team == "A" and c.is_alive and c.has_spike),
                None,
            )
            for key in list(self.pending):
                phase, name = key
                char = chars[name]
                pos, hp, progress, kills, alive = before[name]
                reward = (
                    -0.05
                    - 0.025 * max(0, hp - char.hp)
                    + 0.5 * max(0, char.kills - kills)
                )
                dead = alive and not char.is_alive
                if dead:
                    reward -= 8.0
                if game.round_over and not game.is_planted and game.round_timer <= 0:
                    reward -= TIMEOUT_PENALTY
                # During an actual duel, standing still preserves aim and
                # lets the engine's automatic shooting resolve the fight.
                # Teach this through reward, rather than an inference-side
                # movement lock. Carry/Guard use action 0 for STAY; Escort 4.
                action = self.pending[key].action
                fighting = engaged.get(name, False)
                reward += combat_reward(phase, action, fighting)
                if (
                    orb_assignment is not None
                    and orb_assignment[0].name == name
                    and phase in ("carry", "escort")
                    and orb_assignment[3] > 0
                    and char.is_alive
                    and not fighting
                ):
                    _collector, orb, distances, old_distance = orb_assignment
                    new_pos = tuple(map(int, char.pos))
                    new_distance = (
                        int(distances[new_pos]) if distances is not None
                        else abs(new_pos[0] - orb[0]) + abs(new_pos[1] - orb[1])
                    )
                    orb_approach_opportunities += 1
                    if new_distance >= 0:
                        progress_to_orb = max(-1, min(1, old_distance - new_distance))
                        reward += ORB_APPROACH_STEP_REWARD * progress_to_orb
                        orb_approach_steps += int(progress_to_orb > 0)
                if name in combat_start_aim:
                    angle = combat_start_aim[name]
                    # 0° is a full reward; at the engine's firing-cone edge
                    # it is neutral; outside that cone it becomes a small
                    # penalty.  This is sampled only once per new contact,
                    # preventing a sustained duel from farming reward.
                    alignment = max(
                        0.0, 1.0 - angle / float(SHOOTING_SITE_DIGREE)
                    )
                    aim_reward = (
                        AIM_CONTACT_REWARD * alignment
                        - AIM_CONTACT_MISS_PENALTY * (1.0 - alignment)
                    )
                    reward += aim_reward
                    combat_start_events += 1
                    combat_start_angle_total += angle
                    combat_start_alignment_total += alignment
                    combat_start_reward_total += aim_reward
                ultimate_action = {
                    "carry": runtime.ULTIMATE_ACTION_INDEX,
                    "escort": escort_runtime.ACTION_ULTIMATE,
                    "guard": guard_runtime.ULTIMATE_ACTION_INDEX,
                }[phase]
                if action == ultimate_action:
                    spent_ultimate = (
                        ultimate_before.get(name, 0)
                        >= getattr(char, "ultimate_cost", 1)
                        and getattr(char, "ultimate_points", 0) < ultimate_before[name]
                    )
                    if spent_ultimate:
                        ultimate_uses += 1
                        ultimate_uses_by_phase[phase] += 1
                        preentry_ultimate_uses += int(not planted_before)
                        learned_context = ultimate_context_from_observation(
                            phase, self.pending[key].obs
                        )
                        tactical_window = tactical_ultimate_window(
                            char, learned_context
                        )
                        tactical_ultimate_uses += int(tactical_window)
                        tactical_ultimate_uses_by_phase[phase] += int(tactical_window)
                        # v15 learned that spending points was generally good:
                        # 85 casts, but only six in a tactical window.  Make a
                        # premature cast materially worse than saving the ult.
                        reward += 3.0 if tactical_window else -5.0
                    else:
                        reward -= 2.0
                orb_action = {
                    "carry": runtime.COLLECT_ORB_ACTION_INDEX,
                    "escort": escort_runtime.ACTION_COLLECT_ORB,
                    "guard": guard_runtime.COLLECT_ORB_ACTION_INDEX,
                }[phase]
                if (
                    phase in ("carry", "escort")
                    and self.orb_eligible_pending.get(key, False)
                    and action != orb_action
                ):
                    # Charge the *decision that walked away or waited* rather
                    # than relying only on a remote round-end team outcome.
                    reward -= ORB_ELIGIBLE_MISS_PENALTY
                    progress_index = (
                        runtime.FACING_HEAD_OBS_DIM + 1 if phase == "carry"
                        else escort_runtime.FAKE_WAIT_SUPPORT_OBS_DIM + 1
                    )
                    if self.pending[key].obs[progress_index] > 0:
                        reward -= ORB_INTERRUPTED_COLLECTION_PENALTY
                if action == orb_action:
                    orb_actions += 1
                    # Reward actual collection progress, not merely selecting
                    # the action.  The engine exposes the per-tick flag and
                    # increments ultimate_points only on completion.
                    collecting = bool(getattr(char, "collecting_orb_this_tick", False))
                    gained = getattr(char, "ultimate_points", 0) > ultimate_before.get(name, 0)
                    if gained:
                        orb_collections += 1
                        reward += ORB_COLLECTION_REWARD
                    elif collecting:
                        reward += ORB_COLLECTION_PROGRESS_REWARD
                    else:
                        reward -= ORB_INVALID_ACTION_PENALTY
                if fighting:
                    combat_ticks += 1
                    combat_stops += tuple(char.pos) == pos
                if phase == "carry":
                    route_goal = self.action_goals.get(key, target)
                    if route_goal is not None:
                        distance = self.distances(route_goal)
                        if distance[pos] >= 0 and distance[tuple(char.pos)] >= 0:
                            reward += self.progress.step(
                                name,
                                route_goal,
                                int(distance[pos]),
                                int(distance[tuple(char.pos)]),
                                fighting,
                                required=int(distance[tuple(char.pos)]) > 1
                                and char.plant_timer == 0,
                            )
                            if fighting:
                                reward += 0.20 * max(
                                    0,
                                    int(distance[pos]) - int(distance[tuple(char.pos)]),
                                )
                    carry_ticks += 1
                    entry_event = bool(
                        (game.is_planted and not planted_before)
                        or (entry_tick is not None and entry_tick == game.battle_tick)
                    )
                    no_entry = bool(entry_tick is None and not game.is_planted)
                    if entry_event:
                        reward += CARRY_ROUTE_PRIORITY.carry_site_entry_bonus
                    if (
                        not no_entry
                        and route_goal is not None
                        and distance[tuple(char.pos)] >= 0
                    ):
                        reward += CARRY_ROUTE_PRIORITY.plant_progress_bonus * max(
                            0.0,
                            1.0
                            - distance[tuple(char.pos)] / max(1, sum(game.grid.shape)),
                        )
                    if (
                        not no_entry
                        and game.planted_pos is not None
                        and getattr(char, "has_spike", False)
                        and tuple(char.pos) == tuple(game.planted_pos)
                    ):
                        reward += CARRY_ROUTE_PRIORITY.site_hold_bonus
                    if (
                        no_entry
                        and not fighting
                        and tuple(char.pos) == pos
                        and char.plant_timer == 0
                    ):
                        reward -= CARRY_ROUTE_PRIORITY.carry_no_entry_penalty
                    if (
                        game.round_over
                        and not game.is_planted
                        and game.round_timer <= 0
                    ):
                        reward -= CARRY_ROUTE_PRIORITY.timeout_penalty
                    if (
                        game.is_planted
                        and game.planted_pos is not None
                        and tuple(char.pos) == tuple(game.planted_pos)
                    ):
                        reward += CARRY_ROUTE_PRIORITY.site_hold_bonus
                    recent = carry_positions.setdefault(name, deque(maxlen=3))
                    if not recent:
                        recent.append(pos)
                    recent.append(tuple(char.pos))
                    reversed_move = (
                        len(recent) == 3
                        and recent[0] == recent[2]
                        and recent[0] != recent[1]
                    )
                    carry_reversals += int(reversed_move)
                    quiet_stall = (
                        tuple(char.pos) == pos
                        and not fighting
                        and char.plant_timer == 0
                        and not new_plant
                    )
                    carry_quiet_stalls += int(quiet_stall)
                    if quiet_stall and not pre_sync_needed:
                        reward -= CARRY_QUIET_STALL_PENALTY
                        if game.round_timer <= LATE_ROUND_THRESHOLD:
                            reward -= LATE_QUIET_STALL_EXTRA_PENALTY
                    if reversed_move and not fighting:
                        reward -= 0.15
                    spawn_distance = self.distances(source_positions[name])[
                        tuple(char.pos)
                    ]
                    carry_spawn_ticks += int(0 <= spawn_distance <= 6)
                    if (
                        0 <= spawn_distance <= 6
                        and game.battle_tick > 20
                        and not fighting
                    ):
                        # Waiting briefly for information is valid. Spending
                        # the attack clock at spawn without a duel is not.
                        reward -= 0.15
                    if (
                        name == getattr(holder, "name", None)
                        and pre_sync_needed
                        and not fighting
                    ):
                        if tuple(char.pos) == pos:
                            reward -= 0.30
                        elif (
                            route_goal is not None
                            and distance[tuple(char.pos)] < distance[pos]
                        ):
                            reward += 0.30
                    reward += 0.8 * max(0, char.plant_timer - progress)
                    if (
                        progress > 0
                        and char.plant_timer < progress
                        and not game.is_planted
                    ):
                        reward -= 0.6 * progress
                elif phase == "escort" and holder_pos is not None:
                    route_goal = self.action_goals.get(key)
                    fake_wait_bodyguard = bool(
                        holder is not None
                        and holder.is_alive
                        and char.is_alive
                        and navigation_intent(game, holder)[2] == "FAKE_WAIT"
                        and navigation_intent(game, char)[2] == "FAKE_WAIT"
                    )
                    if fake_wait_bodyguard and not fighting:
                        # The fake sellers deliberately leave the Spike behind.
                        # Teach a waiting teammate to *learn* to guard the holder
                        # before contact, without overriding production actions.
                        carrier_distance = self.distances(holder_pos)
                        before_support = int(carrier_distance[pos])
                        after_support = int(carrier_distance[tuple(char.pos)])
                        if before_support >= 0 and after_support >= 0:
                            reward += 0.45 * max(0, before_support - after_support)
                            reward -= 0.65 * max(0, after_support - before_support)
                            if (
                                before_support > FAKE_WAIT_SUPPORT_RADIUS
                                and after_support <= FAKE_WAIT_SUPPORT_RADIUS
                            ):
                                reward += 1.0
                            elif (
                                before_support <= FAKE_WAIT_SUPPORT_RADIUS
                                and after_support > FAKE_WAIT_SUPPORT_RADIUS
                            ):
                                reward -= 1.5
                    pre_designated = (
                        pre_screen_status is not None
                        and pre_screen_status["designated"] is not None
                        and name == pre_screen_status["designated"].name
                    )
                    designated = (
                        screen_status is not None
                        and screen_status["designated"] is not None
                        and name == screen_status["designated"].name
                    )
                    formation_active = bool(
                        designated
                        and FORMATION_MIN_FINAL_DISTANCE
                        < screen_status["final_distance"]
                        <= SCREEN_COMMITMENT_MAX_FINAL_DISTANCE
                    )
                    route_blocking = bool(
                        designated and designated_route_blocking(screen_status, holder)
                    )
                    if (
                        route_goal is not None
                        and not formation_active
                        and not fake_wait_bodyguard
                    ):
                        distance = self.distances(route_goal)
                        reward += self.progress.step(
                            name,
                            route_goal,
                            int(distance[pos]),
                            int(distance[tuple(char.pos)]),
                            fighting,
                            required=int(distance[tuple(char.pos)]) > 1,
                        )
                    if holder is not None and not fighting and holder.is_alive:
                        carry_goal = self.action_goals.get(("carry", holder.name))
                        if carry_goal is not None:
                            carry_distance = self.distances(carry_goal)
                            holder_cell = tuple(holder.pos)
                            escort_cell = tuple(char.pos)
                            if (
                                max(
                                    abs(escort_cell[0] - holder_cell[0]),
                                    abs(escort_cell[1] - holder_cell[1]),
                                )
                                == 1
                                and 0
                                <= carry_distance[escort_cell]
                                < carry_distance[holder_cell]
                            ):
                                reward -= 0.15  # Teach escorts to leave the carrier's next route cells clear.
                    if holder is not None and not holder.is_alive:
                        reward -= 8.0 if pre_designated else 3.0
                    if route_blocking and not fighting:
                        reward -= 1.0
                    if pre_designated and pre_route_blocking and not fighting:
                        moved = tuple(char.pos) != pos
                        post_blocking = bool(
                            screen_status is not None
                            and designated_route_blocking(screen_status, holder)
                        )
                        if (
                            holder is not None
                            and holder.is_alive
                            and moved
                            and not post_blocking
                        ):
                            reward += 0.75
                            designated_route_clear_moves += 1
                        elif holder is not None and holder.is_alive and not moved:
                            reward -= 0.75
                    if pre_designated and first_carrier_threat_this_tick:
                        reward += 3.0 if pre_screen_status["screen_ready"] else -2.0
                    if (
                        pre_screen_guidance is not None
                        and name == pre_screen_guidance[0]
                    ):
                        preferred_action = pre_screen_guidance[1]
                        if action == preferred_action:
                            reward += 1.25
                            screen_guidance_follows += 1
                        elif action not in (
                            escort_runtime.ACTION_ABILITY,
                            escort_runtime.ACTION_ULTIMATE,
                        ):
                            reward -= 0.75
                    pre_formation_active = bool(
                        pre_designated
                        and FORMATION_MIN_FINAL_DISTANCE
                        < pre_screen_status["final_distance"]
                        <= SCREEN_COMMITMENT_MAX_FINAL_DISTANCE
                    )
                    if pre_formation_active and not pre_route_blocking and not fighting:
                        # Credit the selected move against the target visible
                        # when it was chosen.  v15 measured against the target
                        # after Carrier also moved, which erased correct steps.
                        formation_distance = pre_screen_status[
                            "formation_target_distance"
                        ]
                        before_distance = int(formation_distance[pos])
                        after_distance = int(formation_distance[tuple(char.pos)])
                        if before_distance >= 0 and after_distance >= 0:
                            reward += 0.60 * max(0, before_distance - after_distance)
                            reward -= 0.40 * max(0, after_distance - before_distance)
                        post_ready = bool(
                            screen_status and screen_status["screen_ready"]
                        )
                        reward += 0.40 if post_ready else -0.15
                        if pre_screen_status["final_distance"] <= 16:
                            reward += 0.60 if post_ready else -0.25
                elif phase == "guard" and name in goals:
                    goal = goals[name]
                    distance = self.distances(goal)
                    if distance[pos] >= 0 and distance[tuple(char.pos)] >= 0:
                        reward += self.progress.step(
                            name,
                            goal,
                            int(distance[pos]),
                            int(distance[tuple(char.pos)]),
                            fighting,
                            required=tuple(char.pos) != goal,
                        )
                    guard_ticks += 1
                    if tuple(char.pos) == goal and char.is_alive:
                        guard_arrivals += 1
                        if name not in self.arrived:
                            reward += 1.0
                            self.arrived.add(name)
                if new_plant and phase in ("carry", "escort"):
                    # A small ordinary-plant reward and a materially larger
                    # registered-site reward prevent rushing any safe cell.
                    registered = tuple(game.planted_pos) in REGISTERED_PLANT_CELLS
                    reward += 3.0 + (17.0 if registered else 0.0)
                self.add_reward(key, reward)
                next_phase = phase_of(char, game.is_planted, next_holder)
                if dead or game.round_over or next_phase != phase:
                    self.flush(key, self.pending.pop(key))
        # Remember opportunities at decision time. At round end all attackers
        # may be dead or the orb may be gone, but those later facts must not
        # erase a missed collectable chance earlier in the round.
        orb_opportunity_names.update(self.orb_eligible_actor_names)
        orb_round_missed = bool(
            orb_collections == 0
            and orb_opportunity_names
        )
        orb_round_miss_penalty = (
            ORB_ROUND_MISS_PENALTY
            if orb_round_missed and self.collect_demonstrations
            else 0.0
        )
        # Give each participant's last action the actual round outcome, including
        # Carry/Escort whose phases ended at planting, and actors who died earlier.
        won = game.attacker_wins > initial_wins
        orb_penalty_targets = {}
        for phase in ("carry", "escort"):
            for index, (name, _row) in enumerate(self.transitions[phase]):
                if name in orb_opportunity_names:
                    orb_penalty_targets[name] = (phase, index)
        per_actor_orb_penalty = orb_round_miss_penalty / max(1, len(orb_penalty_targets))
        for phase, rows in self.transitions.items():
            last = {name: i for i, (name, _) in enumerate(rows)}
            for index in last.values():
                name, row = rows[index]
                obs, action, reward, next_obs, mask, terminal, ticks = row
                missed_orb_share = (
                    per_actor_orb_penalty
                    if orb_penalty_targets.get(name) == (phase, index) else 0.0
                )
                rows[index] = (
                    name,
                    (
                        obs,
                        action,
                        reward + (20.0 if won else -20.0) - missed_orb_share,
                        next_obs,
                        mask,
                        terminal,
                        ticks,
                    ),
                )
        no_entry = entry_tick is None and not game.is_planted
        team_wiped = not any(c.team == "A" and c.is_alive for c in game.chars)
        return {
            "attacker_win": won,
            "planted": bool(game.is_planted),
            "registered": bool(
                game.is_planted and tuple(game.planted_pos) in REGISTERED_PLANT_CELLS
            ),
            "ticks": game.battle_tick,
            "guard_ticks": guard_ticks,
            "guard_position_ticks": guard_arrivals,
            "opponent_stats": list(stats),
            "combat_ticks": combat_ticks,
            "combat_stop_ticks": combat_stops,
            "combat_start_events": combat_start_events,
            "combat_start_angle_total": combat_start_angle_total,
            "combat_start_alignment_total": combat_start_alignment_total,
            "combat_start_reward_total": combat_start_reward_total,
            "carry_ticks": carry_ticks,
            "carry_spawn_ticks": carry_spawn_ticks,
            "carry_reversals": carry_reversals,
            "carry_quiet_stalls": carry_quiet_stalls,
            "carry_entry_tick": entry_tick,
            "plant_tick": plant_tick,
            "timed_out": bool(not game.is_planted and game.round_timer <= 0),
            "carrier_preentry_death": carrier_preentry_death,
            "carrier_preentry_death_engaged": carrier_preentry_death_engaged,
            "carrier_preentry_death_support_near": carrier_preentry_death_support_near,
            "carrier_preentry_death_support_ahead": carrier_preentry_death_support_ahead,
            "first_carrier_threat_tick": first_carrier_threat_tick,
            "first_carrier_threat_support_near": first_carrier_threat_support_near,
            "first_carrier_threat_support_ahead": first_carrier_threat_support_ahead,
            "carrier_preentry_death_screen_ready": carrier_preentry_death_screen_ready,
            "first_carrier_threat_screen_ready": first_carrier_threat_screen_ready,
            "carrier_preentry_death_screen_state": carrier_preentry_death_screen_state,
            "first_carrier_threat_screen_state": first_carrier_threat_screen_state,
            "carrier_preentry_death_final_distance": carrier_preentry_death_final_distance,
            "first_carrier_threat_final_distance": first_carrier_threat_final_distance,
            "carrier_preentry_death_alive_allies": carrier_preentry_death_alive_allies,
            "first_carrier_threat_alive_allies": first_carrier_threat_alive_allies,
            "carrier_preentry_death_macro_context": carrier_preentry_death_macro_context,
            "first_carrier_threat_macro_context": first_carrier_threat_macro_context,
            "screened_approach_ticks": screening_opportunities,
            "screened_approach_success_ticks": screened_opportunities,
            "formation_approach_ticks": formation_opportunities,
            "formation_ready_ticks": formation_ready_opportunities,
            "entry_formation_ticks": entry_formation_opportunities,
            "entry_formation_ready_ticks": entry_formation_ready_opportunities,
            "entry_sync_ticks": entry_sync_opportunities,
            "entry_sync_hold_ticks": entry_sync_holds,
            "designated_route_block_ticks": designated_route_block_ticks,
            "designated_route_clear_opportunities": designated_route_clear_opportunities,
            "designated_route_clear_moves": designated_route_clear_moves,
            "screen_guidance_opportunities": screen_guidance_opportunities,
            "screen_guidance_follows": screen_guidance_follows,
            "ultimate_ready_ticks": ultimate_ready_ticks,
            "orb_actions": orb_actions,
            "orb_collections": orb_collections,
            "orb_approach_opportunities": orb_approach_opportunities,
            "orb_approach_steps": orb_approach_steps,
            "orb_eligible_decisions": self.orb_eligible_decisions,
            "orb_eligible_misses": self.orb_eligible_misses,
            "orb_greedy_choices": self.orb_greedy_choices,
            "orb_q_margin_mean": (
                self.orb_q_margin_sum / self.orb_q_margin_count
                if self.orb_q_margin_count else None
            ),
            "orb_q_margin_sum": self.orb_q_margin_sum,
            "orb_q_margin_count": self.orb_q_margin_count,
            "orb_round_missed": orb_round_missed,
            "orb_round_miss_penalty": orb_round_miss_penalty,
            "orb_visible_ticks": self.orb_visible_ticks,
            "orb_teacher_actions": self.orb_teacher_actions,
            "orb_teacher_forced_actions": self.orb_teacher_forced_actions,
            "orb_teacher_round_enabled": self.orb_teacher_round_enabled,
            "ultimate_uses": ultimate_uses,
            "tactical_ultimate_uses": tactical_ultimate_uses,
            "preentry_ultimate_uses": preentry_ultimate_uses,
            "ultimate_uses_by_phase": ultimate_uses_by_phase,
            "tactical_ultimate_uses_by_phase": tactical_ultimate_uses_by_phase,
            "facing_decisions_by_phase": dict(self.facing_decisions),
            "facing_teacher_matches_by_phase": dict(self.facing_teacher_matches),
            "facing_confident_decisions_by_phase": dict(
                self.facing_confident_decisions
            ),
            "facing_confident_matches_by_phase": dict(self.facing_confident_matches),
            "facing_confidence_mass_by_phase": dict(self.facing_confidence_mass),
            "facing_weighted_matches_by_phase": dict(self.facing_weighted_matches),
            "facing_context_counts_by_phase": {
                phase: dict(values)
                for phase, values in self.facing_context_counts.items()
            },
            "facing_context_matches_by_phase": {
                phase: dict(values)
                for phase, values in self.facing_context_matches.items()
            },
            "no_entry_carrier_death": bool(no_entry and carrier_preentry_death),
            "no_entry_team_wipe": bool(no_entry and team_wiped),
            "no_entry_other": bool(
                no_entry
                and not carrier_preentry_death
                and not team_wiped
                and game.round_timer > 0
            ),
            "postplant_win": bool(game.is_planted and won),
        }


def evaluate(session, episodes, seed, final):
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        rows = [session.play(seed + i, final, training=False) for i in range(episodes)]
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    death_rows = [r for r in rows if r.get("carrier_preentry_death")]
    threat_rows = [r for r in rows if r.get("first_carrier_threat_tick") is not None]
    no_candidate_threat_rows = [
        r
        for r in threat_rows
        if r.get("first_carrier_threat_screen_state") == "no_same_route_candidate"
    ]
    no_candidate_death_rows = [
        r
        for r in death_rows
        if r.get("carrier_preentry_death_screen_state") == "no_same_route_candidate"
    ]
    fake_wait_no_candidate_threat_rows = [
        r
        for r in no_candidate_threat_rows
        if (r.get("first_carrier_threat_macro_context") or {}).get("carrier_role")
        == "FAKE_WAIT"
    ]
    fake_wait_threat_distances = [
        (r.get("first_carrier_threat_macro_context") or {}).get(
            "nearest_same_role_distance"
        )
        for r in threat_rows
        if (r.get("first_carrier_threat_macro_context") or {}).get("carrier_role")
        == "FAKE_WAIT"
    ]
    fake_wait_threat_close_count = sum(
        distance is not None and distance <= FAKE_WAIT_SUPPORT_RADIUS
        for distance in fake_wait_threat_distances
    )

    def screen_state_counts(event_rows, key):
        states = [r.get(key) for r in event_rows]
        return {state: states.count(state) for state in sorted(set(states) - {None})}

    def mean_contact_distance(event_rows, key):
        distances = [r[key] for r in event_rows if r.get(key) is not None]
        return float(np.mean(distances)) if distances else None

    facing_context_counts = {
        phase: {
            source: sum(
                r.get("facing_context_counts_by_phase", {})
                .get(phase, {})
                .get(source, 0)
                for r in rows
            )
            for source in sorted(
                {
                    source
                    for r in rows
                    for source in r.get("facing_context_counts_by_phase", {}).get(
                        phase, {}
                    )
                }
            )
        }
        for phase in FACING_PHASES
    }
    facing_context_matches = {
        phase: {
            source: sum(
                r.get("facing_context_matches_by_phase", {})
                .get(phase, {})
                .get(source, 0)
                for r in rows
            )
            for source in facing_context_counts[phase]
        }
        for phase in FACING_PHASES
    }
    return {
        "episodes": episodes,
        "seed": seed,
        "opponent_stats": list(final),
        "round_win_rate": sum(r["attacker_win"] for r in rows) / episodes,
        "plant_rate": sum(r["planted"] for r in rows) / episodes,
        "registered_plant_rate": sum(r["registered"] for r in rows) / episodes,
        "guard_position_tick_rate": sum(r["guard_position_ticks"] for r in rows)
        / max(1, sum(r["guard_ticks"] for r in rows)),
        "mean_ticks": sum(r["ticks"] for r in rows) / episodes,
        "carry_no_entry_rate": sum(not entered_site(r) for r in rows) / episodes,
        "carry_spawn_tick_rate": sum(r.get("carry_spawn_ticks", 0) for r in rows)
        / max(1, sum(r.get("carry_ticks", 0) for r in rows)),
        "carry_reversal_tick_rate": sum(r.get("carry_reversals", 0) for r in rows)
        / max(1, sum(r.get("carry_ticks", 0) for r in rows)),
        "carry_quiet_stall_tick_rate": sum(r.get("carry_quiet_stalls", 0) for r in rows)
        / max(1, sum(r.get("carry_ticks", 0) for r in rows)),
        "timeout_rate": sum(r.get("timed_out", False) for r in rows) / episodes,
        "carrier_preentry_death_rate": sum(
            r.get("carrier_preentry_death", False) for r in rows
        )
        / episodes,
        "no_entry_carrier_death_rate": sum(
            r.get("no_entry_carrier_death", False) for r in rows
        )
        / episodes,
        "no_entry_team_wipe_rate": sum(r.get("no_entry_team_wipe", False) for r in rows)
        / episodes,
        "no_entry_other_rate": sum(r.get("no_entry_other", False) for r in rows)
        / episodes,
        "mean_death_support_near": (
            float(
                np.mean([r["carrier_preentry_death_support_near"] for r in death_rows])
            )
            if death_rows
            else 0.0
        ),
        "mean_death_support_ahead": (
            float(
                np.mean([r["carrier_preentry_death_support_ahead"] for r in death_rows])
            )
            if death_rows
            else 0.0
        ),
        "mean_first_threat_support_near": (
            float(
                np.mean([r["first_carrier_threat_support_near"] for r in threat_rows])
            )
            if threat_rows
            else 0.0
        ),
        "mean_first_threat_support_ahead": (
            float(
                np.mean([r["first_carrier_threat_support_ahead"] for r in threat_rows])
            )
            if threat_rows
            else 0.0
        ),
        "death_screen_ready_rate": sum(
            r.get("carrier_preentry_death_screen_ready", False) for r in death_rows
        )
        / max(1, len(death_rows)),
        "first_threat_screen_ready_rate": sum(
            r.get("first_carrier_threat_screen_ready", False) for r in threat_rows
        )
        / max(1, len(threat_rows)),
        "first_threat_screen_state_counts": screen_state_counts(
            threat_rows, "first_carrier_threat_screen_state"
        ),
        "death_screen_state_counts": screen_state_counts(
            death_rows, "carrier_preentry_death_screen_state"
        ),
        "no_candidate_threat_strategy_counts": dict(
            Counter(
                (r.get("first_carrier_threat_macro_context") or {}).get("strategy", "")
                for r in no_candidate_threat_rows
            )
        ),
        "no_candidate_threat_ally_role_counts": dict(
            Counter(
                role
                for r in no_candidate_threat_rows
                for role in (r.get("first_carrier_threat_macro_context") or {}).get(
                    "alive_ally_roles", ()
                )
            )
        ),
        "no_candidate_death_strategy_counts": dict(
            Counter(
                (r.get("carrier_preentry_death_macro_context") or {}).get(
                    "strategy", ""
                )
                for r in no_candidate_death_rows
            )
        ),
        "no_candidate_death_carrier_role_counts": dict(
            Counter(
                (r.get("carrier_preentry_death_macro_context") or {}).get(
                    "carrier_role", ""
                )
                for r in no_candidate_death_rows
            )
        ),
        "no_candidate_threat_carrier_role_counts": dict(
            Counter(
                (r.get("first_carrier_threat_macro_context") or {}).get(
                    "carrier_role", ""
                )
                for r in no_candidate_threat_rows
            )
        ),
        "fake_wait_threat_same_role_distance_counts": dict(
            Counter(
                str(
                    (r.get("first_carrier_threat_macro_context") or {}).get(
                        "nearest_same_role_distance"
                    )
                )
                for r in fake_wait_no_candidate_threat_rows
            )
        ),
        "fake_wait_threat_count": len(fake_wait_threat_distances),
        "fake_wait_threat_close_count": fake_wait_threat_close_count,
        "fake_wait_threat_close_rate": fake_wait_threat_close_count / max(
            1, len(fake_wait_threat_distances)
        ),
        "first_threat_final_distance_mean": mean_contact_distance(
            threat_rows, "first_carrier_threat_final_distance"
        ),
        "death_final_distance_mean": mean_contact_distance(
            death_rows, "carrier_preentry_death_final_distance"
        ),
        "screened_approach_rate": sum(
            r.get("screened_approach_success_ticks", 0) for r in rows
        )
        / max(1, sum(r.get("screened_approach_ticks", 0) for r in rows)),
        "screened_approach_ticks": sum(
            r.get("screened_approach_ticks", 0) for r in rows
        ),
        "screened_approach_success_ticks": sum(
            r.get("screened_approach_success_ticks", 0) for r in rows
        ),
        "formation_ready_approach_rate": sum(
            r.get("formation_ready_ticks", 0) for r in rows
        )
        / max(1, sum(r.get("formation_approach_ticks", 0) for r in rows)),
        "formation_approach_ticks": sum(
            r.get("formation_approach_ticks", 0) for r in rows
        ),
        "formation_ready_ticks": sum(r.get("formation_ready_ticks", 0) for r in rows),
        "entry_formation_ready_rate": sum(
            r.get("entry_formation_ready_ticks", 0) for r in rows
        )
        / max(1, sum(r.get("entry_formation_ticks", 0) for r in rows)),
        "entry_formation_ticks": sum(r.get("entry_formation_ticks", 0) for r in rows),
        "entry_formation_ready_ticks": sum(
            r.get("entry_formation_ready_ticks", 0) for r in rows
        ),
        "entry_sync_hold_rate": sum(r.get("entry_sync_hold_ticks", 0) for r in rows)
        / max(1, sum(r.get("entry_sync_ticks", 0) for r in rows)),
        "entry_sync_ticks": sum(r.get("entry_sync_ticks", 0) for r in rows),
        "entry_sync_hold_ticks": sum(r.get("entry_sync_hold_ticks", 0) for r in rows),
        "designated_route_block_rate": sum(
            r.get("designated_route_block_ticks", 0) for r in rows
        )
        / max(1, sum(r.get("formation_approach_ticks", 0) for r in rows)),
        "designated_route_block_ticks": sum(
            r.get("designated_route_block_ticks", 0) for r in rows
        ),
        "designated_route_clear_rate": sum(
            r.get("designated_route_clear_moves", 0) for r in rows
        )
        / max(1, sum(r.get("designated_route_clear_opportunities", 0) for r in rows)),
        "designated_route_clear_opportunities": sum(
            r.get("designated_route_clear_opportunities", 0) for r in rows
        ),
        "designated_route_clear_moves": sum(
            r.get("designated_route_clear_moves", 0) for r in rows
        ),
        "screen_guidance_follow_rate": sum(
            r.get("screen_guidance_follows", 0) for r in rows
        )
        / max(1, sum(r.get("screen_guidance_opportunities", 0) for r in rows)),
        "screen_guidance_opportunities": sum(
            r.get("screen_guidance_opportunities", 0) for r in rows
        ),
        "screen_guidance_follows": sum(
            r.get("screen_guidance_follows", 0) for r in rows
        ),
        "ultimate_use_rate": sum(r.get("ultimate_uses", 0) for r in rows)
        / max(1, sum(r.get("ultimate_ready_ticks", 0) for r in rows)),
        "ultimate_ready_ticks": sum(r.get("ultimate_ready_ticks", 0) for r in rows),
        "ultimate_uses": sum(r.get("ultimate_uses", 0) for r in rows),
        "orb_actions": sum(r.get("orb_actions", 0) for r in rows),
        "orb_collections": sum(r.get("orb_collections", 0) for r in rows),
        "orb_approach_opportunities": sum(
            r.get("orb_approach_opportunities", 0) for r in rows
        ),
        "orb_approach_steps": sum(r.get("orb_approach_steps", 0) for r in rows),
        "orb_eligible_decisions": sum(
            r.get("orb_eligible_decisions", 0) for r in rows
        ),
        "orb_eligible_misses": sum(r.get("orb_eligible_misses", 0) for r in rows),
        "orb_greedy_choices": sum(r.get("orb_greedy_choices", 0) for r in rows),
        "orb_q_margin_sum": sum(r.get("orb_q_margin_sum", 0.0) for r in rows),
        "orb_q_margin_count": sum(r.get("orb_q_margin_count", 0) for r in rows),
        "orb_q_margin_mean": sum(r.get("orb_q_margin_sum", 0.0) for r in rows)
        / max(1, sum(r.get("orb_q_margin_count", 0) for r in rows)),
        "orb_round_missed": sum(r.get("orb_round_missed", False) for r in rows),
        "orb_round_miss_penalty": sum(
            r.get("orb_round_miss_penalty", 0.0) for r in rows
        ),
        "orb_visible_ticks": sum(r.get("orb_visible_ticks", 0) for r in rows),
        "orb_teacher_actions": sum(r.get("orb_teacher_actions", 0) for r in rows),
        "orb_teacher_forced_actions": sum(
            r.get("orb_teacher_forced_actions", 0) for r in rows
        ),
        "tactical_ultimate_rate": sum(r.get("tactical_ultimate_uses", 0) for r in rows)
        / max(1, sum(r.get("ultimate_uses", 0) for r in rows)),
        "tactical_ultimate_uses": sum(r.get("tactical_ultimate_uses", 0) for r in rows),
        "preentry_ultimate_uses": sum(r.get("preentry_ultimate_uses", 0) for r in rows),
        "ultimate_uses_by_phase": {
            phase: sum(r.get("ultimate_uses_by_phase", {}).get(phase, 0) for r in rows)
            for phase in PHASES
        },
        "tactical_ultimate_uses_by_phase": {
            phase: sum(
                r.get("tactical_ultimate_uses_by_phase", {}).get(phase, 0) for r in rows
            )
            for phase in PHASES
        },
        "facing_decisions_by_phase": {
            phase: sum(
                r.get("facing_decisions_by_phase", {}).get(phase, 0) for r in rows
            )
            for phase in FACING_PHASES
        },
        "facing_teacher_matches_by_phase": {
            phase: sum(
                r.get("facing_teacher_matches_by_phase", {}).get(phase, 0) for r in rows
            )
            for phase in FACING_PHASES
        },
        "facing_teacher_match_rate_by_phase": {
            phase: sum(
                r.get("facing_teacher_matches_by_phase", {}).get(phase, 0) for r in rows
            )
            / max(
                1,
                sum(r.get("facing_decisions_by_phase", {}).get(phase, 0) for r in rows),
            )
            for phase in FACING_PHASES
        },
        "facing_confident_decisions_by_phase": {
            phase: sum(
                r.get("facing_confident_decisions_by_phase", {}).get(phase, 0)
                for r in rows
            )
            for phase in FACING_PHASES
        },
        "facing_confident_matches_by_phase": {
            phase: sum(
                r.get("facing_confident_matches_by_phase", {}).get(phase, 0)
                for r in rows
            )
            for phase in FACING_PHASES
        },
        "facing_confident_match_rate_by_phase": {
            phase: sum(
                r.get("facing_confident_matches_by_phase", {}).get(phase, 0)
                for r in rows
            )
            / max(
                1,
                sum(
                    r.get("facing_confident_decisions_by_phase", {}).get(phase, 0)
                    for r in rows
                ),
            )
            for phase in FACING_PHASES
        },
        "facing_confidence_mass_by_phase": {
            phase: sum(
                r.get("facing_confidence_mass_by_phase", {}).get(phase, 0.0)
                for r in rows
            )
            for phase in FACING_PHASES
        },
        "facing_weighted_matches_by_phase": {
            phase: sum(
                r.get("facing_weighted_matches_by_phase", {}).get(phase, 0.0)
                for r in rows
            )
            for phase in FACING_PHASES
        },
        "facing_weighted_match_rate_by_phase": {
            phase: sum(
                r.get("facing_weighted_matches_by_phase", {}).get(phase, 0.0)
                for r in rows
            )
            / max(
                1e-6,
                sum(
                    r.get("facing_confidence_mass_by_phase", {}).get(phase, 0.0)
                    for r in rows
                ),
            )
            for phase in FACING_PHASES
        },
        "facing_context_counts_by_phase": facing_context_counts,
        "facing_context_matches_by_phase": facing_context_matches,
        "facing_context_match_rate_by_phase": {
            phase: {
                source: facing_context_matches[phase][source] / max(1, count)
                for source, count in facing_context_counts[phase].items()
            }
            for phase in FACING_PHASES
        },
        "combat_start_events": sum(r.get("combat_start_events", 0) for r in rows),
        "combat_start_mean_aim_angle": sum(
            r.get("combat_start_angle_total", 0.0) for r in rows
        ) / max(1, sum(r.get("combat_start_events", 0) for r in rows)),
        "combat_start_aim_alignment": sum(
            r.get("combat_start_alignment_total", 0.0) for r in rows
        ) / max(1, sum(r.get("combat_start_events", 0) for r in rows)),
        "combat_start_aim_reward": sum(
            r.get("combat_start_reward_total", 0.0) for r in rows
        ),
        "combat_stop_rate": sum(r.get("combat_stop_ticks", 0) for r in rows)
        / max(1, sum(r.get("combat_ticks", 0) for r in rows)),
        "postplant_win_rate": sum(r.get("postplant_win", False) for r in rows)
        / max(1, sum(r["planted"] for r in rows)),
    }


def evaluate_multi(session, episodes, seeds, final):
    """Evaluate independent seed blocks and report both mean and worst block."""
    blocks = [evaluate(session, episodes, int(seed), final) for seed in seeds]

    def total_screen_states(key):
        labels = sorted({label for block in blocks for label in block[key]})
        return {
            label: sum(block[key].get(label, 0) for block in blocks) for label in labels
        }

    plants = sum(b["episodes"] * b["plant_rate"] for b in blocks)
    postplant_wins = sum(
        b["episodes"] * b["plant_rate"] * b["postplant_win_rate"] for b in blocks
    )
    facing_context_counts = {
        phase: {
            source: sum(
                block.get("facing_context_counts_by_phase", {})
                .get(phase, {})
                .get(source, 0)
                for block in blocks
            )
            for source in sorted(
                {
                    source
                    for block in blocks
                    for source in block.get("facing_context_counts_by_phase", {}).get(
                        phase, {}
                    )
                }
            )
        }
        for phase in FACING_PHASES
    }
    facing_context_matches = {
        phase: {
            source: sum(
                block.get("facing_context_matches_by_phase", {})
                .get(phase, {})
                .get(source, 0)
                for block in blocks
            )
            for source in facing_context_counts[phase]
        }
        for phase in FACING_PHASES
    }
    return {
        "episodes": episodes * len(blocks),
        "seeds": [int(s) for s in seeds],
        "opponent_stats": list(final),
        "round_win_rate": float(np.mean([b["round_win_rate"] for b in blocks])),
        "worst_round_win_rate": float(min(b["round_win_rate"] for b in blocks)),
        "plant_rate": float(np.mean([b["plant_rate"] for b in blocks])),
        "registered_plant_rate": float(
            np.mean([b["registered_plant_rate"] for b in blocks])
        ),
        "worst_registered_plant_rate": float(
            min(b["registered_plant_rate"] for b in blocks)
        ),
        "worst_carry_no_entry_rate": float(
            max(b["carry_no_entry_rate"] for b in blocks)
        ),
        "worst_timeout_rate": float(max(b["timeout_rate"] for b in blocks)),
        "worst_carrier_preentry_death_rate": float(
            max(b["carrier_preentry_death_rate"] for b in blocks)
        ),
        "worst_no_entry_carrier_death_rate": float(
            max(b["no_entry_carrier_death_rate"] for b in blocks)
        ),
        "guard_position_tick_rate": float(
            np.mean([b["guard_position_tick_rate"] for b in blocks])
        ),
        "mean_ticks": float(np.mean([b["mean_ticks"] for b in blocks])),
        "per_seed": blocks,
        "first_threat_screen_state_counts": total_screen_states(
            "first_threat_screen_state_counts"
        ),
        "death_screen_state_counts": total_screen_states("death_screen_state_counts"),
        "no_candidate_threat_strategy_counts": total_screen_states(
            "no_candidate_threat_strategy_counts"
        ),
        "no_candidate_threat_ally_role_counts": total_screen_states(
            "no_candidate_threat_ally_role_counts"
        ),
        "no_candidate_death_strategy_counts": total_screen_states(
            "no_candidate_death_strategy_counts"
        ),
        "no_candidate_death_carrier_role_counts": total_screen_states(
            "no_candidate_death_carrier_role_counts"
        ),
        "no_candidate_threat_carrier_role_counts": total_screen_states(
            "no_candidate_threat_carrier_role_counts"
        ),
        "fake_wait_threat_same_role_distance_counts": total_screen_states(
            "fake_wait_threat_same_role_distance_counts"
        ),
        "fake_wait_threat_count": sum(
            b.get("fake_wait_threat_count", 0) for b in blocks
        ),
        "fake_wait_threat_close_count": sum(
            b.get("fake_wait_threat_close_count", 0) for b in blocks
        ),
        "fake_wait_threat_close_rate": sum(
            b.get("fake_wait_threat_close_count", 0) for b in blocks
        ) / max(1, sum(b.get("fake_wait_threat_count", 0) for b in blocks)),
        "postplant_win_rate": float(postplant_wins / max(1, plants)),
        "screened_approach_rate": sum(
            b["screened_approach_success_ticks"] for b in blocks
        )
        / max(1, sum(b["screened_approach_ticks"] for b in blocks)),
        "formation_ready_approach_rate": sum(b["formation_ready_ticks"] for b in blocks)
        / max(1, sum(b["formation_approach_ticks"] for b in blocks)),
        "entry_formation_ready_rate": sum(
            b["entry_formation_ready_ticks"] for b in blocks
        )
        / max(1, sum(b["entry_formation_ticks"] for b in blocks)),
        "entry_sync_hold_rate": sum(b["entry_sync_hold_ticks"] for b in blocks)
        / max(1, sum(b["entry_sync_ticks"] for b in blocks)),
        "designated_route_block_rate": sum(
            b["designated_route_block_ticks"] for b in blocks
        )
        / max(1, sum(b["formation_approach_ticks"] for b in blocks)),
        "designated_route_clear_rate": sum(
            b["designated_route_clear_moves"] for b in blocks
        )
        / max(1, sum(b["designated_route_clear_opportunities"] for b in blocks)),
        "designated_route_clear_opportunities": sum(
            b["designated_route_clear_opportunities"] for b in blocks
        ),
        "designated_route_clear_moves": sum(
            b["designated_route_clear_moves"] for b in blocks
        ),
        "screen_guidance_follow_rate": sum(b["screen_guidance_follows"] for b in blocks)
        / max(1, sum(b["screen_guidance_opportunities"] for b in blocks)),
        "screen_guidance_opportunities": sum(
            b["screen_guidance_opportunities"] for b in blocks
        ),
        "screen_guidance_follows": sum(b["screen_guidance_follows"] for b in blocks),
        "ultimate_use_rate": sum(b["ultimate_uses"] for b in blocks)
        / max(1, sum(b["ultimate_ready_ticks"] for b in blocks)),
        "ultimate_ready_ticks": sum(b["ultimate_ready_ticks"] for b in blocks),
        "ultimate_uses": sum(b["ultimate_uses"] for b in blocks),
        "orb_actions": sum(b.get("orb_actions", 0) for b in blocks),
        "orb_collections": sum(b.get("orb_collections", 0) for b in blocks),
        "orb_approach_opportunities": sum(
            b.get("orb_approach_opportunities", 0) for b in blocks
        ),
        "orb_approach_steps": sum(b.get("orb_approach_steps", 0) for b in blocks),
        "orb_eligible_decisions": sum(
            b.get("orb_eligible_decisions", 0) for b in blocks
        ),
        "orb_eligible_misses": sum(
            b.get("orb_eligible_misses", 0) for b in blocks
        ),
        "orb_greedy_choices": sum(
            b.get("orb_greedy_choices", 0) for b in blocks
        ),
        "orb_q_margin_sum": sum(b.get("orb_q_margin_sum", 0.0) for b in blocks),
        "orb_q_margin_count": sum(b.get("orb_q_margin_count", 0) for b in blocks),
        "orb_q_margin_mean": sum(b.get("orb_q_margin_sum", 0.0) for b in blocks)
        / max(1, sum(b.get("orb_q_margin_count", 0) for b in blocks)),
        "orb_round_missed": sum(b.get("orb_round_missed", 0) for b in blocks),
        "orb_round_miss_penalty": sum(
            b.get("orb_round_miss_penalty", 0.0) for b in blocks
        ),
        "orb_visible_ticks": sum(b.get("orb_visible_ticks", 0) for b in blocks),
        "orb_teacher_actions": sum(b.get("orb_teacher_actions", 0) for b in blocks),
        "orb_teacher_forced_actions": sum(
            b.get("orb_teacher_forced_actions", 0) for b in blocks
        ),
        "combat_start_events": sum(
            b.get("combat_start_events", 0) for b in blocks
        ),
        "combat_start_mean_aim_angle": sum(
            b.get("combat_start_mean_aim_angle", 0.0)
            * b.get("combat_start_events", 0)
            for b in blocks
        ) / max(1, sum(b.get("combat_start_events", 0) for b in blocks)),
        "combat_start_aim_alignment": sum(
            b.get("combat_start_aim_alignment", 0.0)
            * b.get("combat_start_events", 0)
            for b in blocks
        ) / max(1, sum(b.get("combat_start_events", 0) for b in blocks)),
        "combat_start_aim_reward": sum(
            b.get("combat_start_aim_reward", 0.0) for b in blocks
        ),
        "tactical_ultimate_rate": sum(b["tactical_ultimate_uses"] for b in blocks)
        / max(1, sum(b["ultimate_uses"] for b in blocks)),
        "tactical_ultimate_uses": sum(b["tactical_ultimate_uses"] for b in blocks),
        "preentry_ultimate_uses": sum(b["preentry_ultimate_uses"] for b in blocks),
        "ultimate_uses_by_phase": {
            phase: sum(b["ultimate_uses_by_phase"][phase] for b in blocks)
            for phase in PHASES
        },
        "tactical_ultimate_uses_by_phase": {
            phase: sum(b["tactical_ultimate_uses_by_phase"][phase] for b in blocks)
            for phase in PHASES
        },
        "facing_teacher_match_rate_by_phase": {
            phase: sum(b["facing_teacher_matches_by_phase"][phase] for b in blocks)
            / max(1, sum(b["facing_decisions_by_phase"][phase] for b in blocks))
            for phase in FACING_PHASES
        },
        "facing_decisions_by_phase": {
            phase: sum(b["facing_decisions_by_phase"][phase] for b in blocks)
            for phase in FACING_PHASES
        },
        "facing_teacher_matches_by_phase": {
            phase: sum(b["facing_teacher_matches_by_phase"][phase] for b in blocks)
            for phase in FACING_PHASES
        },
        "facing_confident_match_rate_by_phase": {
            phase: sum(b["facing_confident_matches_by_phase"][phase] for b in blocks)
            / max(
                1,
                sum(b["facing_confident_decisions_by_phase"][phase] for b in blocks),
            )
            for phase in FACING_PHASES
        },
        "facing_confident_decisions_by_phase": {
            phase: sum(b["facing_confident_decisions_by_phase"][phase] for b in blocks)
            for phase in FACING_PHASES
        },
        "facing_confident_matches_by_phase": {
            phase: sum(b["facing_confident_matches_by_phase"][phase] for b in blocks)
            for phase in FACING_PHASES
        },
        "facing_weighted_match_rate_by_phase": {
            phase: sum(b["facing_weighted_matches_by_phase"][phase] for b in blocks)
            / max(
                1e-6,
                sum(b["facing_confidence_mass_by_phase"][phase] for b in blocks),
            )
            for phase in FACING_PHASES
        },
        "facing_confidence_mass_by_phase": {
            phase: sum(b["facing_confidence_mass_by_phase"][phase] for b in blocks)
            for phase in FACING_PHASES
        },
        "facing_weighted_matches_by_phase": {
            phase: sum(b["facing_weighted_matches_by_phase"][phase] for b in blocks)
            for phase in FACING_PHASES
        },
        "facing_context_counts_by_phase": facing_context_counts,
        "facing_context_matches_by_phase": facing_context_matches,
        "facing_context_match_rate_by_phase": {
            phase: {
                source: facing_context_matches[phase][source] / max(1, count)
                for source, count in facing_context_counts[phase].items()
            }
            for phase in FACING_PHASES
        },
        **{
            key: float(np.mean([b[key] for b in blocks]))
            for key in (
                "carry_no_entry_rate",
                "carry_spawn_tick_rate",
                "carry_reversal_tick_rate",
                "carry_quiet_stall_tick_rate",
                "timeout_rate",
                "carrier_preentry_death_rate",
                "no_entry_carrier_death_rate",
                "no_entry_team_wipe_rate",
                "no_entry_other_rate",
                "mean_death_support_near",
                "mean_death_support_ahead",
                "mean_first_threat_support_near",
                "mean_first_threat_support_ahead",
                "death_screen_ready_rate",
                "first_threat_screen_ready_rate",
                "combat_stop_rate",
            )
        },
    }


def train(args):
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    torch.manual_seed(args.seed)
    data_fingerprint = runtime_data_fingerprint()

    def ensure_runtime_data_unchanged():
        current = runtime_data_fingerprint()
        if current != data_fingerprint:
            changed = sorted(
                name
                for name in set(data_fingerprint) | set(current)
                if data_fingerprint.get(name) != current.get(name)
            )
            raise RuntimeError(
                "runtime data changed during training; refusing a mixed-revision "
                f"evaluation: {changed}"
            )

    sources = {p: getattr(args, "init_" + p).resolve() for p in PHASES}
    checkpoints = {
        p: torch.load(s, map_location="cpu", weights_only=False)
        for p, s in sources.items()
    }
    changed_source_data = [
        phase
        for phase in PHASES
        if checkpoints[phase].get("runtime_data_fingerprint") != data_fingerprint
    ]
    if changed_source_data:
        print(
            "[SOURCE DATA REVISION] " + json.dumps({"phases": changed_source_data}),
            flush=True,
        )
    hashes = {p: hashlib.sha256(s.read_bytes()).hexdigest() for p, s in sources.items()}
    policies = {
        "carry": runtime.AttackerCarryDuelingDQN(
            obs_dim=runtime.ORB_OBS_DIM, action_dim=runtime.ACTION_DIM
        ),
        "escort": escort_runtime.DuelingQNetwork(
            escort_runtime.ORB_OBS_DIM, escort_runtime.N_ACTIONS
        ),
        "guard": guard_runtime.AttackerGuardDuelingDQN(
            obs_dim=guard_runtime.ORB_OBS_DIM,
            action_dim=guard_runtime.ACTION_DIM,
        ),
    }
    for phase in PHASES:
        state = expand_policy_state(
            checkpoints[phase],
            policies[phase].feature[0].in_features,
            policies[phase].advantage_head[-1].out_features,
        )
        incompatible = policies[phase].load_state_dict(state, strict=False)
        missing = [
            key
            for key in incompatible.missing_keys
            if not (phase in FACING_PHASES and is_facing_parameter(key))
        ]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"{phase} checkpoint keys mismatch: missing={missing}, "
                f"unexpected={list(incompatible.unexpected_keys)}"
            )
    import copy

    feature_columns = {
        "carry": range(runtime.ENTRY_SYNC_OBS_DIM, runtime.FACING_HEAD_OBS_DIM),
        "escort": list(
            range(escort_runtime.CLEARANCE_OBS_DIM, escort_runtime.FACING_HEAD_OBS_DIM)
        )
        + list(
            range(
                escort_runtime.FACING_HEAD_OBS_DIM,
                escort_runtime.FAKE_WAIT_SUPPORT_OBS_DIM,
            )
        ),
        "guard": range(guard_runtime.OBS_DIM, guard_runtime.ULTIMATE_CONTEXT_OBS_DIM),
    }
    ultimate_rows = {
        "carry": (runtime.ULTIMATE_ACTION_INDEX,),
        "escort": (escort_runtime.ACTION_ULTIMATE,),
        "guard": (guard_runtime.ULTIMATE_ACTION_INDEX,),
    }
    restricted = set(args.feature_only_phases) | set(args.action_only_phases)
    for phase in restricted:
        restrict_policy_updates(
            policies[phase],
            feature_columns.get(phase, ()) if phase in args.feature_only_phases else (),
            ultimate_rows[phase] if phase in args.action_only_phases else (),
        )
    for phase in args.facing_only_phases:
        restrict_policy_to_facing_head(policies[phase])
    for phase in args.movement_only_phases:
        rows = MOVEMENT_ACTION_ROWS[phase]
        if phase in args.reset_movement_head_phases:
            reset_movement_rows(policies[phase], rows)
        if phase == "escort":
            # Keep the previously added fake-wait features trainable and also
            # expose the four orb-context features to the movement head.
            input_columns = range(
                escort_runtime.FACING_HEAD_OBS_DIM,
                escort_runtime.ORB_OBS_DIM,
            )
        elif phase == "carry":
            # Carry v12 appends orb distance/availability context after the
            # facing head.  Without these columns the new action row can
            # never condition on an orb, even when demonstrations exist.
            input_columns = range(
                runtime.FACING_HEAD_OBS_DIM,
                runtime.ORB_OBS_DIM,
            )
        else:
            input_columns = ()
        restrict_policy_to_movement_rows(
            policies[phase],
            rows,
            input_columns,
            train_facing=(
                args.train_facing_with_movement and phase in FACING_PHASES
            ),
        )
    # A reset branch must start its target network from the same freshly
    # initialized movement rows, not from the pre-reset source checkpoint.
    targets = {p: copy.deepcopy(net) for p, net in policies.items()}
    optimizers = {
        p: torch.optim.Adam(
            [parameter for parameter in net.parameters() if parameter.requires_grad],
            lr=args.lr,
        )
        for p, net in policies.items()
    }
    orb_collection_optimizers = {
        phase: torch.optim.Adam(
            (
                policies[phase].advantage_head[-1].weight,
                policies[phase].advantage_head[-1].bias,
            ),
            lr=args.lr * args.orb_collection_lr_multiplier,
        )
        for phase in ("carry", "escort")
        if phase in args.movement_only_phases
    }
    replays = {p: deque(maxlen=100_000) for p in PHASES}
    demonstrations = {p: deque(maxlen=20_000) for p in PHASES}
    orb_demonstrations = {p: deque(maxlen=8_000) for p in PHASES}
    orb_approach_demonstrations = {p: deque(maxlen=8_000) for p in PHASES}
    orb_collection_demonstrations = {p: deque(maxlen=2_000) for p in PHASES}
    fake_wait_demonstrations = deque(maxlen=5_000)
    facing_demonstrations = {p: deque(maxlen=30_000) for p in FACING_PHASES}
    ultimate_positives = {p: deque(maxlen=10_000) for p in PHASES}
    ultimate_negatives = {p: deque(maxlen=10_000) for p in PHASES}
    session = CurriculumSession(sources, policies, args.gamma)
    session.frozen_phases = set(args.freeze_phases)
    session.frozen_facing_phases = set(args.freeze_phases) | set(
        args.movement_only_phases
    )
    for phase in FACING_PHASES:
        if phase in session.frozen_phases:
            session.controllers[phase].facing_head_enabled = (
                int(checkpoints[phase].get("facing_head_version", 0)) >= 1
            )
    macro_path = Path(
        session.game.attacker_controller.inner_controller.macro_controller.model_path
    ).resolve()
    macro_source = {
        "path": str(macro_path),
        "sha256": hashlib.sha256(macro_path.read_bytes()).hexdigest(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite_output and any(
        args.output_dir.resolve() in source.parents for source in sources.values()
    ):
        raise ValueError("output-dir must not be a source checkpoint directory")
    if any(args.output_dir.iterdir()) and not args.overwrite_output:
        raise ValueError(
            "output-dir must be empty unless --overwrite-output is explicit"
        )
    history, evaluations = [], []
    best = None
    best_episode = None
    baseline_metrics = None
    evaluations_without_improvement = 0
    guardrail_violations = 0
    stop_reason = None
    stopped_episode = None
    best_facing_scores = {phase: None for phase in FACING_PHASES}
    best_facing_episodes = {phase: None for phase in FACING_PHASES}
    started = time.monotonic()

    def checkpoint_payload(phase, episode, metrics):
        payload = (
            dict(checkpoints[phase]) if "model_state_dict" in checkpoints[phase] else {}
        )
        payload.update(
            model_state_dict=policies[phase].state_dict(),
            obs_dim=policies[phase].feature[0].in_features,
            n_actions=policies[phase].advantage_head[-1].out_features,
            positioning_version=VERSIONS[phase],
            episode=episode,
            training_environment="actual_engine_full_round_curriculum",
            training_revision=(
                "carry_escort_joint_movement"
                if set(args.movement_only_phases) == {"carry", "escort"}
                else (
                    "escort_support_input_columns_v22"
                    if set(args.movement_only_phases) == {"escort"}
                    else (
                        "movement_rows_v21"
                        if set(args.movement_only_phases) == {"carry"}
                        else TRAINING_REVISION
                    )
                )
            ),
            evaluation=metrics,
            source_models={p: str(s) for p, s in sources.items()},
            source_hashes=hashes,
            source_runtime_data_fingerprints={
                p: checkpoints[p].get("runtime_data_fingerprint") for p in PHASES
            },
            macro_source=macro_source,
            runtime_data_fingerprint=data_fingerprint,
            training_parameters=vars_for_json(args),
        )
        if phase in FACING_PHASES:
            payload["facing_head_version"] = runtime.FACING_HEAD_VERSION
        return payload

    def save(kind, episode, metrics):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "episode": episode,
            "evaluation": metrics,
            "models": {},
            "macro_source": macro_source,
            "runtime_data_fingerprint": data_fingerprint,
        }
        for phase in PHASES:
            payload = checkpoint_payload(phase, episode, metrics)
            filename = f"dqn_attacker_{phase}_gc_{kind}.pt"
            torch.save(payload, args.output_dir / filename)
            bundle["models"][phase] = filename
        (args.output_dir / (kind + "_bundle.json")).write_text(
            json.dumps(bundle, indent=2), encoding="utf-8"
        )

    def save_phase_facing(phase, episode, metrics, score):
        filename = f"dqn_attacker_{phase}_gc_best_facing.pt"
        payload = checkpoint_payload(phase, episode, metrics)
        payload["phase_facing_selection_score"] = list(score)
        torch.save(payload, args.output_dir / filename)
        record = {
            "phase": phase,
            "episode": episode,
            "model": filename,
            "score": list(score),
            "evaluation": metrics,
            "runtime_data_fingerprint": data_fingerprint,
        }
        (args.output_dir / f"best_facing_{phase}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    for episode in range(args.episodes + 1):
        global_episode = args.episode_offset + episode
        schedule_episode = episode if args.relative_schedules else global_episode
        if episode:
            stats = opponent_stats(
                schedule_episode,
                args.curriculum_episodes,
                args.start_stats,
                args.final_stats,
            )
            # Expose the policy to the final opponent before the curriculum is
            # complete, avoiding a late collapse at the final-strength eval.
            if (
                schedule_episode > args.curriculum_episodes // 2
                and random.random() < args.final_mix
            ):
                stats = tuple(args.final_stats)
            # Keep exploration available when the opponent reaches full strength.
            epsilon = max(
                0.03,
                0.20
                * (1.0 - (schedule_episode - 1) / max(1, args.curriculum_episodes)),
            )
            teacher_probability = (
                0.80
                * max(
                    0.0,
                    1.0 - (schedule_episode - 1) / args.navigation_bootstrap_episodes,
                )
                if args.navigation_bootstrap_episodes
                else 0.0
            )
            orb_teacher_probability = scheduled_orb_teacher_probability(
                schedule_episode,
                args.navigation_bootstrap_episodes,
                args.orb_teacher_final_probability,
            )
            learning_rate = scheduled_learning_rate(
                args.lr,
                schedule_episode,
                args.lr_decay_start,
                args.lr_decay_episodes,
                args.lr_final_scale,
            )
            for optimizer in optimizers.values():
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
            for optimizer in orb_collection_optimizers.values():
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * args.orb_collection_lr_multiplier
            row = session.play(
                args.seed + global_episode,
                stats,
                epsilon,
                teacher_probability=teacher_probability,
                orb_teacher_probability=orb_teacher_probability,
            )
            row["episode"] = global_episode
            row["navigation_teacher_probability"] = teacher_probability
            row["orb_teacher_probability"] = orb_teacher_probability
            row["orb_teacher_round_enabled"] = session.orb_teacher_round_enabled
            row["learning_rate"] = learning_rate
            row["demonstration_counts"] = {
                p: len(session.demonstrations[p]) for p in PHASES
            }
            row["orb_demonstration_counts"] = {
                p: len(session.orb_demonstrations[p]) for p in PHASES
            }
            row["orb_approach_demonstration_counts"] = {
                p: len(session.orb_approach_demonstrations[p]) for p in PHASES
            }
            row["orb_collection_demonstration_counts"] = {
                p: len(session.orb_collection_demonstrations[p]) for p in PHASES
            }
            row["fake_wait_demonstration_count"] = int(sum(
                len(obs) >= escort_runtime.FAKE_WAIT_SUPPORT_OBS_DIM
                and obs[escort_runtime.FACING_HEAD_OBS_DIM] > 0.5
                for obs, _action, _mask in session.demonstrations["escort"]
            ))
            row["facing_demonstration_counts"] = {
                p: len(session.facing_examples[p]) for p in FACING_PHASES
            }
            row["ultimate_classification_counts"] = {
                p: {
                    "positive": sum(
                        label for _obs, _mask, label in session.ultimate_examples[p]
                    ),
                    "negative": sum(
                        not label for _obs, _mask, label in session.ultimate_examples[p]
                    ),
                }
                for p in PHASES
            }
            row["navigation_demo_multiplier"] = max(
                teacher_probability, args.navigation_retention_weight
            )
            history.append(row)
            for phase in PHASES:
                if phase in args.freeze_phases:
                    continue
                if phase in args.facing_only_phases:
                    facing_demonstrations[phase].extend(session.facing_examples[phase])
                    facing_updates = min(
                        args.max_updates,
                        max(1, len(session.facing_examples[phase]) // 2),
                    )
                    for _ in range(facing_updates):
                        optimize_facing(
                            policies[phase],
                            optimizers[phase],
                            facing_demonstrations[phase],
                            args.facing_supervision_weight,
                        )
                    continue
                transitions = session.transitions[phase]
                phase_demonstrations = session.demonstrations[phase]
                allowed_actions = None
                if phase in args.movement_only_phases:
                    allowed_actions = MOVEMENT_ACTION_ROWS[phase]
                    phase_demonstrations = movement_only_demonstrations(
                        phase_demonstrations, allowed_actions
                    )
                phase_orb_demonstrations = session.orb_demonstrations[phase]
                if allowed_actions is not None:
                    phase_orb_demonstrations = movement_only_demonstrations(
                        phase_orb_demonstrations, allowed_actions
                    )
                replays[phase].extend(
                    n_step_transitions(
                        transitions,
                        args.gamma,
                        args.n_step,
                        allowed_start_actions=allowed_actions,
                    )
                )
                demonstrations[phase].extend(phase_demonstrations)
                orb_demonstrations[phase].extend(phase_orb_demonstrations)
                orb_approach_demonstrations[phase].extend(
                    session.orb_approach_demonstrations[phase]
                )
                orb_collection_demonstrations[phase].extend(
                    session.orb_collection_demonstrations[phase]
                )
                if phase == "escort":
                    fake_wait_demonstrations.extend(
                        sample for sample in phase_demonstrations
                        if len(sample[0]) >= escort_runtime.FAKE_WAIT_SUPPORT_OBS_DIM
                        and sample[0][escort_runtime.FACING_HEAD_OBS_DIM] > 0.5
                    )
                if phase in FACING_PHASES:
                    facing_demonstrations[phase].extend(session.facing_examples[phase])
                for obs, mask, tactical in session.ultimate_examples[phase]:
                    destination = ultimate_positives if tactical else ultimate_negatives
                    destination[phase].append((obs, mask))
                update_transition_count = (
                    len(movement_only_transitions(transitions, allowed_actions))
                    if allowed_actions is not None
                    else len(transitions)
                )
                updates = min(args.max_updates, max(1, update_transition_count // 2))
                td_updates = updates
                demo_updates = updates
                if phase in args.movement_only_phases:
                    td_updates = min(td_updates, args.movement_td_updates)
                    demo_updates = min(demo_updates, args.movement_demo_updates)
                for update_index in range(
                    max(
                        td_updates,
                        demo_updates,
                        args.orb_demo_updates if phase in ("carry", "escort") else 0,
                        args.orb_collection_updates if phase in ("carry", "escort") else 0,
                    )
                ):
                    if update_index < td_updates:
                        optimize(
                            policies[phase],
                            targets[phase],
                            optimizers[phase],
                            replays[phase],
                            args.gamma,
                        )
                    if update_index < demo_updates:
                        optimize_demonstrations(
                            policies[phase],
                            optimizers[phase],
                            demonstrations[phase],
                            args.navigation_demo_weight
                            * max(
                                teacher_probability,
                                args.navigation_retention_weight,
                            ),
                            focus_samples=(
                                fake_wait_demonstrations if phase == "escort" else None
                            ),
                        )
                    if (
                        phase in ("carry", "escort")
                        and update_index < args.orb_demo_updates
                    ):
                        # Approach is the scarce behavior at inference time:
                        # train legal steps toward a nearby orb alongside the
                        # final COLLECT action, not only the latter.
                        optimize_demonstrations(
                            policies[phase],
                            optimizers[phase],
                            orb_approach_demonstrations[phase]
                            if orb_approach_demonstrations[phase]
                            else orb_demonstrations[phase],
                            args.orb_demo_weight,
                            batch_size=64,
                            focus_samples=orb_collection_demonstrations[phase],
                            focus_fraction=0.40,
                            focus_with_replacement=True,
                        )
                    if (
                        phase in orb_collection_optimizers
                        and update_index < args.orb_collection_updates
                    ):
                        # Rare executable COLLECT labels need a faster,
                        # row-local update.  Other action rows stay fixed.
                        optimize_orb_collection_head(
                            policies[phase],
                            orb_collection_optimizers[phase],
                            orb_collection_demonstrations[phase],
                            runtime.COLLECT_ORB_ACTION_INDEX if phase == "carry"
                            else escort_runtime.ACTION_COLLECT_ORB,
                        )
                    if (
                        update_index < td_updates
                        and phase in FACING_PHASES
                        and (
                            phase not in args.movement_only_phases
                            or args.train_facing_with_movement
                        )
                    ):
                        optimize_facing(
                            policies[phase],
                            optimizers[phase],
                            facing_demonstrations[phase],
                            args.facing_supervision_weight,
                        )
                    if (
                        update_index < td_updates
                        and phase not in args.movement_only_phases
                    ):
                        optimize_ultimate_classification(
                            policies[phase],
                            optimizers[phase],
                            ultimate_positives[phase],
                            ultimate_negatives[phase],
                            ultimate_rows[phase][0],
                            args.ultimate_classification_weight,
                        )
            if global_episode % 20 == 0:
                for p in PHASES:
                    targets[p].load_state_dict(policies[p].state_dict())
            if global_episode % 10 == 0:
                recent = history[-50:]
                print(
                    f"[EP {global_episode}] opponent={tuple(round(x, 1) for x in stats)} "
                    f"win={sum(r['attacker_win'] for r in recent)/len(recent):.3f} "
                    f"plant={sum(r['planted'] for r in recent)/len(recent):.3f} "
                    f"epsilon={epsilon:.3f} teacher={teacher_probability:.3f} "
                    f"orb_teacher_prob={orb_teacher_probability:.3f} "
                    f"lr={learning_rate:.2e} "
                    f"demo={row['navigation_demo_multiplier']:.3f} "
                    f"fake_wait_demo={sum(r.get('fake_wait_demonstration_count', 0) for r in recent)/len(recent):.1f} "
                    f"orb_actions={sum(r.get('orb_actions', 0) for r in recent)} "
                    f"orb_collections={sum(r.get('orb_collections', 0) for r in recent)} "
                    f"orb_approach={sum(r.get('orb_approach_steps', 0) for r in recent)}/"
                    f"{sum(r.get('orb_approach_opportunities', 0) for r in recent)} "
                    f"orb_teacher={sum(r.get('orb_teacher_actions', 0) for r in recent)} "
                    f"orb_forced={sum(r.get('orb_teacher_forced_actions', 0) for r in recent)} "
                    f"orb_teacher_rounds={sum(r.get('orb_teacher_round_enabled', False) for r in recent)} "
                    f"orb_eligible={sum(r.get('orb_eligible_decisions', 0) for r in recent)} "
                    f"orb_misses={sum(r.get('orb_eligible_misses', 0) for r in recent)} "
                    f"orb_greedy={sum(r.get('orb_greedy_choices', 0) for r in recent)} "
                    f"orb_q_margin={sum(r.get('orb_q_margin_sum', 0.0) for r in recent) / max(1, sum(r.get('orb_q_margin_count', 0) for r in recent)):.2f} "
                    f"orb_missed={sum(r.get('orb_round_missed', False) for r in recent)} "
                    f"orb_demo={sum(sum(r.get('orb_demonstration_counts', {}).values()) for r in recent)} "
                    f"orb_approach_demo={sum(sum(r.get('orb_approach_demonstration_counts', {}).values()) for r in recent)} "
                    f"orb_collect_demo={sum(sum(r.get('orb_collection_demonstration_counts', {}).values()) for r in recent)} "
                    f"no_entry={sum(not entered_site(r) for r in recent)/len(recent):.3f} "
                    f"timeout={sum(r['timed_out'] for r in recent)/len(recent):.3f} "
                    f"seconds={time.monotonic()-started:.0f}",
                    flush=True,
                )
        if (
            episode == 0
            or global_episode % args.eval_interval == 0
            or episode == args.episodes
        ):
            ensure_runtime_data_unchanged()
            metrics = evaluate_multi(
                session, args.eval_episodes, args.eval_seeds, args.final_stats
            )
            metrics["episode"] = global_episode
            metrics["entry_quality_passed"] = entry_quality_passed(
                metrics, args.max_no_entry_rate, args.max_timeout_rate
            )
            evaluations.append(metrics)
            if baseline_metrics is None:
                baseline_metrics = metrics
            if set(args.movement_only_phases) == {"carry"}:
                score = carry_movement_selection_score(
                    metrics, args.max_no_entry_rate, args.max_timeout_rate
                )
            elif set(args.movement_only_phases) == {"escort"}:
                score = escort_support_selection_score(
                    metrics, args.max_no_entry_rate, args.max_timeout_rate
                )
            elif set(args.movement_only_phases) == {"carry", "escort"}:
                score = joint_movement_selection_score(
                    metrics,
                    baseline_metrics,
                    args.max_no_entry_rate,
                    args.max_timeout_rate,
                    args.max_quiet_stall_increase,
                    args.max_carrier_death_increase,
                    args.max_plant_regression,
                )
            else:
                score = selection_score(
                    metrics, args.max_no_entry_rate, args.max_timeout_rate
                )
            save("latest", global_episode, metrics)
            improved = best is None or score > best
            if improved:
                best = score
                best_episode = global_episode
                evaluations_without_improvement = 0
                save("best_by_eval", global_episode, metrics)
            elif episode > 0:
                evaluations_without_improvement += 1
            if episode > 0 and set(args.movement_only_phases) == {"carry"}:
                if carry_movement_guardrail_violated(
                    metrics,
                    baseline_metrics,
                    args.max_timeout_rate,
                    args.max_quiet_stall_increase,
                ):
                    guardrail_violations += 1
                else:
                    guardrail_violations = 0
            elif episode > 0 and set(args.movement_only_phases) == {
                "carry", "escort"
            }:
                if joint_movement_guardrail_violated(
                    metrics,
                    baseline_metrics,
                    args.max_timeout_rate,
                    args.max_quiet_stall_increase,
                    args.max_carrier_death_increase,
                    args.max_plant_regression,
                ):
                    guardrail_violations += 1
                else:
                    guardrail_violations = 0
            elif episode > 0 and set(args.movement_only_phases) == {"escort"}:
                if escort_support_guardrail_violated(
                    metrics,
                    baseline_metrics,
                    args.max_timeout_rate,
                    args.max_carrier_death_increase,
                    args.max_plant_regression,
                ):
                    guardrail_violations += 1
                else:
                    guardrail_violations = 0
            for phase in FACING_PHASES:
                phase_score = phase_facing_selection_score(
                    phase,
                    metrics,
                    baseline_metrics,
                    args.max_no_entry_rate,
                    args.max_timeout_rate,
                    args.facing_regression_tolerance,
                )
                if (
                    best_facing_scores[phase] is None
                    or phase_score > best_facing_scores[phase]
                ):
                    best_facing_scores[phase] = phase_score
                    best_facing_episodes[phase] = global_episode
                    save_phase_facing(phase, global_episode, metrics, phase_score)
            print("[FINAL-STRENGTH EVAL] " + json.dumps(metrics), flush=True)
            (args.output_dir / "training_history.json").write_text(
                json.dumps(history), encoding="utf-8"
            )
            (args.output_dir / "evaluation_history.json").write_text(
                json.dumps(evaluations, indent=2), encoding="utf-8"
            )
            if episode < args.episodes:
                if (
                    args.movement_guardrail_patience > 0
                    and guardrail_violations >= args.movement_guardrail_patience
                ):
                    stop_reason = "movement_guardrail"
                elif (
                    args.early_stop_patience > 0
                    and evaluations_without_improvement >= args.early_stop_patience
                ):
                    stop_reason = "no_evaluation_improvement"
                if stop_reason is not None:
                    stopped_episode = global_episode
                    print(
                        "[EARLY STOP] "
                        + json.dumps(
                            {
                                "reason": stop_reason,
                                "episode": stopped_episode,
                                "best_episode": best_episode,
                                "evaluations_without_improvement": evaluations_without_improvement,
                                "consecutive_guardrail_violations": guardrail_violations,
                            }
                        ),
                        flush=True,
                    )
                    break
    stop_report = {
        "reason": stop_reason or "completed",
        "stopped_episode": stopped_episode,
        "last_evaluated_episode": evaluations[-1]["episode"],
        "best_episode": best_episode,
        "evaluations_without_improvement": evaluations_without_improvement,
        "consecutive_guardrail_violations": guardrail_violations,
        "early_stop_patience": args.early_stop_patience,
        "movement_guardrail_patience": args.movement_guardrail_patience,
        "max_timeout_rate": args.max_timeout_rate,
        "max_quiet_stall_increase": args.max_quiet_stall_increase,
        "max_carrier_death_increase": args.max_carrier_death_increase,
        "max_plant_regression": args.max_plant_regression,
        "baseline_quiet_stall_tick_rate": baseline_metrics.get(
            "carry_quiet_stall_tick_rate", 0.0
        ),
    }
    (args.output_dir / "training_stop.json").write_text(
        json.dumps(stop_report, indent=2), encoding="utf-8"
    )
    ensure_runtime_data_unchanged()
    # Report fresh held-out rounds for the selected joint policy; never use this
    # seed to select another checkpoint or present training wins as validation.
    for phase in PHASES:
        payload = torch.load(
            args.output_dir / f"dqn_attacker_{phase}_gc_best_by_eval.pt",
            weights_only=False,
        )
        policies[phase].load_state_dict(payload["model_state_dict"])
    holdout = evaluate_multi(
        session, args.holdout_episodes, args.holdout_seeds, args.final_stats
    )
    holdout["meets_win_rate_target"] = (
        holdout["worst_round_win_rate"] >= args.target_win_rate
    )
    holdout["entry_quality_passed"] = entry_quality_passed(
        holdout, args.max_no_entry_rate, args.max_timeout_rate
    )
    holdout["ready_for_match_validation"] = (
        holdout["meets_win_rate_target"] and holdout["entry_quality_passed"]
    )
    holdout["target_win_rate"] = args.target_win_rate
    holdout["selected_episode"] = json.loads(
        (args.output_dir / "best_by_eval_bundle.json").read_text(encoding="utf-8")
    )["episode"]
    (args.output_dir / "holdout_evaluation.json").write_text(
        json.dumps(holdout, indent=2), encoding="utf-8"
    )
    print("[HOLDOUT] " + json.dumps(holdout), flush=True)

    # Compose independently selected Carry/Escort facing heads.  All movement
    # tensors are frozen in facing-only runs, so this combines only phase-local
    # view policies.  Evaluation seeds validate the combination; holdout remains
    # a one-shot generalization report and never selects another checkpoint.
    if args.facing_only_phases:
        for phase in FACING_PHASES:
            payload = torch.load(
                args.output_dir / f"dqn_attacker_{phase}_gc_best_facing.pt",
                weights_only=False,
            )
            policies[phase].load_state_dict(payload["model_state_dict"])
            policies[phase].facing_head_version = int(
                payload.get("facing_head_version", runtime.FACING_HEAD_VERSION)
            )
        guard_payload = torch.load(
            args.output_dir / "dqn_attacker_guard_gc_best_by_eval.pt",
            weights_only=False,
        )
        policies["guard"].load_state_dict(guard_payload["model_state_dict"])
        composite_eval = evaluate_multi(
            session, args.eval_episodes, args.eval_seeds, args.final_stats
        )
        composite_eval["selected_episodes_by_phase"] = dict(best_facing_episodes)
        composite_episode = max(best_facing_episodes.values())
        composite_bundle = {
            "episode": composite_episode,
            "selected_episodes_by_phase": dict(best_facing_episodes),
            "evaluation": composite_eval,
            "models": {},
            "runtime_data_fingerprint": data_fingerprint,
        }
        for phase in PHASES:
            filename = f"dqn_attacker_{phase}_gc_best_phase_facing.pt"
            payload = checkpoint_payload(phase, composite_episode, composite_eval)
            if phase in FACING_PHASES:
                payload["selected_facing_episode"] = best_facing_episodes[phase]
            torch.save(payload, args.output_dir / filename)
            composite_bundle["models"][phase] = filename
        (args.output_dir / "best_phase_facing_bundle.json").write_text(
            json.dumps(composite_bundle, indent=2), encoding="utf-8"
        )
        ensure_runtime_data_unchanged()
        composite_holdout = evaluate_multi(
            session, args.holdout_episodes, args.holdout_seeds, args.final_stats
        )
        composite_holdout["selected_episodes_by_phase"] = dict(best_facing_episodes)
        composite_holdout["entry_quality_passed"] = entry_quality_passed(
            composite_holdout, args.max_no_entry_rate, args.max_timeout_rate
        )
        (args.output_dir / "holdout_phase_facing.json").write_text(
            json.dumps(composite_holdout, indent=2), encoding="utf-8"
        )
        print("[PHASE-FACING HOLDOUT] " + json.dumps(composite_holdout), flush=True)


def vars_for_json(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for phase in PHASES:
        parser.add_argument(
            "--init-" + phase,
            type=Path,
            default=ROOT
            / "gc_v1"
            / "data"
            / f"attacker_{phase}_gc_data"
            / f"dqn_attacker_{phase}_gc_best_by_eval.pt",
        )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="overwrite checkpoints and reports in an existing output directory",
    )
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help="global episode already completed by init checkpoints; --episodes is additional work",
    )
    parser.add_argument("--curriculum-episodes", type=int, default=6000)
    parser.add_argument(
        "--start-stats",
        type=float,
        nargs=3,
        default=(50, 10, 0),
        metavar=("HIT", "HS", "DODGE"),
    )
    parser.add_argument(
        "--final-stats",
        type=float,
        nargs=3,
        default=(100, 60, 40),
        metavar=("HIT", "HS", "DODGE"),
    )
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--holdout-episodes", type=int, default=200)
    parser.add_argument("--target-win-rate", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=2026091700)
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=(3026091700, 5026091700, 6026091700),
    )
    parser.add_argument(
        "--holdout-seeds",
        type=int,
        nargs="+",
        default=(9026092000, 10026092000, 11026092000),
    )
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument(
        "--lr-decay-start",
        type=int,
        default=0,
        help="schedule-relative episode at which linear learning-rate decay begins",
    )
    parser.add_argument(
        "--lr-decay-episodes",
        type=int,
        default=1,
        help="episodes over which lr reaches lr-final-scale",
    )
    parser.add_argument(
        "--lr-final-scale",
        type=float,
        default=1.0,
        help="final learning rate as a fraction of --lr",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-updates", type=int, default=64)
    parser.add_argument("--final-mix", type=float, default=0.20)
    parser.add_argument("--max-no-entry-rate", type=float, default=0.25)
    parser.add_argument("--max-timeout-rate", type=float, default=0.10)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="stop after this many consecutive evaluations without a better checkpoint; zero disables",
    )
    parser.add_argument(
        "--movement-guardrail-patience",
        type=int,
        default=0,
        help="stop movement training after this many consecutive safety regressions; zero disables",
    )
    parser.add_argument(
        "--max-quiet-stall-increase",
        type=float,
        default=0.05,
        help="maximum Carry quiet-stall increase over the episode-zero evaluation",
    )
    parser.add_argument(
        "--max-carrier-death-increase",
        type=float,
        default=0.05,
        help="maximum worst-seed pre-entry carrier-death increase over baseline",
    )
    parser.add_argument(
        "--max-plant-regression",
        type=float,
        default=0.05,
        help="maximum worst-seed registered-plant loss versus baseline",
    )
    parser.add_argument(
        "--navigation-bootstrap-episodes",
        type=int,
        default=500,
        help="fade training-only quiet navigation demonstrations to zero; eval is always policy-only",
    )
    parser.add_argument("--navigation-demo-weight", type=float, default=1.0)
    parser.add_argument(
        "--orb-demo-weight",
        type=float,
        default=3.0,
        help="dedicated demonstration-loss weight for every teacher step toward an orb",
    )
    parser.add_argument(
        "--orb-demo-updates",
        type=int,
        default=2,
        help="extra balanced orb imitation updates per active phase and episode",
    )
    parser.add_argument(
        "--orb-collection-updates",
        type=int,
        default=4,
        help="extra row-local on-orb COLLECT updates per active phase and episode",
    )
    parser.add_argument(
        "--orb-collection-lr-multiplier",
        type=float,
        default=20.0,
        help="learning-rate multiplier for the isolated COLLECT action row",
    )
    parser.add_argument(
        "--orb-teacher-final-probability",
        type=float,
        default=0.05,
        help="fraction of orb teacher-guided rounds after bootstrap",
    )
    parser.add_argument(
        "--navigation-retention-weight",
        type=float,
        default=0.15,
        help="minimum demo loss multiplier after bootstrap; labels never replace eval/inference actions",
    )
    parser.add_argument(
        "--ultimate-classification-weight",
        type=float,
        default=1.0,
        help="balanced margin loss for tactical ULT use versus explicit saving",
    )
    parser.add_argument(
        "--facing-supervision-weight",
        type=float,
        default=0.5,
        help="cross-entropy weight for Carry/Escort phase-specific facing heads",
    )
    parser.add_argument(
        "--facing-regression-tolerance",
        type=float,
        default=0.05,
        help="maximum tolerated regression when selecting each phase's facing head",
    )
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument(
        "--freeze-phases",
        nargs="*",
        choices=PHASES,
        default=(),
        help="keep selected phase weights fixed while preserving their deployed behavior",
    )
    parser.add_argument(
        "--feature-only-phases",
        nargs="*",
        choices=PHASES,
        default=(),
        help="train only coordination-feature columns; zero-feature behavior stays exact",
    )
    parser.add_argument(
        "--action-only-phases",
        nargs="*",
        choices=PHASES,
        default=(),
        help="also train only the new ultimate advantage-output row",
    )
    parser.add_argument(
        "--facing-only-phases",
        nargs="*",
        choices=FACING_PHASES,
        default=(),
        help="freeze movement/value/features and update only each phase's facing head",
    )
    parser.add_argument(
        "--movement-only-phases",
        nargs="*",
        choices=tuple(MOVEMENT_ACTION_ROWS),
        default=(),
        help=(
            "train phase displacement rows while freezing tactical/facing weights; "
            "Escort may also learn its new support-only input columns"
        ),
    )
    parser.add_argument(
        "--train-facing-with-movement",
        action="store_true",
        help=(
            "also update the independent Carry/Escort facing heads during "
            "movement-only training; required for aim-contact rewards to "
            "improve pre-aim rather than movement alone"
        ),
    )
    parser.add_argument(
        "--reset-movement-head-phases",
        nargs="*",
        choices=tuple(MOVEMENT_ACTION_ROWS),
        default=(),
        help="reinitialize selected displacement rows before movement-only training",
    )
    parser.add_argument(
        "--movement-td-updates",
        type=int,
        default=1,
        help="maximum TD optimizer steps per episode for a movement-only phase",
    )
    parser.add_argument(
        "--movement-demo-updates",
        type=int,
        default=4,
        help="maximum navigation-demo optimizer steps per episode for a movement-only phase",
    )
    parser.add_argument(
        "--relative-schedules",
        action="store_true",
        help="restart curriculum/exploration/demo schedules for this additional run",
    )
    args = parser.parse_args()
    if (
        min(
            args.episodes,
            args.eval_interval,
            args.eval_episodes,
            args.holdout_episodes,
            args.max_updates,
        )
        < 1
    ):
        parser.error(
            "episode counts, evaluation interval and max-updates must be positive"
        )
    if args.episode_offset < 0:
        parser.error("episode-offset must be nonnegative")
    total_episodes = args.episode_offset + args.episodes
    schedule_episodes = args.episodes if args.relative_schedules else total_episodes
    if not 0 <= args.curriculum_episodes < schedule_episodes:
        parser.error(
            "curriculum-episodes must be >= 0 and smaller than the active schedule"
        )
    if (
        args.navigation_bootstrap_episodes < 0
        or args.navigation_demo_weight < 0
        or args.orb_demo_weight < 0
        or args.orb_demo_updates < 0
        or args.orb_collection_updates < 0
        or args.orb_collection_lr_multiplier <= 0
        or not 0 <= args.orb_teacher_final_probability <= 0.80
        or args.ultimate_classification_weight < 0
        or args.facing_supervision_weight < 0
        or args.movement_td_updates < 0
        or args.movement_demo_updates < 0
        or args.early_stop_patience < 0
        or args.movement_guardrail_patience < 0
        or args.lr_decay_start < 0
        or args.lr_decay_episodes < 1
        or args.max_quiet_stall_increase < 0
        or args.max_carrier_death_increase < 0
        or args.max_plant_regression < 0
        or not 0 <= args.facing_regression_tolerance <= 1
        or not 0 <= args.navigation_retention_weight <= 1
        or args.n_step < 1
    ):
        parser.error(
            "navigation bootstrap counts/weights must be nonnegative and n-step positive"
        )
    if args.navigation_bootstrap_episodes >= schedule_episodes:
        parser.error(
            "navigation-bootstrap-episodes must be smaller than the active schedule (reserve policy-only training)"
        )
    if (
        not 0 < args.gamma <= 1
        or args.lr <= 0
        or not 0 < args.lr_final_scale <= 1
        or not 0 <= args.target_win_rate <= 1
        or not 0 <= args.final_mix <= 1
        or not 0 <= args.max_no_entry_rate <= 1
        or not 0 <= args.max_timeout_rate <= 1
    ):
        parser.error("invalid gamma, lr or target-win-rate")
    if any(not 0 <= a <= b <= 100 for a, b in zip(args.start_stats, args.final_stats)):
        parser.error("combat percentages must satisfy 0 <= start <= final <= 100")
    if set(args.freeze_phases) & set(args.feature_only_phases):
        parser.error("a phase cannot be both frozen and feature-only")
    if set(args.freeze_phases) & set(args.action_only_phases):
        parser.error("a phase cannot be both frozen and action-only")
    if set(args.freeze_phases) & set(args.facing_only_phases):
        parser.error("a phase cannot be both frozen and facing-only")
    if set(args.feature_only_phases) & set(args.facing_only_phases):
        parser.error("a phase cannot be both feature-only and facing-only")
    if set(args.action_only_phases) & set(args.facing_only_phases):
        parser.error("a phase cannot be both action-only and facing-only")
    if set(args.movement_only_phases) & (
        set(args.freeze_phases)
        | set(args.feature_only_phases)
        | set(args.action_only_phases)
        | set(args.facing_only_phases)
    ):
        parser.error("a phase cannot use movement-only with another restricted mode")
    if not set(args.reset_movement_head_phases) <= set(args.movement_only_phases):
        parser.error(
            "reset-movement-head-phases must be a subset of movement-only-phases"
        )
    if args.movement_guardrail_patience > 0 and set(args.movement_only_phases) not in (
        {"carry"},
        {"escort"},
        {"carry", "escort"},
    ):
        parser.error(
            "movement-guardrail-patience requires Carry and/or Escort movement-only training"
        )
    seed_sets = [
        set(
            range(
                args.seed + args.episode_offset + 1,
                args.seed + args.episode_offset + args.episodes + 1,
            )
        )
    ]
    seed_sets.extend(set(range(s, s + args.eval_episodes)) for s in args.eval_seeds)
    seed_sets.extend(
        set(range(s, s + args.holdout_episodes)) for s in args.holdout_seeds
    )
    if any(
        seed_sets[i] & seed_sets[j] for i in range(len(seed_sets)) for j in range(i)
    ):
        parser.error("training, evaluation and holdout seed ranges must not overlap")
    train(args)


if __name__ == "__main__":
    main()
