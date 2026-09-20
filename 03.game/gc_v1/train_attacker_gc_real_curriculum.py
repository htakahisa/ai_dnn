"""Learn Carry/Escort/Guard together in real rounds against progressively stronger foes.

Each phase keeps its own network, optimizer and replay. Production weights are
never overwritten. All three best checkpoints come from the same evaluation.
"""

from __future__ import annotations

import argparse
from collections import deque
import contextlib
import hashlib
import io
import json
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
from gc_facing import FACING_DIRS, facing_towards
from carry_route_priority import current_carry_route_priority
from positioning_gc import REGISTERED_PLANT_CELLS, team_plant_target
from navigation_intent_gc import (
    navigation_intent,
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
)
from ultimate_tactics_gc import tactical_ultimate_window

PHASES = ("carry", "escort", "guard")
VERSIONS = {"carry": 11, "escort": 11, "guard": 4}
FACING_PHASES = ("carry", "escort")
TRAINING_REVISION = "phase_specific_facing_heads_v19_head_only"

CARRY_ROUTE_PRIORITY = current_carry_route_priority()

# Stage-2 navigation shaping. Waiting for a synchronized entry or fighting is
# still valid; an unforced Carry stall becomes increasingly costly late in the
# round, and timing out is materially worse than an ordinary losing tick.
CARRY_QUIET_STALL_PENALTY = 0.08
LATE_ROUND_THRESHOLD = 30
LATE_QUIET_STALL_EXTRA_PENALTY = 0.12
TIMEOUT_PENALTY = 20.0


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
    if hasattr(policy, "facing_head"):
        for parameter in policy.facing_head.parameters():
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
    if not hasattr(policy, "facing_head"):
        raise ValueError("facing-only training requires a factorized facing head")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.facing_head.parameters():
        parameter.requires_grad_(True)


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


def n_step_transitions(rows, gamma, horizon):
    """Back up rewards within one actor/phase segment, never across terminal."""
    by_actor = {}
    for name, row in rows:
        by_actor.setdefault(name, []).append(row)
    for actor_rows in by_actor.values():
        for start, first in enumerate(actor_rows):
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


def observable_teacher_action(phase, obs, mask, controller, context, view=None):
    """Label policy-visited states using only the acting character's IQ view.

    Labels can be trained without executing them. No production controller
    imports or calls this helper, and evaluation collects no demonstrations.
    """
    if context is None:
        return None
    char, state = context
    view = view if view is not None else controller.game
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


def optimize_demonstrations(policy, optimizer, samples, weight, batch_size=64):
    if not samples or weight <= 0:
        return None
    batch = random.sample(list(samples), min(batch_size, len(samples)))
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


def optimize_facing(policy, optimizer, samples, weight, batch_size=64):
    """Train the phase-local facing head without expanding base actions."""
    if not samples or weight <= 0 or not hasattr(policy, "facing_values"):
        return None
    batch = random.sample(list(samples), min(batch_size, len(samples)))
    obs, actions, targets = zip(*batch)
    obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32))
    action_tensor = torch.tensor(actions, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    logits = policy.facing_values(obs_tensor, action_tensor)
    loss = weight * torch.nn.functional.cross_entropy(logits, target_tensor)
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


def entry_quality_passed(metrics, max_no_entry, max_timeout):
    return (
        metrics.get("worst_carry_no_entry_rate", metrics["carry_no_entry_rate"])
        <= max_no_entry
        and metrics.get("worst_timeout_rate", metrics["timeout_rate"]) <= max_timeout
    )


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


def observable_facing_teacher(phase, char, state, controller, goal=None):
    """Return an observable training label; no counterpart runs at inference."""
    chars = state.get("chars", ())
    grid = state["grid"]
    smoke_cells = state.get("smoke_cells", set())
    visible = [
        other
        for other in chars
        if getattr(other, "is_alive", True)
        and other.team != char.team
        and runtime._has_los(grid, char.pos, other.pos, smoke_cells)
    ]
    target = None
    if visible:
        target = min(
            visible,
            key=lambda other: max(
                abs(other.pos[0] - char.pos[0]),
                abs(other.pos[1] - char.pos[1]),
            ),
        ).pos
    elif phase == "carry" and getattr(controller, "_sighting", None) is not None:
        target = controller._sighting["pos"]
    elif goal is not None:
        target = goal
    facing = (
        facing_towards(tuple(char.pos), tuple(target)) if target is not None else None
    )
    if facing is None:
        facing = getattr(char, "facing", "N")
    return FACING_DIRS.index(facing) if facing in FACING_DIRS else 0


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
        self.collect_demonstrations = False
        self.frozen_phases = set()
        self.demonstrations = {phase: [] for phase in PHASES}
        self.ultimate_examples = {phase: [] for phase in PHASES}
        self.facing_examples = {phase: [] for phase in FACING_PHASES}
        self.facing_decisions = {phase: 0 for phase in FACING_PHASES}
        self.facing_teacher_matches = {phase: 0 for phase in FACING_PHASES}
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
            teacher = (
                observable_teacher_action(
                    phase,
                    obs,
                    mask,
                    self.controllers[phase],
                    getattr(self, "decision_context", None),
                    view=view,
                )
                if getattr(self, "collect_demonstrations", False)
                else None
            )
            if teacher is not None:
                self.demonstrations[phase].append((obs.copy(), teacher, mask.copy()))
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
                teacher is not None
                and phase not in getattr(self, "frozen_phases", ())
                and random.random() < self.teacher_probability
            ):
                action = teacher
            elif random.random() < self.epsilon:
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
            teacher = None
            if context is not None:
                char, state = context
                teacher = observable_facing_teacher(
                    phase,
                    char,
                    state,
                    self.controllers[phase],
                    self.action_goals.get(key),
                )
                if getattr(self, "collect_demonstrations", False):
                    self.facing_examples[phase].append(
                        (obs.copy(), int(action), int(teacher))
                    )
            if (
                teacher is not None
                and getattr(self, "collect_demonstrations", False)
                and random.random() < self.teacher_probability
            ):
                index = teacher
            elif random.random() < self.epsilon:
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

    def play(self, seed, stats, epsilon=0.0, training=True, teacher_probability=0.0):
        self.collect_demonstrations = bool(training)
        self.epsilon = epsilon
        self.teacher_probability = teacher_probability if training else 0.0
        self.demonstrations = {phase: [] for phase in PHASES}
        self.ultimate_examples = {phase: [] for phase in PHASES}
        self.facing_examples = {phase: [] for phase in FACING_PHASES}
        self.facing_decisions = {phase: 0 for phase in FACING_PHASES}
        self.facing_teacher_matches = {phase: 0 for phase in FACING_PHASES}
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
        screening_opportunities = screened_opportunities = 0
        formation_opportunities = formation_ready_opportunities = 0
        entry_formation_opportunities = entry_formation_ready_opportunities = 0
        entry_sync_opportunities = entry_sync_holds = 0
        designated_route_block_ticks = 0
        designated_route_clear_opportunities = designated_route_clear_moves = 0
        screen_guidance_opportunities = screen_guidance_follows = 0
        ultimate_ready_ticks = ultimate_uses = tactical_ultimate_uses = 0
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
                    if route_goal is not None and not formation_active:
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
        # Give each participant's last action the actual round outcome, including
        # Carry/Escort whose phases ended at planting, and actors who died earlier.
        won = game.attacker_wins > initial_wins
        for phase, rows in self.transitions.items():
            last = {name: i for i, (name, _) in enumerate(rows)}
            for index in last.values():
                name, row = rows[index]
                obs, action, reward, next_obs, mask, terminal, ticks = row
                rows[index] = (
                    name,
                    (
                        obs,
                        action,
                        reward + (20.0 if won else -20.0),
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
            "ultimate_uses": ultimate_uses,
            "tactical_ultimate_uses": tactical_ultimate_uses,
            "preentry_ultimate_uses": preentry_ultimate_uses,
            "ultimate_uses_by_phase": ultimate_uses_by_phase,
            "tactical_ultimate_uses_by_phase": tactical_ultimate_uses_by_phase,
            "facing_decisions_by_phase": dict(self.facing_decisions),
            "facing_teacher_matches_by_phase": dict(self.facing_teacher_matches),
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
        "combat_stop_rate": sum(r.get("combat_stop_ticks", 0) for r in rows)
        / max(1, sum(r.get("combat_ticks", 0) for r in rows)),
        "postplant_win_rate": sum(r.get("postplant_win", False) for r in rows)
        / max(1, sum(r["planted"] for r in rows)),
    }


def evaluate_multi(session, episodes, seeds, final):
    """Evaluate independent seed blocks and report both mean and worst block."""
    blocks = [evaluate(session, episodes, int(seed), final) for seed in seeds]
    plants = sum(b["episodes"] * b["plant_rate"] for b in blocks)
    postplant_wins = sum(
        b["episodes"] * b["plant_rate"] * b["postplant_win_rate"] for b in blocks
    )
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
    sources = {p: getattr(args, "init_" + p).resolve() for p in PHASES}
    checkpoints = {
        p: torch.load(s, map_location="cpu", weights_only=False)
        for p, s in sources.items()
    }
    hashes = {p: hashlib.sha256(s.read_bytes()).hexdigest() for p, s in sources.items()}
    policies = {
        "carry": runtime.AttackerCarryDuelingDQN(
            obs_dim=runtime.FACING_HEAD_OBS_DIM, action_dim=runtime.ACTION_DIM
        ),
        "escort": escort_runtime.DuelingQNetwork(
            escort_runtime.FACING_HEAD_OBS_DIM, escort_runtime.N_ACTIONS
        ),
        "guard": guard_runtime.AttackerGuardDuelingDQN(
            obs_dim=guard_runtime.ULTIMATE_CONTEXT_OBS_DIM,
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
            if not (phase in FACING_PHASES and key.startswith("facing_head."))
        ]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"{phase} checkpoint keys mismatch: missing={missing}, "
                f"unexpected={list(incompatible.unexpected_keys)}"
            )
    import copy

    targets = {p: copy.deepcopy(net) for p, net in policies.items()}
    feature_columns = {
        "carry": range(runtime.ENTRY_SYNC_OBS_DIM, runtime.FACING_HEAD_OBS_DIM),
        "escort": range(
            escort_runtime.CLEARANCE_OBS_DIM, escort_runtime.FACING_HEAD_OBS_DIM
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
    optimizers = {
        p: torch.optim.Adam(
            [parameter for parameter in net.parameters() if parameter.requires_grad],
            lr=args.lr,
        )
        for p, net in policies.items()
    }
    replays = {p: deque(maxlen=100_000) for p in PHASES}
    demonstrations = {p: deque(maxlen=20_000) for p in PHASES}
    facing_demonstrations = {p: deque(maxlen=30_000) for p in FACING_PHASES}
    ultimate_positives = {p: deque(maxlen=10_000) for p in PHASES}
    ultimate_negatives = {p: deque(maxlen=10_000) for p in PHASES}
    session = CurriculumSession(sources, policies, args.gamma)
    session.frozen_phases = set(args.freeze_phases)
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
    if any(args.output_dir.iterdir()):
        raise ValueError(
            "output-dir must be empty; use a new directory for each training run"
        )
    history, evaluations = [], []
    best = (-1, -float("inf"), -float("inf"), -1, -1, -1, -1)
    started = time.monotonic()

    def save(kind, episode, metrics):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "episode": episode,
            "evaluation": metrics,
            "models": {},
            "macro_source": macro_source,
        }
        for phase in PHASES:
            payload = (
                dict(checkpoints[phase])
                if "model_state_dict" in checkpoints[phase]
                else {}
            )
            payload.update(
                model_state_dict=policies[phase].state_dict(),
                obs_dim=policies[phase].feature[0].in_features,
                n_actions=policies[phase].advantage_head[-1].out_features,
                positioning_version=VERSIONS[phase],
                episode=episode,
                training_environment="actual_engine_full_round_curriculum",
                training_revision=TRAINING_REVISION,
                evaluation=metrics,
                source_models={p: str(s) for p, s in sources.items()},
                source_hashes=hashes,
                macro_source=macro_source,
                training_parameters=vars_for_json(args),
            )
            if phase in FACING_PHASES:
                payload["facing_head_version"] = runtime.FACING_HEAD_VERSION
            filename = f"dqn_attacker_{phase}_gc_{kind}.pt"
            torch.save(payload, args.output_dir / filename)
            bundle["models"][phase] = filename
        (args.output_dir / (kind + "_bundle.json")).write_text(
            json.dumps(bundle, indent=2), encoding="utf-8"
        )

    for episode in range(args.episodes + 1):
        global_episode = args.episode_offset + episode
        if episode:
            stats = opponent_stats(
                global_episode,
                args.curriculum_episodes,
                args.start_stats,
                args.final_stats,
            )
            # Expose the policy to the final opponent before the curriculum is
            # complete, avoiding a late collapse at the final-strength eval.
            if (
                global_episode > args.curriculum_episodes // 2
                and random.random() < args.final_mix
            ):
                stats = tuple(args.final_stats)
            # Keep exploration available when the opponent reaches full strength.
            epsilon = max(
                0.03,
                0.20 * (1.0 - (global_episode - 1) / max(1, args.curriculum_episodes)),
            )
            teacher_probability = (
                0.80
                * max(
                    0.0, 1.0 - (global_episode - 1) / args.navigation_bootstrap_episodes
                )
                if args.navigation_bootstrap_episodes
                else 0.0
            )
            row = session.play(
                args.seed + global_episode,
                stats,
                epsilon,
                teacher_probability=teacher_probability,
            )
            row["episode"] = global_episode
            row["navigation_teacher_probability"] = teacher_probability
            row["demonstration_counts"] = {
                p: len(session.demonstrations[p]) for p in PHASES
            }
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
                    facing_demonstrations[phase].extend(
                        session.facing_examples[phase]
                    )
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
                replays[phase].extend(
                    n_step_transitions(transitions, args.gamma, args.n_step)
                )
                demonstrations[phase].extend(session.demonstrations[phase])
                if phase in FACING_PHASES:
                    facing_demonstrations[phase].extend(session.facing_examples[phase])
                for obs, mask, tactical in session.ultimate_examples[phase]:
                    destination = ultimate_positives if tactical else ultimate_negatives
                    destination[phase].append((obs, mask))
                for _ in range(min(args.max_updates, max(1, len(transitions) // 2))):
                    optimize(
                        policies[phase],
                        targets[phase],
                        optimizers[phase],
                        replays[phase],
                        args.gamma,
                    )
                    optimize_demonstrations(
                        policies[phase],
                        optimizers[phase],
                        demonstrations[phase],
                        args.navigation_demo_weight
                        * max(teacher_probability, args.navigation_retention_weight),
                    )
                    if phase in FACING_PHASES:
                        optimize_facing(
                            policies[phase],
                            optimizers[phase],
                            facing_demonstrations[phase],
                            args.facing_supervision_weight,
                        )
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
                    f"demo={row['navigation_demo_multiplier']:.3f} "
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
            metrics = evaluate_multi(
                session, args.eval_episodes, args.eval_seeds, args.final_stats
            )
            metrics["episode"] = global_episode
            metrics["entry_quality_passed"] = entry_quality_passed(
                metrics, args.max_no_entry_rate, args.max_timeout_rate
            )
            evaluations.append(metrics)
            score = selection_score(
                metrics, args.max_no_entry_rate, args.max_timeout_rate
            )
            save("latest", global_episode, metrics)
            if score > best:
                best = score
                save("best_by_eval", global_episode, metrics)
            print("[FINAL-STRENGTH EVAL] " + json.dumps(metrics), flush=True)
            (args.output_dir / "training_history.json").write_text(
                json.dumps(history), encoding="utf-8"
            )
            (args.output_dir / "evaluation_history.json").write_text(
                json.dumps(evaluations, indent=2), encoding="utf-8"
            )
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
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-updates", type=int, default=64)
    parser.add_argument("--final-mix", type=float, default=0.20)
    parser.add_argument("--max-no-entry-rate", type=float, default=0.25)
    parser.add_argument("--max-timeout-rate", type=float, default=0.10)
    parser.add_argument(
        "--navigation-bootstrap-episodes",
        type=int,
        default=500,
        help="fade training-only quiet navigation demonstrations to zero; eval is always policy-only",
    )
    parser.add_argument("--navigation-demo-weight", type=float, default=1.0)
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
    if not 0 <= args.curriculum_episodes < total_episodes:
        parser.error(
            "curriculum-episodes must be >= 0 and smaller than total episodes (offset + additional episodes)"
        )
    if (
        args.navigation_bootstrap_episodes < 0
        or args.navigation_demo_weight < 0
        or args.ultimate_classification_weight < 0
        or args.facing_supervision_weight < 0
        or not 0 <= args.navigation_retention_weight <= 1
        or args.n_step < 1
    ):
        parser.error(
            "navigation bootstrap counts/weights must be nonnegative and n-step positive"
        )
    if args.navigation_bootstrap_episodes >= total_episodes:
        parser.error(
            "navigation-bootstrap-episodes must be smaller than total episodes (reserve policy-only training)"
        )
    if (
        not 0 < args.gamma <= 1
        or args.lr <= 0
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
