"""Train Ghost Champions Defender Opening Macro with real VisualFPSBattle matches.

学習対象
--------
1) Selection:
   Smoke / Flash / Recon それぞれについて
   NONE または登録済みPatternを選択する。

2) Execution:
   実行可能なOpening planに対して
   WAIT / EXECUTE / CANCEL を選択する。

設計思想
--------
- 簡易専用シミュレータではなく、実際の VisualFPSBattle をheadlessで回す。
- Search Controllerを下位方策として使い、Opening Macroだけを学習する。
- ラウンド勝敗を主報酬にし、アビリティ実行・無駄撃ち・キャンセルなどを補助報酬にする。
- Smoke/Flash/Reconは独立に選べるため、複数定石を同一ラウンドで組み合わせ可能。
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from collections import deque, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from controllers import DefaultAttackerController, DefaultDefenderController

try:
    from attacker_v3.multi_role_attacker_controller import MultiRoleAttackerController
    from defender_v3.multi_role_defender_controller import MultiRoleDefenderController
except Exception:
    MultiRoleAttackerController = None
    MultiRoleDefenderController = None
from map_data import NEW_MAZE_STR
from roster_utils import build_two_balanced_rosters
from party_presets import get_preset
from run_game import VisualFPSBattle
from team_ai import DualRoleTeamAI

from learning_defender_opening_macro_gc import (
    ABILITY_ORDER,
    OBS_DIM,
    EXEC_WAIT,
    EXECUTE,
    EXEC_CANCEL,
    EXEC_ACTION_DIM,
    LearningDefenderOpeningMacroGCController,
    _ability_charge,
    _bfs_dist_map,
)

try:
    from learning_defender_search_gc import LearningDefenderSearchGCController
except Exception:
    LearningDefenderSearchGCController = None


# This trainer lives in gc_v1/. Resolve GC phase models relative to this file,
# never relative to the current working directory (03.game).
_GC_DIR = Path(__file__).resolve().parent
_SEARCH_MODEL_CANDIDATES = (
    _GC_DIR / "data" / "defender_search_gc_data" / "dqn_defender_search_gc_best_by_eval.pt",
    _GC_DIR / "data" / "defender_search_gc_data" / "dqn_defender_search_gc_latest.pt",
    _GC_DIR / "data" / "defender_search_gc_data" / "dqn_defender_search_gc_final.pt",
)


def _first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.is_file():
            return path
    return None


# ============================================================================
# Paths / hyperparameters
# ============================================================================

MODEL_DIR = Path("gc_v1/data/defender_opening_macro_gc_data")
BEST_MODEL = MODEL_DIR / "dqn_defender_opening_macro_gc_best.pt"
LATEST_MODEL = MODEL_DIR / "dqn_defender_opening_macro_gc_latest.pt"
FINAL_MODEL = MODEL_DIR / "dqn_defender_opening_macro_gc_final.pt"
INTERRUPT_MODEL = MODEL_DIR / "dqn_defender_opening_macro_gc_interrupt.pt"
LOG_FILE = MODEL_DIR / "training_log.jsonl"

GAMMA = 0.985
LR = 2.5e-4
BATCH_SIZE = 128
REPLAY_CAPACITY = 120_000

# SelectionとExecutionでは経験生成量が大きく違うため、別々にwarmupする。
SELECTION_LEARNING_STARTS = 1_000
EXECUTION_LEARNING_STARTS = 400

TRAIN_EVERY = 4
TARGET_UPDATE_EVERY = 1_000

EPS_START = 0.90
EPS_END = 0.05
EPS_DECAY_STEPS = 120_000

# 主報酬
ROUND_WIN_REWARD = 20.0
ROUND_LOSS_REWARD = -20.0

# 補助報酬。勝敗より小さくする。
ABILITY_EXECUTE_REWARD = 0.30
PLAN_CANCEL_PENALTY = -0.15
OPENING_TIMEOUT_PENALTY = -0.25
UNUSED_PLAN_PENALTY = -0.10

# Execution Head専用報酬。
# 勝敗報酬から切り離し、「Openingを実行できたか」を直接学習する。
WAIT_COST = -0.03
EXECUTION_SUCCESS_REWARD = 2.00
EXECUTION_TIMEOUT_PENALTY = -2.00
EXECUTION_TACTICAL_CANCEL_REWARD = 0.00
EXECUTION_OTHER_CANCEL_PENALTY = -0.25

# Execution Head はSelectionより経験が少なくなりやすい。
# 学習初期はEXECUTEを十分に試させ、CANCEL偏重を避ける。
EXEC_RANDOM_WAIT_P = 0.25
EXEC_RANDOM_EXECUTE_P = 0.55
EXEC_RANDOM_CANCEL_P = 0.20

# Execution replay は1 decisionだけでなく、各active planごとのdecisionを記録する。
# これにより Smoke / Flash / Recon の複数planを同一ラウンドで学習できる。
EXECUTE_BONUS = 0.15

# 1 match = 最大13勝先取なので、学習の単位は内部ラウンド。
EVAL_MATCHES = 8


# ============================================================================
# Networks
# ============================================================================

class OpeningSelectionQNet(nn.Module):
    """Shared trunk + ability-specific selection heads."""

    def __init__(self, obs_dim: int, action_dims: dict[str, int]):
        super().__init__()
        self.action_dims = dict(action_dims)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({
            ability: nn.Linear(128, int(dim))
            for ability, dim in self.action_dims.items()
        })

    def forward(self, x):
        h = self.trunk(x)
        return {ability: head(h) for ability, head in self.heads.items()}


class OpeningExecutionQNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, action_dim=EXEC_ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Replay
# ============================================================================

@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: float
    ability: str | None = None


class ReplayBuffer:
    def __init__(self, capacity):
        self.data = deque(maxlen=int(capacity))

    def push(self, *args):
        self.data.append(Transition(*args))

    def sample(self, n):
        idx = np.random.choice(len(self.data), size=n, replace=False)
        return [self.data[i] for i in idx]

    def __len__(self):
        return len(self.data)


# ============================================================================
# Training controller
# ============================================================================

class TrainableOpeningMacro(LearningDefenderOpeningMacroGCController):
    """Opening runtime with external trainable policies and rollout recording."""

    def __init__(
        self,
        selection_net,
        execution_net,
        device,
        *,
        epsilon=0.0,
        training=True,
        verbose=False,
        seed=0,
    ):
        super().__init__(model_path=None, greedy=True, verbose=verbose, seed=seed)
        self.selection_net = selection_net
        self.execution_net = execution_net
        self.device = device
        self.epsilon = float(epsilon)
        self.training_mode = bool(training)

        # ability -> stable action catalog: [None, pattern2, ...]
        self.selection_catalog = {
            ability: [None] + sorted(self.patterns.get(ability, {}).keys())
            for ability in ABILITY_ORDER
        }

        self.round_selection_steps = []
        self.round_execution_steps = []
        self.round_aux_reward = 0.0
        self.round_initialized_policy = False
        self.round_plan_failures = []

    def reset_round(self):
        # Parent reset may run during __init__ before our custom fields exist.
        super().reset_round()
        self.round_selection_steps = []
        self.round_execution_steps = []
        self.round_aux_reward = 0.0
        self.round_initialized_policy = False
        self.round_plan_failures = []

    def _epsilon_action(self, q_values):
        if self.training_mode and random.random() < self.epsilon:
            return random.randrange(len(q_values))
        return int(np.argmax(q_values))

    def _choose_selection_action(self, ability, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.selection_net(x)[ability][0].cpu().numpy()
        return self._epsilon_action(q)

    def _diagnose_plan_failure(self, ability, pattern_id, game_state):
        ability = str(ability).upper()
        pattern = self.patterns.get(ability, {}).get(int(pattern_id))

        diagnosis = {
            "ability": ability,
            "pattern_id": int(pattern_id),
            "pattern_exists": pattern is not None,
            "origins": [],
            "targets": [],
            "caster_candidates": [],
            "reason": None,
        }

        if pattern is None:
            diagnosis["reason"] = "PATTERN_NOT_FOUND"
            return diagnosis

        diagnosis["origins"] = [tuple(map(int, p)) for p in pattern.get("origins", [])]
        diagnosis["targets"] = [tuple(map(int, p)) for p in pattern.get("targets", [])]

        if not diagnosis["targets"]:
            diagnosis["reason"] = "NO_TARGETS"
            return diagnosis

        candidates = []
        for c in game_state.get("chars", []):
            if getattr(c, "team", None) != "D":
                continue
            if not getattr(c, "is_alive", True):
                continue

            cname = str(getattr(c, "name", ""))
            ability_name = str(getattr(c, "ability_name", "")).upper()
            charge = _ability_charge(c, ability)

            entry = {
                "name": cname,
                "ability_name": ability_name,
                "charge": int(charge),
                "pos": tuple(map(int, c.pos)),
                "matches_ability": ability_name == ability,
            }

            if ability != "SMOKE" and diagnosis["origins"]:
                try:
                    dist_map = _bfs_dist_map(
                        game_state["grid"],
                        diagnosis["origins"],
                    )
                    r, col = map(int, c.pos)
                    entry["origin_distance"] = int(dist_map[r, col])
                except Exception as exc:
                    entry["origin_distance"] = None
                    entry["origin_distance_error"] = str(exc)

            candidates.append(entry)

        diagnosis["caster_candidates"] = candidates

        matching = [
            c for c in candidates
            if c["matches_ability"]
        ]
        if not matching:
            diagnosis["reason"] = "NO_MATCHING_ABILITY_CASTER"
            return diagnosis

        charged = [
            c for c in matching
            if c["charge"] > 0
        ]
        if not charged:
            diagnosis["reason"] = "NO_CHARGE"
            return diagnosis

        if ability != "SMOKE":
            if not diagnosis["origins"]:
                diagnosis["reason"] = "NO_ORIGINS"
                return diagnosis

            reachable = [
                c for c in charged
                if c.get("origin_distance") is not None
                and c.get("origin_distance", -1) >= 0
            ]
            if not reachable:
                diagnosis["reason"] = "ORIGIN_UNREACHABLE"
                return diagnosis

        diagnosis["reason"] = "UNKNOWN_SET_PLAN_FAILURE"
        return diagnosis

    def _print_plan_failure(self, diagnosis):
        print(
            "[OPENING PLAN FAIL] "
            f"ability={diagnosis.get('ability')} "
            f"pattern={diagnosis.get('pattern_id')} "
            f"reason={diagnosis.get('reason')}"
        )
        print(
            "  origins="
            f"{diagnosis.get('origins')} "
            "targets="
            f"{diagnosis.get('targets')}"
        )
        for caster in diagnosis.get("caster_candidates", []):
            print(
                "  caster="
                f"{caster.get('name')} "
                f"ability_name={caster.get('ability_name')} "
                f"charge={caster.get('charge')} "
                f"pos={caster.get('pos')} "
                f"matches={caster.get('matches_ability')} "
                f"origin_distance={caster.get('origin_distance', '-')}"
            )

    def _policy_execution_action(self, game_state):
        obs = self.build_observation(game_state)
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.execution_net(x)[0].cpu().numpy()

        if self.training_mode and random.random() < self.epsilon:
            # Uniform randomだと未学習時にCANCELへ偏るケースがあるため、
            # 初期探索では実際にAbilityを試すEXECUTEをやや多めにする。
            action = random.choices(
                [EXEC_WAIT, EXECUTE, EXEC_CANCEL],
                weights=[
                    EXEC_RANDOM_WAIT_P,
                    EXEC_RANDOM_EXECUTE_P,
                    EXEC_RANDOM_CANCEL_P,
                ],
                k=1,
            )[0]
        else:
            action = int(np.argmax(q))

        # 評価時もWAIT/EXECUTE/CANCELの実際の選択を記録する。
        active_abilities = [
            ability
            for ability, plan in self.plans.items()
            if plan.active
        ]
        self.round_execution_steps.append({
            "obs": obs.copy(),
            "action": int(action),
            "active_abilities": tuple(active_abilities),
        })

        # Execution action rewards are assigned in finalize_round() per ability.
        # Do not mix them into round_aux_reward, because that reward is also
        # consumed by Selection transitions.
        return action

    def _initialize_round_policy(self, game_state):
        if self.round_initialized_policy:
            return
        self.round_initialized_policy = True

        obs = self.build_observation(game_state)

        # independent heads => Smoke + Recon etc. can coexist.
        # GC固定ロスターにそのAbility持ちが存在しない場合（現在はFLASH）は
        # そもそも選択/学習対象にしない。
        for ability in ABILITY_ORDER:
            catalog = self.selection_catalog[ability]
            if len(catalog) <= 1:
                continue

            feasible_casters = self._alive_defenders_with_ability(
                ability,
                game_state,
            )
            if not feasible_casters:
                continue

            action = self._choose_selection_action(ability, obs)
            pid = catalog[action]

            # 評価時も行動統計を残す。Replayへ入れるかどうかは呼び出し側で分離。
            self.round_selection_steps.append({
                "ability": ability,
                "obs": obs.copy(),
                "action": int(action),
            })

            if pid is not None:
                ok = self.set_plan(ability, int(pid), game_state)
                if not ok:
                    diagnosis = self._diagnose_plan_failure(
                        ability,
                        int(pid),
                        game_state,
                    )
                    self.round_plan_failures.append(diagnosis)
                    self._print_plan_failure(diagnosis)
                    if self.training_mode:
                        self.round_aux_reward -= 0.5

    def coordinate(self, char, game_state, base_result):
        self._initialize_round_policy(game_state)

        before = {
            a: (
                self.plans[a].state
                if a in self.plans else None
            )
            for a in ABILITY_ORDER
        }

        result = super().coordinate(char, game_state, base_result)

        after = {
            a: (
                self.plans[a].state
                if a in self.plans else None
            )
            for a in ABILITY_ORDER
        }

        if self.training_mode:
            for ability in ABILITY_ORDER:
                if before[ability] != "EXECUTED" and after[ability] == "EXECUTED":
                    self.round_aux_reward += ABILITY_EXECUTE_REWARD
                if (
                    before[ability] not in {"CANCELLED", "EXECUTED"}
                    and after[ability] == "CANCELLED"
                    and self.plans[ability].cancel_reason == "OPENING_TIMEOUT"
                ):
                    self.round_aux_reward += OPENING_TIMEOUT_PENALTY

        return result

    def finalize_round(self, defender_won: bool, terminal_obs=None):
        """Convert one real game round into replay-ready transitions."""
        terminal_reward = (
            ROUND_WIN_REWARD if defender_won else ROUND_LOSS_REWARD
        ) + self.round_aux_reward

        # Chosen but never executed plans receive a tiny penalty.
        for plan in self.plans.values():
            if plan.state not in {"EXECUTED", "CANCELLED"}:
                terminal_reward += UNUSED_PLAN_PENALTY

        if terminal_obs is None:
            terminal_obs = self._last_obs
        if terminal_obs is None:
            terminal_obs = np.zeros(OBS_DIM, dtype=np.float32)

        selections = []
        for step in self.round_selection_steps:
            selections.append(
                Transition(
                    obs=step["obs"],
                    action=step["action"],
                    reward=float(terminal_reward),
                    next_obs=terminal_obs.copy(),
                    done=1.0,
                    ability=step["ability"],
                )
            )

        executions = []
        # Execution Headはラウンド勝敗から切り離す。
        # Searchだけで勝ったラウンドがWAITを強化しないよう、各Ability Planの
        # 最終結果（実行成功 / timeout / 戦術キャンセル）を直接教師信号にする。
        tactical_cancel_reasons = {
            "DROPPED_SPIKE",
            "MAIN_FORCE_X3",
            "MAIN_FORCE_X4",
            "MAIN_FORCE_X5",
        }

        for step in self.round_execution_steps:
            action = int(step["action"])
            active_abilities = tuple(step.get("active_abilities") or ())

            for ability in active_abilities:
                plan = self.plans.get(ability)
                if plan is None:
                    continue

                # Outcome reward: this is the dominant signal.
                if plan.state == "EXECUTED":
                    outcome_reward = EXECUTION_SUCCESS_REWARD
                elif plan.state == "CANCELLED":
                    reason = plan.cancel_reason or "UNKNOWN"
                    if reason == "OPENING_TIMEOUT":
                        outcome_reward = EXECUTION_TIMEOUT_PENALTY
                    elif reason in tactical_cancel_reasons:
                        outcome_reward = EXECUTION_TACTICAL_CANCEL_REWARD
                    else:
                        outcome_reward = EXECUTION_OTHER_CANCEL_PENALTY
                else:
                    outcome_reward = UNUSED_PLAN_PENALTY

                # Small action-local shaping. EXECUTE is rewarded only when the
                # plan actually ended EXECUTED; WAIT always carries time cost.
                local_reward = 0.0
                if action == EXEC_WAIT:
                    local_reward += WAIT_COST
                elif action == EXEC_CANCEL:
                    local_reward += PLAN_CANCEL_PENALTY
                elif action == EXECUTE and plan.state == "EXECUTED":
                    local_reward += EXECUTE_BONUS

                executions.append(
                    Transition(
                        obs=step["obs"],
                        action=action,
                        reward=float(outcome_reward + local_reward),
                        next_obs=terminal_obs.copy(),
                        done=1.0,
                        ability=ability,
                    )
                )

        info = {
            "reward": float(terminal_reward),
            "defender_won": bool(defender_won),
            "selection_actions": {
                step["ability"]: int(step["action"])
                for step in self.round_selection_steps
            },
            "execution_actions": [
                int(step["action"]) for step in self.round_execution_steps
            ],
            "plan_failures": list(self.round_plan_failures),
            "plans": {
                a: {
                    "pattern_id": p.pattern_id,
                    "state": p.state,
                    "cancel_reason": p.cancel_reason,
                    "executed_tick": p.executed_tick,
                }
                for a, p in self.plans.items()
            },
        }
        return selections, executions, info


class OpeningWrappedDefender(DefaultDefenderController):
    """Search base controller + trainable Opening Macro above it."""

    def __init__(self, macro):
        super().__init__()
        self.macro = macro

        self.search = None
        if LearningDefenderSearchGCController is not None:
            search_model = _first_existing(_SEARCH_MODEL_CANDIDATES)
            if search_model is None:
                print(
                    "[OPENING TRAIN][WARN] GC Search checkpoint not found; "
                    "falling back to DefaultDefenderController. Searched: "
                    + ", ".join(str(p) for p in _SEARCH_MODEL_CANDIDATES)
                )
            else:
                try:
                    sig = inspect.signature(LearningDefenderSearchGCController)
                    kwargs = {}
                    if "model_path" in sig.parameters:
                        kwargs["model_path"] = str(search_model)
                    if "greedy" in sig.parameters:
                        kwargs["greedy"] = True
                    self.search = LearningDefenderSearchGCController(**kwargs)
                    print(
                        "[OPENING TRAIN] loaded GC Search: "
                        f"{search_model}"
                    )
                except Exception as exc:
                    print(
                        "[OPENING TRAIN][WARN] failed to load GC Search "
                        f"{search_model}: {exc}"
                    )
                    self.search = None

    def set_game(self, game):
        self.game = game
        self.macro.set_game(game)
        if self.search is not None and hasattr(self.search, "set_game"):
            self.search.set_game(game)

    def reset_round(self):
        self.macro.reset_round()
        if self.search is not None and hasattr(self.search, "reset_round"):
            self.search.reset_round()

    def decide_move(self, char, game_state):
        if game_state.get("is_planted"):
            # Opening ends at plant. For training, default retake fallback is enough;
            # GC retake can be inserted later without changing the Macro interface.
            return super().decide_move(char, game_state)

        if self.search is not None:
            base = self.search.decide_move(char, game_state)
        else:
            base = super().decide_move(char, game_state)

        return self.macro.coordinate(char, game_state, base)


# ============================================================================
# Helpers
# ============================================================================

def epsilon_by_step(step):
    frac = min(1.0, step / max(1, EPS_DECAY_STEPS))
    return EPS_START + frac * (EPS_END - EPS_START)


def _select_action_dims(patterns):
    return {
        ability: 1 + len(patterns.get(ability, {}))
        for ability in ABILITY_ORDER
    }


def optimize_selection(net, target, optimizer, replay, device):
    if len(replay) < max(BATCH_SIZE, SELECTION_LEARNING_STARTS):
        return None

    batch = replay.sample(BATCH_SIZE)
    losses = []

    optimizer.zero_grad(set_to_none=True)

    # Transitions belong to different heads, so optimize head by head.
    for ability in ABILITY_ORDER:
        subset = [t for t in batch if t.ability == ability]
        if not subset:
            continue

        obs = torch.as_tensor(
            np.stack([t.obs for t in subset]),
            dtype=torch.float32,
            device=device,
        )
        act = torch.tensor([t.action for t in subset], dtype=torch.long, device=device)
        rew = torch.tensor([t.reward for t in subset], dtype=torch.float32, device=device)
        nxt = torch.as_tensor(
            np.stack([t.next_obs for t in subset]),
            dtype=torch.float32,
            device=device,
        )
        done = torch.tensor([t.done for t in subset], dtype=torch.float32, device=device)

        q = net(obs)[ability].gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_online = net(nxt)[ability]
            next_act = next_online.argmax(dim=1)
            next_q = target(nxt)[ability].gather(1, next_act.unsqueeze(1)).squeeze(1)
            y = rew + GAMMA * (1.0 - done) * next_q

        losses.append(F.smooth_l1_loss(q, y))

    if not losses:
        return None

    loss = torch.stack(losses).mean()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())


def optimize_execution(net, target, optimizer, replay, device):
    if len(replay) < max(BATCH_SIZE, EXECUTION_LEARNING_STARTS):
        return None

    batch = replay.sample(BATCH_SIZE)

    obs = torch.as_tensor(
        np.stack([t.obs for t in batch]), dtype=torch.float32, device=device
    )
    act = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
    rew = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
    nxt = torch.as_tensor(
        np.stack([t.next_obs for t in batch]), dtype=torch.float32, device=device
    )
    done = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

    q = net(obs).gather(1, act.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_online = net(nxt)
        next_act = next_online.argmax(dim=1)
        next_q = target(nxt).gather(1, next_act.unsqueeze(1)).squeeze(1)
        y = rew + GAMMA * (1.0 - done) * next_q

    loss = F.smooth_l1_loss(q, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())


def _build_training_attacker(opponent_key):
    key = str(opponent_key or "default").strip().lower()

    if key == "default":
        return DefaultAttackerController()

    if key == "toru_ai_v3":
        if MultiRoleAttackerController is None:
            raise RuntimeError(
                "Toru AI v3 attacker controller could not be imported. "
                "Check the module that defines MultiRoleAttackerController."
            )
        return MultiRoleAttackerController()

    raise ValueError(f"Unknown training opponent: {opponent_key}")


def _resolve_opponent(opponent_mode, rng):
    mode = str(opponent_mode or "default").strip().lower()

    if mode in {"default", "toru_ai_v3"}:
        return mode

    if mode == "mixed":
        # Start with a simple robust mixture:
        # 70% Toru AI v3, 30% Default.
        return "toru_ai_v3" if rng.random() < 0.70 else "default"

    raise ValueError(
        "--opponent must be one of: default, toru_ai_v3, mixed"
    )


def _ghost_champions_roster():
    preset = get_preset("Ghost Champions")
    if preset is None:
        raise RuntimeError(
            'party_presets.py に "Ghost Champions" プリセットが見つかりません'
        )
    players = list(preset.players)
    if len(players) != 5:
        raise RuntimeError(
            f"Ghost Champions roster must contain 5 players: {players}"
        )
    return players


def make_game(defender_controller, opponent_key="default"):
    attacker_roster, _unused_defender_roster = build_two_balanced_rosters()
    defender_roster = _ghost_champions_roster()

    attacker_team = DualRoleTeamAI(
        f"OpeningMacroTrain-A[{opponent_key}]",
        attacker_factory=lambda: _build_training_attacker(opponent_key),
        defender_factory=lambda: DefaultDefenderController(),
        use_iq_perception=False,
    )
    defender_team = DualRoleTeamAI(
        "OpeningMacroTrain-D",
        attacker_factory=lambda: DefaultAttackerController(),
        defender_factory=lambda: defender_controller,
        use_iq_perception=False,
    )

    kwargs: dict[str, Any] = {
        "headless": True,
        "attacker_roster": attacker_roster,
        "defender_roster": defender_roster,
    }
    if "disable_side_swap" in inspect.signature(VisualFPSBattle.__init__).parameters:
        kwargs["disable_side_swap"] = True

    game = VisualFPSBattle(
        NEW_MAZE_STR,
        attacker_team,
        defender_team,
        **kwargs,
    )
    defender_controller.set_game(game)
    return game


def save_checkpoint(
    path,
    selection,
    execution,
    selection_target,
    execution_target,
    selection_optimizer,
    execution_optimizer,
    *,
    episode,
    global_step,
    best_win_rate,
    action_dims,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "gc_defender_opening_macro_dqn_v1",
        "obs_dim": OBS_DIM,
        "action_dims": action_dims,
        "execution_action_dim": EXEC_ACTION_DIM,
        "selection_state_dict": selection.state_dict(),
        "execution_state_dict": execution.state_dict(),
        "selection_target_state_dict": selection_target.state_dict(),
        "execution_target_state_dict": execution_target.state_dict(),
        "selection_optimizer_state_dict": selection_optimizer.state_dict(),
        "execution_optimizer_state_dict": execution_optimizer.state_dict(),
        "episode": int(episode),
        "global_step": int(global_step),
        "best_win_rate": float(best_win_rate),
    }, path)


def load_checkpoint(
    path,
    selection,
    execution,
    selection_target,
    execution_target,
    selection_optimizer,
    execution_optimizer,
    device,
):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if int(ckpt.get("obs_dim", -1)) != OBS_DIM:
        raise ValueError("checkpoint OBS_DIM mismatch")

    selection.load_state_dict(ckpt["selection_state_dict"])
    execution.load_state_dict(ckpt["execution_state_dict"])
    selection_target.load_state_dict(
        ckpt.get("selection_target_state_dict", ckpt["selection_state_dict"])
    )
    execution_target.load_state_dict(
        ckpt.get("execution_target_state_dict", ckpt["execution_state_dict"])
    )

    if "selection_optimizer_state_dict" in ckpt:
        selection_optimizer.load_state_dict(ckpt["selection_optimizer_state_dict"])
    if "execution_optimizer_state_dict" in ckpt:
        execution_optimizer.load_state_dict(ckpt["execution_optimizer_state_dict"])

    return (
        int(ckpt.get("episode", 0)),
        int(ckpt.get("global_step", 0)),
        float(ckpt.get("best_win_rate", -1.0)),
    )


# ============================================================================
# One actual match
# ============================================================================

def run_match(
    selection_net,
    execution_net,
    device,
    epsilon,
    training,
    seed,
    opponent_mode="default",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    macro = TrainableOpeningMacro(
        selection_net,
        execution_net,
        device,
        epsilon=epsilon,
        training=training,
        verbose=False,
        seed=seed,
    )
    defender = OpeningWrappedDefender(macro)
    opponent_rng = random.Random(seed + 12345)
    opponent_key = _resolve_opponent(opponent_mode, opponent_rng)
    game = make_game(defender, opponent_key=opponent_key)

    # To learn per-round reward without invasive game edits, intercept reset_round.
    # Battle code calls reset_round after score update. Store previous scoreboard.
    round_records = []
    prev_a = int(game.attacker_wins)
    prev_d = int(game.defender_wins)

    original_reset = defender.reset_round

    def reset_and_record():
        nonlocal prev_a, prev_d

        now_a = int(game.attacker_wins)
        now_d = int(game.defender_wins)

        if now_a != prev_a or now_d != prev_d:
            defender_won = now_d > prev_d
            sels, execs, info = macro.finalize_round(defender_won)
            round_records.append((sels, execs, info))
            prev_a, prev_d = now_a, now_d

        original_reset()

    defender.reset_round = reset_and_record

    game.run_headless_loop()

    # Some versions finish match without another reset_round after final score.
    now_a = int(game.attacker_wins)
    now_d = int(game.defender_wins)
    if now_a != prev_a or now_d != prev_d:
        defender_won = now_d > prev_d
        sels, execs, info = macro.finalize_round(defender_won)
        round_records.append((sels, execs, info))

    return {
        "attacker_wins": int(game.attacker_wins),
        "defender_wins": int(game.defender_wins),
        "match_win": int(game.defender_wins > game.attacker_wins),
        "round_records": round_records,
        "opponent": opponent_key,
    }


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(selection, execution, device, matches=EVAL_MATCHES, seed=100000, opponent_mode="default"):
    selection.eval()
    execution.eval()

    match_wins = 0
    rounds_a = 0
    rounds_d = 0
    rewards = []
    pattern_counts = Counter()
    states = Counter()
    opponent_counts = Counter()
    selection_counts = {
        ability: Counter()
        for ability in ABILITY_ORDER
    }
    execution_counts = Counter()
    plan_failure_counts = Counter()
    ability_state_counts = {
        ability: Counter()
        for ability in ABILITY_ORDER
    }
    ability_cancel_reason_counts = {
        ability: Counter()
        for ability in ABILITY_ORDER
    }

    for i in range(matches):
        result = run_match(
            selection,
            execution,
            device,
            epsilon=0.0,
            training=False,
            seed=seed + i,
            opponent_mode=opponent_mode,
        )
        match_wins += result["match_win"]
        rounds_a += result["attacker_wins"]
        rounds_d += result["defender_wins"]
        opponent_counts[result.get("opponent", "unknown")] += 1

        for _, _, info in result["round_records"]:
            rewards.append(info["reward"])

            for ability, action in info.get("selection_actions", {}).items():
                selection_counts[ability][int(action)] += 1

            for action in info.get("execution_actions", []):
                execution_counts[int(action)] += 1

            for failure in info.get("plan_failures", []):
                key = (
                    f"{failure.get('ability')}_"
                    f"{failure.get('pattern_id')}_"
                    f"{failure.get('reason')}"
                )
                plan_failure_counts[key] += 1

            for ability, p in info["plans"].items():
                pattern_counts[f"{ability}_{p['pattern_id']}"] += 1
                states[p["state"]] += 1

                ability_state_counts[ability][p["state"]] += 1

                if p["state"] == "CANCELLED":
                    reason = p.get("cancel_reason") or "UNKNOWN"
                    ability_cancel_reason_counts[ability][reason] += 1

    selection.train()
    execution.train()

    return {
        "win_rate": match_wins / max(1, matches),
        "avg_rounds_a": rounds_a / max(1, matches),
        "avg_rounds_d": rounds_d / max(1, matches),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "patterns": dict(pattern_counts),
        "plan_states": dict(states),
        "selection_actions": {
            ability: dict(counter)
            for ability, counter in selection_counts.items()
        },
        "execution_actions": {
            "WAIT": int(execution_counts.get(EXEC_WAIT, 0)),
            "EXECUTE": int(execution_counts.get(EXECUTE, 0)),
            "CANCEL": int(execution_counts.get(EXEC_CANCEL, 0)),
        },
        "plan_failures": dict(plan_failure_counts),
        "ability_outcomes": {
            ability: {
                "EXECUTED": int(ability_state_counts[ability].get("EXECUTED", 0)),
                "CANCELLED": int(ability_state_counts[ability].get("CANCELLED", 0)),
                "PLANNED": int(ability_state_counts[ability].get("PLANNED", 0)),
                "MOVING": int(ability_state_counts[ability].get("MOVING", 0)),
                "READY": int(ability_state_counts[ability].get("READY", 0)),
            }
            for ability in ABILITY_ORDER
        },
        "cancel_reasons": {
            ability: dict(ability_cancel_reason_counts[ability])
            for ability in ABILITY_ORDER
        },
        "opponents": dict(opponent_counts),
    }


# ============================================================================
# Train
# ============================================================================

def train(args):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    # Probe runtime patterns to establish head sizes.
    probe = LearningDefenderOpeningMacroGCController()
    action_dims = _select_action_dims(probe.patterns)

    selection = OpeningSelectionQNet(OBS_DIM, action_dims).to(device)
    selection_target = OpeningSelectionQNet(OBS_DIM, action_dims).to(device)
    execution = OpeningExecutionQNet().to(device)
    execution_target = OpeningExecutionQNet().to(device)

    selection_target.load_state_dict(selection.state_dict())
    execution_target.load_state_dict(execution.state_dict())
    selection_target.eval()
    execution_target.eval()

    sel_opt = optim.AdamW(selection.parameters(), lr=LR)
    exe_opt = optim.AdamW(execution.parameters(), lr=LR)

    sel_replay = ReplayBuffer(REPLAY_CAPACITY)
    exe_replay = ReplayBuffer(REPLAY_CAPACITY)

    start_episode = 1
    global_step = 0
    best_win_rate = -1.0

    if args.resume:
        ep, global_step, best_win_rate = load_checkpoint(
            Path(args.resume),
            selection,
            execution,
            selection_target,
            execution_target,
            sel_opt,
            exe_opt,
            device,
        )
        start_episode = ep + 1
        print(
            f"[RESUME] episode={ep} global_step={global_step} "
            f"best_win_rate={best_win_rate:.3f}"
        )

    recent_match_wins = deque(maxlen=50)
    recent_round_rewards = deque(maxlen=200)

    print(f"device={device}")
    print(f"OBS_DIM={OBS_DIM}")
    print(f"selection action dims={action_dims}")
    print(f"execution actions={EXEC_ACTION_DIM} (WAIT/EXECUTE/CANCEL)")
    print(
        f"learning starts: selection={SELECTION_LEARNING_STARTS} "
        f"execution={EXECUTION_LEARNING_STARTS}"
    )
    print(f"GC defender roster={_ghost_champions_roster()}")
    print(f"opponent mode={args.opponent}")

    try:
        for episode in range(start_episode, args.episodes + 1):
            epsilon = epsilon_by_step(global_step)

            result = run_match(
                selection,
                execution,
                device,
                epsilon=epsilon,
                training=True,
                seed=args.seed + episode,
                opponent_mode=args.opponent,
            )

            recent_match_wins.append(result["match_win"])

            sel_added = 0
            exe_added = 0
            sel_losses = []
            exe_losses = []

            for selections, executions, info in result["round_records"]:
                recent_round_rewards.append(info["reward"])

                for t in selections:
                    sel_replay.push(
                        t.obs, t.action, t.reward, t.next_obs, t.done, t.ability
                    )
                    sel_added += 1
                    global_step += 1

                for t in executions:
                    exe_replay.push(
                        t.obs, t.action, t.reward, t.next_obs, t.done, t.ability
                    )
                    exe_added += 1
                    global_step += 1

                if global_step % TRAIN_EVERY == 0:
                    sl = optimize_selection(
                        selection, selection_target, sel_opt, sel_replay, device
                    )
                    el = optimize_execution(
                        execution, execution_target, exe_opt, exe_replay, device
                    )
                    if sl is not None:
                        sel_losses.append(sl)
                    if el is not None:
                        exe_losses.append(el)

                if global_step % TARGET_UPDATE_EVERY == 0:
                    selection_target.load_state_dict(selection.state_dict())
                    execution_target.load_state_dict(execution.state_dict())

            recent_wr = float(np.mean(recent_match_wins)) if recent_match_wins else 0.0
            recent_r = float(np.mean(recent_round_rewards)) if recent_round_rewards else 0.0

            print(
                f"[{episode:5d}/{args.episodes}] "
                f"A {result['attacker_wins']:2d}-{result['defender_wins']:2d} D "
                f"| eps={epsilon:.3f} "
                f"| recentWR={recent_wr:.3f} "
                f"| R={recent_r:+.2f} "
                f"| opp={result.get('opponent', 'unknown')} "
                f"| replay={len(sel_replay)}/{len(exe_replay)} "
                f"| add={sel_added}/{exe_added}"
            )

            if episode % args.save_every == 0:
                save_checkpoint(
                    LATEST_MODEL,
                    selection,
                    execution,
                    selection_target,
                    execution_target,
                    sel_opt,
                    exe_opt,
                    episode=episode,
                    global_step=global_step,
                    best_win_rate=best_win_rate,
                    action_dims=action_dims,
                )

            if episode % args.eval_every == 0:
                metrics = evaluate(
                    selection,
                    execution,
                    device,
                    matches=args.eval_matches,
                    seed=args.seed + 500000 + episode * 100,
                    opponent_mode=args.opponent,
                )
                print("[EVAL]", metrics)

                with LOG_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "episode": episode,
                        "global_step": global_step,
                        **metrics,
                    }, ensure_ascii=False) + "\n")

                if metrics["win_rate"] > best_win_rate:
                    best_win_rate = metrics["win_rate"]
                    save_checkpoint(
                        BEST_MODEL,
                        selection,
                        execution,
                        selection_target,
                        execution_target,
                        sel_opt,
                        exe_opt,
                        episode=episode,
                        global_step=global_step,
                        best_win_rate=best_win_rate,
                        action_dims=action_dims,
                    )
                    print(f"[BEST] win_rate={best_win_rate:.3f}")

        save_checkpoint(
            FINAL_MODEL,
            selection,
            execution,
            selection_target,
            execution_target,
            sel_opt,
            exe_opt,
            episode=args.episodes,
            global_step=global_step,
            best_win_rate=best_win_rate,
            action_dims=action_dims,
        )

    except KeyboardInterrupt:
        episode_now = locals().get("episode", start_episode - 1)
        save_checkpoint(
            INTERRUPT_MODEL,
            selection,
            execution,
            selection_target,
            execution_target,
            sel_opt,
            exe_opt,
            episode=episode_now,
            global_step=global_step,
            best_win_rate=best_win_rate,
            action_dims=action_dims,
        )
        print(f"\n[INTERRUPT] saved: {INTERRUPT_MODEL}")
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", default=None)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-matches", type=int, default=8)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument(
        "--opponent",
        choices=("default", "toru_ai_v3", "mixed"),
        default="toru_ai_v3",
        help="Training/evaluation opponent. mixed = 70% Toru AI v3, 30% Default.",
    )
    args = p.parse_args()

    train(args)


if __name__ == "__main__":
    main()
