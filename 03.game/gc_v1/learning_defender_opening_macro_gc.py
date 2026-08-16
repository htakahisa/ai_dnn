"""Ghost Champions Defender Opening Ability Macro.

Purpose
-------
Defender Search / Retake の上位に置く「開幕定石」レイヤー。

このモジュールは、
1. map_data_defender_opening_ability_gc.py に定義された定石候補を読む
2. RLポリシーが「どの定石を採用するか / WAIT / CANCEL / EXECUTE」を決める
3. Flash / Recon担当だけ一時的にOriginへBFS移動させる
4. Smokeは現在位置を崩さずTargetへ実行する
5. 緊急情報が入ったら定石をキャンセルしてSearchへ制御を返す
ための実戦ランタイムを提供する。

現段階では「学習環境そのもの」ではない。
学習スクリプトはこのクラスが公開する observation/action 定義を利用して
別ファイル train_defender_opening_macro_gc.py から学習する想定。

Action design
-------------
Opening選択:
    0 = NONE
    1..N = 定義済みPattern

実行中:
    0 = WAIT
    1 = EXECUTE
    2 = CANCEL

複数Abilityを同一ラウンドで採用できるよう、Smoke / Flash / Reconを
独立スロットとして扱う。将来のRLは各スロットを別々に選択可能。

Controller integration
----------------------
既存Searchから得た base_result を coordinate() に渡す。
Openingが何もしない場合は base_result をそのまま返す。

    result = opening_macro.coordinate(char, game_state, base_result)

返却形式は既存Controllerに合わせる:
    move only:
        [row, col]
    ability:
        ([row, col], {"ability": "SMOKE", "target": (row, col)})
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import deque
import math
import random
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # 学習済みモデル未導入でもルール/デバッグ使用可能
    torch = None
    nn = None

from defender_opening_ability_patterns_gc import (
    OPENING_ABILITY_PATTERNS,
    validate_opening_ability_patterns,
)


ABILITY_ORDER = ("SMOKE", "FLASH", "RECON")
CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))

# 開幕定石として扱う最大時間。
# 固定の「実行Tick」ではなく、これを超えたら定石そのものを捨てる安全弁。
OPENING_MAX_TICKS = 30

# Origin到達判定。Flash / Recon はそのマスへ着いてから実行する。
ORIGIN_READY_RADIUS = 0

# 緊急情報によるキャンセル。
# 3人以上同時確認は本隊相当としてOpeningよりSearch情報Commitmentを優先。
MAIN_FORCE_CANCEL_COUNT = 3

# 落下Spikeは最重要情報。確認できるならOpeningを破棄する。
CANCEL_ON_DROPPED_SPIKE = True

# 自分自身が敵を見ている場合は定石移動を強制しない。
# 交戦をSearch/DQNへ返す。
PRESERVE_COMBAT = True

# RL observation:
# team/global 16 + ability slot 3 * 10 = 46
OBS_DIM = 46

# 実行フェーズの行動
EXEC_WAIT = 0
EXECUTE = 1
EXEC_CANCEL = 2
EXEC_ACTION_DIM = 3


@dataclass
class OpeningPlan:
    ability: str
    pattern_id: int
    caster_name: str | None = None
    origin: tuple[int, int] | None = None
    target: tuple[int, int] | None = None
    state: str = "PLANNED"       # PLANNED/MOVING/READY/EXECUTED/CANCELLED
    chosen_tick: int = 0
    executed_tick: int | None = None
    cancel_reason: str | None = None

    @property
    def active(self):
        return self.state not in {"EXECUTED", "CANCELLED"}


class OpeningMacroQNet(nn.Module if nn is not None else object):
    """Small DQN head used by future training script.

    The network chooses one of three execution decisions:
    WAIT / EXECUTE / CANCEL.

    Pattern selection is intentionally separated from execution timing.
    This keeps the action space compact and allows several ability plans
    (e.g. Smoke + Recon) to coexist in one round.
    """

    def __init__(self, obs_dim=OBS_DIM, action_dim=EXEC_ACTION_DIM):
        if nn is None:
            raise RuntimeError("PyTorch is required to construct OpeningMacroQNet")
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


def _ability_charge(char, ability):
    ability = str(ability).upper()
    attr = {
        "SMOKE": "smoke_charges",
        "FLASH": "flash_charges",
        "RECON": "recon_charges",
    }.get(ability)
    if attr is None:
        return 0
    return int(getattr(char, attr, 0) or 0)


def _cheb(a, b):
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _has_los(grid, start, end):
    """Conservative grid LOS used only for combat-preservation checks."""
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        if (r, c) != (r0, c0) and grid[r, c] == 1:
            return False
        if (r, c) == (r1, c1):
            return True
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def _visible_enemies(char, game_state):
    grid = game_state["grid"]
    chars = game_state.get("chars", [])
    return [
        e for e in chars
        if getattr(e, "is_alive", True)
        and getattr(e, "team", None) != getattr(char, "team", None)
        and _has_los(grid, tuple(char.pos), tuple(e.pos))
    ]


def _team_visible_enemies(my_team, game_state):
    grid = game_state["grid"]
    chars = game_state.get("chars", [])
    allies = [
        c for c in chars
        if getattr(c, "is_alive", True)
        and getattr(c, "team", None) == my_team
    ]
    enemies = [
        c for c in chars
        if getattr(c, "is_alive", True)
        and getattr(c, "team", None) != my_team
    ]
    visible = {}
    for ally in allies:
        for enemy in enemies:
            if _has_los(grid, tuple(ally.pos), tuple(enemy.pos)):
                visible[getattr(enemy, "name", id(enemy))] = enemy
    return list(visible.values())


def _bfs_dist_map(grid, goals):
    h, w = grid.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    q = deque()

    for r, c in goals:
        r, c = int(r), int(c)
        if 0 <= r < h and 0 <= c < w and grid[r, c] != 1:
            dist[r, c] = 0
            q.append((r, c))

    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if grid[nr, nc] == 1 or dist[nr, nc] >= 0:
                continue
            dist[nr, nc] = nd
            q.append((nr, nc))
    return dist


def _bfs_step_towards(grid, char, chars, goal):
    """One legal cardinal step that strictly reduces BFS distance."""
    dist = _bfs_dist_map(grid, [goal])
    r0, c0 = map(int, char.pos)
    cur = int(dist[r0, c0])
    if cur <= 0:
        return [r0, c0]

    occupied = {
        tuple(map(int, c.pos))
        for c in chars
        if c is not char and getattr(c, "is_alive", True)
    }

    candidates = []
    for dr, dc in CARDINAL:
        nr, nc = r0 + dr, c0 + dc
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            continue
        if grid[nr, nc] == 1 or (nr, nc) in occupied:
            continue
        d = int(dist[nr, nc])
        if 0 <= d < cur:
            candidates.append((d, nr, nc))

    if not candidates:
        return [r0, c0]

    _, nr, nc = min(candidates)
    return [nr, nc]


class LearningDefenderOpeningMacroGCController:
    """Upper-layer Opening coordinator for Ghost Champions Defender."""

    def __init__(
        self,
        model_path: str | None = None,
        greedy: bool = True,
        verbose: bool = False,
        seed: int | None = None,
    ):
        self.model_path = str(model_path) if model_path else None
        self.greedy = bool(greedy)
        self.verbose = bool(verbose)
        self.rng = random.Random(seed)
        self.game = None

        self.patterns = OPENING_ABILITY_PATTERNS
        warnings = validate_opening_ability_patterns()
        if warnings:
            raise ValueError("Invalid opening ability patterns: " + "; ".join(warnings))

        self.device = None
        self.model = None
        if self.model_path:
            self._load_model(self.model_path)

        self.reset_round()

    def _load_model(self, model_path):
        if torch is None:
            raise RuntimeError("PyTorch is required to load Opening Macro model")
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)

        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload)
        obs_dim = int(payload.get("obs_dim", OBS_DIM)) if isinstance(payload, dict) else OBS_DIM
        action_dim = int(payload.get("action_dim", EXEC_ACTION_DIM)) if isinstance(payload, dict) else EXEC_ACTION_DIM

        self.model = OpeningMacroQNet(obs_dim=obs_dim, action_dim=action_dim)
        self.model.load_state_dict(state)
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def set_game(self, game):
        self.game = game

    def reset_round(self):
        self.tick = 0
        self.plans: dict[str, OpeningPlan] = {}
        self._team_tick_signature = None
        self._round_initialized = False
        self._opening_cancelled_globally = False
        self._global_cancel_reason = None
        self._last_obs = None

    # ------------------------------------------------------------------
    # Pattern / caster selection
    # ------------------------------------------------------------------
    def available_patterns(self, ability):
        return self.patterns.get(str(ability).upper(), {})

    def _alive_defenders_with_ability(self, ability, game_state):
        return [
            c for c in game_state.get("chars", [])
            if getattr(c, "is_alive", True)
            and getattr(c, "team", None) == "D"
            and str(getattr(c, "ability_name", "")).upper() == ability
            and _ability_charge(c, ability) > 0
        ]

    def _best_origin_for_caster(self, caster, pattern, grid):
        origins = list(pattern.get("origins", []))
        if not origins:
            return None, 0

        dist = _bfs_dist_map(grid, origins)
        r, c = map(int, caster.pos)
        d = int(dist[r, c])
        if d < 0:
            return None, 10**9

        best = None
        for origin in origins:
            od = int(_bfs_dist_map(grid, [origin])[r, c])
            if od >= 0 and (best is None or od < best[0]):
                best = (od, origin)
        if best is None:
            return None, 10**9
        return tuple(best[1]), int(best[0])

    def _choose_caster_and_geometry(self, ability, pattern, game_state):
        grid = game_state["grid"]
        candidates = self._alive_defenders_with_ability(ability, game_state)
        if not candidates:
            return None, None, None

        targets = list(pattern.get("targets", []))
        if not targets:
            return None, None, None

        # Smoke is position-independent in this game: choose any alive smoker.
        if ability == "SMOKE":
            caster = min(candidates, key=lambda c: str(getattr(c, "name", "")))
            target = self.rng.choice(targets)
            return caster, None, tuple(target)

        ranked = []
        for caster in candidates:
            origin, dist = self._best_origin_for_caster(caster, pattern, grid)
            if origin is not None:
                ranked.append((dist, str(getattr(caster, "name", "")), caster, origin))
        if not ranked:
            return None, None, None

        _, _, caster, origin = min(ranked)
        # Multiple targets are allowed; RL can later learn target index.
        # Runtime default uses the nearest target to chosen origin.
        target = min(targets, key=lambda p: _cheb(origin, p))
        return caster, tuple(origin), tuple(target)

    def set_plan(self, ability, pattern_id, game_state, chosen_tick=None):
        """Explicitly install a plan. Training/runtime policy calls this."""
        ability = str(ability).upper()
        pattern_id = int(pattern_id)
        pattern = self.patterns.get(ability, {}).get(pattern_id)
        if pattern is None:
            return False

        caster, origin, target = self._choose_caster_and_geometry(
            ability, pattern, game_state
        )
        if caster is None or target is None:
            return False

        self.plans[ability] = OpeningPlan(
            ability=ability,
            pattern_id=pattern_id,
            caster_name=str(caster.name),
            origin=origin,
            target=target,
            state="PLANNED",
            chosen_tick=self.tick if chosen_tick is None else int(chosen_tick),
        )
        if self.verbose:
            print(
                f"[GC D-OPENING] PLAN {ability}#{pattern_id} "
                f"caster={caster.name} origin={origin} target={target}"
            )
        return True

    def clear_plan(self, ability, reason="policy_cancel"):
        ability = str(ability).upper()
        plan = self.plans.get(ability)
        if plan is None or not plan.active:
            return
        plan.state = "CANCELLED"
        plan.cancel_reason = str(reason)
        if self.verbose:
            print(
                f"[GC D-OPENING] CANCEL {ability}#{plan.pattern_id} "
                f"reason={reason}"
            )

    # ------------------------------------------------------------------
    # Observation for RL
    # ------------------------------------------------------------------
    def build_observation(self, game_state):
        chars = game_state.get("chars", [])
        defenders = [
            c for c in chars
            if getattr(c, "is_alive", True) and getattr(c, "team", None) == "D"
        ]
        attackers = [
            c for c in chars
            if getattr(c, "is_alive", True) and getattr(c, "team", None) == "A"
        ]
        visible = _team_visible_enemies("D", game_state)

        spike_pos = game_state.get("spike_pos")
        planted = bool(game_state.get("is_planted", False))

        global_features = [
            min(self.tick / max(1, OPENING_MAX_TICKS), 1.0),
            len(defenders) / 5.0,
            len(attackers) / 5.0,
            len(visible) / 5.0,
            float(planted),
            float(spike_pos is not None),
            float(any(getattr(a, "has_spike", False) for a in attackers)),
            float(self._opening_cancelled_globally),
            float(bool(self.plans.get("SMOKE") and self.plans["SMOKE"].active)),
            float(bool(self.plans.get("FLASH") and self.plans["FLASH"].active)),
            float(bool(self.plans.get("RECON") and self.plans["RECON"].active)),
            float(bool(self.plans.get("SMOKE") and self.plans["SMOKE"].state == "EXECUTED")),
            float(bool(self.plans.get("FLASH") and self.plans["FLASH"].state == "EXECUTED")),
            float(bool(self.plans.get("RECON") and self.plans["RECON"].state == "EXECUTED")),
            0.0,
            0.0,
        ]

        slot_features = []
        grid = game_state["grid"]
        for ability in ABILITY_ORDER:
            plan = self.plans.get(ability)
            caster = None
            if plan is not None:
                caster = next(
                    (
                        c for c in defenders
                        if str(getattr(c, "name", "")) == plan.caster_name
                    ),
                    None,
                )

            active = float(plan is not None and plan.active)
            executed = float(plan is not None and plan.state == "EXECUTED")
            cancelled = float(plan is not None and plan.state == "CANCELLED")
            caster_alive = float(caster is not None and getattr(caster, "is_alive", True))
            charge = float(_ability_charge(caster, ability) > 0) if caster is not None else 0.0

            dist_norm = 0.0
            at_origin = 0.0
            sees_enemy = 0.0
            if caster is not None:
                sees_enemy = float(bool(_visible_enemies(caster, game_state)))
                if plan is not None and plan.origin is not None:
                    dist = _bfs_dist_map(grid, [plan.origin])
                    r, c = map(int, caster.pos)
                    d = int(dist[r, c])
                    if d >= 0:
                        dist_norm = min(d / 30.0, 1.0)
                        at_origin = float(d <= ORIGIN_READY_RADIUS)
                elif ability == "SMOKE":
                    at_origin = 1.0

            pid_norm = (
                float(plan.pattern_id) / 9.0
                if plan is not None else 0.0
            )
            age_norm = (
                min(max(0, self.tick - plan.chosen_tick) / OPENING_MAX_TICKS, 1.0)
                if plan is not None else 0.0
            )

            slot_features.extend([
                active,
                executed,
                cancelled,
                caster_alive,
                charge,
                dist_norm,
                at_origin,
                sees_enemy,
                pid_norm,
                age_norm,
            ])

        obs = np.asarray(global_features + slot_features, dtype=np.float32)
        if obs.shape != (OBS_DIM,):
            raise RuntimeError(f"Opening Macro obs mismatch: {obs.shape} != {(OBS_DIM,)}")
        self._last_obs = obs
        return obs

    # ------------------------------------------------------------------
    # Policy decisions
    # ------------------------------------------------------------------
    def _policy_execution_action(self, game_state):
        """Choose WAIT/EXECUTE/CANCEL from loaded DQN.

        No model yet => conservative runtime:
        - ready plan executes
        - otherwise waits
        Training will replace this heuristic with learned Q-values.
        """
        if self.model is None:
            return EXECUTE

        obs = self.build_observation(game_state)
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.model(x)[0].cpu().numpy()
        if self.greedy:
            return int(np.argmax(q))
        probs = np.exp(q - np.max(q))
        probs = probs / probs.sum()
        return int(self.rng.choices(range(len(q)), weights=probs, k=1)[0])

    def _emergency_cancel_reason(self, game_state):
        if bool(game_state.get("is_planted", False)):
            return "PLANTED"

        if self.tick > OPENING_MAX_TICKS:
            return "OPENING_TIMEOUT"

        visible = _team_visible_enemies("D", game_state)
        if len(visible) >= MAIN_FORCE_CANCEL_COUNT:
            return f"MAIN_FORCE_X{len(visible)}"

        if CANCEL_ON_DROPPED_SPIKE:
            spike_pos = game_state.get("spike_pos")
            if spike_pos is not None:
                attackers = [
                    c for c in game_state.get("chars", [])
                    if getattr(c, "team", None) == "A"
                    and getattr(c, "is_alive", True)
                ]
                living_holder = next(
                    (a for a in attackers if getattr(a, "has_spike", False)),
                    None,
                )
                if living_holder is None:
                    return "DROPPED_SPIKE"

        return None

    def _maybe_advance_tick(self, char, game_state):
        """Increment once per team cycle, not once per Defender."""
        signature = (
            int(game_state.get("battle_tick", -1)),
            int(getattr(self.game, "battle_tick", -1)) if self.game is not None else -1,
        )
        if signature != self._team_tick_signature:
            self._team_tick_signature = signature
            self.tick += 1

    def _caster_for_plan(self, plan, game_state):
        return next(
            (
                c for c in game_state.get("chars", [])
                if getattr(c, "is_alive", True)
                and str(getattr(c, "name", "")) == plan.caster_name
            ),
            None,
        )

    def _ability_result(self, char, plan):
        return (
            list(map(int, char.pos)),
            {
                "ability": plan.ability,
                "target": tuple(map(int, plan.target)),
            },
        )

    def _protect_reserved_ability(self, base_result, plan):
        """Prevent lower Search layer from consuming an Opening-reserved charge.

        Search may return:
            ([row, col], {"ability": "SMOKE/FLASH/RECON", "target": ...})

        While an Opening plan is active for that same ability, keep Search's
        movement destination but strip only the ability action.

        Once the plan is EXECUTED/CANCELLED, coordinate() no longer calls this
        helper for that plan, so Search regains normal ability ownership.
        """
        if plan is None or not plan.active:
            return base_result

        if (
            isinstance(base_result, tuple)
            and len(base_result) == 2
            and isinstance(base_result[1], dict)
        ):
            ability = str(base_result[1].get("ability", "")).upper()
            if ability == str(plan.ability).upper():
                move_part = base_result[0]
                try:
                    return list(map(int, move_part))
                except Exception:
                    return move_part

        return base_result

    # ------------------------------------------------------------------
    # Main upper-layer hook
    # ------------------------------------------------------------------
    def coordinate(self, char, game_state, base_result):
        """Apply Opening Macro above Search.

        Only the assigned caster is overridden.
        Everybody else keeps Search's normal positioning/commitment result.
        """
        if not getattr(char, "is_alive", True):
            return base_result

        self._maybe_advance_tick(char, game_state)

        emergency = self._emergency_cancel_reason(game_state)
        if emergency is not None:
            self._opening_cancelled_globally = True
            self._global_cancel_reason = emergency
            for ability in ABILITY_ORDER:
                self.clear_plan(ability, emergency)
            return base_result

        # No active plan assigned to this character => Search keeps ownership.
        plan = next(
            (
                p for p in self.plans.values()
                if p.active and p.caster_name == str(getattr(char, "name", ""))
            ),
            None,
        )
        if plan is None:
            return base_result

        # Caster died / lost charge: cancel only that plan.
        caster = self._caster_for_plan(plan, game_state)
        if caster is None:
            self.clear_plan(plan.ability, "CASTER_DEAD")
            return base_result
        if _ability_charge(caster, plan.ability) <= 0:
            self.clear_plan(plan.ability, "NO_CHARGE")
            return base_result

        # Current firefight has higher priority than opening lineup movement.
        if PRESERVE_COMBAT and _visible_enemies(char, game_state):
            return self._protect_reserved_ability(base_result, plan)

        # Smoke does not require a cast position.
        if plan.ability == "SMOKE":
            action = self._policy_execution_action(game_state)
            if action == EXEC_CANCEL:
                self.clear_plan(plan.ability, "POLICY_CANCEL")
                return base_result
            if action == EXEC_WAIT:
                return self._protect_reserved_ability(base_result, plan)

            plan.state = "EXECUTED"
            plan.executed_tick = self.tick
            if self.verbose:
                print(
                    f"[GC D-OPENING] EXECUTE SMOKE#{plan.pattern_id} "
                    f"{char.name} -> {plan.target} tick={self.tick}"
                )
            return self._ability_result(char, plan)

        # Flash / Recon: temporarily prioritize lineup origin over Search position.
        if plan.origin is None:
            self.clear_plan(plan.ability, "NO_ORIGIN")
            return base_result

        dist_map = _bfs_dist_map(game_state["grid"], [plan.origin])
        r, c = map(int, char.pos)
        d = int(dist_map[r, c])

        if d < 0:
            self.clear_plan(plan.ability, "ORIGIN_UNREACHABLE")
            return base_result

        if d > ORIGIN_READY_RADIUS:
            plan.state = "MOVING"
            next_pos = _bfs_step_towards(
                game_state["grid"],
                char,
                game_state.get("chars", []),
                plan.origin,
            )
            if next_pos != list(map(int, char.pos)):
                return next_pos
            # Teammate blockage etc. => let Search move/hold, but do not let
            # Search consume the ability charge reserved for this Opening plan.
            return self._protect_reserved_ability(base_result, plan)

        plan.state = "READY"
        action = self._policy_execution_action(game_state)

        if action == EXEC_CANCEL:
            self.clear_plan(plan.ability, "POLICY_CANCEL")
            return base_result
        if action == EXEC_WAIT:
            return self._protect_reserved_ability(base_result, plan)

        plan.state = "EXECUTED"
        plan.executed_tick = self.tick
        if self.verbose:
            print(
                f"[GC D-OPENING] EXECUTE {plan.ability}#{plan.pattern_id} "
                f"{char.name} {plan.origin}->{plan.target} tick={self.tick}"
            )
        return self._ability_result(char, plan)

    # ------------------------------------------------------------------
    # Helpers for future trainer
    # ------------------------------------------------------------------
    def get_pattern_action_catalog(self):
        """Stable flat catalog used by train_defender_opening_macro_gc.py.

        0 is always NONE. Remaining entries are deterministic by ability/id.
        """
        catalog = [("NONE", 0)]
        for ability in ABILITY_ORDER:
            for pid in sorted(self.patterns.get(ability, {})):
                catalog.append((ability, int(pid)))
        return tuple(catalog)

    def install_catalog_action(self, action_index, game_state):
        """Install one pattern from flat catalog; 0 means no new plan."""
        catalog = self.get_pattern_action_catalog()
        idx = int(action_index)
        if not (0 <= idx < len(catalog)):
            return False
        ability, pid = catalog[idx]
        if ability == "NONE":
            return True
        return self.set_plan(ability, pid, game_state)


# Backward/short alias.
DefenderOpeningMacroGCController = LearningDefenderOpeningMacroGCController


if __name__ == "__main__":
    ctrl = LearningDefenderOpeningMacroGCController(verbose=True)
    print("OBS_DIM:", OBS_DIM)
    print("Execution actions: WAIT / EXECUTE / CANCEL")
    print("Pattern catalog:")
    for i, item in enumerate(ctrl.get_pattern_action_catalog()):
        print(f"  {i:2d}: {item}")
