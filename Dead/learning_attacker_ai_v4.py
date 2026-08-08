# learning_attacker_ai_v4.py
"""
Phase3/Phase4系Dueling DQNを、画面付きゲームへ接続するアタッカーAI。

主な仕様
--------
- 観測: 131次元
    - 先頭125次元は learning_attacker_ai_v2.py の実戦観測を再利用
    - 末尾6次元はPhase3戦術観測
- 行動: 14
    0..3  : 4方向移動
    4     : WAIT
    5     : PLANT
    6..13 : 固定8個の戦術アビリティ候補
- battle_logic.py が理解する戻り値を返す
    MOVE    -> ([row, col], "MOVE")
    PLANT   -> ([row, col], "PLANT")
    ABILITY -> ([row, col], "ABILITY", payload)

重要
----
現在のPhase3学習コードでは、VALID_PHASE1_ACTIONSが0..5に限定されている版があります。
そのモデルでは6..13のアビリティ出力は未学習です。
アビリティまで学習したPhase4モデルを使うまでは、
enable_abilities=False で動作確認することを推奨します。
"""

from __future__ import annotations

import importlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from controllers import BaseController
from learning_attacker_ai_v2 import LearningAttackerAIv2Controller

Position = Tuple[int, int]

# ---------------------------------------------------------------------
# Phase4側で固定する入出力
# ---------------------------------------------------------------------

PHASE2_OBS_DIM = 125
PHASE3_EXTRA_OBS = 6
OBS_DIM = PHASE2_OBS_DIM + PHASE3_EXTRA_OBS

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_WAIT = 4
ACTION_PLANT = 5
ACTION_ABILITY_0 = 6
ACTION_ABILITY_1 = 7
ACTION_ABILITY_2 = 8
ACTION_ABILITY_3 = 9
ACTION_ABILITY_4 = 10
ACTION_ABILITY_5 = 11
ACTION_ABILITY_6 = 12
ACTION_ABILITY_7 = 13

N_ACTIONS = 14
BASE_ACTION_COUNT = 6
ABILITY_TARGET_COUNT = 8

MOVE_DELTAS: Dict[int, Position] = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}

ACTION_NAMES = {
    ACTION_UP: "MOVE_UP",
    ACTION_DOWN: "MOVE_DOWN",
    ACTION_LEFT: "MOVE_LEFT",
    ACTION_RIGHT: "MOVE_RIGHT",
    ACTION_WAIT: "WAIT",
    ACTION_PLANT: "PLANT",
    ACTION_ABILITY_0: "ABILITY_SELF_FRONT",
    ACTION_ABILITY_1: "ABILITY_ENEMY_1",
    ACTION_ABILITY_2: "ABILITY_ENEMY_2",
    ACTION_ABILITY_3: "ABILITY_SITE_CENTER",
    ACTION_ABILITY_4: "ABILITY_PLANT_POSITION",
    ACTION_ABILITY_5: "ABILITY_NEAREST_ENEMY_MIDPOINT",
    ACTION_ABILITY_6: "ABILITY_ENEMY_PAIR_MIDPOINT",
    ACTION_ABILITY_7: "ABILITY_NEAREST_ENEMY_NEIGHBOR",
}

SUPPORTED_ACTIVE_ABILITIES = {"SMOKE", "FLASH", "RECON"}

# 実ゲーム側の値。game_core.pyから取得できなければPhase3学習値を使う。
try:
    from game_core import (
        ROUND_DURATION_TICKS as RUNTIME_ROUND_DURATION_TICKS,
        SPIKE_DETONATION_TICKS as RUNTIME_SPIKE_DETONATION_TICKS,
    )
except ImportError:
    RUNTIME_ROUND_DURATION_TICKS = 120
    RUNTIME_SPIKE_DETONATION_TICKS = 35


@dataclass(frozen=True)
class RuntimeAbilityTarget:
    """DQNの固定インデックスに対応するアビリティ候補。"""

    name: str
    target: Position
    valid: bool = True


def _inside(pos: Position, grid: np.ndarray) -> bool:
    return 0 <= int(pos[0]) < int(grid.shape[0]) and 0 <= int(pos[1]) < int(
        grid.shape[1]
    )


def _walkable(pos: Position, grid: np.ndarray) -> bool:
    return _inside(pos, grid) and int(grid[pos]) != 1


def _is_alive(char) -> bool:
    return bool(getattr(char, "is_alive", getattr(char, "alive", False)))


def _ability_name(char) -> str:
    return str(getattr(char, "ability_name", "")).strip().upper()


def _ability_charges(char) -> int:
    ability = _ability_name(char)
    if ability == "SMOKE":
        return int(getattr(char, "smoke_charges", 0))
    if ability == "FLASH":
        return int(getattr(char, "flash_charges", 0))
    if ability == "RECON":
        return int(getattr(char, "recon_charges", 0))
    return 0


def _chebyshev(a: Position, b: Position) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _midpoint(a: Position, b: Position) -> Position:
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def _normalize_direction(dr: int, dc: int) -> Position:
    return (
        0 if dr == 0 else (1 if dr > 0 else -1),
        0 if dc == 0 else (1 if dc > 0 else -1),
    )


def _nearest_walkable(origin: Position, grid: np.ndarray) -> Optional[Position]:
    """壁や範囲外の候補を、近くの有効セルへ補正する。"""
    if _walkable(origin, grid):
        return origin

    height, width = grid.shape
    max_radius = max(height, width)

    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in (-radius, radius):
                candidate = (origin[0] + dr, origin[1] + dc)
                if _walkable(candidate, grid):
                    return candidate

        for dc in range(-radius + 1, radius):
            for dr in (-radius, radius):
                candidate = (origin[0] + dr, origin[1] + dc)
                if _walkable(candidate, grid):
                    return candidate

    return None


class RuntimeAbilityTargetGenerator:
    """実ゲームのgame_stateから、固定8個の候補を生成する。"""

    def generate(
        self,
        char,
        game_state: dict,
        grid: np.ndarray,
    ) -> List[RuntimeAbilityTarget]:
        current = (int(char.pos[0]), int(char.pos[1]))
        chars = list(game_state.get("chars", []))

        enemies = sorted(
            [
                other
                for other in chars
                if other is not char
                and _is_alive(other)
                and getattr(other, "team", None) != getattr(char, "team", None)
            ],
            key=lambda enemy: (
                _manhattan(current, tuple(enemy.pos)),
                str(getattr(enemy, "name", "")),
            ),
        )

        targets = [
            self._make(
                "SELF_FRONT",
                self._front_target(char, current, enemies, grid),
                grid,
            ),
            self._enemy_target("ENEMY_1", enemies, 0, current, grid),
            self._enemy_target("ENEMY_2", enemies, 1, current, grid),
            self._make(
                "SITE_CENTER",
                self._site_center(game_state, grid),
                grid,
            ),
            self._make(
                "PLANT_POSITION",
                self._plant_position(game_state, grid),
                grid,
            ),
            self._nearest_enemy_midpoint(current, enemies, grid),
            self._enemy_pair_midpoint(enemies, current, grid),
            self._nearest_enemy_neighbor(enemies, current, grid),
        ]

        if len(targets) != ABILITY_TARGET_COUNT:
            raise RuntimeError(f"ability candidate count mismatch: {len(targets)}")
        return targets

    def _make(
        self,
        name: str,
        target: Optional[Position],
        grid: np.ndarray,
    ) -> RuntimeAbilityTarget:
        if target is None:
            return RuntimeAbilityTarget(name, (0, 0), False)

        normalized = (int(target[0]), int(target[1]))
        nearest = _nearest_walkable(normalized, grid)
        if nearest is None:
            return RuntimeAbilityTarget(name, normalized, False)

        return RuntimeAbilityTarget(name, nearest, True)

    def _enemy_target(
        self,
        name: str,
        enemies: Sequence,
        index: int,
        fallback: Position,
        grid: np.ndarray,
    ) -> RuntimeAbilityTarget:
        if index >= len(enemies):
            return RuntimeAbilityTarget(name, fallback, False)
        return self._make(name, tuple(enemies[index].pos), grid)

    def _front_target(
        self,
        char,
        current: Position,
        enemies: Sequence,
        grid: np.ndarray,
    ) -> Position:
        # Characterに向きがある場合はそれを優先。
        if hasattr(char, "dir_r") and hasattr(char, "dir_c"):
            direction = _normalize_direction(
                int(char.dir_r),
                int(char.dir_c),
            )
        else:
            raw = getattr(char, "direction", None)
            if isinstance(raw, (tuple, list)) and len(raw) == 2:
                direction = _normalize_direction(
                    int(raw[0]),
                    int(raw[1]),
                )
            else:
                facing = str(
                    getattr(
                        char,
                        "facing",
                        getattr(char, "direction_name", ""),
                    )
                ).upper()
                direction = {
                    "UP": (-1, 0),
                    "DOWN": (1, 0),
                    "LEFT": (0, -1),
                    "RIGHT": (0, 1),
                }.get(facing, (0, 0))

        # 向き情報がなければ、最寄り敵の方向を使う。
        if direction == (0, 0) and enemies:
            enemy_pos = tuple(enemies[0].pos)
            dr = enemy_pos[0] - current[0]
            dc = enemy_pos[1] - current[1]
            if abs(dr) >= abs(dc):
                direction = _normalize_direction(dr, 0)
            else:
                direction = _normalize_direction(0, dc)

        # 敵も向きもなければサイト方向。
        if direction == (0, 0):
            plant = self._plant_position({}, grid)
            if plant is not None:
                dr = plant[0] - current[0]
                dc = plant[1] - current[1]
                if abs(dr) >= abs(dc):
                    direction = _normalize_direction(dr, 0)
                else:
                    direction = _normalize_direction(0, dc)

        if direction == (0, 0):
            return current

        last_valid = current
        max_steps = max(grid.shape)
        for step in range(1, max_steps + 1):
            candidate = (
                current[0] + direction[0] * step,
                current[1] + direction[1] * step,
            )
            if not _walkable(candidate, grid):
                break
            last_valid = candidate

        return last_valid

    def _site_center(
        self,
        game_state: dict,
        grid: np.ndarray,
    ) -> Optional[Position]:
        target = game_state.get("target_plant_pos")
        if isinstance(target, (list, tuple)) and len(target) == 2:
            return int(target[0]), int(target[1])

        plant_cells = np.argwhere(grid == 2)
        if len(plant_cells) == 0:
            return None

        return (
            int(round(float(plant_cells[:, 0].mean()))),
            int(round(float(plant_cells[:, 1].mean()))),
        )

    def _plant_position(
        self,
        game_state: dict,
        grid: np.ndarray,
    ) -> Optional[Position]:
        for key in ("planted_pos", "target_plant_pos", "spike_pos"):
            value = game_state.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return int(value[0]), int(value[1])

        plant_cells = np.argwhere(grid == 2)
        if len(plant_cells) == 0:
            return None
        return int(plant_cells[0][0]), int(plant_cells[0][1])

    def _nearest_enemy_midpoint(
        self,
        current: Position,
        enemies: Sequence,
        grid: np.ndarray,
    ) -> RuntimeAbilityTarget:
        if not enemies:
            return RuntimeAbilityTarget(
                "NEAREST_ENEMY_MIDPOINT",
                current,
                False,
            )
        return self._make(
            "NEAREST_ENEMY_MIDPOINT",
            _midpoint(current, tuple(enemies[0].pos)),
            grid,
        )

    def _enemy_pair_midpoint(
        self,
        enemies: Sequence,
        fallback: Position,
        grid: np.ndarray,
    ) -> RuntimeAbilityTarget:
        if len(enemies) < 2:
            return RuntimeAbilityTarget(
                "ENEMY_PAIR_MIDPOINT",
                fallback,
                False,
            )
        return self._make(
            "ENEMY_PAIR_MIDPOINT",
            _midpoint(tuple(enemies[0].pos), tuple(enemies[1].pos)),
            grid,
        )

    def _nearest_enemy_neighbor(
        self,
        enemies: Sequence,
        fallback: Position,
        grid: np.ndarray,
    ) -> RuntimeAbilityTarget:
        if not enemies:
            return RuntimeAbilityTarget(
                "NEAREST_ENEMY_NEIGHBOR",
                fallback,
                False,
            )

        enemy_pos = tuple(enemies[0].pos)
        # 使用者に近い隣接マスを優先。
        neighbors = [
            (enemy_pos[0] - 1, enemy_pos[1]),
            (enemy_pos[0] + 1, enemy_pos[1]),
            (enemy_pos[0], enemy_pos[1] - 1),
            (enemy_pos[0], enemy_pos[1] + 1),
        ]
        neighbors.sort(key=lambda pos: _manhattan(fallback, pos))

        for candidate in neighbors:
            if _walkable(candidate, grid):
                return RuntimeAbilityTarget(
                    "NEAREST_ENEMY_NEIGHBOR",
                    candidate,
                    True,
                )

        return RuntimeAbilityTarget(
            "NEAREST_ENEMY_NEIGHBOR",
            enemy_pos,
            False,
        )


class LearningAttackerAIv4Controller(LearningAttackerAIv2Controller):
    """Phase3/Phase4モデルを実ゲームへ接続するコントローラー。"""

    is_attacker_ai_v2 = True
    is_attacker_ai_v4 = True

    def __init__(
        self,
        model_path: str = (
            "attacker_ai_v3_phase3_data/" "dqn_attacker_ai_v3_phase3_best.pt"
        ),
        greedy: bool = True,
        temperature: float = 0.35,
        device: Optional[str] = None,
        verbose: bool = False,
        enable_abilities: bool = False,
        mask_invalid_movement: bool = False,
    ) -> None:
        # 親クラスの__init__はv2ネットワークを読み込むため呼ばない。
        BaseController.__init__(self)

        if temperature <= 0:
            raise ValueError("temperatureは0より大きくしてください")

        selected_device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = torch.device(selected_device)
        self.greedy = bool(greedy)
        self.temperature = float(temperature)
        self.verbose = bool(verbose)
        self.enable_abilities = bool(enable_abilities)
        self.mask_invalid_movement = bool(mask_invalid_movement)

        path = Path(model_path)
        self.model_path = (
            path if path.is_absolute() else Path(__file__).resolve().parent / path
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                "Phase3/Phase4モデルが見つかりません。\n"
                f"探した場所: {self.model_path}"
            )

        self.model_class = self._load_network_class()
        self.model = self.model_class(
            obs_dim=OBS_DIM,
            n_actions=N_ACTIONS,
        ).to(self.device)
        self._load_model(self.model_path)
        self.model.eval()

        # 親クラスの実戦観測構築に必要な状態。
        self.last_actions: Dict[str, int] = {}
        self.enemy_memories = {}
        self.requested_moves = {}
        self.last_move_failed: Dict[str, bool] = {}
        self.last_move_fail_reason: Dict[str, str] = {}
        self._distance_cache_key = None
        self._distance_cache = None
        self._fallback_tick = 0
        self._last_seen_game_tick = None

        self.target_generator = RuntimeAbilityTargetGenerator()
        self.last_candidates: Dict[str, List[RuntimeAbilityTarget]] = {}

        print(
            f"[AI v4] model={self.model_path} "
            f"device={self.device} greedy={self.greedy} "
            f"abilities={self.enable_abilities}"
        )

        if not self.enable_abilities:
            print(
                "[AI v4] ability actions 6..13 are masked. "
                "アビリティ学習済みモデルでTrueにしてください。"
            )

    @staticmethod
    def _load_network_class():
        """canonicalな学習ファイルからDuelingQNetworkを取得する。"""
        module_candidates = (
            "train_attacker_ai_v3_phase3_transfer_epsilon",
            "train_attacker_ai_v3_phase3",
            "train_attacker_ai_v3_phase3_complete",
        )

        errors: List[str] = []
        for module_name in module_candidates:
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                errors.append(f"{module_name}: {exc}")
                continue

            network_class = getattr(module, "DuelingQNetwork", None)
            if network_class is not None:
                return network_class

            errors.append(f"{module_name}: DuelingQNetworkなし")

        joined = "\n".join(errors)
        raise ImportError(
            "Phase3のDuelingQNetworkを読み込めません。\n"
            "学習ファイル名を次のどれかにしてください:\n"
            "  train_attacker_ai_v3_phase3_transfer_epsilon.py\n"
            "  train_attacker_ai_v3_phase3.py\n"
            "  train_attacker_ai_v3_phase3_complete.py\n"
            f"詳細:\n{joined}"
        )

    def _load_model(self, path: Path) -> None:
        try:
            loaded = torch.load(
                str(path),
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            # 古いPyTorch向け。
            loaded = torch.load(
                str(path),
                map_location=self.device,
            )

        state_dict = self._extract_state_dict(loaded)

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                "モデル構造がPhase3/Phase4ネットワークと一致しません。\n"
                f"期待: OBS_DIM={OBS_DIM}, N_ACTIONS={N_ACTIONS}\n"
                f"model: {path}\n"
                "training_state_latest.ptの場合はpolicyキーを自動取得します。"
            ) from exc

    @staticmethod
    def _extract_state_dict(loaded):
        if not isinstance(loaded, dict):
            return loaded

        for key in (
            "policy",
            "model_state_dict",
            "state_dict",
            "model",
        ):
            value = loaded.get(key)
            if isinstance(value, dict):
                return value

        # raw state_dictもdictなので、Tensor値が含まれるか確認。
        if loaded and all(isinstance(value, torch.Tensor) for value in loaded.values()):
            return loaded

        raise ValueError(
            "checkpointからモデル重みを取得できません。"
            "policy/model_state_dict/state_dict/modelキーを確認してください。"
        )

    def reset_round(self) -> None:
        super().reset_round()
        self.last_candidates.clear()

    def decide_move(self, char, game_state: dict):
        grid = np.asarray(game_state["grid"])
        chars = list(game_state["chars"])
        tick_id = self._get_tick_id(game_state)

        self._update_previous_move_result(
            char,
            grid,
            chars,
            tick_id,
        )

        observation = self._make_observation_v4(
            char=char,
            game_state=game_state,
            grid=grid,
            chars=chars,
        )

        candidates = self.target_generator.generate(
            char,
            game_state,
            grid,
        )
        self.last_candidates[str(char.name)] = candidates

        valid_mask = self._valid_action_mask(
            char=char,
            game_state=game_state,
            grid=grid,
            chars=chars,
            candidates=candidates,
        )

        action, q_values = self._select_action_v4(
            observation,
            valid_mask,
        )
        self.last_actions[str(char.name)] = action

        command = self._action_to_game_command_v4(
            char=char,
            action=action,
            grid=grid,
            game_state=game_state,
            tick_id=tick_id,
            candidates=candidates,
        )

        if self.verbose:
            valid_names = [
                ACTION_NAMES[index] for index, valid in enumerate(valid_mask) if valid
            ]
            print(
                f"[AI v4] tick={tick_id} char={char.name} "
                f"action={action}:{ACTION_NAMES.get(action)} "
                f"valid={valid_names} "
                f"q={np.round(q_values, 3).tolist()} "
                f"command={command}"
            )

        return command

    def _make_observation_v4(
        self,
        char,
        game_state: dict,
        grid: np.ndarray,
        chars: Sequence,
    ) -> np.ndarray:
        """v2実戦観測125次元の末尾へ、Phase3戦術6要素を追加する。"""
        base = super()._make_observation(
            char=char,
            game_state=game_state,
            grid=grid,
            chars=chars,
        )

        if base.shape != (PHASE2_OBS_DIM,):
            raise RuntimeError(
                "Phase2 observation mismatch: "
                f"{base.shape}, expected {(PHASE2_OBS_DIM,)}"
            )

        battle_tick = float(
            game_state.get(
                "battle_tick",
                self._fallback_tick,
            )
        )
        round_remaining = max(
            RUNTIME_ROUND_DURATION_TICKS - battle_tick,
            0.0,
        ) / max(RUNTIME_ROUND_DURATION_TICKS, 1)

        is_planted = bool(game_state.get("is_planted", False))
        spike_timer = float(game_state.get("spike_timer", 0.0))
        spike_remaining = (
            max(spike_timer, 0.0) / max(RUNTIME_SPIKE_DETONATION_TICKS, 1)
            if is_planted
            else 0.0
        )

        defenders = [other for other in chars if getattr(other, "team", None) == "D"]
        alive_defenders = [defender for defender in defenders if _is_alive(defender)]
        alive_defender_ratio = len(alive_defenders) / max(len(defenders), 1)

        active_defuser_name = game_state.get("active_defuser_name")
        active_defusers = [
            defender
            for defender in alive_defenders
            if float(getattr(defender, "defuse_timer", 0)) > 0
            or (
                active_defuser_name is not None
                and str(getattr(defender, "name", "")) == str(active_defuser_name)
            )
        ]
        defuse_active = float(bool(active_defusers))

        spike_distance = 0.0
        planted_pos = game_state.get("planted_pos")
        if (
            is_planted
            and isinstance(planted_pos, (list, tuple))
            and len(planted_pos) == 2
        ):
            current = (
                int(char.pos[0]),
                int(char.pos[1]),
            )
            spike = (
                int(planted_pos[0]),
                int(planted_pos[1]),
            )
            raw_distance = self._distance_to_objective(
                current,
                spike,
                grid,
            )
            if not math.isfinite(raw_distance):
                spike_distance = 1.0
            else:
                spike_distance = min(
                    raw_distance / max(grid.shape[0] + grid.shape[1], 1),
                    1.0,
                )

        extra = np.asarray(
            [
                float(round_remaining),
                float(spike_remaining),
                float(alive_defender_ratio),
                float(defuse_active),
                float(spike_distance),
                float(is_planted),
            ],
            dtype=np.float32,
        )

        observation = np.concatenate([base, extra])
        if observation.shape != (OBS_DIM,):
            raise RuntimeError(
                "AI v4 observation mismatch: "
                f"{observation.shape}, expected {(OBS_DIM,)}"
            )
        return observation

    def _valid_action_mask(
        self,
        char,
        game_state: dict,
        grid: np.ndarray,
        chars: Sequence,
        candidates: Sequence[RuntimeAbilityTarget],
    ) -> np.ndarray:
        mask = np.ones(N_ACTIONS, dtype=np.bool_)

        current = (
            int(char.pos[0]),
            int(char.pos[1]),
        )

        if self.mask_invalid_movement:
            occupied = {
                tuple(other.pos)
                for other in chars
                if other is not char and _is_alive(other)
            }
            for action, (dr, dc) in MOVE_DELTAS.items():
                target = (
                    current[0] + dr,
                    current[1] + dc,
                )
                mask[action] = _walkable(target, grid) and target not in occupied

        # WAITは常に可能。
        mask[ACTION_WAIT] = True

        # PLANTは学習環境に合わせて、不正場所では無効化する。
        mask[ACTION_PLANT] = (
            bool(getattr(char, "has_spike", False))
            and not bool(game_state.get("is_planted", False))
            and self._is_runtime_plant_position(
                char,
                game_state,
                grid,
            )
        )

        ability_valid = (
            self.enable_abilities
            and _ability_name(char) in SUPPORTED_ACTIVE_ABILITIES
            and _ability_charges(char) > 0
        )

        for index in range(ABILITY_TARGET_COUNT):
            action = BASE_ACTION_COUNT + index
            mask[action] = bool(
                ability_valid and index < len(candidates) and candidates[index].valid
            )

        # 全Falseは避ける。
        if not bool(mask.any()):
            mask[ACTION_WAIT] = True

        return mask

    def _select_action_v4(
        self,
        observation: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[int, np.ndarray]:
        state = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state).squeeze(0).detach().cpu().numpy()

        if q_values.shape != (N_ACTIONS,):
            raise RuntimeError(f"Q output mismatch: {q_values.shape}")

        valid_indices = np.flatnonzero(valid_mask)
        if len(valid_indices) == 0:
            return ACTION_WAIT, q_values

        masked = q_values.copy()
        masked[~valid_mask] = -np.inf

        if self.greedy:
            action = int(np.argmax(masked))
            return action, q_values

        valid_q = q_values[valid_indices]
        centered = valid_q - np.max(valid_q)
        logits = centered / self.temperature
        probabilities = np.exp(logits)
        total = float(probabilities.sum())

        if not np.isfinite(total) or total <= 0:
            action = int(random.choice(valid_indices.tolist()))
        else:
            probabilities /= total
            action = int(
                np.random.choice(
                    valid_indices,
                    p=probabilities,
                )
            )

        return action, q_values

    def _action_to_game_command_v4(
        self,
        char,
        action: int,
        grid: np.ndarray,
        game_state: dict,
        tick_id: Optional[int],
        candidates: Sequence[RuntimeAbilityTarget],
    ):
        current = (
            int(char.pos[0]),
            int(char.pos[1]),
        )
        name = str(char.name)

        if action in MOVE_DELTAS:
            dr, dc = MOVE_DELTAS[action]
            target = (
                current[0] + dr,
                current[1] + dc,
            )

            # 親クラスのRequestedMove型は内部更新処理で必要。
            from learning_attacker_ai_v2 import RequestedMove

            self.requested_moves[name] = RequestedMove(
                origin=current,
                target=target,
                action=action,
                tick_id=tick_id,
            )
            return [target[0], target[1]], "MOVE"

        self.requested_moves.pop(name, None)

        if action == ACTION_PLANT:
            return [current[0], current[1]], "PLANT"

        if action == ACTION_WAIT:
            return [current[0], current[1]], "MOVE"

        if BASE_ACTION_COUNT <= action < N_ACTIONS:
            index = action - BASE_ACTION_COUNT
            if index >= len(candidates):
                return [current[0], current[1]], "MOVE"

            candidate = candidates[index]
            if not candidate.valid:
                return [current[0], current[1]], "MOVE"

            ability = _ability_name(char)
            if ability not in SUPPORTED_ACTIVE_ABILITIES or _ability_charges(char) <= 0:
                return [current[0], current[1]], "MOVE"

            payload = {
                "ability": ability,
                "target": [
                    int(candidate.target[0]),
                    int(candidate.target[1]),
                ],
                "candidate_index": index,
                "candidate_name": candidate.name,
            }
            return (
                [current[0], current[1]],
                "ABILITY",
                payload,
            )

        return [current[0], current[1]], "MOVE"


# run_game.pyで短い名前を使えるようにする。
LearningAttackerV4Controller = LearningAttackerAIv4Controller


__all__ = [
    "LearningAttackerAIv4Controller",
    "LearningAttackerV4Controller",
    "RuntimeAbilityTarget",
    "RuntimeAbilityTargetGenerator",
]
