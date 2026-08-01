"""
BC / DAgger モデル実戦評価スクリプト

実行:
    python evaluate_bc_dagger.py
    python evaluate_bc_dagger.py --matches 30
    python evaluate_bc_dagger.py --models policy_dagger_final.pt
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from controllers import DefaultAttackerController, DefaultDefenderController
from map_data import NEW_MAZE_STR
from roster_utils import build_two_balanced_rosters
from run_game import VisualFPSBattle
from train_bc import NUM_ACTIONS, PolicyNetwork, observation_to_vector

# ============================================================================
# 学習時と同じモデル構造・観測ベクトル化をtrain_bc.pyから共有する。
# ============================================================================


DIRECTION_BY_ACTION = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


LOCAL_MAP_RADIUS = 3

LOCAL_EMPTY = 0
LOCAL_WALL = 1
LOCAL_SITE = 2
LOCAL_ALLY = 3
LOCAL_ENEMY = 4
LOCAL_SELF = 5
LOCAL_OUT_OF_MAP = 6
LOCAL_SPIKE = 7


def occupied_positions(game: Any, viewer: Any | None = None) -> dict[tuple[int, int], Any]:
    occupied: dict[tuple[int, int], Any] = {}

    for char in getattr(game, "chars", []):
        if not character_is_alive(char) or char is viewer:
            continue

        pos = getattr(char, "pos", None)
        if pos is None or len(pos) != 2:
            continue

        occupied[(int(pos[0]), int(pos[1]))] = char

    return occupied


def is_valid_destination(game: Any, viewer: Any, row: int, col: int) -> bool:
    row = int(row)
    col = int(col)
    grid = game.grid

    if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
        return False

    if grid[row, col] == 1:
        return False

    if (row, col) in occupied_positions(game, viewer):
        return False

    return True


def build_valid_move_mask(game: Any, viewer: Any) -> list[int]:
    row = int(viewer.pos[0])
    col = int(viewer.pos[1])

    return [
        int(is_valid_destination(game, viewer, row + dr, col + dc))
        for dr, dc in DIRECTION_BY_ACTION.values()
    ]




def build_action_mask(game: Any, char: Any, game_state: dict) -> torch.Tensor:
    """現在の状態で実行可能な9アクションをTrueで返す。"""
    mask = torch.ones(NUM_ACTIONS, dtype=torch.bool)

    # 0-3: 上下左右。壁・範囲外・占有マスを除外する。
    move_mask = build_valid_move_mask(game, char)
    for action_idx in range(4):
        mask[action_idx] = bool(move_mask[action_idx])

    # 4-6: 所持チャージがないアビリティを除外する。
    mask[4] = getattr(char, "smoke_charges", 0) > 0
    mask[5] = getattr(char, "flash_charges", 0) > 0
    mask[6] = getattr(char, "recon_charges", 0) > 0

    # 7: STOPは常に有効。
    mask[7] = True

    # 8: スパイク所持・設置地点上・未設置の場合のみPLANT可能。
    row, col = int(char.pos[0]), int(char.pos[1])
    mask[8] = bool(
        getattr(char, "has_spike", False)
        and game_state["grid"][row, col] == 2
        and not bool(game_state.get("is_planted", False))
    )

    return mask


def masked_action_probabilities(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """無効アクションの確率を0にして再正規化する。"""
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError(f"expected logits shape [1, N], got {tuple(logits.shape)}")
    if action_mask.numel() != logits.shape[1]:
        raise ValueError(
            f"action mask size mismatch: mask={action_mask.numel()}, "
            f"logits={logits.shape[1]}"
        )

    mask = action_mask.to(device=logits.device).unsqueeze(0)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(masked_logits, dim=1)

def objective_override_action(
    game: Any,
    char: Any,
    game_state: dict,
    helper: DefaultAttackerController,
) -> int | None:
    """設置と落下スパイク回収の最後の操作だけを確定させる。"""
    grid = game_state["grid"]
    row, col = int(char.pos[0]), int(char.pos[1])

    if (
        getattr(char, "has_spike", False)
        and grid[row, col] == 2
        and not bool(game_state.get("is_planted", False))
    ):
        return 8

    spike_pos = game_state.get("spike_pos")
    holder = next((
        other for other in game_state.get("chars", [])
        if character_is_alive(other)
        and getattr(other, "team", None) == char.team
        and getattr(other, "has_spike", False)
    ), None)
    if holder is not None or spike_pos is None:
        return None

    alive_attackers = [
        other for other in game_state.get("chars", [])
        if character_is_alive(other) and getattr(other, "team", None) == char.team
    ]
    if not alive_attackers:
        return None
    retriever = min(
        alive_attackers,
        key=lambda other: (
            helper.shortest_path_distance(other.pos, spike_pos, grid),
            str(getattr(other, "name", "")),
        ),
    )
    if retriever is not char:
        return None

    next_pos = helper.move_towards_target(
        char.pos, spike_pos, grid,
        chars=game_state.get("chars", []), moving_char=char,
    )
    dr = int(next_pos[0]) - row
    dc = int(next_pos[1]) - col
    return {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}.get((dr, dc))


def build_local_map(
    game: Any,
    viewer: Any,
    radius: int = LOCAL_MAP_RADIUS,
) -> list[list[int]]:
    grid = game.grid
    height, width = grid.shape

    viewer_row = int(viewer.pos[0])
    viewer_col = int(viewer.pos[1])
    viewer_team = viewer.team

    all_characters: dict[tuple[int, int], Any] = {}
    for char in getattr(game, "chars", []):
        if character_is_alive(char):
            all_characters[(int(char.pos[0]), int(char.pos[1]))] = char

    spike = getattr(game, "spike_pos", None)
    spike_position = (
        (int(spike[0]), int(spike[1]))
        if spike is not None
        else None
    )

    local_map: list[list[int]] = []

    for dr in range(-radius, radius + 1):
        row_values: list[int] = []

        for dc in range(-radius, radius + 1):
            row = viewer_row + dr
            col = viewer_col + dc
            position = (row, col)

            if not (0 <= row < height and 0 <= col < width):
                value = LOCAL_OUT_OF_MAP
            elif position == (viewer_row, viewer_col):
                value = LOCAL_SELF
            elif position in all_characters:
                other = all_characters[position]
                value = (
                    LOCAL_ALLY
                    if getattr(other, "team", None) == viewer_team
                    else LOCAL_ENEMY
                )
            elif spike_position is not None and position == spike_position:
                value = LOCAL_SPIKE
            elif grid[row, col] == 1:
                value = LOCAL_WALL
            elif grid[row, col] == 2:
                value = LOCAL_SITE
            else:
                value = LOCAL_EMPTY

            row_values.append(int(value))

        local_map.append(row_values)

    return local_map


def get_game_observation(game: Any, viewer: Any) -> dict:
    """学習・DAgger収集時と同一形式の観測を生成する。"""
    if viewer is None:
        raise ValueError("viewer must not be None")

    viewer_team = viewer.team
    enemy_team = "D" if viewer_team == "A" else "A"

    viewer_row = int(viewer.pos[0])
    viewer_col = int(viewer.pos[1])

    obs_dict = {
        "grid": game.grid.flatten().tolist(),
        "allies": [],
        "visible_enemies": [],
        "game_state": [],
        "spike_pos": [0, 0],
        "target_plant_pos": [0, 0],
        "visible_enemy_count": 0,
        "distance_to_site": 0.0,
        "viewer_pos": [viewer_row, viewer_col],
        "local_map": build_local_map(game, viewer),
        "valid_move_mask": build_valid_move_mask(game, viewer),

        # スパイク状態を明示する3特徴
        "spike_on_ground": (
            1 if getattr(game, "spike_pos", None) is not None else 0
        ),
        "ally_has_spike": (
            1
            if any(
                getattr(other, "team", None) == viewer_team
                and character_is_alive(other)
                and getattr(other, "has_spike", False)
                for other in game.chars
            )
            else 0
        ),
        "viewer_has_spike": (
            1 if getattr(viewer, "has_spike", False) else 0
        ),
    }

    allies = [viewer] + [
        char
        for char in game.chars
        if char.team == viewer_team and char is not viewer
    ]

    for char in allies:
        obs_dict["allies"].append(
            {
                "name": char.name,
                "pos": [int(char.pos[0]), int(char.pos[1])],
                "rel_pos": [
                    int(char.pos[0]) - viewer_row,
                    int(char.pos[1]) - viewer_col,
                ],
                "hp": int(char.hp),
                "is_alive": 1 if character_is_alive(char) else 0,
                "has_spike": 1 if getattr(char, "has_spike", False) else 0,
                "recon_cd": 1 if getattr(char, "recon_charges", 0) > 0 else 0,
                "flash_cd": 1 if getattr(char, "flash_charges", 0) > 0 else 0,
                "smoke_cd": 1 if getattr(char, "smoke_charges", 0) > 0 else 0,
            }
        )

    for char in game.chars:
        if char.team == enemy_team and character_is_alive(char):
            obs_dict["visible_enemies"].append(
                {
                    "name": char.name,
                    "rel_pos": [
                        int(char.pos[0]) - viewer_row,
                        int(char.pos[1]) - viewer_col,
                    ],
                    "hp": int(char.hp),
                }
            )

    obs_dict["visible_enemy_count"] = len(obs_dict["visible_enemies"])

    spike_pos = getattr(game, "spike_pos", None)
    if spike_pos is not None:
        obs_dict["spike_pos"] = [
            int(spike_pos[0]) - viewer_row,
            int(spike_pos[1]) - viewer_col,
        ]

    target_plant_pos = getattr(game, "target_plant_pos", None)
    if target_plant_pos is not None:
        plant_dr = int(target_plant_pos[0]) - viewer_row
        plant_dc = int(target_plant_pos[1]) - viewer_col

        obs_dict["target_plant_pos"] = [plant_dr, plant_dc]
        obs_dict["distance_to_site"] = float(abs(plant_dr) + abs(plant_dc))

    obs_dict["game_state"] = [
        float(getattr(game, "round_timer", 0.0)) / 100.0,
        1.0 if bool(getattr(game, "is_planted", False)) else 0.0,
    ]

    return obs_dict


def choose_ability_target(
    ability_name: str, char: Any, game_state: dict, expert_result: Any
) -> tuple[int, int]:
    """分類モデルが持たないアビリティ標的位置を既存AI等で補完する。"""
    if (
        isinstance(expert_result, tuple)
        and len(expert_result) == 2
        and isinstance(expert_result[1], dict)
        and str(expert_result[1].get("ability", "")).upper() == ability_name
    ):
        target = expert_result[1].get("target")
        if target is not None:
            return int(target[0]), int(target[1])

    enemies = [
        other
        for other in game_state.get("chars", [])
        if character_is_alive(other) and other.team != char.team
    ]
    if ability_name in {"SMOKE", "FLASH"} and enemies:
        closest = min(
            enemies,
            key=lambda enemy: max(
                abs(enemy.pos[0] - char.pos[0]),
                abs(enemy.pos[1] - char.pos[1]),
            ),
        )
        return int(closest.pos[0]), int(closest.pos[1])

    if ability_name == "RECON":
        target = game_state.get("planted_pos") or game_state.get("target_plant_pos")
        if target is not None:
            return int(target[0]), int(target[1])

    return int(char.pos[0]), int(char.pos[1])


def _unwrap_state_dict(loaded: Any) -> dict:
    """state_dict直保存とcheckpoint辞書の両方を受け付ける。"""
    if not isinstance(loaded, dict):
        raise TypeError("モデルファイルが辞書形式ではありません。")
    for key in ("state_dict", "model_state_dict", "policy_state_dict"):
        candidate = loaded.get(key)
        if isinstance(candidate, dict):
            return candidate
    return loaded


def load_policy(model_path: Path, device: str) -> tuple[PolicyNetwork, int]:
    """新旧チェックポイントから学習時と同じPolicyNetworkを復元する。"""
    loaded = torch.load(model_path, map_location=device)

    if not isinstance(loaded, dict):
        raise TypeError("モデルファイルが辞書形式ではありません。")

    if isinstance(loaded.get("model_state_dict"), dict):
        state_dict = loaded["model_state_dict"]
        obs_size = loaded.get("obs_size")
        num_actions = int(loaded.get("num_actions", NUM_ACTIONS))
    elif isinstance(loaded.get("state_dict"), dict):
        state_dict = loaded["state_dict"]
        obs_size = loaded.get("obs_size")
        num_actions = int(loaded.get("num_actions", NUM_ACTIONS))
    elif isinstance(loaded.get("policy_state_dict"), dict):
        state_dict = loaded["policy_state_dict"]
        obs_size = loaded.get("obs_size")
        num_actions = int(loaded.get("num_actions", NUM_ACTIONS))
    else:
        # 旧形式: state_dictを直接保存
        state_dict = loaded
        obs_size = None
        num_actions = NUM_ACTIONS

    if any(str(key).startswith("module.") for key in state_dict):
        state_dict = {
            str(key).removeprefix("module."): value
            for key, value in state_dict.items()
        }

    first_weight = state_dict.get("net.0.weight")
    if first_weight is None:
        raise KeyError(
            "モデル内に net.0.weight がありません。"
            "現在のPolicyNetwork形式か確認してください。"
        )

    if obs_size is None:
        obs_size = int(first_weight.shape[1])
    else:
        obs_size = int(obs_size)

    if num_actions != NUM_ACTIONS:
        raise ValueError(
            f"モデルの出力数が{NUM_ACTIONS}ではありません: {num_actions}"
        )

    model = PolicyNetwork(
        obs_size=obs_size,
        num_actions=num_actions,
    ).to(device)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "モデル構造が現在のtrain_bc.pyと一致しません。"
            "古い観測・旧ネットワークで学習したモデルは再学習が必要です。"
        ) from exc

    model.eval()
    return model, obs_size


ACTION_NAMES = {
    0: "MOVE_UP",
    1: "MOVE_DOWN",
    2: "MOVE_LEFT",
    3: "MOVE_RIGHT",
    4: "SMOKE",
    5: "FLASH",
    6: "RECON",
    7: "STOP",
    8: "PLANT",
}

ABILITY_BY_ACTION = {4: "SMOKE", 5: "FLASH", 6: "RECON"}
GROUP_ORDER = ["MOVE", "STOP", "PLANT", "SMOKE", "FLASH", "RECON", "UNKNOWN"]


def group_action_index(action_idx: int) -> str:
    if action_idx in DIRECTION_BY_ACTION:
        return "MOVE"
    return {4: "SMOKE", 5: "FLASH", 6: "RECON", 7: "STOP", 8: "PLANT"}.get(
        action_idx, "UNKNOWN"
    )


def group_controller_result(char: Any, result: Any) -> str:
    if result is None:
        return "UNKNOWN"
    if isinstance(result, tuple) and len(result) == 2:
        second = result[1]
        if isinstance(second, dict):
            ability = str(second.get("ability", "")).upper()
            return ability if ability in {"SMOKE", "FLASH", "RECON"} else "UNKNOWN"
        if isinstance(second, str):
            return "PLANT" if second.upper() == "PLANT" else "UNKNOWN"
    try:
        move_pos = result[0] if isinstance(result, tuple) else result
        return (
            "STOP" if list(map(int, move_pos)) == list(map(int, char.pos)) else "MOVE"
        )
    except (TypeError, ValueError, IndexError):
        return "UNKNOWN"


def character_is_alive(char: Any) -> bool:
    return bool(getattr(char, "is_alive", True))


def decode_evaluation_action(
    action_idx: int,
    char: Any,
    game_state: dict,
    expert_result: Any,
) -> Any:
    grid = game_state["grid"]

    if action_idx in DIRECTION_BY_ACTION:
        dr, dc = DIRECTION_BY_ACTION[action_idx]
        nr = int(char.pos[0] + dr)
        nc = int(char.pos[1] + dc)

        valid = (
            0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] != 1
        )
        if valid:
            for other in game_state.get("chars", []):
                if other is char or not character_is_alive(other):
                    continue
                if int(other.pos[0]) == nr and int(other.pos[1]) == nc:
                    valid = False
                    break
        return [nr, nc] if valid else list(char.pos)

    if action_idx in ABILITY_BY_ACTION:
        ability_name = ABILITY_BY_ACTION[action_idx]
        charge_name = {
            "SMOKE": "smoke_charges",
            "FLASH": "flash_charges",
            "RECON": "recon_charges",
        }[ability_name]
        if getattr(char, charge_name, 0) <= 0:
            return list(char.pos)
        target = choose_ability_target(ability_name, char, game_state, expert_result)
        return list(char.pos), {"ability": ability_name, "target": target}

    if action_idx == 8:
        row, col = int(char.pos[0]), int(char.pos[1])
        can_plant = (
            bool(getattr(char, "has_spike", False))
            and game_state["grid"][row, col] == 2
            and not bool(game_state.get("is_planted", False))
        )
        return (list(char.pos), "PLANT") if can_plant else list(char.pos)

    return list(char.pos)


@dataclass
class ControllerStatistics:
    predicted: Counter = field(default_factory=Counter)
    executed: Counter = field(default_factory=Counter)
    detailed: Counter = field(default_factory=Counter)
    invalid_detailed: Counter = field(default_factory=Counter)
    invalid_reasons: Counter = field(default_factory=Counter)
    total_predictions: int = 0
    predicted_moves: int = 0
    invalid_moves: int = 0
    confidence_sum: float = 0.0
    probability_sums: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)
    )
    plant_predictions: int = 0
    plant_executions: int = 0
    observed_plant_transitions: int = 0

    def merge(self, other: "ControllerStatistics") -> None:
        self.predicted.update(other.predicted)
        self.executed.update(other.executed)
        self.detailed.update(other.detailed)
        self.invalid_detailed.update(other.invalid_detailed)
        self.invalid_reasons.update(other.invalid_reasons)
        self.total_predictions += other.total_predictions
        self.predicted_moves += other.predicted_moves
        self.invalid_moves += other.invalid_moves
        self.confidence_sum += other.confidence_sum
        self.probability_sums += other.probability_sums
        self.plant_predictions += other.plant_predictions
        self.plant_executions += other.plant_executions
        self.observed_plant_transitions += other.observed_plant_transitions


class EvaluationAttackerController:
    """学習済みモデルだけでアタッカーを操作し、行動統計を収集する。"""

    def __init__(self, model: torch.nn.Module, obs_size: int, device: str) -> None:
        self.model = model
        self.obs_size = int(obs_size)
        self.device = device
        self.target_helper = DefaultAttackerController()
        self.game = None
        self.stats = ControllerStatistics()
        self._plant_seen_this_round = False
        self._last_round_marker = None

        # 無効移動デバッグ
        # 逐次表示せず、先頭の一定件数を保存して試合終了後にまとめて出力する。
        self.invalid_debug_limit = 20
        self.invalid_debug_records: list[dict[str, Any]] = []

    def set_game(self, game: VisualFPSBattle) -> None:
        self.game = game

    def reset_round(self) -> None:
        if hasattr(self.target_helper, "reset_round"):
            self.target_helper.reset_round()
        self._plant_seen_this_round = False
        self._last_round_marker = getattr(self.game, "current_round", None)

    def _update_real_plant_transition(self) -> None:
        if self.game is None:
            return
        current_round = getattr(self.game, "current_round", None)
        if current_round != self._last_round_marker:
            self._last_round_marker = current_round
            self._plant_seen_this_round = False
        if (
            bool(getattr(self.game, "is_planted", False))
            and not self._plant_seen_this_round
        ):
            self.stats.observed_plant_transitions += 1
            self._plant_seen_this_round = True

    def _invalid_move_reason(
        self,
        action_idx: int,
        char: Any,
        game_state: dict,
    ) -> tuple[bool, str, tuple[int, int]]:
        """無効移動か、理由、移動先座標を返す。"""
        if action_idx not in DIRECTION_BY_ACTION:
            return False, "NOT_MOVE", (int(char.pos[0]), int(char.pos[1]))

        grid = game_state["grid"]
        dr, dc = DIRECTION_BY_ACTION[action_idx]
        nr = int(char.pos[0] + dr)
        nc = int(char.pos[1] + dc)
        target = (nr, nc)

        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            return True, "OUT_OF_MAP", target

        if grid[nr, nc] == 1:
            return True, "WALL", target

        for other in game_state.get("chars", []):
            if other is char or not character_is_alive(other):
                continue
            if int(other.pos[0]) == nr and int(other.pos[1]) == nc:
                relation = "ALLY_BLOCK" if other.team == char.team else "ENEMY_BLOCK"
                other_name = str(getattr(other, "name", "UNKNOWN"))
                return True, f"{relation} ({other_name})", target

        return False, "FREE", target

    def _render_local_map(
        self,
        char: Any,
        game_state: dict,
        radius: int = 2,
    ) -> str:
        """キャラクター周辺をASCIIマップとして返す。"""
        grid = game_state["grid"]
        center_r = int(char.pos[0])
        center_c = int(char.pos[1])

        occupied: dict[tuple[int, int], str] = {}
        for other in game_state.get("chars", []):
            if not character_is_alive(other):
                continue
            pos = (int(other.pos[0]), int(other.pos[1]))
            if other is char:
                occupied[pos] = "P"
            elif other.team == char.team:
                occupied[pos] = "A"
            else:
                occupied[pos] = "E"

        lines: list[str] = []
        for r in range(center_r - radius, center_r + radius + 1):
            row: list[str] = []
            for c in range(center_c - radius, center_c + radius + 1):
                if (r, c) in occupied:
                    row.append(occupied[(r, c)])
                elif not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                    row.append("X")
                elif grid[r, c] == 1:
                    row.append("#")
                else:
                    row.append(".")
            lines.append("".join(row))
        return "\n".join(lines)

    def _possible_cardinal_moves(
        self,
        char: Any,
        game_state: dict,
    ) -> dict[str, tuple[bool, str]]:
        """上下左右それぞれの移動可否と理由を返す。"""
        result: dict[str, tuple[bool, str]] = {}
        for action_idx in range(4):
            invalid, reason, _ = self._invalid_move_reason(action_idx, char, game_state)
            result[ACTION_NAMES[action_idx]] = (not invalid, reason)
        return result

    def _record_invalid_move_debug(
        self,
        action_idx: int,
        char: Any,
        game_state: dict,
        probability_array: np.ndarray,
        reason: str,
        target: tuple[int, int],
    ) -> None:
        """最初の一定件数の無効移動を保存する。"""
        if len(self.invalid_debug_records) >= self.invalid_debug_limit:
            return

        current_pos = (int(char.pos[0]), int(char.pos[1]))
        tick = getattr(self.game, "tick", getattr(self.game, "current_tick", "UNKNOWN"))
        round_no = getattr(self.game, "current_round", "UNKNOWN")

        possible_moves: dict[str, dict[str, Any]] = {}
        for name, (possible, move_reason) in self._possible_cardinal_moves(
            char, game_state
        ).items():
            possible_moves[name] = {
                "possible": bool(possible),
                "reason": str(move_reason),
            }

        probabilities = {
            ACTION_NAMES[idx]: (
                float(probability_array[idx])
                if idx < len(probability_array)
                else 0.0
            )
            for idx in range(NUM_ACTIONS)
        }

        self.invalid_debug_records.append(
            {
                "index": len(self.invalid_debug_records) + 1,
                "character": str(getattr(char, "name", "UNKNOWN")),
                "team": str(getattr(char, "team", "UNKNOWN")),
                "round": round_no,
                "tick": tick,
                "position": current_pos,
                "predicted": ACTION_NAMES.get(action_idx, str(action_idx)),
                "target": (int(target[0]), int(target[1])),
                "reason": str(reason),
                "probabilities": probabilities,
                "possible_cardinal_moves": possible_moves,
                "local_map": self._render_local_map(char, game_state, radius=2),
            }
        )

    def print_invalid_move_debug_records(self) -> None:
        """保存した無効移動を試合終了後にまとめて表示する。"""
        if not self.invalid_debug_records:
            print("\nINVALID MOVE DEBUG: 記録なし")
            return

        print("\n" + "=" * 76)
        print(
            f"INVALID MOVE DEBUG RECORDS "
            f"({len(self.invalid_debug_records)}/{self.invalid_debug_limit})"
        )
        print("=" * 76)

        for record in self.invalid_debug_records:
            print("\n" + "=" * 68)
            print(f"INVALID MOVE DEBUG #{record['index']}")
            print("=" * 68)
            print(f"Character : {record['character']}")
            print(f"Team      : {record['team']}")
            print(f"Round     : {record['round']}")
            print(f"Tick      : {record['tick']}")
            print(f"Position  : {record['position']}")
            print(f"Predicted : {record['predicted']}")
            print(f"Target    : {record['target']}")
            print(f"Reason    : {record['reason']}")

            print("\nNetwork probabilities")
            print("-" * 40)
            for idx in range(NUM_ACTIONS):
                name = ACTION_NAMES[idx]
                print(f"{name:<18}: {record['probabilities'][name]:.4f}")

            print("\nPossible cardinal moves")
            print("-" * 40)
            for action_idx in range(4):
                name = ACTION_NAMES[action_idx]
                move = record["possible_cardinal_moves"][name]
                possible_text = "YES" if move["possible"] else "NO "
                print(f"{name:<18}: {possible_text:<3} ({move['reason']})")

            print("\n5x5 local map")
            print("-" * 40)
            print(record["local_map"])
            print("Legend: P=self A=ally E=enemy #=wall .=free X=out_of_map")
            print("=" * 68)

    def save_invalid_move_debug_records(self, path: Path) -> None:
        """保存した無効移動をJSONファイルへ出力する。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                self.invalid_debug_records,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def decide_move(self, char: Any, game_state: dict) -> Any:
        if self.game is None:
            raise RuntimeError(
                "EvaluationAttackerController.set_game(game) が未実行です。"
            )

        self._update_real_plant_transition()
        observation = get_game_observation(self.game, char)
        obs_vec = observation_to_vector(observation)
        if obs_vec.shape[0] != self.obs_size:
            raise ValueError(
                f"観測次元がモデルと一致しません: observation={obs_vec.shape[0]}, "
                f"model={self.obs_size}"
            )

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_vec).unsqueeze(0).to(self.device)
            logits = self.model(obs_tensor)
            action_mask = build_action_mask(self.game, char, game_state)
            probabilities = masked_action_probabilities(logits, action_mask)
            action_idx = int(probabilities.argmax(dim=1).item())
            forced_action = objective_override_action(
                self.game, char, game_state, self.target_helper
            )
            if forced_action is not None:
                action_idx = forced_action
            confidence = float(probabilities[0, action_idx].item())
            probability_array = probabilities[0].detach().cpu().numpy()

        action_group = group_action_index(action_idx)
        self.stats.total_predictions += 1
        self.stats.predicted[action_group] += 1
        self.stats.detailed[ACTION_NAMES.get(action_idx, f"UNKNOWN_{action_idx}")] += 1
        self.stats.confidence_sum += confidence
        if len(probability_array) == NUM_ACTIONS:
            self.stats.probability_sums += probability_array

        if action_idx in DIRECTION_BY_ACTION:
            self.stats.predicted_moves += 1
            is_invalid, invalid_reason, target_cell = self._invalid_move_reason(
                action_idx,
                char,
                game_state,
            )
            if is_invalid:
                self.stats.invalid_moves += 1
                self.stats.invalid_detailed[ACTION_NAMES[action_idx]] += 1
                reason_group = invalid_reason.split(" (", 1)[0]
                self.stats.invalid_reasons[reason_group] += 1

                self._record_invalid_move_debug(
                    action_idx=action_idx,
                    char=char,
                    game_state=game_state,
                    probability_array=probability_array,
                    reason=invalid_reason,
                    target=target_cell,
                )
        if action_idx == 8:
            self.stats.plant_predictions += 1

        # 現モデルは標的座標を出力しないため、アビリティ標的だけ既存AIで補う。
        expert_result = self.target_helper.decide_move(char, game_state)
        result = decode_evaluation_action(action_idx, char, game_state, expert_result)
        executed_group = group_controller_result(char, result)
        self.stats.executed[executed_group] += 1
        if executed_group == "PLANT":
            self.stats.plant_executions += 1
        return result


@dataclass
class ModelEvaluationResult:
    model_path: str
    matches: int = 0
    attacker_match_wins: int = 0
    defender_match_wins: int = 0
    draws_or_unknown: int = 0
    attacker_rounds: int = 0
    defender_rounds: int = 0
    attacker_kills: int = 0
    attacker_deaths: int = 0
    attacker_13_0: int = 0
    defender_13_0: int = 0
    matches_with_plant: int = 0
    controller_stats: ControllerStatistics = field(default_factory=ControllerStatistics)

    @property
    def win_rate(self) -> float:
        return self.attacker_match_wins / self.matches if self.matches else 0.0

    @property
    def average_attacker_rounds(self) -> float:
        return self.attacker_rounds / self.matches if self.matches else 0.0

    @property
    def average_defender_rounds(self) -> float:
        return self.defender_rounds / self.matches if self.matches else 0.0

    @property
    def average_round_difference(self) -> float:
        return self.average_attacker_rounds - self.average_defender_rounds

    @property
    def average_attacker_kills(self) -> float:
        return self.attacker_kills / self.matches if self.matches else 0.0

    @property
    def average_attacker_deaths(self) -> float:
        return self.attacker_deaths / self.matches if self.matches else 0.0


class ModelEvaluator:
    """複数のBC / DAggerモデルをHeadless試合で比較する。"""

    def __init__(
        self,
        model_paths: list[Path],
        matches_per_model: int = 20,
        device: str | None = None,
        seed: int = 42,
        invalid_debug_limit: int = 20,
        invalid_debug_dir: Path = Path("evaluation_debug"),
        print_invalid_debug: bool = True,
    ) -> None:
        if matches_per_model <= 0:
            raise ValueError("matches_per_modelは1以上にしてください。")
        self.model_paths = [Path(path) for path in model_paths]
        self.matches_per_model = int(matches_per_model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(seed)
        self.invalid_debug_limit = max(0, int(invalid_debug_limit))
        self.invalid_debug_dir = Path(invalid_debug_dir)
        self.print_invalid_debug = bool(print_invalid_debug)

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _make_game(self, controller: EvaluationAttackerController) -> VisualFPSBattle:
        attacker_roster, defender_roster = build_two_balanced_rosters()
        kwargs = {
            "headless": True,
            "attacker_roster": attacker_roster,
            "defender_roster": defender_roster,
        }
        signature = inspect.signature(VisualFPSBattle.__init__)
        if "disable_side_swap" in signature.parameters:
            kwargs["disable_side_swap"] = True
        game = VisualFPSBattle(
            NEW_MAZE_STR,
            controller,
            DefaultDefenderController(),
            **kwargs,
        )
        controller.set_game(game)
        return game

    def _extract_team_kd(self, game: VisualFPSBattle, team: str) -> tuple[int, int]:
        names = {
            str(getattr(char, "name", ""))
            for char in getattr(game, "chars", [])
            if getattr(char, "team", None) == team
        }
        kills = deaths = 0
        match_stats = getattr(game, "match_stats", {})
        if isinstance(match_stats, dict):
            for name in names:
                data = match_stats.get(name, {})
                if isinstance(data, dict):
                    kills += int(data.get("kills", 0))
                    deaths += int(data.get("deaths", 0))
        return kills, deaths

    def evaluate_model(self, model_path: Path) -> ModelEvaluationResult:
        if not model_path.exists():
            raise FileNotFoundError(f"モデルがありません: {model_path}")
        model, obs_size = load_policy(model_path, self.device)
        result = ModelEvaluationResult(model_path=str(model_path))

        print("\n" + "=" * 76)
        print(f"評価開始: {model_path}")
        print(f"device={self.device} matches={self.matches_per_model}")
        print("=" * 76)

        for match_index in range(self.matches_per_model):
            self._seed_everything(self.seed + match_index)
            controller = EvaluationAttackerController(model, obs_size, self.device)
            controller.invalid_debug_limit = self.invalid_debug_limit
            game = self._make_game(controller)
            game.run_headless_loop()
            controller._update_real_plant_transition()

            debug_path = (
                self.invalid_debug_dir
                / model_path.stem
                / f"match_{match_index + 1:03d}_invalid_moves.json"
            )
            controller.save_invalid_move_debug_records(debug_path)
            if self.print_invalid_debug:
                controller.print_invalid_move_debug_records()
            print(f"Invalid move debug saved: {debug_path}")

            attacker_score = int(getattr(game, "attacker_wins", 0))
            defender_score = int(getattr(game, "defender_wins", 0))
            result.matches += 1
            result.attacker_rounds += attacker_score
            result.defender_rounds += defender_score

            if attacker_score > defender_score:
                result.attacker_match_wins += 1
            elif defender_score > attacker_score:
                result.defender_match_wins += 1
            else:
                result.draws_or_unknown += 1

            result.attacker_13_0 += int(attacker_score == 13 and defender_score == 0)
            result.defender_13_0 += int(defender_score == 13 and attacker_score == 0)
            result.matches_with_plant += int(
                controller.stats.observed_plant_transitions > 0
            )

            kills, deaths = self._extract_team_kd(game, "A")
            result.attacker_kills += kills
            result.attacker_deaths += deaths
            result.controller_stats.merge(controller.stats)

            print(
                f"[{match_index + 1:>3}/{self.matches_per_model}] "
                f"A {attacker_score:>2} - {defender_score:<2} D | "
                f"plant={controller.stats.observed_plant_transitions} | "
                f"invalid={controller.stats.invalid_moves}/"
                f"{controller.stats.predicted_moves}"
            )
        return result

    def evaluate_all(self) -> list[ModelEvaluationResult]:
        return [self.evaluate_model(path) for path in self.model_paths]

    def _print_counter(self, title: str, counter: Counter, total: int) -> None:
        print(f"\n{title}\n" + "-" * 60)
        for name in GROUP_ORDER:
            count = int(counter.get(name, 0))
            percentage = count / total * 100.0 if total else 0.0
            print(f"{name:<18}: {count:>9} ({percentage:>6.2f}%)")

    def _print_detailed_action_counter(
        self,
        title: str,
        counter: Counter,
        total: int,
    ) -> None:
        print(f"\n{title}\n" + "-" * 60)
        for action_idx in range(NUM_ACTIONS):
            name = ACTION_NAMES[action_idx]
            count = int(counter.get(name, 0))
            percentage = count / total * 100.0 if total else 0.0
            print(f"{name:<18}: {count:>9} ({percentage:>6.2f}%)")

    def _print_invalid_by_direction(self, stats: ControllerStatistics) -> None:
        print("\nInvalid MOVE by Direction\n" + "-" * 60)
        for action_idx in range(4):
            name = ACTION_NAMES[action_idx]
            predicted = int(stats.detailed.get(name, 0))
            invalid = int(stats.invalid_detailed.get(name, 0))
            rate = invalid / predicted * 100.0 if predicted else 0.0
            print(
                f"{name:<18}: predicted={predicted:>7} "
                f"invalid={invalid:>7} ({rate:>6.2f}%)"
            )

    def _print_invalid_by_reason(self, stats: ControllerStatistics) -> None:
        print("\nInvalid MOVE by Reason\n" + "-" * 60)
        total_invalid = stats.invalid_moves
        known = ("WALL", "ALLY_BLOCK", "ENEMY_BLOCK", "OUT_OF_MAP")
        for reason in known:
            count = int(stats.invalid_reasons.get(reason, 0))
            rate = count / total_invalid * 100.0 if total_invalid else 0.0
            print(f"{reason:<18}: {count:>9} ({rate:>6.2f}%)")

        other_count = sum(
            int(count)
            for reason, count in stats.invalid_reasons.items()
            if reason not in known
        )
        if other_count:
            rate = other_count / total_invalid * 100.0 if total_invalid else 0.0
            print(f"{'OTHER':<18}: {other_count:>9} ({rate:>6.2f}%)")

    def print_result(self, result: ModelEvaluationResult) -> None:
        stats = result.controller_stats
        total = stats.total_predictions
        print("\n" + "=" * 76)
        print(f"MODEL: {result.model_path}")
        print("=" * 76)
        print(f"試合数                  : {result.matches}")
        print(f"アタッカー勝利          : {result.attacker_match_wins}")
        print(f"ディフェンダー勝利      : {result.defender_match_wins}")
        print(f"勝率                    : {result.win_rate * 100.0:.2f}%")
        print(f"平均取得ラウンド        : {result.average_attacker_rounds:.2f}")
        print(f"平均失点ラウンド        : {result.average_defender_rounds:.2f}")
        print(f"平均ラウンド差          : {result.average_round_difference:+.2f}")
        print(f"アタッカー13-0          : {result.attacker_13_0}")
        print(f"ディフェンダー13-0      : {result.defender_13_0}")
        print(f"平均アタッカーキル      : {result.average_attacker_kills:.2f}")
        print(f"平均アタッカーデス      : {result.average_attacker_deaths:.2f}")
        print(f"設置が確認された試合    : {result.matches_with_plant}/{result.matches}")

        self._print_counter("Model Predicted Actions", stats.predicted, total)
        self._print_detailed_action_counter(
            "Model Predicted Action Details",
            stats.detailed,
            total,
        )
        executed_total = sum(stats.executed.values())
        self._print_counter("Executed Actions", stats.executed, executed_total)

        invalid_rate = (
            stats.invalid_moves / stats.predicted_moves
            if stats.predicted_moves
            else 0.0
        )
        average_confidence = stats.confidence_sum / total if total else 0.0
        print("\nInvalid Move Statistics\n" + "-" * 60)
        print(f"Predicted MOVE          : {stats.predicted_moves}")
        print(f"Invalid MOVE            : {stats.invalid_moves}")
        print(f"Invalid / MOVE          : {invalid_rate * 100.0:.2f}%")
        self._print_invalid_by_direction(stats)
        self._print_invalid_by_reason(stats)
        print("\nPlant / Confidence\n" + "-" * 60)
        print(f"PLANT predictions       : {stats.plant_predictions}")
        print(f"PLANT executions        : {stats.plant_executions}")
        print(f"Observed plant rounds   : {stats.observed_plant_transitions}")
        print(f"Average max probability : {average_confidence:.4f}")

        if total:
            average_probabilities = stats.probability_sums / total
            print("\nAverage Probability by Action\n" + "-" * 60)
            for action_idx in range(NUM_ACTIONS):
                print(
                    f"{ACTION_NAMES[action_idx]:<18}: {average_probabilities[action_idx]:.4f}"
                )

    def print_comparison(self, results: list[ModelEvaluationResult]) -> None:
        print("\n" + "=" * 100)
        print("MODEL COMPARISON")
        print("=" * 100)
        print(
            f"{'Model':<32}{'Win%':>8}{'Avg A':>8}{'Avg D':>8}{'Diff':>8}"
            f"{'Invalid%':>10}{'PlantMatch%':>13}{'Confidence':>12}"
        )
        print("-" * 100)
        for result in results:
            stats = result.controller_stats
            invalid_rate = (
                stats.invalid_moves / stats.predicted_moves
                if stats.predicted_moves
                else 0.0
            )
            plant_match_rate = (
                result.matches_with_plant / result.matches if result.matches else 0.0
            )
            confidence = (
                stats.confidence_sum / stats.total_predictions
                if stats.total_predictions
                else 0.0
            )
            print(
                f"{Path(result.model_path).name:<32}"
                f"{result.win_rate * 100.0:>7.2f}%"
                f"{result.average_attacker_rounds:>8.2f}"
                f"{result.average_defender_rounds:>8.2f}"
                f"{result.average_round_difference:>+8.2f}"
                f"{invalid_rate * 100.0:>9.2f}%"
                f"{plant_match_rate * 100.0:>12.2f}%"
                f"{confidence:>12.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        type=Path,
        default=[
            Path("policy_bc_final.pt"),
            Path("policy_dagger_iter_1.pt"),
            Path("policy_dagger_iter_2.pt"),
            Path("policy_dagger_iter_3.pt"),
            Path("policy_dagger_final.pt"),
        ],
    )
    parser.add_argument("--matches", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--invalid-debug-limit",
        type=int,
        default=20,
        help="各試合で保存する無効移動の先頭件数。0で無効化。",
    )
    parser.add_argument(
        "--invalid-debug-dir",
        type=Path,
        default=Path("evaluation_debug"),
        help="無効移動JSONの保存先。",
    )
    parser.add_argument(
        "--no-print-invalid-debug",
        action="store_true",
        help="試合終了後の詳細表示を省略する。JSON保存は行う。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing_models = [path for path in args.models if path.exists()]
    for path in args.models:
        if not path.exists():
            print(f"スキップ: モデルがありません: {path}")
    if not existing_models:
        raise FileNotFoundError("評価可能なモデルが1つもありません。")

    evaluator = ModelEvaluator(
        model_paths=existing_models,
        matches_per_model=args.matches,
        device=args.device,
        seed=args.seed,
        invalid_debug_limit=args.invalid_debug_limit,
        invalid_debug_dir=args.invalid_debug_dir,
        print_invalid_debug=not args.no_print_invalid_debug,
    )
    results = evaluator.evaluate_all()
    for result in results:
        evaluator.print_result(result)
    evaluator.print_comparison(results)


if __name__ == "__main__":
    main()
