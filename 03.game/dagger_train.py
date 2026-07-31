"""
DAgger（Dataset Aggregation）完全版

対応ファイル:
- controllers.py
- collect_demos.py
- train_bc.py
- run_game.py
- map_data.py
- roster_utils.py

対応済み:
- viewer_pos
- local_map
- valid_move_mask
- teacher_action_valid
- 新しいチェックポイント形式
- train_bc.observation_to_vector() の共通利用
- 占有マスを含む無効移動判定
- DAgger統計表示
"""

import inspect
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from controllers import DefaultAttackerController, DefaultDefenderController
from map_data import NEW_MAZE_STR
from roster_utils import build_two_balanced_rosters
from run_game import VisualFPSBattle
from train_bc import (
    ACTION_NAMES,
    BCTrainer,
    NUM_ACTIONS,
    PolicyNetwork,
    observation_to_vector,
    set_global_seed,
)


# ============================================================================
# 設定
# ============================================================================

BASE_DEMO_FILE = Path("demos/rule_based_demos.json")
AGGREGATED_DEMO_FILE = Path("demos/dagger_aggregated_demos.json")

INITIAL_MODEL_FILE = Path("policy_bc_final.pt")
FINAL_MODEL_FILE = Path("policy_dagger_final.pt")

SAMPLES_PER_ITERATION = 12000
DAGGER_ITERATIONS = 3
BETA_SCHEDULE = [0.8, 0.6, 0.4]

MAX_MATCHES_PER_ITERATION = 20

TRAIN_EPOCHS = 80
TRAIN_BATCH_SIZE = 64
TRAIN_VAL_SPLIT = 0.10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
set_global_seed(SEED)


# ============================================================================
# 定数
# ============================================================================

DIRECTION_BY_ACTION = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}

ABILITY_BY_ACTION = {
    4: "SMOKE",
    5: "FLASH",
    6: "RECON",
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


# ============================================================================
# Observation
# ============================================================================

def character_is_alive(char):
    return bool(getattr(char, "is_alive", True))


def occupied_positions(game, viewer=None):
    occupied = {}

    for char in getattr(game, "chars", []):
        if not character_is_alive(char):
            continue
        if char is viewer:
            continue

        pos = getattr(char, "pos", None)
        if pos is None or len(pos) != 2:
            continue

        occupied[(int(pos[0]), int(pos[1]))] = char

    return occupied


def is_valid_destination(game, viewer, row, col):
    row = int(row)
    col = int(col)

    grid = game.grid
    height, width = grid.shape

    if not (0 <= row < height and 0 <= col < width):
        return False

    if grid[row, col] == 1:
        return False

    if (row, col) in occupied_positions(game, viewer):
        return False

    return True


def build_valid_move_mask(game, viewer):
    row = int(viewer.pos[0])
    col = int(viewer.pos[1])

    mask = []
    for action_idx in range(4):
        dr, dc = DIRECTION_BY_ACTION[action_idx]
        mask.append(
            1 if is_valid_destination(game, viewer, row + dr, col + dc) else 0
        )

    return mask




def build_action_mask(game, char, game_state):
    """現在の状態で実行可能な9アクションをTrueで返す。"""
    mask = torch.ones(NUM_ACTIONS, dtype=torch.bool)

    move_mask = build_valid_move_mask(game, char)
    for action_idx in range(4):
        mask[action_idx] = bool(move_mask[action_idx])

    mask[4] = getattr(char, "smoke_charges", 0) > 0
    mask[5] = getattr(char, "flash_charges", 0) > 0
    mask[6] = getattr(char, "recon_charges", 0) > 0
    mask[7] = True

    target = game_state.get("target_plant_pos")
    mask[8] = bool(
        getattr(char, "has_spike", False)
        and target is not None
        and list(map(int, char.pos)) == list(map(int, target))
        and not bool(game_state.get("is_planted", False))
    )

    return mask


def masked_action_probabilities(logits, action_mask):
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

def build_local_map(game, viewer, radius=LOCAL_MAP_RADIUS):
    grid = game.grid
    height, width = grid.shape

    viewer_row = int(viewer.pos[0])
    viewer_col = int(viewer.pos[1])
    viewer_team = viewer.team

    occupied = {}
    for char in getattr(game, "chars", []):
        if not character_is_alive(char):
            continue

        occupied[
            (int(char.pos[0]), int(char.pos[1]))
        ] = char

    spike_pos = getattr(game, "spike_pos", None)
    if spike_pos is not None:
        spike_pos = (int(spike_pos[0]), int(spike_pos[1]))

    local_map = []

    for dr in range(-radius, radius + 1):
        row_values = []

        for dc in range(-radius, radius + 1):
            world_row = viewer_row + dr
            world_col = viewer_col + dc
            world_pos = (world_row, world_col)

            if not (0 <= world_row < height and 0 <= world_col < width):
                value = LOCAL_OUT_OF_MAP
            elif world_pos == (viewer_row, viewer_col):
                value = LOCAL_SELF
            elif world_pos in occupied:
                other = occupied[world_pos]
                value = (
                    LOCAL_ALLY
                    if getattr(other, "team", None) == viewer_team
                    else LOCAL_ENEMY
                )
            elif spike_pos is not None and world_pos == spike_pos:
                value = LOCAL_SPIKE
            elif grid[world_row, world_col] == 1:
                value = LOCAL_WALL
            elif grid[world_row, world_col] == 2:
                value = LOCAL_SITE
            else:
                value = LOCAL_EMPTY

            row_values.append(int(value))

        local_map.append(row_values)

    return local_map


def get_game_observation(game, viewer):
    """collect_demos.pyと同じ観測形式を作る。"""
    if viewer is None:
        raise ValueError("viewer must not be None")

    viewer_team = viewer.team
    enemy_team = "D" if viewer_team == "A" else "A"

    viewer_row = int(viewer.pos[0])
    viewer_col = int(viewer.pos[1])

    obs = {
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
    }

    allies = [viewer] + [
        char
        for char in game.chars
        if char.team == viewer_team and char is not viewer
    ]

    for char in allies:
        obs["allies"].append(
            {
                "name": char.name,
                "pos": [int(char.pos[0]), int(char.pos[1])],
                "rel_pos": [
                    int(char.pos[0] - viewer_row),
                    int(char.pos[1] - viewer_col),
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
            obs["visible_enemies"].append(
                {
                    "name": char.name,
                    "rel_pos": [
                        int(char.pos[0] - viewer_row),
                        int(char.pos[1] - viewer_col),
                    ],
                    "hp": int(char.hp),
                }
            )

    obs["visible_enemy_count"] = len(obs["visible_enemies"])

    if getattr(game, "spike_pos", None) is not None:
        obs["spike_pos"] = [
            int(game.spike_pos[0] - viewer_row),
            int(game.spike_pos[1] - viewer_col),
        ]

    if getattr(game, "target_plant_pos", None) is not None:
        plant_dr = int(game.target_plant_pos[0] - viewer_row)
        plant_dc = int(game.target_plant_pos[1] - viewer_col)

        obs["target_plant_pos"] = [plant_dr, plant_dc]
        obs["distance_to_site"] = float(abs(plant_dr) + abs(plant_dc))

    obs["game_state"] = [
        float(getattr(game, "round_timer", 0.0)) / 100.0,
        1.0 if bool(getattr(game, "is_planted", False)) else 0.0,
    ]

    return obs


# ============================================================================
# Action変換
# ============================================================================

def result_to_action_data(char, result):
    action_data = {
        "char": char.name,
        "team": char.team,
        "ability": None,
        "special": None,
    }

    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[1], dict)
    ):
        move_pos, ability = result

        action_data["move"] = [
            int(move_pos[0]),
            int(move_pos[1]),
        ]
        action_data["ability"] = ability.get("ability")

        target = ability.get("target", char.pos)
        action_data["ability_target"] = [
            int(target[0]),
            int(target[1]),
        ]

    elif (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[1], str)
    ):
        move_pos, special = result

        action_data["move"] = [
            int(move_pos[0]),
            int(move_pos[1]),
        ]
        action_data["special"] = special

    else:
        if not isinstance(result, (list, tuple, np.ndarray)) or len(result) != 2:
            raise ValueError(f"未対応のController戻り値: {result!r}")

        action_data["move"] = [
            int(result[0]),
            int(result[1]),
        ]

    return action_data


def teacher_action_is_valid(game, viewer, result):
    if result is None:
        return False

    if isinstance(result, tuple) and len(result) == 2:
        if isinstance(result[1], (dict, str)):
            return True
        move_pos = result[0]
    else:
        move_pos = result

    if not isinstance(move_pos, (list, tuple, np.ndarray)):
        return False

    if len(move_pos) != 2:
        return False

    nr = int(move_pos[0])
    nc = int(move_pos[1])

    current = (int(viewer.pos[0]), int(viewer.pos[1]))
    if (nr, nc) == current:
        return True

    return is_valid_destination(game, viewer, nr, nc)


def choose_ability_target(ability_name, char, game_state, expert_result):
    if (
        isinstance(expert_result, tuple)
        and len(expert_result) == 2
        and isinstance(expert_result[1], dict)
        and expert_result[1].get("ability") == ability_name
    ):
        target = expert_result[1].get("target")
        if target is not None:
            return int(target[0]), int(target[1])

    enemies = [
        other
        for other in game_state.get("chars", [])
        if character_is_alive(other)
        and getattr(other, "team", None) != char.team
    ]

    if ability_name in {"SMOKE", "FLASH"} and enemies:
        closest = min(
            enemies,
            key=lambda enemy: max(
                abs(int(enemy.pos[0]) - int(char.pos[0])),
                abs(int(enemy.pos[1]) - int(char.pos[1])),
            ),
        )
        return int(closest.pos[0]), int(closest.pos[1])

    if ability_name == "RECON":
        target = (
            game_state.get("planted_pos")
            or game_state.get("target_plant_pos")
        )
        if target is not None:
            return int(target[0]), int(target[1])

    return int(char.pos[0]), int(char.pos[1])


def decode_model_action(action_idx, char, game_state, expert_result):
    """モデル出力をController戻り値へ変換する。"""
    if action_idx in DIRECTION_BY_ACTION:
        dr, dc = DIRECTION_BY_ACTION[action_idx]
        nr = int(char.pos[0]) + dr
        nc = int(char.pos[1]) + dc

        grid = game_state["grid"]

        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            return list(char.pos)

        if grid[nr, nc] == 1:
            return list(char.pos)

        for other in game_state.get("chars", []):
            if other is char or not character_is_alive(other):
                continue

            if (
                int(other.pos[0]) == nr
                and int(other.pos[1]) == nc
            ):
                return list(char.pos)

        return [nr, nc]

    if action_idx in ABILITY_BY_ACTION:
        ability_name = ABILITY_BY_ACTION[action_idx]

        charge_attribute = {
            "SMOKE": "smoke_charges",
            "FLASH": "flash_charges",
            "RECON": "recon_charges",
        }[ability_name]

        if getattr(char, charge_attribute, 0) <= 0:
            return list(char.pos)

        target = choose_ability_target(
            ability_name,
            char,
            game_state,
            expert_result,
        )

        return list(char.pos), {
            "ability": ability_name,
            "target": target,
        }

    if action_idx == 8:
        # 不正な場所でのPLANT予測は停止へ変換
        target = game_state.get("target_plant_pos")
        if (
            getattr(char, "has_spike", False)
            and target is not None
            and list(map(int, char.pos)) == list(map(int, target))
        ):
            return list(char.pos), "PLANT"

        return list(char.pos)

    return list(char.pos)


# ============================================================================
# モデル読み込み
# ============================================================================

def load_policy(model_path, device):
    model_path = Path(model_path)
    checkpoint = torch.load(model_path, map_location=device)

    # 新形式
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
        obs_size = checkpoint.get("obs_size")

        if obs_size is None:
            first_weight = state_dict.get("net.0.weight")
            if first_weight is None:
                raise KeyError("net.0.weightが見つかりません")
            obs_size = int(first_weight.shape[1])

    # 旧state_dict形式も一応許可
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        first_weight = state_dict.get("net.0.weight")

        if first_weight is None:
            raise KeyError(
                "モデル内にmodel_state_dictまたはnet.0.weightがありません"
            )

        obs_size = int(first_weight.shape[1])

    else:
        raise TypeError("未対応のモデル保存形式です")

    model = PolicyNetwork(
        obs_size=int(obs_size),
        num_actions=NUM_ACTIONS,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model, int(obs_size)


# ============================================================================
# DAgger Controller
# ============================================================================

class DAggerAttackerController:
    GROUPED_ACTION_NAMES = [
        "MOVE",
        "STOP",
        "PLANT",
        "SMOKE",
        "FLASH",
        "RECON",
        "UNKNOWN",
    ]

    def __init__(
        self,
        model,
        obs_size,
        device,
        beta,
        record_buffer,
        target_samples,
    ):
        self.model = model
        self.obs_size = int(obs_size)
        self.device = torch.device(device)
        self.beta = float(beta)
        self.record_buffer = record_buffer
        self.target_samples = int(target_samples)

        self.expert = DefaultAttackerController()
        self.game = None

        self.expert_executions = 0
        self.model_executions = 0

        self.predicted_action_counts = Counter()
        self.executed_action_counts = Counter()
        self.expert_action_counts = Counter()
        self.predicted_detailed_counts = Counter()

        self.confusion_counts = defaultdict(Counter)

        self.predicted_move_count = 0
        self.invalid_move_count = 0
        self.invalid_teacher_count = 0

        self.confidence_sum = 0.0
        self.confidence_count = 0

        self.probability_sums = np.zeros(
            NUM_ACTIONS,
            dtype=np.float64,
        )
        self.probability_count = 0

    def set_game(self, game):
        self.game = game

    def reset_round(self):
        if hasattr(self.expert, "reset_round"):
            self.expert.reset_round()

    def _group_action_index(self, action_idx):
        if action_idx in DIRECTION_BY_ACTION:
            return "MOVE"
        if action_idx == 8:
            return "SMOKE"
        if action_idx == 9:
            return "FLASH"
        if action_idx == 10:
            return "RECON"
        if action_idx == 11:
            return "STOP"
        if action_idx == 8:
            return "PLANT"
        return "UNKNOWN"

    def _group_controller_result(self, char, result):
        if result is None:
            return "UNKNOWN"

        if isinstance(result, tuple) and len(result) == 2:
            second = result[1]

            if isinstance(second, dict):
                ability = second.get("ability")
                if ability in {"SMOKE", "FLASH", "RECON"}:
                    return ability
                return "UNKNOWN"

            if isinstance(second, str):
                if second == "PLANT":
                    return "PLANT"
                return "UNKNOWN"

        try:
            move_pos = result[0] if isinstance(result, tuple) else result

            if (
                int(move_pos[0]) == int(char.pos[0])
                and int(move_pos[1]) == int(char.pos[1])
            ):
                return "STOP"

            return "MOVE"
        except (TypeError, ValueError, IndexError):
            return "UNKNOWN"

    def _is_invalid_model_move(self, action_idx, char, game_state):
        if action_idx not in DIRECTION_BY_ACTION:
            return False

        dr, dc = DIRECTION_BY_ACTION[action_idx]
        nr = int(char.pos[0]) + dr
        nc = int(char.pos[1]) + dc

        grid = game_state["grid"]

        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            return True

        if grid[nr, nc] == 1:
            return True

        for other in game_state.get("chars", []):
            if other is char or not character_is_alive(other):
                continue

            if (
                int(other.pos[0]) == nr
                and int(other.pos[1]) == nc
            ):
                return True

        return False

    def decide_move(self, char, game_state):
        if self.game is None:
            raise RuntimeError("set_game(game)が実行されていません")

        # 状態を先に記録する
        observation = get_game_observation(self.game, char)

        # 同じ状態で教師へ問い合わせる
        expert_result = self.expert.decide_move(char, game_state)
        expert_valid = teacher_action_is_valid(
            self.game,
            char,
            expert_result,
        )

        if not expert_valid:
            self.invalid_teacher_count += 1

        expert_group = self._group_controller_result(
            char,
            expert_result,
        )
        self.expert_action_counts[expert_group] += 1

        if len(self.record_buffer) < self.target_samples:
            self.record_buffer.append(
                {
                    "observation": observation,
                    "action": result_to_action_data(
                        char,
                        expert_result,
                    ),
                    "teacher_action_valid": bool(expert_valid),
                }
            )

        obs_vec = observation_to_vector(observation)

        if len(obs_vec) != self.obs_size:
            raise ValueError(
                "観測次元がモデルと一致しません: "
                f"observation={len(obs_vec)}, model={self.obs_size}"
            )

        with torch.no_grad():
            obs_tensor = (
                torch.from_numpy(obs_vec)
                .unsqueeze(0)
                .to(self.device)
            )

            logits = self.model(obs_tensor)
            action_mask = build_action_mask(self.game, char, game_state)
            probabilities = masked_action_probabilities(logits, action_mask)

            model_action_idx = int(
                probabilities.argmax(dim=1).item()
            )

            probability_array = (
                probabilities[0]
                .detach()
                .cpu()
                .numpy()
            )

        confidence = float(probability_array[model_action_idx])

        model_group = self._group_action_index(model_action_idx)
        model_name = ACTION_NAMES[model_action_idx]

        self.predicted_action_counts[model_group] += 1
        self.predicted_detailed_counts[model_name] += 1
        self.confusion_counts[expert_group][model_group] += 1

        self.confidence_sum += confidence
        self.confidence_count += 1
        self.probability_sums += probability_array
        self.probability_count += 1

        if model_action_idx in DIRECTION_BY_ACTION:
            self.predicted_move_count += 1

            if self._is_invalid_model_move(
                model_action_idx,
                char,
                game_state,
            ):
                self.invalid_move_count += 1

        use_expert = random.random() < self.beta

        if use_expert:
            self.expert_executions += 1
            executed_result = expert_result
        else:
            self.model_executions += 1
            executed_result = decode_model_action(
                model_action_idx,
                char,
                game_state,
                expert_result,
            )

        executed_group = self._group_controller_result(
            char,
            executed_result,
        )
        self.executed_action_counts[executed_group] += 1

        return executed_result

    def _print_counter(
        self,
        title,
        counter,
        total,
        ordered_names=None,
    ):
        print()
        print(title)
        print("-" * 56)

        if total <= 0:
            print("データなし")
            return

        if ordered_names is None:
            items = counter.most_common()
        else:
            items = [
                (name, counter.get(name, 0))
                for name in ordered_names
            ]

        for name, count in items:
            percentage = count / total * 100.0
            print(
                f"{name:<18}: {count:>8} "
                f"({percentage:>6.2f}%)"
            )

    def print_statistics(self):
        total_predictions = sum(
            self.predicted_action_counts.values()
        )
        total_executed = sum(
            self.executed_action_counts.values()
        )
        total_expert = sum(
            self.expert_action_counts.values()
        )
        total_sources = (
            self.expert_executions
            + self.model_executions
        )

        print()
        print("=" * 72)
        print("DAgger Action Statistics")
        print("=" * 72)
        print(f"Total predictions       : {total_predictions}")
        print(f"Expert executions       : {self.expert_executions}")
        print(f"Model executions        : {self.model_executions}")
        print(f"Invalid teacher actions : {self.invalid_teacher_count}")

        if total_sources:
            print(
                f"Actual expert ratio     : "
                f"{self.expert_executions / total_sources:.4f}"
            )

        self._print_counter(
            "Model Predicted Actions",
            self.predicted_action_counts,
            total_predictions,
            self.GROUPED_ACTION_NAMES,
        )

        self._print_counter(
            "Executed Actions",
            self.executed_action_counts,
            total_executed,
            self.GROUPED_ACTION_NAMES,
        )

        self._print_counter(
            "Expert Actions",
            self.expert_action_counts,
            total_expert,
            self.GROUPED_ACTION_NAMES,
        )

        self._print_counter(
            "Model Predicted Action Details",
            self.predicted_detailed_counts,
            total_predictions,
            ACTION_NAMES,
        )

        print()
        print("Invalid Move Statistics")
        print("-" * 56)
        print(f"Predicted MOVE          : {self.predicted_move_count}")
        print(f"Invalid MOVE            : {self.invalid_move_count}")

        if self.predicted_move_count:
            ratio = (
                self.invalid_move_count
                / self.predicted_move_count
            )
            print(
                f"Invalid / MOVE          : "
                f"{ratio:.4f} ({ratio * 100.0:.2f}%)"
            )

        print()
        print("Model Confidence")
        print("-" * 56)

        if self.confidence_count:
            print(
                f"Average max probability : "
                f"{self.confidence_sum / self.confidence_count:.4f}"
            )
        else:
            print("データなし")

        if self.probability_count:
            averages = (
                self.probability_sums
                / self.probability_count
            )

            print()
            print("Average Probability by Action")
            print("-" * 56)

            for idx, name in enumerate(ACTION_NAMES):
                print(f"{name:<18}: {averages[idx]:.4f}")

        print()
        print("Expert -> Model Prediction")
        print("-" * 72)

        for expert_name in self.GROUPED_ACTION_NAMES:
            row = self.confusion_counts.get(expert_name, {})
            row_total = sum(row.values())

            if row_total <= 0:
                continue

            print()
            print(f"Expert {expert_name} (total={row_total})")

            for predicted_name in self.GROUPED_ACTION_NAMES:
                count = row.get(predicted_name, 0)

                if count <= 0:
                    continue

                percentage = count / row_total * 100.0

                print(
                    f"  -> {predicted_name:<12}: "
                    f"{count:>7} ({percentage:>6.2f}%)"
                )

        print()
        print("=" * 72)


# ============================================================================
# Collection / Training
# ============================================================================

def make_game(attacker_controller):
    attacker_roster, defender_roster = (
        build_two_balanced_rosters()
    )

    kwargs = {
        "headless": True,
        "attacker_roster": attacker_roster,
        "defender_roster": defender_roster,
    }

    signature = inspect.signature(
        VisualFPSBattle.__init__
    )

    if "disable_side_swap" in signature.parameters:
        kwargs["disable_side_swap"] = True

    game = VisualFPSBattle(
        NEW_MAZE_STR,
        attacker_controller,
        DefaultDefenderController(),
        **kwargs,
    )

    attacker_controller.set_game(game)
    return game


def collect_dagger_samples(model_path, beta, target_samples):
    model, obs_size = load_policy(
        model_path,
        DEVICE,
    )

    new_records = []

    controller = DAggerAttackerController(
        model=model,
        obs_size=obs_size,
        device=DEVICE,
        beta=beta,
        record_buffer=new_records,
        target_samples=target_samples,
    )

    match_count = 0

    while (
        len(new_records) < target_samples
        and match_count < MAX_MATCHES_PER_ITERATION
    ):
        match_count += 1

        print(
            f"  DAgger試合 {match_count} 開始 "
            f"({len(new_records)} / {target_samples})"
        )

        game = make_game(controller)
        game.run_headless_loop()

        print(
            f"  DAgger試合 {match_count} 終了 "
            f"({len(new_records)} / {target_samples})"
        )

    total_executions = (
        controller.expert_executions
        + controller.model_executions
    )

    expert_ratio = (
        controller.expert_executions / total_executions
        if total_executions
        else 0.0
    )

    print(
        f"  収集完了: {len(new_records)}件 | "
        f"expert実行率={expert_ratio:.3f}"
    )

    controller.print_statistics()

    if len(new_records) < target_samples:
        print(
            f"  警告: 試合上限により目標未達です "
            f"({len(new_records)} / {target_samples})"
        )

    return new_records


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError(
            f"Object of type {obj.__class__.__name__} "
            "is not JSON serializable"
        )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            default=json_default,
        )

    os.replace(temporary_path, path)


def prepare_aggregated_dataset():
    if not BASE_DEMO_FILE.exists():
        raise FileNotFoundError(
            f"元デモがありません: {BASE_DEMO_FILE}"
        )

    if AGGREGATED_DEMO_FILE.exists():
        print(
            f"既存の集約データを継続使用します: "
            f"{AGGREGATED_DEMO_FILE}"
        )
        return

    AGGREGATED_DEMO_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        BASE_DEMO_FILE,
        AGGREGATED_DEMO_FILE,
    )

    print(
        f"元デモを集約データとしてコピーしました: "
        f"{AGGREGATED_DEMO_FILE}"
    )


def retrain_on_aggregated_data(iteration):
    trainer = BCTrainer(
        device=DEVICE,
        seed=SEED + iteration,
        use_class_weights=True,
    )

    observations, actions = trainer.load_demos(
        AGGREGATED_DEMO_FILE,
        team="A",
        skip_invalid_teacher=True,
    )

    best_path = Path(
        f"policy_dagger_iter_{iteration}_best.pt"
    )

    trainer.train(
        observations,
        actions,
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        val_split=TRAIN_VAL_SPLIT,
        best_model_path=best_path,
    )

    iteration_model = Path(
        f"policy_dagger_iter_{iteration}.pt"
    )

    trainer.save_model(iteration_model)
    trainer.plot_losses(
        f"training_loss_dagger_iter_{iteration}.png"
    )
    trainer.plot_validation_accuracy(
        f"validation_accuracy_dagger_iter_{iteration}.png"
    )

    return iteration_model


def main():
    print(f"使用デバイス: {DEVICE}")

    if not INITIAL_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"初期BCモデルがありません: "
            f"{INITIAL_MODEL_FILE}"
        )

    prepare_aggregated_dataset()

    current_model = INITIAL_MODEL_FILE

    for iteration in range(
        1,
        DAGGER_ITERATIONS + 1,
    ):
        beta = BETA_SCHEDULE[
            min(
                iteration - 1,
                len(BETA_SCHEDULE) - 1,
            )
        ]

        print()
        print("=" * 72)
        print(
            f"DAgger iteration "
            f"{iteration}/{DAGGER_ITERATIONS} "
            f"| beta={beta:.2f}"
        )
        print("=" * 72)

        new_records = collect_dagger_samples(
            model_path=current_model,
            beta=beta,
            target_samples=SAMPLES_PER_ITERATION,
        )

        aggregated = load_json(
            AGGREGATED_DEMO_FILE
        )

        old_size = len(aggregated)
        aggregated.extend(new_records)

        save_json(
            AGGREGATED_DEMO_FILE,
            aggregated,
        )

        print(
            f"集約データ更新: "
            f"{old_size} -> {len(aggregated)}"
        )

        current_model = retrain_on_aggregated_data(
            iteration
        )

    shutil.copy2(
        current_model,
        FINAL_MODEL_FILE,
    )

    print()
    print("DAgger完了")
    print(f"最終モデル: {FINAL_MODEL_FILE}")
    print(f"集約データ: {AGGREGATED_DEMO_FILE}")


if __name__ == "__main__":
    main()
