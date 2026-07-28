"""
DAgger（Dataset Aggregation）実行スクリプト

流れ:
1. 現在のBCモデルでアタッカーを動かす
2. その状態でルールAIに正解行動を問い合わせる
3. (observation, expert_action) を既存デモへ追加する
4. 集約データ全体で再学習する
5. 数回繰り返す

前提:
    run_game.py
    map_data.py
    controllers.py
    roster_utils.py
    train_bc.py
    policy_bc_final.pt
    demos/rule_based_demos.json

と同じフォルダに置いて実行してください。

実行:
    python dagger_train.py
"""

import inspect
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from controllers import DefaultAttackerController, DefaultDefenderController
from map_data import NEW_MAZE_STR
from roster_utils import build_two_balanced_rosters
from run_game import VisualFPSBattle
from train_bc import BCTrainer, PolicyNetwork

# ============================================================================
# 設定
# ============================================================================

BASE_DEMO_FILE = Path("demos/rule_based_demos.json")
AGGREGATED_DEMO_FILE = Path("demos/dagger_aggregated_demos.json")

INITIAL_MODEL_FILE = Path("policy_bc_final.pt")
FINAL_MODEL_FILE = Path("policy_dagger_final.pt")

# 1反復あたりに追加する「アタッカー側」の教師ラベル数
SAMPLES_PER_ITERATION = 12000

# DAgger反復回数
DAGGER_ITERATIONS = 3

# expertを実際の操作に使う確率。
# 学習が進むほどモデル自身で歩かせる割合を増やす。
BETA_SCHEDULE = [0.8, 0.6, 0.4]

MAX_MATCHES_PER_ITERATION = 20

TRAIN_EPOCHS = 80
TRAIN_BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 再現性をある程度確保
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================================
# Observation
# ============================================================================


def get_game_observation(game, viewer):
    """行動するキャラクターviewer自身の視点で観測を作る。"""
    if viewer is None:
        raise ValueError("viewer must not be None")

    viewer_team = viewer.team
    enemy_team = "D" if viewer_team == "A" else "A"

    obs_dict = {
        "grid": game.grid.flatten().tolist(),
        "allies": [],
        "visible_enemies": [],
        "game_state": [],
        "spike_pos": [0, 0],
        "target_plant_pos": [0, 0],
        "visible_enemy_count": 0,
        "distance_to_site": 0.0,
    }

    # viewer自身を必ず先頭にする。
    allies = [viewer] + [
        char for char in game.chars if char.team == viewer_team and char is not viewer
    ]

    for char in allies:
        obs_dict["allies"].append(
            {
                "name": char.name,
                # 教師アクションの方向計算用
                "pos": [int(char.pos[0]), int(char.pos[1])],
                # NN入力用
                "rel_pos": [
                    int(char.pos[0] - viewer.pos[0]),
                    int(char.pos[1] - viewer.pos[1]),
                ],
                "hp": int(char.hp),
                "has_spike": 1 if char.has_spike else 0,
                "recon_cd": 1 if char.recon_charges > 0 else 0,
                "flash_cd": 1 if char.flash_charges > 0 else 0,
                "smoke_cd": 1 if char.smoke_charges > 0 else 0,
            }
        )

    # 現在のBCデータ形式と揃えるため、生存敵を相対座標で記録する。
    for char in game.chars:
        if char.team == enemy_team and char.is_alive:
            obs_dict["visible_enemies"].append(
                {
                    "rel_pos": [
                        int(char.pos[0] - viewer.pos[0]),
                        int(char.pos[1] - viewer.pos[1]),
                    ],
                    "hp": int(char.hp),
                }
            )

    obs_dict["visible_enemy_count"] = len(obs_dict["visible_enemies"])

    if game.spike_pos is not None:
        obs_dict["spike_pos"] = [
            int(game.spike_pos[0] - viewer.pos[0]),
            int(game.spike_pos[1] - viewer.pos[1]),
        ]

    if game.target_plant_pos is not None:
        plant_dr = int(game.target_plant_pos[0] - viewer.pos[0])
        plant_dc = int(game.target_plant_pos[1] - viewer.pos[1])

        obs_dict["target_plant_pos"] = [plant_dr, plant_dc]
        obs_dict["distance_to_site"] = float(abs(plant_dr) + abs(plant_dc))

    obs_dict["game_state"] = [
        game.round_timer / 100.0,
        1.0 if game.is_planted else 0.0,
    ]

    return obs_dict


def observation_to_vector(obs):
    """train_bc.pyと同じ1197次元の入力ベクトルへ変換する。"""
    grid_vec = np.asarray(obs["grid"], dtype=np.float32)

    ally_vec = np.zeros(30, dtype=np.float32)
    for i, ally in enumerate(obs["allies"][:5]):
        rel_pos = ally.get("rel_pos", ally.get("pos", [0, 0]))
        ally_vec[i * 6 : (i + 1) * 6] = [
            rel_pos[0] / 25.0,
            rel_pos[1] / 35.0,
            ally["hp"] / 100.0,
            float(ally.get("has_spike", 0)),
            float(ally.get("recon_cd", 0)),
            float(ally.get("flash_cd", 0)),
        ]

    enemy_vec = np.zeros(15, dtype=np.float32)
    for i, enemy in enumerate(obs["visible_enemies"][:5]):
        enemy_vec[i * 3 : (i + 1) * 3] = [
            enemy["rel_pos"][0] / 25.0,
            enemy["rel_pos"][1] / 35.0,
            enemy["hp"] / 100.0,
        ]

    game_state_vec = np.asarray(obs["game_state"], dtype=np.float32)

    spike_pos = obs.get("spike_pos", [0, 0])
    spike_vec = np.asarray(
        [spike_pos[0] / 25.0, spike_pos[1] / 35.0],
        dtype=np.float32,
    )

    target = obs.get("target_plant_pos", [0, 0])
    plant_target_vec = np.asarray(
        [target[0] / 25.0, target[1] / 35.0],
        dtype=np.float32,
    )

    enemy_count = float(
        obs.get("visible_enemy_count", len(obs.get("visible_enemies", [])))
    )
    enemy_count_vec = np.asarray(
        [min(max(enemy_count, 0.0), 5.0) / 5.0],
        dtype=np.float32,
    )

    site_distance = float(obs.get("distance_to_site", 0.0))
    site_distance_vec = np.asarray(
        [min(max(site_distance, 0.0), 60.0) / 60.0],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            grid_vec,
            ally_vec,
            enemy_vec,
            game_state_vec,
            spike_vec,
            plant_target_vec,
            enemy_count_vec,
            site_distance_vec,
        ]
    ).astype(np.float32)


# ============================================================================
# Action変換
# ============================================================================

DIRECTION_BY_ACTION = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
    4: (-1, -1),
    5: (-1, 1),
    6: (1, -1),
    7: (1, 1),
}

ABILITY_BY_ACTION = {
    8: "SMOKE",
    9: "FLASH",
    10: "RECON",
}


def result_to_action_data(char, result):
    """Controllerの戻り値をJSON保存用辞書にする。"""
    action_data = {
        "char": char.name,
        "team": char.team,
        "ability": None,
        "special": None,
    }

    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        move_pos, ability = result
        action_data["move"] = [int(move_pos[0]), int(move_pos[1])]
        action_data["ability"] = ability.get("ability")

        target = ability.get("target", char.pos)
        action_data["ability_target"] = [
            int(target[0]),
            int(target[1]),
        ]

    elif isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], str):
        move_pos, special = result
        action_data["move"] = [int(move_pos[0]), int(move_pos[1])]
        action_data["special"] = special

    else:
        action_data["move"] = [int(result[0]), int(result[1])]

    return action_data


def choose_ability_target(ability_name, char, game_state, expert_result):
    """
    現在の分類モデルはアビリティ種類だけを予測し、targetは予測しない。
    そのため実行時のtargetは以下で補う。

    1. expertも同じアビリティを選んだ場合は、そのtarget
    2. SMOKE/FLASHは最も近い生存敵
    3. RECONはプラント地点
    """
    if (
        isinstance(expert_result, tuple)
        and len(expert_result) == 2
        and isinstance(expert_result[1], dict)
        and expert_result[1].get("ability") == ability_name
    ):
        target = expert_result[1].get("target")
        if target is not None:
            return (int(target[0]), int(target[1]))

    enemies = [c for c in game_state["chars"] if c.is_alive and c.team != char.team]

    if ability_name in {"SMOKE", "FLASH"} and enemies:
        closest = min(
            enemies,
            key=lambda enemy: max(
                abs(enemy.pos[0] - char.pos[0]),
                abs(enemy.pos[1] - char.pos[1]),
            ),
        )
        return (int(closest.pos[0]), int(closest.pos[1]))

    if ability_name == "RECON":
        target = game_state.get("planted_pos") or game_state.get("target_plant_pos")
        if target is not None:
            return (int(target[0]), int(target[1]))

    return (int(char.pos[0]), int(char.pos[1]))


def decode_model_action(action_idx, char, game_state, expert_result):
    """離散アクション番号をゲームControllerの戻り値へ変換する。"""
    grid = game_state["grid"]

    if action_idx in DIRECTION_BY_ACTION:
        dr, dc = DIRECTION_BY_ACTION[action_idx]
        nr = int(char.pos[0] + dr)
        nc = int(char.pos[1] + dc)

        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] != 1:
            return [nr, nc]

        return list(char.pos)

    if action_idx in ABILITY_BY_ACTION:
        ability_name = ABILITY_BY_ACTION[action_idx]

        charge_name = {
            "SMOKE": "smoke_charges",
            "FLASH": "flash_charges",
            "RECON": "recon_charges",
        }[ability_name]

        if getattr(char, charge_name, 0) <= 0:
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

    if action_idx == 12:
        return list(char.pos), "PLANT"

    # 11=停止
    return list(char.pos)


# ============================================================================
# Model
# ============================================================================


def load_policy(model_path, device):
    """state_dictから入力次元を読み取り、PolicyNetworkを復元する。"""
    state_dict = torch.load(model_path, map_location=device)

    first_weight = state_dict.get("net.0.weight")
    if first_weight is None:
        raise KeyError(
            "モデル内に net.0.weight がありません。"
            "現在のPolicyNetwork形式か確認してください。"
        )

    obs_size = int(first_weight.shape[1])
    model = PolicyNetwork(obs_size=obs_size, num_actions=13).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, obs_size


# ============================================================================
# DAgger Controller
# ============================================================================


from collections import Counter, defaultdict

import numpy as np
import torch


class DAggerAttackerController:
    """
    モデルが作った状態を訪問しながら、
    各状態に対するルールAIの正解行動を保存する。

    さらに以下の統計を記録する。

    - モデルが予測した行動
    - 実際に実行された行動
    - 教師AIが選択した行動
    - 無効移動数
    - 教師行動とモデル予測の対応
    - モデルの平均確信度
    """

    ACTION_NAMES = {
        0: "MOVE_UP",
        1: "MOVE_DOWN",
        2: "MOVE_LEFT",
        3: "MOVE_RIGHT",
        4: "MOVE_UP_LEFT",
        5: "MOVE_UP_RIGHT",
        6: "MOVE_DOWN_LEFT",
        7: "MOVE_DOWN_RIGHT",
        8: "SMOKE",
        9: "FLASH",
        10: "RECON",
        11: "STOP",
        12: "PLANT",
    }

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
        self.obs_size = obs_size
        self.device = device
        self.beta = float(beta)
        self.record_buffer = record_buffer
        self.target_samples = int(target_samples)

        self.expert = DefaultAttackerController()
        self.game = None

        # 実行主体の統計
        self.expert_executions = 0
        self.model_executions = 0

        # 行動統計
        self.predicted_action_counts = Counter()
        self.executed_action_counts = Counter()
        self.expert_action_counts = Counter()

        # 方向別の詳細統計
        self.predicted_detailed_counts = Counter()

        # 教師行動 -> モデル予測
        self.confusion_counts = defaultdict(Counter)

        # 無効移動
        self.predicted_move_count = 0
        self.invalid_move_count = 0

        # モデルの確信度
        self.confidence_sum = 0.0
        self.confidence_count = 0

        # 最大確率だけでなく、各クラスの平均確率も記録
        self.probability_sums = np.zeros(13, dtype=np.float64)
        self.probability_count = 0

    def set_game(self, game):
        self.game = game

    def reset_round(self):
        if hasattr(self.expert, "reset_round"):
            self.expert.reset_round()

    # ========================================================================
    # 行動分類
    # ========================================================================

    def _group_action_index(self, action_idx):
        """
        モデルの離散アクション番号を、大分類へ変換する。
        """
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

        if action_idx == 12:
            return "PLANT"

        return "UNKNOWN"

    def _group_controller_result(self, char, result):
        """
        Controllerが返した行動を、大分類へ変換する。

        対応形式:
            [row, col]
            ([row, col], {"ability": ...})
            ([row, col], "PLANT")
        """
        if result is None:
            return "UNKNOWN"

        # アビリティまたはspecial
        if isinstance(result, tuple) and len(result) == 2:
            second_value = result[1]

            if isinstance(second_value, dict):
                ability_name = second_value.get("ability")

                if ability_name == "SMOKE":
                    return "SMOKE"

                if ability_name == "FLASH":
                    return "FLASH"

                if ability_name == "RECON":
                    return "RECON"

                return "UNKNOWN"

            if isinstance(second_value, str):
                if second_value == "PLANT":
                    return "PLANT"

                return "UNKNOWN"

        # 通常移動または停止
        try:
            move_pos = result[0] if isinstance(result, tuple) else result

            new_row = int(move_pos[0])
            new_col = int(move_pos[1])

            current_row = int(char.pos[0])
            current_col = int(char.pos[1])

            if new_row == current_row and new_col == current_col:
                return "STOP"

            return "MOVE"

        except (TypeError, ValueError, IndexError):
            return "UNKNOWN"

    # ========================================================================
    # 無効移動判定
    # ========================================================================

    def _is_invalid_model_move(
        self,
        action_idx,
        char,
        game_state,
    ):
        """
        モデルが移動を予測したが、その移動先が無効か判定する。
        """
        if action_idx not in DIRECTION_BY_ACTION:
            return False

        grid = game_state["grid"]

        dr, dc = DIRECTION_BY_ACTION[action_idx]

        nr = int(char.pos[0] + dr)
        nc = int(char.pos[1] + dc)

        # マップ外
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            return True

        # 壁
        if grid[nr, nc] == 1:
            return True

        # 他キャラクターとの重なりも無効にしたい場合
        for other in game_state.get("chars", []):
            if other is char:
                continue

            if not other.is_alive:
                continue

            if int(other.pos[0]) == nr and int(other.pos[1]) == nc:
                return True

        return False

    # ========================================================================
    # 1ステップの判断
    # ========================================================================

    def decide_move(self, char, game_state):
        if self.game is None:
            raise RuntimeError(
                "DAggerAttackerController.set_game(game) " "が実行されていません。"
            )

        # ------------------------------------------------------------
        # 1. 教師行動
        # ------------------------------------------------------------

        expert_result = self.expert.decide_move(
            char,
            game_state,
        )

        expert_group = self._group_controller_result(
            char,
            expert_result,
        )

        self.expert_action_counts[expert_group] += 1

        # ------------------------------------------------------------
        # 2. 観測を作る
        # ------------------------------------------------------------

        observation = get_game_observation(
            self.game,
            char,
        )

        # モデルが訪れた状態へ、教師の正解ラベルを付けて保存する
        if len(self.record_buffer) < self.target_samples:
            self.record_buffer.append(
                {
                    "observation": observation,
                    "action": result_to_action_data(
                        char,
                        expert_result,
                    ),
                }
            )

        obs_vec = observation_to_vector(observation)

        if obs_vec.shape[0] != self.obs_size:
            raise ValueError(
                "観測次元がモデルと一致しません: "
                f"observation={obs_vec.shape[0]}, "
                f"model={self.obs_size}"
            )

        # ------------------------------------------------------------
        # 3. モデル予測
        # ------------------------------------------------------------

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_vec).unsqueeze(0).to(self.device)

            logits = self.model(obs_tensor)
            probabilities = torch.softmax(logits, dim=1)

            model_action_idx = int(probabilities.argmax(dim=1).item())

            confidence = float(probabilities[0, model_action_idx].item())

            probability_array = probabilities[0].detach().cpu().numpy()

        model_group = self._group_action_index(model_action_idx)

        model_detailed_name = self.ACTION_NAMES.get(
            model_action_idx,
            f"UNKNOWN_{model_action_idx}",
        )

        # モデル予測統計
        self.predicted_action_counts[model_group] += 1
        self.predicted_detailed_counts[model_detailed_name] += 1

        # 教師とモデル予測の組み合わせ
        self.confusion_counts[expert_group][model_group] += 1

        # 確信度
        self.confidence_sum += confidence
        self.confidence_count += 1

        if len(probability_array) == 13:
            self.probability_sums += probability_array
            self.probability_count += 1

        # ------------------------------------------------------------
        # 4. 無効移動判定
        # ------------------------------------------------------------

        if model_action_idx in DIRECTION_BY_ACTION:
            self.predicted_move_count += 1

            if self._is_invalid_model_move(
                model_action_idx,
                char,
                game_state,
            ):
                self.invalid_move_count += 1

        # ------------------------------------------------------------
        # 5. betaに応じて実行行動を決定
        # ------------------------------------------------------------

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

    # ========================================================================
    # ログ出力
    # ========================================================================

    def _print_counter(
        self,
        title,
        counter,
        total,
        ordered_names=None,
    ):
        print()
        print(title)
        print("-" * 52)

        if total <= 0:
            print("データなし")
            return

        if ordered_names is None:
            items = counter.most_common()
        else:
            items = [(name, counter.get(name, 0)) for name in ordered_names]

        for name, count in items:
            percentage = count / total * 100.0

            print(f"{name:<18}: " f"{count:>8} " f"({percentage:>6.2f}%)")

    def print_statistics(self):
        """
        収集終了後に、モデル・教師・実行行動の統計を表示する。
        """
        total_predictions = sum(self.predicted_action_counts.values())

        total_executed = sum(self.executed_action_counts.values())

        total_expert = sum(self.expert_action_counts.values())

        total_execution_sources = self.expert_executions + self.model_executions

        print()
        print("=" * 72)
        print("DAgger Action Statistics")
        print("=" * 72)

        print(f"Total predictions       : " f"{total_predictions}")

        print(f"Expert executions       : " f"{self.expert_executions}")

        print(f"Model executions        : " f"{self.model_executions}")

        if total_execution_sources > 0:
            actual_expert_ratio = self.expert_executions / total_execution_sources

            print(f"Actual expert ratio     : " f"{actual_expert_ratio:.4f}")

        # モデル予測
        self._print_counter(
            title="Model Predicted Actions",
            counter=self.predicted_action_counts,
            total=total_predictions,
            ordered_names=self.GROUPED_ACTION_NAMES,
        )

        # 実際に採用された行動
        self._print_counter(
            title="Executed Actions",
            counter=self.executed_action_counts,
            total=total_executed,
            ordered_names=self.GROUPED_ACTION_NAMES,
        )

        # 教師行動
        self._print_counter(
            title="Expert Actions",
            counter=self.expert_action_counts,
            total=total_expert,
            ordered_names=self.GROUPED_ACTION_NAMES,
        )

        # 移動方向詳細
        self._print_counter(
            title="Model Predicted Action Details",
            counter=self.predicted_detailed_counts,
            total=total_predictions,
            ordered_names=[
                "MOVE_UP",
                "MOVE_DOWN",
                "MOVE_LEFT",
                "MOVE_RIGHT",
                "MOVE_UP_LEFT",
                "MOVE_UP_RIGHT",
                "MOVE_DOWN_LEFT",
                "MOVE_DOWN_RIGHT",
                "SMOKE",
                "FLASH",
                "RECON",
                "STOP",
                "PLANT",
            ],
        )

        print()
        print("Invalid Move Statistics")
        print("-" * 52)

        print(f"Predicted MOVE          : " f"{self.predicted_move_count}")

        print(f"Invalid MOVE            : " f"{self.invalid_move_count}")

        if self.predicted_move_count > 0:
            invalid_rate = self.invalid_move_count / self.predicted_move_count

            print(
                f"Invalid / MOVE          : "
                f"{invalid_rate:.4f} "
                f"({invalid_rate * 100.0:.2f}%)"
            )

        if total_predictions > 0:
            total_invalid_rate = self.invalid_move_count / total_predictions

            print(
                f"Invalid / All actions   : "
                f"{total_invalid_rate:.4f} "
                f"({total_invalid_rate * 100.0:.2f}%)"
            )

        # 確信度
        print()
        print("Model Confidence")
        print("-" * 52)

        if self.confidence_count > 0:
            average_confidence = self.confidence_sum / self.confidence_count

            print(f"Average max probability : " f"{average_confidence:.4f}")
        else:
            print("データなし")

        # 各クラスの平均確率
        if self.probability_count > 0:
            average_probabilities = self.probability_sums / self.probability_count

            print()
            print("Average Probability by Action")
            print("-" * 52)

            for action_idx in range(13):
                action_name = self.ACTION_NAMES[action_idx]

                print(f"{action_name:<18}: " f"{average_probabilities[action_idx]:.4f}")

        # 教師とモデル予測の対応
        print()
        print("Expert -> Model Prediction")
        print("-" * 72)

        for expert_name in self.GROUPED_ACTION_NAMES:
            row = self.confusion_counts.get(
                expert_name,
                {},
            )

            row_total = sum(row.values())

            if row_total <= 0:
                continue

            print()
            print(f"Expert {expert_name} " f"(total={row_total})")

            for predicted_name in self.GROUPED_ACTION_NAMES:
                count = row.get(predicted_name, 0)

                if count <= 0:
                    continue

                percentage = count / row_total * 100.0

                print(
                    f"  -> {predicted_name:<12}: "
                    f"{count:>7} "
                    f"({percentage:>6.2f}%)"
                )

        print()
        print("=" * 72)


# ============================================================================
# Collection / Training
# ============================================================================


def make_game(attacker_controller):
    attacker_roster, defender_roster = build_two_balanced_rosters()

    kwargs = {
        "headless": True,
        "attacker_roster": attacker_roster,
        "defender_roster": defender_roster,
    }

    # run_game.pyの版によって有無が違う引数は、対応している場合だけ渡す。
    signature = inspect.signature(VisualFPSBattle.__init__)
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
    model, obs_size = load_policy(model_path, DEVICE)
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
    while len(new_records) < target_samples and match_count < MAX_MATCHES_PER_ITERATION:
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

    total_executions = controller.expert_executions + controller.model_executions

    if total_executions > 0:
        expert_ratio = controller.expert_executions / total_executions
    else:
        expert_ratio = 0.0

    print(f"  収集完了: {len(new_records)}件 | " f"expert実行率={expert_ratio:.3f}")

    controller.print_statistics()

    if len(new_records) < target_samples:
        print(
            f"  警告: 試合上限により目標未達です "
            f"({len(new_records)} / {target_samples})"
        )

    return new_records


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    def json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            default=json_default,
        )

    os.replace(temporary_path, path)


def prepare_aggregated_dataset():
    if not BASE_DEMO_FILE.exists():
        raise FileNotFoundError(f"元デモがありません: {BASE_DEMO_FILE}")

    if AGGREGATED_DEMO_FILE.exists():
        print(f"既存の集約データを継続使用します: " f"{AGGREGATED_DEMO_FILE}")
        return

    AGGREGATED_DEMO_FILE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASE_DEMO_FILE, AGGREGATED_DEMO_FILE)
    print(f"元デモを集約データとしてコピーしました: " f"{AGGREGATED_DEMO_FILE}")


def retrain_on_aggregated_data(iteration):
    trainer = BCTrainer(device=DEVICE)
    observations, actions = trainer.load_demos(str(AGGREGATED_DEMO_FILE))

    trainer.train(
        observations,
        actions,
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
    )

    iteration_model = Path(f"policy_dagger_iter_{iteration}.pt")
    trainer.save_model(str(iteration_model))
    trainer.plot_losses(f"training_loss_dagger_iter_{iteration}.png")

    return iteration_model


def main():
    print(f"使用デバイス: {DEVICE}")

    if not INITIAL_MODEL_FILE.exists():
        raise FileNotFoundError(f"初期BCモデルがありません: {INITIAL_MODEL_FILE}")

    prepare_aggregated_dataset()

    current_model = INITIAL_MODEL_FILE

    for iteration in range(1, DAGGER_ITERATIONS + 1):
        beta = BETA_SCHEDULE[min(iteration - 1, len(BETA_SCHEDULE) - 1)]

        print()
        print("=" * 72)
        print(f"DAgger iteration {iteration}/{DAGGER_ITERATIONS} " f"| beta={beta:.2f}")
        print("=" * 72)

        new_records = collect_dagger_samples(
            model_path=current_model,
            beta=beta,
            target_samples=SAMPLES_PER_ITERATION,
        )

        aggregated = load_json(AGGREGATED_DEMO_FILE)
        old_size = len(aggregated)
        aggregated.extend(new_records)
        save_json(AGGREGATED_DEMO_FILE, aggregated)

        print(f"集約データ更新: {old_size} -> {len(aggregated)}")

        current_model = retrain_on_aggregated_data(iteration)

    shutil.copy2(current_model, FINAL_MODEL_FILE)

    print()
    print("DAgger完了")
    print(f"最終モデル: {FINAL_MODEL_FILE}")
    print(f"集約データ: {AGGREGATED_DEMO_FILE}")


if __name__ == "__main__":
    main()
