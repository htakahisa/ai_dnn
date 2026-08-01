"""
Behavioral Cloning（模倣学習）完全版

対応するデモ形式:
- viewer_pos
- local_map（7x7）
- valid_move_mask（4方向）
- spike_on_ground / ally_has_spike / viewer_has_spike
- teacher_action_valid
- 従来のgrid / allies / visible_enemies / game_state 等

主な改善:
- 観測ベクトル化を observation_to_vector() に統一
- 新しい観測項目を学習入力へ追加
- teacher_action_valid=False のデータを除外
- アタッカー側のみを学習
- 訓練/検証を層化分割
- クラス不均衡を考慮した重み付きCrossEntropyLoss
- 最良モデルとメタデータを保存
- 入力次元を自動決定
- 再現性のため乱数シード固定
"""

import json
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# 設定
# ============================================================================

ACTION_NAMES = [
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "SMOKE",
    "FLASH",
    "RECON",
    "STOP",
    "PLANT",
]

NUM_ACTIONS = len(ACTION_NAMES)

GRID_VALUE_SCALE = 7.0
LOCAL_MAP_VALUE_SCALE = 7.0

DEFAULT_GRID_ROWS = 25
DEFAULT_GRID_COLS = 35

ALLY_COUNT = 5
ALLY_FEATURES = 8

ENEMY_COUNT = 5
ENEMY_FEATURES = 3

LOCAL_MAP_SIZE = 7
LOCAL_MAP_CELLS = LOCAL_MAP_SIZE * LOCAL_MAP_SIZE

VALID_MOVE_COUNT = 4


# ============================================================================
# 共通処理
# ============================================================================

def set_global_seed(seed=42):
    """Python / NumPy / PyTorchの乱数シードを固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _normalize_position(value, scale):
    return safe_float(value, 0.0) / float(scale)


def _prepare_fixed_length(values, length, fill_value=0.0):
    """
    可変長配列を指定長へ切り詰め・ゼロ埋めする。
    """
    result = np.full(length, fill_value, dtype=np.float32)
    if values is None:
        return result

    array = np.asarray(values, dtype=np.float32).reshape(-1)
    copy_length = min(length, len(array))
    result[:copy_length] = array[:copy_length]
    return result


# ============================================================================
# 観測ベクトル化
# ============================================================================

def observation_to_vector(obs):
    """
    collect_demos_fixed.py の観測辞書を固定長ベクトルへ変換する。

    出力構成:
    1. grid
    2. viewer_pos
    3. local_map
    4. valid_move_mask
    5. allies
    6. visible_enemies
    7. game_state
    8. spike_pos
    9. target_plant_pos
    10. visible_enemy_count
    11. distance_to_site
    12. spike_on_ground
    13. ally_has_spike
    14. viewer_has_spike
    """

    if not isinstance(obs, dict):
        raise TypeError(f"observation must be dict, got {type(obs).__name__}")

    # ------------------------------------------------------------------------
    # 1. グローバルマップ
    # ------------------------------------------------------------------------
    grid_raw = obs.get("grid", [])

    if len(grid_raw) == 0:
        raise ValueError("observationにgridがありません")

    grid_vec = np.asarray(grid_raw, dtype=np.float32).reshape(-1)

    # 値の大きさを抑える
    grid_vec = np.clip(grid_vec, 0.0, GRID_VALUE_SCALE) / GRID_VALUE_SCALE

    # ------------------------------------------------------------------------
    # 2. viewer絶対座標
    # ------------------------------------------------------------------------
    viewer_pos = obs.get("viewer_pos")

    if viewer_pos is None:
        # 古いデータ向けフォールバック:
        # allies[0].pos がviewer自身である前提
        allies = obs.get("allies", [])
        if allies:
            viewer_pos = allies[0].get("pos", [0, 0])
        else:
            viewer_pos = [0, 0]

    viewer_pos_vec = np.array(
        [
            _normalize_position(viewer_pos[0], DEFAULT_GRID_ROWS),
            _normalize_position(viewer_pos[1], DEFAULT_GRID_COLS),
        ],
        dtype=np.float32,
    )

    # ------------------------------------------------------------------------
    # 3. viewer中心ローカルマップ
    # ------------------------------------------------------------------------
    local_map = obs.get("local_map")

    if local_map is None:
        # 古いデータ向け互換
        local_map_vec = np.zeros(LOCAL_MAP_CELLS, dtype=np.float32)
    else:
        local_map_vec = _prepare_fixed_length(
            local_map,
            LOCAL_MAP_CELLS,
            fill_value=0.0,
        )
        local_map_vec = (
            np.clip(local_map_vec, 0.0, LOCAL_MAP_VALUE_SCALE)
            / LOCAL_MAP_VALUE_SCALE
        )

    # ------------------------------------------------------------------------
    # 4. 8方向の移動可否
    # ------------------------------------------------------------------------
    valid_move_mask = obs.get("valid_move_mask", [1] * VALID_MOVE_COUNT)
    valid_move_vec = _prepare_fixed_length(
        valid_move_mask,
        VALID_MOVE_COUNT,
        fill_value=0.0,
    )
    valid_move_vec = np.clip(valid_move_vec, 0.0, 1.0)

    # ------------------------------------------------------------------------
    # 5. 味方情報
    # 1人あたり:
    # rel_row, rel_col, hp, alive, spike, recon, flash, smoke
    # ------------------------------------------------------------------------
    ally_vec = np.zeros(
        ALLY_COUNT * ALLY_FEATURES,
        dtype=np.float32,
    )

    for i, ally in enumerate(obs.get("allies", [])[:ALLY_COUNT]):
        rel_pos = ally.get("rel_pos")

        if rel_pos is None:
            absolute_pos = ally.get("pos", [0, 0])
            rel_pos = [
                safe_int(absolute_pos[0]) - safe_int(viewer_pos[0]),
                safe_int(absolute_pos[1]) - safe_int(viewer_pos[1]),
            ]

        start = i * ALLY_FEATURES
        ally_vec[start:start + ALLY_FEATURES] = [
            _normalize_position(rel_pos[0], DEFAULT_GRID_ROWS),
            _normalize_position(rel_pos[1], DEFAULT_GRID_COLS),
            np.clip(safe_float(ally.get("hp", 0)) / 100.0, 0.0, 2.0),
            float(bool(ally.get("is_alive", ally.get("hp", 0) > 0))),
            float(bool(ally.get("has_spike", 0))),
            float(bool(ally.get("recon_cd", 0))),
            float(bool(ally.get("flash_cd", 0))),
            float(bool(ally.get("smoke_cd", 0))),
        ]

    # ------------------------------------------------------------------------
    # 6. 敵情報
    # 1人あたり:
    # rel_row, rel_col, hp
    # ------------------------------------------------------------------------
    enemy_vec = np.zeros(
        ENEMY_COUNT * ENEMY_FEATURES,
        dtype=np.float32,
    )

    for i, enemy in enumerate(
        obs.get("visible_enemies", [])[:ENEMY_COUNT]
    ):
        rel_pos = enemy.get("rel_pos", [0, 0])

        start = i * ENEMY_FEATURES
        enemy_vec[start:start + ENEMY_FEATURES] = [
            _normalize_position(rel_pos[0], DEFAULT_GRID_ROWS),
            _normalize_position(rel_pos[1], DEFAULT_GRID_COLS),
            np.clip(safe_float(enemy.get("hp", 0)) / 100.0, 0.0, 2.0),
        ]

    # ------------------------------------------------------------------------
    # 7. ゲーム状態
    # ------------------------------------------------------------------------
    game_state_vec = _prepare_fixed_length(
        obs.get("game_state", [0.0, 0.0]),
        2,
        fill_value=0.0,
    )

    # ------------------------------------------------------------------------
    # 8. 落下スパイク相対位置
    # ------------------------------------------------------------------------
    spike_pos = obs.get("spike_pos", [0, 0])
    spike_vec = np.array(
        [
            _normalize_position(spike_pos[0], DEFAULT_GRID_ROWS),
            _normalize_position(spike_pos[1], DEFAULT_GRID_COLS),
        ],
        dtype=np.float32,
    )

    # ------------------------------------------------------------------------
    # 9. プラント目標相対位置
    # ------------------------------------------------------------------------
    target_plant_pos = obs.get("target_plant_pos", [0, 0])
    plant_target_vec = np.array(
        [
            _normalize_position(target_plant_pos[0], DEFAULT_GRID_ROWS),
            _normalize_position(target_plant_pos[1], DEFAULT_GRID_COLS),
        ],
        dtype=np.float32,
    )

    # ------------------------------------------------------------------------
    # 10. 敵人数
    # ------------------------------------------------------------------------
    visible_enemy_count = safe_float(
        obs.get(
            "visible_enemy_count",
            len(obs.get("visible_enemies", [])),
        )
    )
    visible_enemy_count_vec = np.array(
        [np.clip(visible_enemy_count / 5.0, 0.0, 1.0)],
        dtype=np.float32,
    )

    # ------------------------------------------------------------------------
    # 11. サイトまでの距離
    # ------------------------------------------------------------------------
    distance_to_site = safe_float(obs.get("distance_to_site", 0.0))
    distance_to_site_vec = np.array(
        [np.clip(distance_to_site / 60.0, 0.0, 1.0)],
        dtype=np.float32,
    )

    # ------------------------------------------------------------------------
    # 12-14. スパイク状態
    # ------------------------------------------------------------------------
    spike_state_vec = np.array(
        [
            float(bool(obs.get("spike_on_ground", 0))),
            float(bool(obs.get("ally_has_spike", 0))),
            float(bool(obs.get("viewer_has_spike", 0))),
        ],
        dtype=np.float32,
    )

    full_obs = np.concatenate(
        [
            grid_vec,
            viewer_pos_vec,
            local_map_vec,
            valid_move_vec,
            ally_vec,
            enemy_vec,
            game_state_vec,
            spike_vec,
            plant_target_vec,
            visible_enemy_count_vec,
            distance_to_site_vec,
            spike_state_vec,
        ]
    ).astype(np.float32, copy=False)

    if not np.all(np.isfinite(full_obs)):
        raise ValueError("観測ベクトルにNaNまたはInfが含まれています")

    return full_obs


# ============================================================================
# アクション符号化
# ============================================================================

def encode_action(action_data, obs):
    """
    アクション辞書を0〜8の離散アクションへ変換する。

    0-3 : 上下左右の移動
    4   : SMOKE
    5   : FLASH
    6   : RECON
    7   : STOP
    8   : PLANT
    """
    ability = action_data.get("ability")

    if ability == "SMOKE":
        return 4
    if ability == "FLASH":
        return 5
    if ability == "RECON":
        return 6

    if action_data.get("special") == "PLANT":
        return 8

    char_name = action_data.get("char")
    move_to = action_data.get("move", [0, 0])
    current_pos = None

    for ally in obs.get("allies", []):
        if ally.get("name") == char_name:
            current_pos = ally.get("pos")
            break

    if current_pos is None:
        current_pos = obs.get("viewer_pos")

    if current_pos is None:
        return 7

    dr = safe_int(move_to[0]) - safe_int(current_pos[0])
    dc = safe_int(move_to[1]) - safe_int(current_pos[1])

    direction_map = {
        (-1, 0): 0,
        (1, 0): 1,
        (0, -1): 2,
        (0, 1): 3,
    }

    if (dr, dc) == (0, 0):
        return 7

    # システム上、斜め移動は存在しない。
    # 1回の教師行動が複数マス先を示す場合だけ主方向へ丸める。
    if dr != 0 and dc == 0:
        return 0 if dr < 0 else 1
    if dc != 0 and dr == 0:
        return 2 if dc < 0 else 3

    # 万一斜め座標が記録された場合は、不正な移動クラスを作らずSTOP扱い。
    return 7


# ============================================================================
# モデル
# ============================================================================

class PolicyNetwork(nn.Module):
    """状態ベクトルから9アクションのlogitを出力するMLP。"""

    def __init__(self, obs_size, num_actions=NUM_ACTIONS):
        super().__init__()

        self.obs_size = int(obs_size)
        self.num_actions = int(num_actions)

        self.net = nn.Sequential(
            nn.Linear(self.obs_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, self.num_actions),
        )

    def forward(self, obs):
        return self.net(obs)


# ============================================================================
# Trainer
# ============================================================================

class BCTrainer:
    """Behavioral Cloningトレーナー。"""

    def __init__(
        self,
        device=None,
        seed=42,
        use_class_weights=True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.seed = int(seed)
        self.use_class_weights = bool(use_class_weights)

        set_global_seed(self.seed)

        self.model = None
        self.obs_size = None
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    # ------------------------------------------------------------------------
    # データ読み込み
    # ------------------------------------------------------------------------

    def load_demos(
        self,
        demo_file,
        team="A",
        skip_invalid_teacher=True,
    ):
        demo_file = Path(demo_file)

        if not demo_file.exists():
            raise FileNotFoundError(f"デモファイルが見つかりません: {demo_file}")

        print(f"デモファイル読み込み: {demo_file}")

        with open(demo_file, "r", encoding="utf-8") as file:
            demos = json.load(file)

        if not isinstance(demos, list):
            raise ValueError("デモJSONの最上位はlistである必要があります")

        print(f"読み込んだデモ: {len(demos)}件")

        observations = []
        actions = []

        skipped_team = 0
        skipped_invalid = 0
        skipped_error = 0
        expected_obs_size = None

        for index, demo in enumerate(demos):
            try:
                obs = demo["observation"]
                action = demo["action"]

                if team is not None and action.get("team") != team:
                    skipped_team += 1
                    continue

                if (
                    skip_invalid_teacher
                    and demo.get("teacher_action_valid") is False
                ):
                    skipped_invalid += 1
                    continue

                obs_vector = observation_to_vector(obs)
                action_index = encode_action(action, obs)

                if not (0 <= action_index < NUM_ACTIONS):
                    raise ValueError(
                        f"action index out of range: {action_index}"
                    )

                if expected_obs_size is None:
                    expected_obs_size = len(obs_vector)
                elif len(obs_vector) != expected_obs_size:
                    raise ValueError(
                        f"観測次元不一致: expected={expected_obs_size}, "
                        f"actual={len(obs_vector)}"
                    )

                observations.append(obs_vector)
                actions.append(action_index)

            except Exception as exc:
                skipped_error += 1

                if skipped_error <= 10:
                    print(
                        f"警告: demo[{index}]を除外しました: "
                        f"{type(exc).__name__}: {exc}"
                    )

        if not observations:
            raise RuntimeError("使用可能なデモが1件もありません")

        obs_array = np.stack(observations).astype(np.float32)
        action_array = np.asarray(actions, dtype=np.int64)

        self.obs_size = int(obs_array.shape[1])

        print()
        print("=" * 64)
        print("Dataset Summary")
        print("=" * 64)
        print(f"使用データ数              : {len(obs_array)}")
        print(f"チーム違いで除外          : {skipped_team}")
        print(f"無効教師行動で除外        : {skipped_invalid}")
        print(f"変換エラーで除外          : {skipped_error}")
        print(f"観測ベクトル次元          : {self.obs_size}")
        print(f"観測形状                  : {obs_array.shape}")
        print(f"アクション形状            : {action_array.shape}")
        print("=" * 64)

        self._print_action_distribution(action_array)

        return obs_array, action_array

    # ------------------------------------------------------------------------
    # 分布表示
    # ------------------------------------------------------------------------

    def _print_action_distribution(self, action_array):
        counter = Counter(action_array.tolist())
        total = len(action_array)

        print()
        print("=" * 64)
        print("Teacher Action Distribution")
        print("=" * 64)

        for action_index, action_name in enumerate(ACTION_NAMES):
            count = counter.get(action_index, 0)
            ratio = count / total * 100.0 if total else 0.0
            print(f"{action_name:<18}: {count:8d} ({ratio:6.2f}%)")

        print("-" * 64)

        move_count = sum(counter.get(i, 0) for i in range(4))
        ability_count = sum(counter.get(i, 0) for i in range(4, 7))
        stop_count = counter.get(7, 0)
        plant_count = counter.get(8, 0)

        def ratio(count):
            return count / total * 100.0 if total else 0.0

        print(
            f"{'TOTAL_MOVE':<18}: "
            f"{move_count:8d} ({ratio(move_count):6.2f}%)"
        )
        print(
            f"{'TOTAL_ABILITY':<18}: "
            f"{ability_count:8d} ({ratio(ability_count):6.2f}%)"
        )
        print(
            f"{'TOTAL_STOP':<18}: "
            f"{stop_count:8d} ({ratio(stop_count):6.2f}%)"
        )
        print(
            f"{'TOTAL_PLANT':<18}: "
            f"{plant_count:8d} ({ratio(plant_count):6.2f}%)"
        )
        print(f"{'TOTAL':<18}: {total:8d} (100.00%)")
        print("=" * 64)

        missing = [
            ACTION_NAMES[i]
            for i in range(NUM_ACTIONS)
            if counter.get(i, 0) == 0
        ]

        if missing:
            print(
                "警告: 教師データに一度も存在しないアクション: "
                + ", ".join(missing)
            )

    # ------------------------------------------------------------------------
    # 分割
    # ------------------------------------------------------------------------

    def _stratified_split(self, actions, val_split):
        """
        各アクションクラスの比率をできるだけ維持して分割する。
        """
        rng = np.random.default_rng(self.seed)

        train_indices = []
        val_indices = []

        for action_index in range(NUM_ACTIONS):
            indices = np.flatnonzero(actions == action_index)

            if len(indices) == 0:
                continue

            rng.shuffle(indices)

            if len(indices) == 1:
                train_indices.extend(indices.tolist())
                continue

            val_count = max(1, int(round(len(indices) * val_split)))
            val_count = min(val_count, len(indices) - 1)

            val_indices.extend(indices[:val_count].tolist())
            train_indices.extend(indices[val_count:].tolist())

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        return (
            np.asarray(train_indices, dtype=np.int64),
            np.asarray(val_indices, dtype=np.int64),
        )

    # ------------------------------------------------------------------------
    # クラス重み
    # ------------------------------------------------------------------------

    def _make_class_weights(self, train_actions):
        """
        極端な不均衡を緩和するためのクラス重みを作成する。

        sqrt(total / count) を使用し、最大値を5に制限する。
        未出現クラスの重みは0。
        """
        counts = np.bincount(
            train_actions,
            minlength=NUM_ACTIONS,
        ).astype(np.float64)

        total = counts.sum()
        weights = np.zeros(NUM_ACTIONS, dtype=np.float32)

        for i, count in enumerate(counts):
            if count > 0:
                weights[i] = math.sqrt(total / count)

        nonzero = weights[weights > 0]
        if len(nonzero) > 0:
            weights = weights / nonzero.mean()

        weights = np.clip(weights, 0.0, 5.0)

        print()
        print("Class weights")
        for index, name in enumerate(ACTION_NAMES):
            print(
                f"  {name:<18}: "
                f"count={int(counts[index]):7d}, "
                f"weight={weights[index]:.4f}"
            )

        return torch.tensor(
            weights,
            dtype=torch.float32,
            device=self.device,
        )

    # ------------------------------------------------------------------------
    # 訓練
    # ------------------------------------------------------------------------

    def train(
        self,
        obs_array,
        action_array,
        epochs=100,
        batch_size=64,
        val_split=0.1,
        learning_rate=3e-4,
        weight_decay=1e-5,
        patience=15,
        best_model_path="policy_bc_best.pt",
        initial_model_path=None,
    ):
        if len(obs_array) != len(action_array):
            raise ValueError("obs_arrayとaction_arrayの件数が一致しません")

        if len(obs_array) < 2:
            raise ValueError("学習には最低2件のデータが必要です")

        print()
        print("=" * 64)
        print("Behavioral Cloning Training")
        print("=" * 64)
        print(f"Device          : {self.device}")
        print(f"Epochs          : {epochs}")
        print(f"Batch size      : {batch_size}")
        print(f"Learning rate   : {learning_rate}")
        print(f"Validation split: {val_split}")
        print(f"Observation size: {obs_array.shape[1]}")
        print(
            "Initialization  : "
            + (
                f"continue from {initial_model_path}"
                if initial_model_path is not None
                else "random initialization"
            )
        )
        print("=" * 64)

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        train_indices, val_indices = self._stratified_split(
            action_array,
            val_split,
        )

        if len(val_indices) == 0:
            raise RuntimeError("検証データを作成できませんでした")

        x_train = torch.tensor(
            obs_array[train_indices],
            dtype=torch.float32,
        )
        y_train = torch.tensor(
            action_array[train_indices],
            dtype=torch.long,
        )

        x_val = torch.tensor(
            obs_array[val_indices],
            dtype=torch.float32,
            device=self.device,
        )
        y_val = torch.tensor(
            action_array[val_indices],
            dtype=torch.long,
            device=self.device,
        )

        train_dataset = TensorDataset(x_train, y_train)

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

        self.obs_size = int(obs_array.shape[1])
        self.model = PolicyNetwork(
            obs_size=self.obs_size,
            num_actions=NUM_ACTIONS,
        ).to(self.device)

        if initial_model_path is not None:
            initial_model_path = Path(initial_model_path)

            if not initial_model_path.exists():
                raise FileNotFoundError(
                    f"継続学習元モデルが見つかりません: {initial_model_path}"
                )

            loaded = torch.load(
                initial_model_path,
                map_location=self.device,
            )

            if not isinstance(loaded, dict):
                raise TypeError(
                    "継続学習元モデルが辞書形式ではありません"
                )

            if isinstance(loaded.get("model_state_dict"), dict):
                state_dict = loaded["model_state_dict"]
                checkpoint_obs_size = loaded.get("obs_size")
                checkpoint_num_actions = int(
                    loaded.get("num_actions", NUM_ACTIONS)
                )
            elif isinstance(loaded.get("state_dict"), dict):
                state_dict = loaded["state_dict"]
                checkpoint_obs_size = loaded.get("obs_size")
                checkpoint_num_actions = int(
                    loaded.get("num_actions", NUM_ACTIONS)
                )
            elif isinstance(loaded.get("policy_state_dict"), dict):
                state_dict = loaded["policy_state_dict"]
                checkpoint_obs_size = loaded.get("obs_size")
                checkpoint_num_actions = int(
                    loaded.get("num_actions", NUM_ACTIONS)
                )
            else:
                # 旧形式: state_dictを直接保存
                state_dict = loaded
                checkpoint_obs_size = None
                checkpoint_num_actions = NUM_ACTIONS

            if any(
                str(key).startswith("module.")
                for key in state_dict
            ):
                state_dict = {
                    str(key).removeprefix("module."): value
                    for key, value in state_dict.items()
                }

            first_weight = state_dict.get("net.0.weight")
            if first_weight is None:
                raise KeyError(
                    "継続学習元モデルに net.0.weight がありません"
                )

            inferred_obs_size = int(first_weight.shape[1])
            if checkpoint_obs_size is None:
                checkpoint_obs_size = inferred_obs_size
            else:
                checkpoint_obs_size = int(checkpoint_obs_size)

            if checkpoint_obs_size != self.obs_size:
                raise ValueError(
                    "継続学習元モデルと現在データの観測次元が一致しません: "
                    f"model={checkpoint_obs_size}, data={self.obs_size}"
                )

            if checkpoint_num_actions != NUM_ACTIONS:
                raise ValueError(
                    "継続学習元モデルと現在コードのアクション数が一致しません: "
                    f"model={checkpoint_num_actions}, code={NUM_ACTIONS}"
                )

            self.model.load_state_dict(state_dict)
            print(
                f"継続学習元モデルを読み込みました: "
                f"{initial_model_path}"
            )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        if self.use_class_weights:
            class_weights = self._make_class_weights(
                action_array[train_indices]
            )
        else:
            class_weights = None

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        best_val_loss = float("inf")
        best_epoch = -1
        patience_counter = 0
        best_model_path = Path(best_model_path)

        for epoch in range(1, epochs + 1):
            # ----------------------------------------------------------------
            # Train
            # ----------------------------------------------------------------
            self.model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_count = 0

            for obs_batch, action_batch in train_loader:
                obs_batch = obs_batch.to(
                    self.device,
                    non_blocking=True,
                )
                action_batch = action_batch.to(
                    self.device,
                    non_blocking=True,
                )

                optimizer.zero_grad(set_to_none=True)

                logits = self.model(obs_batch)
                loss = loss_fn(logits, action_batch)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

                batch_size_actual = action_batch.size(0)
                train_loss_sum += loss.item() * batch_size_actual
                train_correct += (
                    logits.argmax(dim=1) == action_batch
                ).sum().item()
                train_count += batch_size_actual

            train_loss = train_loss_sum / max(train_count, 1)
            train_accuracy = train_correct / max(train_count, 1)

            # ----------------------------------------------------------------
            # Validation
            # ----------------------------------------------------------------
            self.model.eval()

            with torch.no_grad():
                val_logits = self.model(x_val)
                val_loss = loss_fn(val_logits, y_val).item()
                val_predictions = val_logits.argmax(dim=1)
                val_accuracy = (
                    (val_predictions == y_val)
                    .float()
                    .mean()
                    .item()
                )

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss {train_loss:.5f} | "
                f"Train Acc {train_accuracy:.4f} | "
                f"Val Loss {val_loss:.5f} | "
                f"Val Acc {val_accuracy:.4f}"
            )

            # ----------------------------------------------------------------
            # Best / early stopping
            # ----------------------------------------------------------------
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                self._save_checkpoint(
                    best_model_path,
                    epoch=epoch,
                    val_loss=val_loss,
                    initial_model_path=initial_model_path,
                )
            else:
                patience_counter += 1

                if patience_counter >= patience:
                    print(
                        f"早期停止: {patience}エポック連続で"
                        "検証損失が改善しませんでした"
                    )
                    break

        checkpoint = torch.load(
            best_model_path,
            map_location=self.device,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])

        print()
        print("=" * 64)
        print("Training Complete")
        print("=" * 64)
        print(f"Best epoch   : {best_epoch}")
        print(f"Best val loss: {best_val_loss:.6f}")
        print("=" * 64)

        self.print_validation_report(x_val, y_val)

    # ------------------------------------------------------------------------
    # 評価表示
    # ------------------------------------------------------------------------

    def print_validation_report(self, x_val, y_val):
        self.model.eval()

        with torch.no_grad():
            logits = self.model(x_val)
            predictions = logits.argmax(dim=1)

        y_true = y_val.detach().cpu().numpy()
        y_pred = predictions.detach().cpu().numpy()

        print()
        print("=" * 64)
        print("Validation Action Accuracy")
        print("=" * 64)

        for action_index, action_name in enumerate(ACTION_NAMES):
            mask = y_true == action_index
            count = int(mask.sum())

            if count == 0:
                accuracy_text = "N/A"
            else:
                accuracy = float((y_pred[mask] == action_index).mean())
                accuracy_text = f"{accuracy:.4f}"

            predicted_count = int((y_pred == action_index).sum())

            print(
                f"{action_name:<18}: "
                f"true={count:6d}, "
                f"pred={predicted_count:6d}, "
                f"acc={accuracy_text}"
            )

        overall_accuracy = float((y_true == y_pred).mean())

        print("-" * 64)
        print(f"Overall validation accuracy: {overall_accuracy:.4f}")
        print("=" * 64)

    # ------------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------------

    def _save_checkpoint(
        self,
        filepath,
        epoch,
        val_loss,
        initial_model_path=None,
    ):
        filepath = Path(filepath)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "obs_size": self.obs_size,
            "num_actions": NUM_ACTIONS,
            "action_names": ACTION_NAMES,
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "observation_version": 2,
            "local_map_size": LOCAL_MAP_SIZE,
            "continued_from": (
                str(initial_model_path)
                if initial_model_path is not None
                else None
            ),
        }

        torch.save(checkpoint, filepath)

    def save_model(self, filepath="policy_bc_final.pt"):
        """
        最終モデルをチェックポイント形式で保存する。

        dagger_train.py / evaluate_bc_dagger.py 側も
        model_state_dict と obs_size を読むように合わせる必要がある。
        """
        if self.model is None:
            raise RuntimeError("保存できるモデルがありません")

        filepath = Path(filepath)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "obs_size": self.obs_size,
            "num_actions": NUM_ACTIONS,
            "action_names": ACTION_NAMES,
            "observation_version": 2,
            "local_map_size": LOCAL_MAP_SIZE,
        }

        torch.save(checkpoint, filepath)
        print(f"モデルを保存しました: {filepath}")

    # ------------------------------------------------------------------------
    # グラフ
    # ------------------------------------------------------------------------

    def plot_losses(self, save_path="training_loss.png"):
        if not self.train_losses:
            raise RuntimeError("訓練履歴がありません")

        epochs = np.arange(1, len(self.train_losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"訓練曲線を保存しました: {save_path}")

    def plot_validation_accuracy(
        self,
        save_path="validation_accuracy.png",
    ):
        if not self.val_accuracies:
            raise RuntimeError("検証精度履歴がありません")

        epochs = np.arange(1, len(self.val_accuracies) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(
            epochs,
            self.val_accuracies,
            label="Validation Accuracy",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"検証精度グラフを保存しました: {save_path}")


# ============================================================================
# 実行
# ============================================================================

def main():
    trainer = BCTrainer(
        device=None,
        seed=42,
        use_class_weights=True,
    )

    observations, actions = trainer.load_demos(
        "demos/rule_based_demos.json",
        team="A",
        skip_invalid_teacher=True,
    )

    trainer.train(
        observations,
        actions,
        epochs=100,
        batch_size=64,
        val_split=0.10,
        learning_rate=3e-4,
        weight_decay=1e-5,
        patience=15,
        best_model_path="policy_bc_best.pt",
    )

    trainer.save_model("policy_bc_final.pt")
    trainer.plot_losses("training_loss.png")
    trainer.plot_validation_accuracy("validation_accuracy.png")


if __name__ == "__main__":
    main()
