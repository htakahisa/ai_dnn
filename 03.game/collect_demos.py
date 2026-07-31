"""
デモ収集スクリプト（観測改善版）

主な改善:
- 各キャラクター本人の視点で観測を生成
- viewer自身の絶対座標 viewer_pos を追加
- viewer中心の7×7ローカルマップ local_map を追加
- 4方向の移動可否 valid_move_mask を追加
- 教師が返した行動が収集時点で有効か teacher_action_valid を記録
- 既存の観測項目は互換性のため維持

使い方:
    python collect_demos.py

前提:
    このファイルを run_game.py と同じフォルダに置いて実行してください。
"""

import inspect
import json
import os
from pathlib import Path

import numpy as np

from controllers import DefaultAttackerController, DefaultDefenderController
from map_data import NEW_MAZE_STR
from roster_utils import build_two_balanced_rosters
from run_game import VisualFPSBattle


# ============================================================================
# 設定
# ============================================================================

TARGET_DEMOS = 60000
MAX_MATCHES = 15

OUTPUT_DIR = Path("demos")
OUTPUT_FILE = "rule_based_demos.json"

LOCAL_MAP_RADIUS = 3
LOCAL_MAP_SIZE = LOCAL_MAP_RADIUS * 2 + 1

# local_map のセル値
LOCAL_EMPTY = 0
LOCAL_WALL = 1
LOCAL_SITE = 2
LOCAL_ALLY = 3
LOCAL_ENEMY = 4
LOCAL_SELF = 5
LOCAL_OUT_OF_MAP = 6
LOCAL_SPIKE = 7

# train_bc.py / dagger_train.py / evaluate_bc_dagger.py と同じ方向順
DIRECTIONS = [
    (-1, 0),   # 0: MOVE_UP
    (1, 0),    # 1: MOVE_DOWN
    (0, -1),   # 2: MOVE_LEFT
    (0, 1),    # 3: MOVE_RIGHT
]

demos = []


# ============================================================================
# 共通ヘルパー
# ============================================================================

def character_is_alive(char):
    return bool(getattr(char, "is_alive", True))


def occupied_positions(game, viewer=None):
    """
    生存キャラクターの占有座標を返す。

    戻り値:
        {(row, col): character}
    """
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
    """指定座標へviewerが移動可能か判定する。"""
    row = int(row)
    col = int(col)

    grid = game.grid
    height, width = grid.shape

    if not (0 <= row < height and 0 <= col < width):
        return False

    if grid[row, col] == 1:
        return False

    occupied = occupied_positions(game, viewer)
    if (row, col) in occupied:
        return False

    return True


def build_valid_move_mask(game, viewer):
    """
    viewerの現在地から上下左右へ移動できるかを0/1で返す。

    順序:
        UP, DOWN, LEFT, RIGHT
    """
    current_row = int(viewer.pos[0])
    current_col = int(viewer.pos[1])

    mask = []
    for dr, dc in DIRECTIONS:
        nr = current_row + dr
        nc = current_col + dc
        mask.append(1 if is_valid_destination(game, viewer, nr, nc) else 0)

    return mask


def build_local_map(game, viewer, radius=LOCAL_MAP_RADIUS):
    """
    viewerを中央にしたローカルマップを生成する。

    セル値:
        0: 空き
        1: 壁
        2: プラントサイト
        3: 味方
        4: 敵
        5: viewer自身
        6: マップ外
        7: 落ちているスパイク

    戻り値:
        7×7の二次元リスト
    """
    grid = game.grid
    height, width = grid.shape

    viewer_row = int(viewer.pos[0])
    viewer_col = int(viewer.pos[1])
    viewer_team = viewer.team

    occupied = {}
    for char in getattr(game, "chars", []):
        if not character_is_alive(char):
            continue

        pos = getattr(char, "pos", None)
        if pos is None or len(pos) != 2:
            continue

        occupied[(int(pos[0]), int(pos[1]))] = char

    spike_pos = getattr(game, "spike_pos", None)
    spike_pos = (
        (int(spike_pos[0]), int(spike_pos[1]))
        if spike_pos is not None
        else None
    )

    local = []

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

        local.append(row_values)

    return local


def teacher_action_is_valid(game, viewer, result):
    """
    教師Controllerの戻り値が、収集時点で実行可能か確認する。

    アビリティ・PLANT・DEFUSEは移動判定の対象外なのでTrue。
    通常移動のみ、壁・マップ外・占有を検査する。
    """
    if result is None:
        return False

    if isinstance(result, tuple) and len(result) == 2:
        second = result[1]

        if isinstance(second, dict):
            return True

        if isinstance(second, str):
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


# ============================================================================
# Observation
# ============================================================================

def get_game_observation(game, viewer):
    """現在のゲーム状態を、行動するviewer本人の視点で特徴量化する。"""
    if viewer is None:
        raise ValueError("viewer must not be None")

    viewer_team = viewer.team
    enemy_team = "D" if viewer_team == "A" else "A"

    viewer_row = int(viewer.pos[0])
    viewer_col = int(viewer.pos[1])

    obs_dict = {
        # 既存形式を維持
        "grid": game.grid.flatten().tolist(),
        "allies": [],
        "visible_enemies": [],
        "game_state": [],
        "spike_pos": [0, 0],
        "target_plant_pos": [0, 0],
        "visible_enemy_count": 0,
        "distance_to_site": 0.0,

        # 新しい観測
        "viewer_pos": [viewer_row, viewer_col],
        "local_map": build_local_map(game, viewer),
        "valid_move_mask": build_valid_move_mask(game, viewer),
    }

    # viewer自身を必ず先頭にする
    allies = [viewer] + [
        char
        for char in game.chars
        if char.team == viewer_team and char is not viewer
    ]

    for char in allies:
        obs_dict["allies"].append(
            {
                "name": char.name,
                # 教師アクションの方向計算用
                "pos": [int(char.pos[0]), int(char.pos[1])],
                # NN入力用のviewer基準相対座標
                "rel_pos": [
                    int(char.pos[0] - viewer_row),
                    int(char.pos[1] - viewer_col),
                ],
                "hp": int(char.hp),
                "is_alive": 1 if character_is_alive(char) else 0,
                "has_spike": 1 if getattr(char, "has_spike", False) else 0,
                "recon_cd": (
                    1 if getattr(char, "recon_charges", 0) > 0 else 0
                ),
                "flash_cd": (
                    1 if getattr(char, "flash_charges", 0) > 0 else 0
                ),
                "smoke_cd": (
                    1 if getattr(char, "smoke_charges", 0) > 0 else 0
                ),
            }
        )

    # 現行形式との互換性のためキー名はvisible_enemiesを維持する
    for char in game.chars:
        if char.team == enemy_team and character_is_alive(char):
            obs_dict["visible_enemies"].append(
                {
                    "name": char.name,
                    "rel_pos": [
                        int(char.pos[0] - viewer_row),
                        int(char.pos[1] - viewer_col),
                    ],
                    "hp": int(char.hp),
                }
            )

    obs_dict["visible_enemy_count"] = len(obs_dict["visible_enemies"])

    spike_pos = getattr(game, "spike_pos", None)
    if spike_pos is not None:
        obs_dict["spike_pos"] = [
            int(spike_pos[0] - viewer_row),
            int(spike_pos[1] - viewer_col),
        ]

    target_plant_pos = getattr(game, "target_plant_pos", None)
    if target_plant_pos is not None:
        plant_dr = int(target_plant_pos[0] - viewer_row)
        plant_dc = int(target_plant_pos[1] - viewer_col)

        obs_dict["target_plant_pos"] = [plant_dr, plant_dc]
        obs_dict["distance_to_site"] = float(abs(plant_dr) + abs(plant_dc))

    round_timer = float(getattr(game, "round_timer", 0.0))
    obs_dict["game_state"] = [
        round_timer / 100.0,
        1.0 if bool(getattr(game, "is_planted", False)) else 0.0,
    ]

    return obs_dict


# ============================================================================
# Action serialization
# ============================================================================

def result_to_action_data(char, result):
    """Controllerの戻り値をJSON保存用のaction辞書へ変換する。"""
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
        move_pos, action_str = result

        action_data["move"] = [
            int(move_pos[0]),
            int(move_pos[1]),
        ]
        action_data["special"] = action_str

    else:
        if not isinstance(result, (list, tuple, np.ndarray)) or len(result) != 2:
            raise ValueError(
                f"未対応のController戻り値です: "
                f"char={char.name}, result={result!r}"
            )

        action_data["move"] = [
            int(result[0]),
            int(result[1]),
        ]

    return action_data


def make_recording_controller(base_controller, game_ref):
    """
    base_controller.decide_move()をラップし、
    呼ばれるたびに(observation, expert action)を保存する。

    Controllerが返した行動自体は変更しない。
    """
    original_decide_move = base_controller.decide_move

    def wrapped_decide_move(char, game_state):
        # decide_moveが呼ばれた時点の状態を記録する
        observation = get_game_observation(game_ref, char)

        # 同じ状態に対する教師行動を取得する
        result = original_decide_move(char, game_state)

        action_data = result_to_action_data(char, result)
        action_valid = teacher_action_is_valid(game_ref, char, result)

        demos.append(
            {
                "observation": observation,
                "action": action_data,
                "teacher_action_valid": bool(action_valid),
            }
        )

        return result

    base_controller.decide_move = wrapped_decide_move
    return base_controller


# ============================================================================
# Match collection
# ============================================================================

def run_one_match():
    """新しい試合を1つ生成し、match_overまで実行する。"""
    attacker_controller = DefaultAttackerController()
    defender_controller = DefaultDefenderController()

    attacker_roster, defender_roster = build_two_balanced_rosters()

    kwargs = {
        "headless": True,
        "attacker_roster": attacker_roster,
        "defender_roster": defender_roster,
    }

    # run_game.pyの版にdisable_side_swapがある場合だけ渡す
    signature = inspect.signature(VisualFPSBattle.__init__)
    if "disable_side_swap" in signature.parameters:
        kwargs["disable_side_swap"] = True

    game = VisualFPSBattle(
        NEW_MAZE_STR,
        attacker_controller,
        defender_controller,
        **kwargs,
    )

    make_recording_controller(attacker_controller, game)
    make_recording_controller(defender_controller, game)

    game.run_headless_loop()


# ============================================================================
# Save / main
# ============================================================================

def json_default(value):
    """numpy型をJSONへ保存可能なPython標準型へ変換する。"""
    if hasattr(value, "item"):
        return value.item()

    if hasattr(value, "tolist"):
        return value.tolist()

    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def save_demos(path, records):
    """
    一時ファイルへ書き込んでから置換し、
    保存途中で元ファイルが壊れにくいようにする。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            default=json_default,
        )

    os.replace(temporary_path, path)


def print_collection_summary():
    total = len(demos)
    attacker_count = sum(
        1 for demo in demos if demo["action"].get("team") == "A"
    )
    defender_count = total - attacker_count
    invalid_teacher_count = sum(
        1 for demo in demos if not demo.get("teacher_action_valid", True)
    )

    print()
    print("=" * 64)
    print("Demo Collection Summary")
    print("=" * 64)
    print(f"Total records           : {total}")
    print(f"Attacker records        : {attacker_count}")
    print(f"Defender records        : {defender_count}")
    print(f"Invalid teacher actions : {invalid_teacher_count}")

    if total > 0:
        invalid_ratio = invalid_teacher_count / total * 100.0
        print(f"Invalid teacher ratio   : {invalid_ratio:.4f}%")

    print("=" * 64)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"デモ収集開始: 目標 {TARGET_DEMOS} 件 "
        f"(local_map={LOCAL_MAP_SIZE}x{LOCAL_MAP_SIZE})"
    )

    match_count = 0

    while len(demos) < TARGET_DEMOS and match_count < MAX_MATCHES:
        match_count += 1

        print(
            f"--- 試合 {match_count} 開始 "
            f"(現在デモ数 {len(demos)}) ---"
        )

        run_one_match()

        print(
            f"    試合 {match_count} 終了 "
            f"(累計デモ数 {len(demos)})"
        )

    if len(demos) >= TARGET_DEMOS:
        print(f"目標デモ数 {TARGET_DEMOS} に到達しました")
    else:
        print(
            f"試合上限 {MAX_MATCHES} に到達しました "
            f"({len(demos)} / {TARGET_DEMOS})"
        )

    output_path = OUTPUT_DIR / OUTPUT_FILE
    save_demos(output_path, demos)
    print_collection_summary()

    print(
        f"{len(demos)}件の(observation, action)ペアを "
        f"{output_path} に保存しました"
    )


if __name__ == "__main__":
    main()
