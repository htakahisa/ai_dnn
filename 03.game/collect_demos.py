"""
デモ収集スクリプト（実行可能版・最終）

使い方:
    python collect_demos_final.py

前提:
    このファイルを run_game.py と同じフォルダに置いて実行してください。
"""

import json
import os

from run_game import VisualFPSBattle
from map_data import NEW_MAZE_STR
from controllers import DefaultAttackerController, DefaultDefenderController
from roster_utils import build_two_balanced_rosters

TARGET_DEMOS = 60000  # 目標ステップ数（5〜7万ステップの中間を目安に設定）
MAX_MATCHES = 15  # 念のための上限（無限ループ防止）
OUTPUT_DIR = "demos"
OUTPUT_FILE = "rule_based_demos.json"

demos = []  # ここに (observation, action) を貯めていく


def get_game_observation(game, viewer):
    """現在のゲーム状態を、行動するキャラクター viewer の視点で特徴量化する。"""
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

    # viewerを必ず先頭にし、以降を同じ順序で並べる。
    # "pos" は教師アクションの方向計算用の絶対座標、
    # "rel_pos" はニューラルネット入力用のviewer基準相対座標。
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

    # 行動するキャラクター自身を基準に敵座標を記録する。
    # 現行データ形式との互換性を保つため、キー名はvisible_enemiesのまま。
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

    # 見えている敵人数
    obs_dict["visible_enemy_count"] = len(obs_dict["visible_enemies"])

    # 落ちているスパイク位置をviewer基準の相対座標で記録する。
    if game.spike_pos is not None:
        obs_dict["spike_pos"] = [
            int(game.spike_pos[0] - viewer.pos[0]),
            int(game.spike_pos[1] - viewer.pos[1]),
        ]

    # プラント目標位置をviewer基準の相対座標で記録する。
    if game.target_plant_pos is not None:
        plant_dr = int(game.target_plant_pos[0] - viewer.pos[0])
        plant_dc = int(game.target_plant_pos[1] - viewer.pos[1])

        obs_dict["target_plant_pos"] = [plant_dr, plant_dc]

        # サイトまでのマンハッタン距離
        obs_dict["distance_to_site"] = float(abs(plant_dr) + abs(plant_dc))

    obs_dict["game_state"] = [
        game.round_timer / 100.0,
        1.0 if game.is_planted else 0.0,
    ]

    return obs_dict


def make_recording_controller(base_controller, game_ref):
    """
    既存のコントローラーの decide_move() をラップし、
    呼ばれるたびに (observation, action) を demos に記録する。
    ゲームの挙動そのものは一切変えない（結果をそのまま返すだけ）。
    """

    original_decide_move = base_controller.decide_move

    def wrapped_decide_move(char, game_state):
        result = original_decide_move(char, game_state)

        obs = get_game_observation(game_ref, char)

        action_data = {"char": char.name, "team": char.team}

        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], dict)
        ):
            # ケース1: アビリティ（辞書型）
            move_pos, ability = result
            action_data["move"] = [int(move_pos[0]), int(move_pos[1])]
            action_data["ability"] = ability["ability"]
            action_data["ability_target"] = [
                int(ability["target"][0]),
                int(ability["target"][1]),
            ]
            action_data["special"] = None
        elif (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], str)
        ):
            # ケース2: DEFUSE/PLANT等の文字列型アクション
            move_pos, action_str = result
            action_data["move"] = [int(move_pos[0]), int(move_pos[1])]
            action_data["ability"] = None
            action_data["special"] = action_str
        else:
            # ケース3: 座標のみ
            action_data["move"] = [int(result[0]), int(result[1])]
            action_data["ability"] = None
            action_data["special"] = None

        demos.append({"observation": obs, "action": action_data})
        return result

    base_controller.decide_move = wrapped_decide_move
    return base_controller


def run_one_match():
    """新しい試合を1つ作って最後まで（match_overまで）実行する"""
    attacker_ctrl = DefaultAttackerController()
    defender_ctrl = DefaultDefenderController()

    # 役割構成を揃え、実在キャラ（能力値あり）から両チームを公平に選出
    attacker_roster, defender_roster = build_two_balanced_rosters()

    game = VisualFPSBattle(
        NEW_MAZE_STR,
        attacker_ctrl,
        defender_ctrl,
        headless=True,
        disable_side_swap=True,  # 学習データ収集中はサイドスワップ無効化
        attacker_roster=attacker_roster,
        defender_roster=defender_roster,
    )

    # decide_move をラップして記録できるようにする
    make_recording_controller(attacker_ctrl, game)
    make_recording_controller(defender_ctrl, game)

    game.run_headless_loop()  # match_over になるまで自動で全ラウンド回る


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"デモ収集開始：目標 {TARGET_DEMOS} ステップ分のデータを集めます")

    match_count = 0
    while True:
        match_count += 1
        print(f"--- 試合 {match_count} 開始 (現在デモ数 {len(demos)}) ---")

        run_one_match()

        print(f"    試合 {match_count} 終了 (累計デモ数 {len(demos)})")

        if len(demos) >= TARGET_DEMOS:
            print(f"✅ 目標ステップ数 {TARGET_DEMOS} に到達しました")
            break
        if match_count >= MAX_MATCHES:
            print(
                f"⚠️ 上限試合数 {MAX_MATCHES} に到達したため終了します（目標未達の可能性あり）"
            )
            break

    filepath = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    def _json_default(o):
        """numpyの整数・浮動小数点・配列型をPython標準型に変換する"""
        if hasattr(o, "item"):  # numpy scalar (int64, float32など)
            return o.item()
        if hasattr(o, "tolist"):  # numpy array
            return o.tolist()
        raise TypeError(
            f"Object of type {o.__class__.__name__} is not JSON serializable"
        )

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(demos, f, ensure_ascii=False, default=_json_default)

    print(
        f"✅ {len(demos)} 個の (observation, action) ペアを {filepath} に保存しました"
    )


if __name__ == "__main__":
    main()
