# evaluate_matchup.py
"""
AttackerコントローラーとDefenderコントローラーの対戦成績を計測するスクリプト。
GUIなし(headless)で高速に何百マッチも回し、勝率を集計する。
"""
import time
import argparse
import os
import glob
import shutil

from run_game import VisualFPSBattle
from map_data import NEW_MAZE_STR
from controllers import DefaultAttackerController, DefaultDefenderController, UserInputController
from learning_attacker import LearningAttackerController
from learning_defender import LearningDefenderController, LearningDefenderAllAIController


def build_controller(kind, side, greedy=False, model_path=None):
    if kind == "default":
        return DefaultAttackerController() if side == "A" else DefaultDefenderController()
    if kind == "learning":
        if side == "A":
            # 外部からパスを指定できるように引数を追加
            path = model_path if model_path else "dqn_attacker_combined_best.pt"
            return LearningAttackerController(model_path=path, greedy=greedy)
        else:
            return LearningDefenderAllAIController(model_path="dqn_defender_combined_best.pt")
    raise ValueError(f"unknown controller kind: {kind}")


def run_matches(att_ctrl, def_ctrl, num_matches, min_win_rate_at_half=0.20):
    """num_matches回のフルマッチ(5ラウンド先取)を実行し、統計を返す。
    途中で望みが薄い場合は早期終了する。"""
    match_att_wins = 0
    match_def_wins = 0
    total_att_rounds = 0
    total_def_rounds = 0

    start = time.time()
    half_matches = num_matches // 2

    for i in range(num_matches):
        game = VisualFPSBattle(NEW_MAZE_STR, att_ctrl, def_ctrl, headless=True)
        game.run()  # match_overになるまで内部でループする

        total_att_rounds += game.attacker_wins
        total_def_rounds += game.defender_wins

        if game.attacker_wins > game.defender_wins:
            match_att_wins += 1
        else:
            match_def_wins += 1

        # 折り返し地点(半分)での早期判定
        if half_matches > 0 and (i + 1) == half_matches:
            current_win_rate = match_att_wins / half_matches
            if current_win_rate < min_win_rate_at_half:
                print(f"  ⚠️ 半分({half_matches}M)経過時点で勝率{current_win_rate*100:.1f}%のため打ち切り")
                elapsed = time.time() - start
                return {
                    "match_att_wins": match_att_wins,
                    "match_def_wins": match_def_wins,
                    "total_att_rounds": total_att_rounds,
                    "total_def_rounds": total_def_rounds,
                    "num_matches": i + 1,  # 実施した数
                    "elapsed": elapsed,
                    "skipped": True
                }

        if (i + 1) % max(1, num_matches // 10) == 0:
            elapsed = time.time() - start
            print(f"  {i+1}/{num_matches} matches done ({elapsed:.1f}s経過) "
                  f"| Att match wins: {match_att_wins} / Def match wins: {match_def_wins}")

    elapsed = time.time() - start
    return {
        "match_att_wins": match_att_wins,
        "match_def_wins": match_def_wins,
        "total_att_rounds": total_att_rounds,
        "total_def_rounds": total_def_rounds,
        "num_matches": num_matches,
        "elapsed": elapsed,
        "skipped": False
    }


def print_summary(stats):
    n = stats["num_matches"]
    att_match_rate = stats["match_att_wins"] / n * 100
    def_match_rate = stats["match_def_wins"] / n * 100
    total_rounds = stats["total_att_rounds"] + stats["total_def_rounds"]
    att_round_rate = stats["total_att_rounds"] / total_rounds * 100 if total_rounds else 0
    def_round_rate = stats["total_def_rounds"] / total_rounds * 100 if total_rounds else 0

    print("\n" + "=" * 50)
    print(f"総マッチ数: {n}  (所要時間: {stats['elapsed']:.1f}秒)")
    print("-" * 50)
    print(f"[マッチ単位の勝率]  ※1マッチ = 5ラウンド先取")
    print(f"  Attacker: {stats['match_att_wins']} 勝 ({att_match_rate:.1f}%)")
    print(f"  Defender: {stats['match_def_wins']} 勝 ({def_match_rate:.1f}%)")
    print("-" * 50)
    print(f"[ラウンド単位の勝率]  ※参考指標(接戦度合いの目安)")
    print(f"  Attacker: {stats['total_att_rounds']} R ({att_round_rate:.1f}%)")
    print(f"  Defender: {stats['total_def_rounds']} R ({def_round_rate:.1f}%)")
    print("=" * 50)


def evaluate_all_saved_models(matches_per_model, greedy=False):
    target_dir = "attacker_data"
    model_files = glob.glob(os.path.join(target_dir, "dqn_attacker_ep*.pt"))
    
    if not model_files:
        print(f"エラー: {target_dir} フォルダ内に評価対象のモデルが見つかりません。")
        return

    print(f"合計 {len(model_files)} 個のモデルを連続評価します。 (各 {matches_per_model} マッチ)")
    print("=" * 50)

    def_ctrl = build_controller("learning", "D")
    
    best_win_rate = -1.0
    best_model_path = ""
    results = {}

    for model_path in sorted(model_files, key=lambda x: int(os.path.basename(x).replace("dqn_attacker_ep", "").replace(".pt", ""))):
        print(f"\n▶ 評価開始: {model_path}")
        att_ctrl = build_controller("learning", "A", greedy=greedy, model_path=model_path)
        
        stats = run_matches(att_ctrl, def_ctrl, matches_per_model)
        win_rate = stats["match_att_wins"] / stats["num_matches"]
        
        # スキップ状態も記録しておく
        results[model_path] = {
            "win_rate": win_rate,
            "skipped": stats.get("skipped", False)
        }
        
        skip_text = " (早期打ち切り)" if stats.get("skipped", False) else ""
        print(f"  => {model_path} の勝率: {win_rate * 100:.1f}%{skip_text}")
        
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_model_path = model_path

    print("\n" + "=" * 50)
    print("🎯 全モデルの評価が完了しました")
    for m, res in results.items():
        skip_text = " (早期打ち切り)" if res["skipped"] else ""
        print(f"  {m}: {res['win_rate'] * 100:.1f}%{skip_text}")
    
    print("-" * 50)
    print(f"👑 最高勝率モデル: {best_model_path} ({best_win_rate * 100:.1f}%)")
    
    # 最高のモデルを combined_best として上書きコピー保存
    final_save_path = os.path.join(target_dir, "dqn_attacker_combined_best.pt")
    shutil.copy(best_model_path, final_save_path)
    print(f"✅ 最高モデルを {final_save_path} に保存・上書きしました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attacker/Defender AIの対戦成績を計測")
    parser.add_argument("--matches", type=int, default=100, help="実行するマッチ数(デフォルト100)")
    parser.add_argument("--attacker", type=str, default="learning", choices=["learning", "default"], help="attacker側コントローラー種別")
    parser.add_argument("--defender", type=str, default="learning", choices=["learning", "default"], help="defender側コントローラー種別")
    parser.add_argument("--greedy", action="store_true", help="attackerをargmax(決定論的)で動かす")
    parser.add_argument("--eval_all", action="store_true", help="attacker_data内の全モデルを連続評価し、最高モデルを保存する")
    
    args = parser.parse_args()

    if args.eval_all:
        # 一括評価モード
        evaluate_all_saved_models(args.matches, greedy=args.greedy)
    else:
        # 通常の1vs1評価モード
        att_ctrl = build_controller(args.attacker, "A", greedy=args.greedy)
        def_ctrl = build_controller(args.defender, "D")

        print(f"対戦カード: Attacker=[{args.attacker}] vs Defender=[{args.defender}]")
        print(f"{args.matches}マッチ実行します...\n")

        stats = run_matches(att_ctrl, def_ctrl, args.matches)
        print_summary(stats)