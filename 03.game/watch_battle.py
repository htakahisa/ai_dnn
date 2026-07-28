"""
ルールベースAI（アタッカー）vs ルールベースAI（ディフェンダー）を
画面表示で確認するスクリプト。

使い方:
    python watch_battle.py
"""

from run_game import VisualFPSBattle
from map_data import NEW_MAZE_STR
from controllers import DefaultAttackerController, DefaultDefenderController
from roster_utils import build_two_balanced_rosters


def main():
    # 役割構成を揃え、実在キャラ（能力値あり）から両チームを公平に選出
    attacker_roster, defender_roster = build_two_balanced_rosters()
    print("Attacker roster:", attacker_roster)
    print("Defender roster:", defender_roster)

    game = VisualFPSBattle(
        NEW_MAZE_STR,
        DefaultAttackerController(),
        DefaultDefenderController(),
        headless=False,  # ← 画面表示モード
        disable_side_swap=True,  # スコア入れ替えなしで、ロジックの素の実力を見る
        attacker_roster=attacker_roster,
        defender_roster=defender_roster,
    )
    game.run()


if __name__ == "__main__":
    main()
