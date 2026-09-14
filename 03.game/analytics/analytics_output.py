"""
試合分析結果を日本語テキスト化するモジュール。

出力レイヤー:
1. チームデータ
2. 試合結果
3. プレイヤー成績
4. ラウンド別分析 + カスタムスコア
5. 戦術分析
6. 反省点・改善提案（AI化待ち、現在ルールベース）
"""

from typing import Tuple
from match_analyzer import MatchSeries
from rating_calculator import (
    calculate_all_player_ratings,
    calculate_all_player_stats,
    calculate_team_rating,
    PlayerRating,
)


def generate_section_1_team_data(series: MatchSeries) -> str:
    """セクション1: チームデータ"""
    output = []
    output.append("=" * 80)
    output.append("【1. チームデータ】")
    output.append("=" * 80)
    output.append("")

    # チーム1
    output.append(f"◆ {series.team1}")
    team1_players = series.get_team_players(series.team1)
    output.append(f"  選手: {', '.join(sorted(team1_players.keys()))}")
    output.append("")

    # チーム2
    output.append(f"◆ {series.team2}")
    team2_players = series.get_team_players(series.team2)
    output.append(f"  選手: {', '.join(sorted(team2_players.keys()))}")
    output.append("")

    return "\n".join(output)


def generate_section_2_result(series: MatchSeries) -> str:
    """セクション2: 試合結果"""
    output = []
    output.append("=" * 80)
    output.append("【2. 試合結果】")
    output.append("=" * 80)
    output.append("")

    output.append(f"最終結果: {series.winner} が勝利")
    output.append(f"マップスコア: {series.team1_wins}勝 vs {series.team2_wins}勝")
    output.append(f"実施マップ数: {series.total_maps}")
    output.append("")

    return "\n".join(output)


def generate_section_3_player_stats(series: MatchSeries) -> str:
    """セクション3: プレイヤー成績"""
    output = []
    output.append("=" * 80)
    output.append("【3. プレイヤー成績（全マップ合計）】")
    output.append("=" * 80)
    output.append("")

    all_players = series.get_all_players()

    # チームごとに出力
    for team in [series.team1, series.team2]:
        output.append(f"▼ {team}")
        team_players = {k: v for k, v in all_players.items() if v.team == team}

        # KD順でソート
        sorted_players = sorted(
            team_players.items(), key=lambda x: (x[1].kills - x[1].deaths), reverse=True
        )

        output.append(f"{'選手名':<15} {'K':<3} {'D':<3} {'KD比':<8}")
        output.append("-" * 40)

        for name, stat in sorted_players:
            output.append(
                f"{name:<15} {stat.kills:<3} {stat.deaths:<3} {stat.kd_ratio:<8.2f}"
            )
            output.append(
                f"      Role:{stat.role or '-':<10} "
                f"Gunfights:{stat.gunfights_participated} "
                f"({stat.gunfights_won}-{stat.gunfights_lost}-{stat.gunfights_draw}) "
                f"Assist:{stat.assists} Cover:{stat.covers}"
            )
            one_v_one_total = (
                stat.one_v_one_won
                + stat.one_v_one_lost
                + stat.one_v_one_draw
            )
            one_v_one_rate = (
                stat.one_v_one_won / one_v_one_total * 100
                if one_v_one_total else 0.0
            )
            output.append(
                f"      1v1:{stat.one_v_one_won}-"
                f"{stat.one_v_one_lost}-{stat.one_v_one_draw} "
                f"WinRate:{one_v_one_rate:.1f}%"
            )
            average_preaim = (
                stat.preaim_angle_sum / stat.preaim_angle_count
                if stat.preaim_angle_count else 0.0
            )
            output.append(
                f"      FirstK:{stat.first_kills} FirstD:{stat.first_deaths} "
                f"PreAim:{average_preaim:.1f}deg"
            )

        output.append("")

    return "\n".join(output)


def generate_section_4_map_analysis(series: MatchSeries, player_ratings: list) -> str:
    """セクション4: ラウンド（マップ）別分析 + カスタムスコア"""
    output = []
    output.append("=" * 80)
    output.append("【4. マップ別分析】")
    output.append("=" * 80)
    output.append("")

    for map_data in series.maps:
        for round_info in map_data.round_records:
            tactic = round_info.get("tactic", {})
            output.append(
                f"R{round_info.get('round_number', '?')}: "
                f"{round_info.get('winner', '?')} "
                f"({round_info.get('reason', 'unknown')}) "
                f"attack={tactic.get('attacker_strategy', 'unknown')} "
                f"site={tactic.get('final_attack_site', '-') }"
            )
        output.append(f"▼ Map {map_data.number}: {map_data.team1} vs {map_data.team2}")
        output.append(f"  結果: {map_data.winner} 勝利")
        output.append(f"  スコア: {map_data.score1} - {map_data.score2}")
        output.append(f"  初期攻撃側: {map_data.initial_attacker}")
        if map_data.overtime:
            output.append(f"  ⚠ オーバータイム")
        output.append("")

        # 各チームのプレイヤー成績
        for side_label, side_data in (
            ("Attacker side", map_data.attacker_side_stats),
            ("Defender side", map_data.defender_side_stats),
        ):
            played = int(side_data.get("rounds_played", 0))
            won = int(side_data.get("rounds_won", 0))
            rate = won / played * 100 if played else 0.0
            output.append(
                f"  {side_label}: rounds {won}/{played} "
                f"win rate {rate:.1f}%"
            )
            if side_label == "Attacker side":
                plants = int(side_data.get("plants", 0))
                postplant = int(side_data.get("postplant_wins", 0))
                output.append(
                    f"    Plant:{plants}/{played} "
                    f"({plants / played * 100 if played else 0.0:.1f}%) "
                    f"PostPlant:{postplant}/{plants} "
                    f"({postplant / plants * 100 if plants else 0.0:.1f}%)"
                )
            else:
                planted_against = int(side_data.get("plants_against", 0))
                retakes = int(side_data.get("retakes_won", 0))
                output.append(
                    f"    PlantedAgainst:{planted_against}/{played} "
                    f"({planted_against / played * 100 if played else 0.0:.1f}%) "
                    f"Retake:{retakes}/{planted_against} "
                    f"({retakes / planted_against * 100 if planted_against else 0.0:.1f}%)"
                )
            for player_name, player in sorted(
                side_data.get("players", {}).items()
            ):
                output.append(
                    f"    {player_name}: K:{player.get('kills', 0)} "
                    f"D:{player.get('deaths', 0)} "
                    f"GF:{player.get('gunfights_participated', 0)} "
                    f"A:{player.get('assists', 0)} "
                    f"C:{player.get('covers', 0)}"
                )

        for team in [map_data.team1, map_data.team2]:
            team_players = map_data.get_team_stats(team)
            output.append(f"  {team}:")

            for player in sorted(team_players, key=lambda p: p.kills, reverse=True):
                output.append(
                    f"    {player.name:<15} K:{player.kills:<2} D:{player.deaths:<2}"
                )

        output.append("")

    # 全体レーティングサマリー
    output.append("=" * 80)
    output.append("【プレイヤーレーティング】")
    output.append("=" * 80)
    output.append("")

    sorted_ratings = sorted(player_ratings, key=lambda x: x.rating, reverse=True)

    output.append(f"{'選手名':<15} {'チーム':<15} {'Rating':>8} {'脅威度':>8}")
    output.append("-" * 60)

    for rating in sorted_ratings:
        output.append(
            f"{rating.name:<15} {rating.team:<15} {rating.rating:>8.1f} {rating.threat_score:>8.1f}"
        )

    output.append("")

    return "\n".join(output)


def generate_section_5_tactical_analysis(
    series: MatchSeries, player_ratings: list
) -> str:
    """セクション5: 戦術分析（ルールベース）"""
    output = []
    output.append("=" * 80)
    output.append("【5. 戦術分析】")
    output.append("=" * 80)
    output.append("")

    winner = series.winner
    loser = series.loser

    # 勝者チームの分析
    winner_ratings = [r for r in player_ratings if r.team == winner]
    loser_ratings = [r for r in player_ratings if r.team == loser]

    # 勝者側のMVP
    top_player = max(winner_ratings, key=lambda x: x.rating)
    output.append(f"【{winner} の勝因】")
    output.append("")
    output.append(f"最強選手: {top_player.name} (Rating: {top_player.rating:.1f})")
    output.append(f"  脅威度スコア: {top_player.threat_score:.1f}/100")
    output.append(f"  KD比: {top_player.kd_ratio:.2f}")
    output.append("")

    # チーム全体の強み
    winner_avg_rating = sum(r.rating for r in winner_ratings) / len(winner_ratings)
    loser_avg_rating = sum(r.rating for r in loser_ratings) / len(loser_ratings)
    rating_diff = winner_avg_rating - loser_avg_rating

    output.append(
        f"チーム平均Rating: {winner_avg_rating:.1f} vs {loser_avg_rating:.1f}"
    )
    output.append(f"  評価差: +{rating_diff:.1f}")
    output.append("")

    if rating_diff > 1000:
        output.append(
            f"→ {winner} は総合的に圧倒的に強かった。全選手がバランス良く活躍。"
        )
    elif rating_diff > 500:
        output.append(f"→ {winner} が全体的な実力で勝利。複数の強選手が機能した。")
    else:
        output.append(f"→ {winner} が接戦を制した。主力選手の活躍が決定打。")

    output.append("")

    # 敗者側の分析
    output.append(f"【{loser} の敗因】")
    output.append("")

    # 最も弱かった選手
    worst_player = min(loser_ratings, key=lambda x: x.rating)
    output.append(
        f"最弱リンク: {worst_player.name} (Rating: {worst_player.rating:.1f})"
    )
    output.append(f"  KD比: {worst_player.kd_ratio:.2f}")

    if worst_player.kd_ratio < 0.5:
        output.append(f"  → デス数が多く、チームの足を引っ張った。")
    else:
        output.append(f"  → 脅威度が低く、相手に警戒されなかった。")

    output.append("")

    # チーム全体の弱み
    output.append(f"チーム平均Rating: {loser_avg_rating:.1f}")
    output.append("")

    consistency_scores = [r.consistency_score for r in loser_ratings]
    avg_consistency = sum(consistency_scores) / len(consistency_scores)

    if avg_consistency < 50:
        output.append(f"→ 選手ごとの成績差が大きく、チームの安定性に欠けた。")
    else:
        output.append(f"→ 全体的に実力不足。パフォーマンス向上が課題。")

    output.append("")

    return "\n".join(output)


def generate_section_6_improvements(
    series: MatchSeries, player_ratings: list, player_stats: list
) -> str:
    """セクション6: 改善提案（現在ルールベース、AI化待ち）"""
    output = []
    output.append("=" * 80)
    output.append("【6. 改善提案】")
    output.append("=" * 80)
    output.append("")

    loser = series.loser
    loser_ratings = [r for r in player_ratings if r.team == loser]
    loser_stats = [r for r in player_stats if r.team == loser]

    output.append(f"【{loser} への改善提案】")
    output.append("")

    # 改善提案1: 低Rating選手への対策
    low_performers = [r for r in loser_ratings if r.rating < 1500]
    if low_performers:
        output.append(f"1. 低パフォーマンス選手への対策")
        for player in low_performers:
            output.append(f"   - {player.name} (Rating: {player.rating:.1f})")
            if player.deaths > player.kills * 2:
                output.append(
                    f"     → 不要なデスを減らす。ポジショニングを改善し、敵と無理に交戦しない。"
                )
            else:
                output.append(f"     → 攻撃性を上げ、敵への圧力をかける。")

        output.append("")

    # 改善提案2: チーム安定性
    consistency_scores = [r.consistency_score for r in loser_ratings]
    avg_consistency = sum(consistency_scores) / len(consistency_scores)

    covers = [r.covers for r in loser_stats]
    avg_cover = sum(covers) / len(covers)

    kds = [r.kd_ratio for r in loser_stats]
    avg_kd = sum(kds) / len(kds)

    output.append(f"2. チーム安定性向上")
    output.append(f"   現在の平均安定性スコア: {avg_consistency:.1f}/100")
    output.append(
        f"   現在のチーム平均カバー数/ラウンド: {avg_cover:.1f}/{series.total_rounds:.1f}"
    )
    output.append(f"   現在のチーム平均KD: {avg_kd:.1f}")

    if avg_consistency < 60:
        output.append(
            f"   → 選手ごとの成績ブレが大きい。マクロで負けていて個人技でなんとかしている印象。"
        )

    if avg_kd > 0.9:
        output.append(f"   →個人技は悪くない。 ")

    if avg_cover > series.total_rounds / 5 and avg_kd < 0.9:
        output.append(
            f"   →カバーやダブルピークのミクロの連携は悪くない。課題は個人技か。 "
        )

    if avg_cover > series.total_rounds / 7 and avg_kd < 0.9:
        output.append(
            f"   →カバーやダブルピークのミクロの連携は完璧。フラッシュを投げてからのピークなどを意識すると改善する可能性。 "
        )

    if avg_cover < series.total_rounds / 5 and avg_kd < 0.9:
        output.append(
            f"   →カバーを取り合う、ダブルピークなど、ミクロの連携で改善する可能性。 "
        )
    output.append("")

    # 改善提案3: MVP逆転
    winner = series.winner
    winner_ratings = [r for r in player_ratings if r.team == winner]
    top_winner = max(winner_ratings, key=lambda x: x.rating)

    output.append(f"3. 相手のMVP対策")
    output.append(
        f"   相手の最強選手: {top_winner.name} (Rating: {top_winner.rating:.1f})"
    )
    output.append(
        f"   → この選手に集中的にマークをつけ、無理をさせるプレイを徹底する。"
    )
    output.append("")

    # 改善提案4: 次のマッチアップ向けアドバイス
    output.append(f"4. 戦術的改善")
    output.append(f"   → 次のマッチでは、チームの弱点を補強した編成を検討する。")
    output.append(
        f"   → 強い相手チームに対しては、敵の主力選手を積極的に狙う戦術を採用。"
    )
    output.append("")

    return "\n".join(output)


def generate_full_report(series: MatchSeries) -> str:
    """全セクションを統合した完全なレポートを生成"""
    player_ratings = calculate_all_player_ratings(series)
    player_stats = calculate_all_player_stats(series)

    sections = [
        generate_section_1_team_data(series),
        generate_section_2_result(series),
        generate_section_3_player_stats(series),
        generate_section_4_map_analysis(series, player_ratings),
        generate_section_5_tactical_analysis(series, player_ratings),
        generate_section_6_improvements(series, player_ratings, player_stats),
    ]

    return "\n\n".join(sections)


def save_report(series: MatchSeries, output_path: str) -> None:
    """レポートをファイルに保存"""
    report = generate_full_report(series)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    from match_analyzer import load_series_json
    from pathlib import Path

    test_file = Path(
        "series_Ghost_Champions_vs_とうやまゲーミング_0-2_20260913_141446.json"
    )
    if test_file.exists():
        series = load_series_json(str(test_file))
        report = generate_full_report(series)
        print(report)

        # ファイルにも保存
        save_report(series, "match_analysis_report.txt")
    else:
        print(f"Test file not found: {test_file}")
