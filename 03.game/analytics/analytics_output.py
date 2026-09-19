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

from collections import Counter
from typing import Tuple
from match_analyzer import MatchSeries
from rating_calculator import (
    calculate_all_player_ratings,
    calculate_all_player_stats,
    calculate_team_rating,
    PlayerRating,
)


def _attacker_team_for_round(map_data, round_number):
    """Return the team attacking in a recorded round."""
    initial_attacker = map_data.initial_attacker
    other_team = (
        map_data.team2 if initial_attacker == map_data.team1 else map_data.team1
    )
    return initial_attacker if int(round_number) <= 12 else other_team


def attacker_plant_rate(series: MatchSeries, team_name: str) -> float:
    """Percentage of the team's attacking rounds that ended in a plant."""
    attacking_rounds = 0
    planted_rounds = 0

    for map_data in series.maps:
        for round_info in map_data.round_records:
            if (
                _attacker_team_for_round(map_data, round_info.get("round_number", 0))
                != team_name
            ):
                continue
            attacking_rounds += 1
            if round_info.get("planted", False):
                planted_rounds += 1

    return planted_rounds / attacking_rounds * 100 if attacking_rounds else 0.0


def attacker_plant_count(series: MatchSeries, team_name: str) -> int:
    """Return the number of plants by the team while attacking."""
    count = 0
    for map_data in series.maps:
        for round_info in map_data.round_records:
            if _attacker_team_for_round(
                map_data, round_info.get("round_number", 0)
            ) == team_name and round_info.get("planted", False):
                count += 1
    return count


def _team_half_stats(map_data, team_name, half):
    """Aggregate one team's player and round stats for one half of a map."""
    rows = {}
    rounds = []
    for record in map_data.round_records:
        number = int(record.get("round_number", 0) or 0)
        if ("first" if number <= 12 else "second") != half:
            continue
        rounds.append(record)
        for name, player in record.get("players", {}).items():
            if player.get("team") != team_name:
                continue
            row = rows.setdefault(name, {"role": player.get("role", "")})
            row["role"] = player.get("role", row["role"])
            for key, value in player.items():
                if key in {"team", "side", "role"}:
                    continue
                row[key] = row.get(key, 0) + value
    return rounds, rows


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
            # 1v1勝率の分母にはDrawを含めない。
            one_v_one_total = stat.one_v_one_won + stat.one_v_one_lost
            one_v_one_rate = (
                stat.one_v_one_won / one_v_one_total * 100 if one_v_one_total else 0.0
            )
            output.append(
                f"      1v1:{stat.one_v_one_won}-"
                f"{stat.one_v_one_lost}-{stat.one_v_one_draw} "
                f"WinRate:{one_v_one_rate:.1f}%"
            )
            average_preaim = (
                stat.preaim_angle_sum / stat.preaim_angle_count
                if stat.preaim_angle_count
                else 0.0
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
        losing_reason_counts = Counter()
        losing_team = (
            map_data.team2 if map_data.winner == map_data.team1 else map_data.team1
        )
        initial_attacker = map_data.initial_attacker
        initial_defender = (
            map_data.team2 if initial_attacker == map_data.team1 else map_data.team1
        )
        for round_info in map_data.round_records:
            round_number = int(round_info.get("round_number", 0) or 0)
            if round_info.get("winner_team"):
                round_winner = round_info["winner_team"]
            else:
                attacker_team = (
                    initial_attacker if round_number <= 12 else initial_defender
                )
                defender_team = (
                    initial_defender if round_number <= 12 else initial_attacker
                )
                round_winner = (
                    attacker_team
                    if round_info.get("winner") == "attacker"
                    else defender_team
                )
            if round_winner == losing_team:
                losing_reason_counts[round_info.get("reason", "unknown")] += 1
        for round_info in map_data.round_records:
            tactic = round_info.get("tactic", {})
            output.append(
                f"R{round_info.get('round_number', '?')}: "
                f"{round_info.get('winner', '?')} "
                f"({round_info.get('reason', 'unknown')}) "
                f"attack={tactic.get('attacker_strategy', 'unknown')} "
                f"site={tactic.get('final_attack_site', '-') } "
                f"def_setup={tactic.get('defender_initial_setup', 'unknown') } "
                f"({tactic.get('defender_initial_setup_axis', 'A-Mid-B')})"
            )
        output.append(f"▼ Map {map_data.number}: {map_data.team1} vs {map_data.team2}")
        output.append(f"  結果: {map_data.winner} 勝利")
        output.append(f"  スコア: {map_data.score1} - {map_data.score2}")
        output.append(f"  初期攻撃側: {map_data.initial_attacker}")
        if losing_reason_counts:
            output.append(f"  Losing team ({losing_team}) round reasons:")
            for reason, count in sorted(losing_reason_counts.items()):
                output.append(f"    {reason}: {count}")
        if map_data.overtime:
            output.append(f"  ⚠ オーバータイム")
        output.append("")

        # 各チームのプレイヤー成績
        output.append("  Team / half statistics:")
        for team in [map_data.team1, map_data.team2]:
            output.append(f"    {team}:")
            for half, label in (("first", "First half"), ("second", "Second half")):
                rounds, players = _team_half_stats(map_data, team, half)
                won = sum(
                    1
                    for record in rounds
                    if (
                        (record.get("winner") == "attacker")
                        == (
                            _attacker_team_for_round(
                                map_data, record.get("round_number", 0)
                            )
                            == team
                        )
                    )
                )
                output.append(
                    f"      {label} ({'Attacker' if _attacker_team_for_round(map_data, 1 if half == 'first' else 13) == team else 'Defender'}): "
                    f"rounds {won}/{len(rounds)} "
                    f"win rate {won / len(rounds) * 100 if rounds else 0.0:.1f}%"
                )
                for name, player in sorted(players.items()):
                    count = int(player.get("preaim_angle_count", 0))
                    average = (
                        float(player.get("preaim_angle_sum", 0.0)) / count
                        if count
                        else 0.0
                    )
                    output.append(
                        f"        {name}: K:{player.get('kills', 0)} D:{player.get('deaths', 0)} "
                        f"GF:{player.get('gunfights_participated', 0)} "
                        f"1v1:{player.get('one_v_one_won', 0)}-"
                        f"{player.get('one_v_one_lost', 0)}-"
                        f"{player.get('one_v_one_draw', 0)} "
                        f"FirstK:{player.get('first_kills', 0)} FirstD:{player.get('first_deaths', 0)} "
                        f"PreAim:{average:.1f}deg A:{player.get('assists', 0)} C:{player.get('covers', 0)}"
                    )

        for side_label, side_data in (
            ("Attacker side", map_data.attacker_side_stats),
            ("Defender side", map_data.defender_side_stats),
        ):
            played = int(side_data.get("rounds_played", 0))
            won = int(side_data.get("rounds_won", 0))
            rate = won / played * 100 if played else 0.0
            output.append(
                f"  {side_label}: rounds {won}/{played} " f"win rate {rate:.1f}%"
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
            for player_name, player in sorted(side_data.get("players", {}).items()):
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
        output.append(f" → デス数が多く、チームの足を引っ張った。")
    else:
        output.append(f" → 脅威度が低く、相手に警戒されなかった。")

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
                    f"   → 不要なデスを減らす。ポジショニングを改善し、敵と無理に交戦しない。"
                )
            else:
                output.append(f"   → 攻撃性を上げ、敵への圧力をかける。")

        output.append("")

    # 改善提案2: チーム安定性
    consistency_scores = [r.consistency_score for r in loser_ratings]
    avg_consistency = sum(consistency_scores) / len(consistency_scores)

    covers = [r.covers for r in loser_stats]
    avg_cover = sum(covers) / len(covers)

    kds = [r.kd_ratio for r in loser_stats]
    avg_kd = sum(kds) / len(kds)

    one_v_one_won = [r.one_v_one_won for r in loser_stats]
    one_v_one_lost = [r.one_v_one_lost for r in loser_stats]
    one_v_one_win_ratio = sum(one_v_one_won) / (
        sum(one_v_one_won) + sum(one_v_one_lost)
    )

    plant_count = attacker_plant_count(
        series,
        loser,
    )
    attacker_round_count = sum(
        1
        for map_data in series.maps
        for round_info in map_data.round_records
        if _attacker_team_for_round(map_data, round_info.get("round_number", 0))
        == loser
    )
    plant_rate = (
        plant_count / attacker_round_count * 100 if attacker_round_count else 0.0
    )

    defused_rate = attacker_defused_rate(
        series,
        loser,
    )

    team_preaim_samples = [
        (
            float(getattr(stat, "preaim_angle_sum", 0.0)),
            int(getattr(stat, "preaim_angle_count", 0)),
        )
        for stat in loser_stats
        if int(getattr(stat, "preaim_angle_count", 0)) > 0
    ]
    preaim_count = sum(count for _, count in team_preaim_samples)
    preaim_angle = (
        sum(total for total, _ in team_preaim_samples) / preaim_count
        if preaim_count
        else 0.0
    )

    output.append(f"2. チーム安定性向上")
    output.append(f"   現在の平均安定性スコア: {avg_consistency:.1f}/100")
    output.append(
        f"   現在のチーム平均カバー数/ラウンド: {avg_cover:.1f}/{series.total_rounds:.1f}"
    )
    output.append(f"   現在のチーム平均KD: {avg_kd:.1f}")

    if preaim_count and preaim_angle >= 45.0:
        output.append(
            f"   チーム全体的にプリエイムを意識しましょう "
            f"(平均ずれ角度 {preaim_angle:.1f}度)"
        )

    output.append(
        f"     アタッカー時プラント成功率: {plant_count}/"
        f"{attacker_round_count} ({plant_rate:.1f}%)"
    )
    if attacker_round_count and plant_rate <= 30.0:
        output.append(
            "   → シリーズ全体でプラントの成功率は{plant_rate:.1f}%で、意識して上げるべき。"
        )

    if attacker_round_count and plant_rate >= 50.0:
        output.append(
            f"   → シリーズ全体でプラントの成功率は{plant_rate:.1f}%と非常に高い。"
        )

    if avg_consistency < 60:
        output.append(f"   → 選手ごとの成績ブレが大きい。マクロで負けている印象。")

    if defused_rate >= 0.5:
        print(
            f"   → シリーズ全体でプラント後、解除率が {defused_rate:.1f}%。リテイク阻止が課題。"
        )

    if defused_rate <= 0.3:
        print(
            f"    → シリーズ全体でプラント後、解除率が {defused_rate:.1f}%リテイクの阻止はできている。"
        )

    retakes_won, planted_against = team_retake_stats(series, loser)
    retake_rate = retakes_won / planted_against * 100 if planted_against else 0.0
    output.append(
        f"   リテイク成功率: {retakes_won}/{planted_against} ({retake_rate:.1f}%)"
    )
    if planted_against and retake_rate < 30.0:
        output.append(
            "   → リテイクが課題かもしれません。人数を揃えて同時に入る手順を見直しましょう。"
        )

    if one_v_one_win_ratio < 0.8:
        output.append(
            f"    → 個人の撃ち合いで負けている。単独勝負ではなくダブルピークなどを意識する。"
        )

    if one_v_one_win_ratio > 1 and avg_kd < 0.9:
        output.append(
            f"    → 個人の撃ち合いでは勝っている。相手が複数人いるところに単独で勝負しないように。"
        )

    if avg_kd > 1:
        output.append(f"    → 個人技は悪くない。マクロなどで大きなミスがあったか。")

    if avg_cover > series.total_rounds / 7 and avg_kd < 0.9:
        output.append(
            f"    → カバーやダブルピークのミクロの連携は完璧。フラッシュを投げてからのピークなどを意識すると改善する可能性。 "
        )
    elif avg_cover > series.total_rounds / 5 and avg_kd < 0.9:
        output.append(
            f"    → カバーやダブルピークのミクロの連携は悪くない。課題は個人技か。 "
        )

    if avg_cover < series.total_rounds / 5 and avg_kd < 0.9:
        output.append(
            f"    → カバーを取り合う、ダブルピークなど、ミクロの連携で改善する可能性。 "
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
    output.append(f"  → この選手に集中的にマークをつけ、無理をさせるプレイを徹底する。")
    output.append("")

    # 改善提案4: 次のマッチアップ向けアドバイス
    output.append(f"4. 戦術的改善")
    output.append(f"  → 次のマッチでは、チームの弱点を補強した編成を検討する。")
    output.append(
        f"  → 強い相手チームに対しては、敵の主力選手を積極的に狙う戦術を採用。"
    )
    output.append("")

    return "\n".join(output)


def attacker_defused_rate(series, my_team):
    planted_rounds = []
    defused_rounds = []

    for map_data in series.maps:
        initial_attacker = map_data.initial_attacker
        initial_defender = (
            map_data.team2 if initial_attacker == map_data.team1 else map_data.team1
        )

        for round_info in map_data.round_records:
            round_number = int(round_info.get("round_number", 0))

            attacker_team = initial_attacker if round_number <= 12 else initial_defender

            # 自分のチームがアタッカーだったラウンドのみ
            if attacker_team != my_team:
                continue

            if round_info.get("planted", False):
                planted_rounds.append(round_info)

                if round_info.get("reason") == "defused":
                    defused_rounds.append(round_info)

    return len(defused_rounds) / len(planted_rounds) * 100 if planted_rounds else 0.0


def team_retake_stats(series: MatchSeries, team_name: str):
    """Return (successful retakes, planted-against rounds) for a team.

    A team is defending whenever the other team is the round's attacker.  A
    planted round won by the defender is a successful retake, regardless of
    whether it ended by defuse or attacker elimination.
    """
    planted_against = 0
    retakes_won = 0
    for map_data in series.maps:
        for round_info in map_data.round_records:
            if not round_info.get("planted", False):
                continue
            if (
                _attacker_team_for_round(map_data, round_info.get("round_number", 0))
                == team_name
            ):
                continue
            planted_against += 1
            if str(round_info.get("winner", "")).lower() in {
                "defender",
                "d",
                str(team_name).lower(),
            }:
                retakes_won += 1
    return retakes_won, planted_against


def team_retake_success_rate(series: MatchSeries, team_name: str) -> float:
    """Return a team's post-plant defensive success rate as a percentage."""
    won, planted_against = team_retake_stats(series, team_name)
    return won / planted_against * 100 if planted_against else 0.0


def build_improvement_suggestions(series: MatchSeries) -> list[str]:
    """Build concise UI-friendly suggestions for both teams in the series."""
    suggestions = []
    for team in (series.team1, series.team2):
        attacking_rounds = sum(
            1
            for map_data in series.maps
            for round_info in map_data.round_records
            if _attacker_team_for_round(map_data, round_info.get("round_number", 0))
            == team
        )
        plants = attacker_plant_count(series, team)
        if attacking_rounds and plants / attacking_rounds * 100 <= 30.0:
            suggestions.append(
                f"{team}: 攻めのプラント成功率が30%以下です。プラントまでの進行とキャリアーの保護を見直しましょう。"
            )

        retakes_won, planted_against = team_retake_stats(series, team)
        if planted_against:
            rate = retakes_won / planted_against * 100
            if rate < 30.0:
                suggestions.append(
                    f"{team}: リテイク成功率が{rate:.1f}%（{retakes_won}/{planted_against}）です。リテイクが課題かもしれません。"
                )

        team_players = [p for p in series.get_all_players().values() if p.team == team]
        preaim_samples = sum(getattr(p, "preaim_angle_count", 0) for p in team_players)
        preaim_total = sum(getattr(p, "preaim_angle_sum", 0.0) for p in team_players)
        if preaim_samples and preaim_total / preaim_samples >= 45.0:
            suggestions.append(
                f"{team}: チーム全体的にプリエイムを意識しましょう（平均ずれ{preaim_total / preaim_samples:.1f}度）。"
            )

    return suggestions or ["現時点で大きな改善提案はありません。"]


def _team_improvement_suggestions(series: MatchSeries, team: str) -> list[str]:
    """Return the same normalized suggestions used by the report and UI."""
    players = [p for p in series.get_all_players().values() if p.team == team]
    suggestions = []

    attacking_rounds = sum(
        1
        for map_data in series.maps
        for record in map_data.round_records
        if _attacker_team_for_round(map_data, record.get("round_number", 0)) == team
    )
    plants = attacker_plant_count(series, team)
    plant_rate = plants / attacking_rounds * 100 if attacking_rounds else 0.0
    if attacking_rounds and plant_rate <= 30.0:
        suggestions.append(
            f"攻めのプラント成功率が{plant_rate:.1f}%です。プラントまでの進行とキャリアーの保護を見直しましょう。"
        )
    elif attacking_rounds and plant_rate >= 50.0:
        suggestions.append(f"攻めのプラント成功率は{plant_rate:.1f}%で良好です。")

    retakes_won, planted_against = team_retake_stats(series, team)
    retake_rate = retakes_won / planted_against * 100 if planted_against else 0.0
    if planted_against and retake_rate < 30.0:
        suggestions.append(
            f"リテイク成功率が{retake_rate:.1f}%（{retakes_won}/{planted_against}）です。リテイクが課題かもしれません。"
        )

    preaim_count = sum(int(getattr(p, "preaim_angle_count", 0)) for p in players)
    preaim_total = sum(float(getattr(p, "preaim_angle_sum", 0.0)) for p in players)
    preaim_angle = preaim_total / preaim_count if preaim_count else 0.0
    if preaim_count and preaim_angle >= 35.0:
        suggestions.append(
            f"チーム全体的にプリエイムを意識しましょう（平均ずれ角度 {preaim_angle:.1f}度）。"
        )

    consistency = []
    for player in players:
        # Rating objects are not needed here; this value is available on the
        # player-stat objects used by the existing report when present.
        value = getattr(player, "consistency_score", None)
        if value is not None:
            consistency.append(float(value))
    if consistency and sum(consistency) / len(consistency) < 60.0:
        suggestions.append(
            "選手ごとの成績のブレが大きいため、ラウンドごとの再現性を高めましょう。"
        )

    kd_values = [float(p.kd_ratio) for p in players]
    avg_kd = sum(kd_values) / len(kd_values) if kd_values else 0.0
    one_v_one_won = sum(int(p.one_v_one_won) for p in players)
    one_v_one_lost = sum(int(p.one_v_one_lost) for p in players)
    one_v_one_total = one_v_one_won + one_v_one_lost
    one_v_one_rate = one_v_one_won / one_v_one_total if one_v_one_total else 0.0
    if one_v_one_total and one_v_one_rate < 0.8:
        suggestions.append(
            "1v1の勝率が低いため、単独勝負ではなくダブルピークを意識しましょう。"
        )

    covers = sum(int(p.covers) for p in players)
    total_rounds = max(1, series.total_rounds)
    if covers < total_rounds / 5 and avg_kd < 0.9:
        suggestions.append(
            "カバーを取り合い、ダブルピークなどのミクロの連携を増やしましょう。"
        )
    elif covers > total_rounds / 7 and avg_kd < 0.9:
        suggestions.append(
            "カバーとダブルピークは機能しています。フラッシュ後のピークなど次の連携を意識しましょう。"
        )

    return suggestions


def build_improvement_suggestions(series: MatchSeries, team_name=None) -> list[str]:
    """Build report/UI suggestions with one consistent bullet-ready format."""
    teams = [team_name] if team_name else [series.team1, series.team2]
    suggestions = []
    for team in teams:
        for suggestion in _team_improvement_suggestions(series, team):
            suggestions.append(f"{team}: {suggestion}")
    return suggestions or ["現時点で大きな改善提案はありません。"]


def generate_section_6_improvements(
    series: MatchSeries, player_ratings: list, player_stats: list
) -> str:
    """Generate the normalized improvement section used by the UI as well."""
    loser = series.loser or series.team2
    output = ["=" * 80, "【6. 改善提案】", "=" * 80, "", f"【{loser}への改善提案】", ""]
    for suggestion in build_improvement_suggestions(series, loser):
        output.append(f"・{suggestion}")
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
