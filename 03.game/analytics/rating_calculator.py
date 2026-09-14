"""
試合データからプレイヤー/チームレーティングを計算するモジュール。

レーティング分布:
- 1000: Top 100% (最低層)
- 1500: 平均
- 2000: Top 25%
- 2500: Top 10%
- 3000: Top 1% (最高層)

評価基準:
1. KD比率（戦闘力の基本）
2. マップ勝率への寄与度（チーム勝利への貢献）
3. 敵チーム内での脅威度（相手から見た危険度）
4. ラウンド別安定性（ブレの少なさ）
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from match_analyzer import MatchSeries, PlayerStat, Map
import statistics


@dataclass
class PlayerRating:
    """プレイヤーのレーティング結果"""

    name: str
    team: str
    rating: float
    kd_ratio: float
    map_win_contribution: float  # そのプレイヤーが参加したマップの勝率
    threat_score: float  # 敵チーム内での脅威度 (0-100)
    consistency_score: float  # ラウンド別安定性 (0-100)
    kills: int
    deaths: int
    maps_played: int

    def __str__(self) -> str:
        return f"{self.name:15} [{self.team:15}] Rating: {self.rating:6.1f} KD:{self.kd_ratio:5.2f} Threat:{self.threat_score:5.1f}"


@dataclass
class TeamRating:
    """チームのレーティング結果"""

    team: str
    rating: float
    player_ratings: List[PlayerRating]
    wins: int
    losses: int

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def __str__(self) -> str:
        wr = self.win_rate * 100
        return f"{self.team:20} Rating: {self.rating:6.1f} Win Rate: {wr:5.1f}% ({self.wins}W-{self.losses}L)"


def _calculate_kd_score(kd_ratio: float) -> float:
    """
    KD比率から 0-100 のスコアに変換

    - KD 2.0 = 70点
    - KD 1.0 = 50点
    - KD 0.5 = 30点
    - KD 3.0 = 85点
    """
    # KD を正規化して 0-100 に収める
    # KD 0.5 = 25, KD 1.0 = 50, KD 2.0 = 75, KD 3.0 = 85
    normalized = (kd_ratio - 0.5) / 0.5  # 0.5 を起点に正規化
    score = 50 + (normalized * 15)  # 50 を中心に、+15点/KD増加
    return max(0.0, min(100.0, score))  # 0-100 に制限


def _calculate_map_contribution(
    player_stats_per_map: List[Dict], series: MatchSeries
) -> float:
    """
    プレイヤーが参加したマップにおける勝率を計算

    そのプレイヤーが参加したマップで、所属チームが勝った割合
    """
    if not player_stats_per_map:
        return 0.0

    wins = 0
    total = len(player_stats_per_map)

    for map_info in player_stats_per_map:
        map_data = map_info["map"]
        player = map_info["player"]

        # そのマップでプレイヤーが所属するチームが勝ったか
        team_won = map_data.winner == player.team
        if team_won:
            wins += 1

    return wins / total if total > 0 else 0.0


def _calculate_threat_score(
    player: PlayerStat, series: MatchSeries, player_maps: List[Map]
) -> float:
    """
    敵チーム内での脅威度を計算 (0-100)

    このプレイヤーがいなかったら敵はどれだけ楽だったか

    指標:
    - このプレイヤーの KDA vs 敵チーム平均 KDA
    - このプレイヤーが敵をキルした割合
    """
    if not player_maps:
        return 0.0

    # 敵チームの平均 KD を計算
    enemy_players = []
    for map_data in player_maps:
        enemy_team = map_data.team2 if player.team == map_data.team1 else map_data.team1
        for enemy_player in map_data.get_team_stats(enemy_team):
            enemy_players.append(enemy_player)

    if not enemy_players:
        return 0.0

    enemy_avg_kills = sum(p.kills for p in enemy_players) / len(enemy_players)
    enemy_avg_deaths = sum(p.deaths for p in enemy_players) / len(enemy_players) or 0.1

    enemy_avg_kd = enemy_avg_kills / enemy_avg_deaths

    # プレイヤーの KD と敵平均 KD の比較
    player_kd = player.kd_ratio
    kd_diff = player_kd - enemy_avg_kd

    # 敵チーム全体のキル数に対する割合
    total_enemy_kills = sum(p.kills for p in enemy_players)
    player_kill_share = player.kills / max(1, total_enemy_kills)

    # 脅威度スコア
    threat = 30 + (kd_diff * 10) + (player_kill_share * 30)
    return max(0.0, min(100.0, threat))


def _calculate_consistency_score(player_stats_per_map: List[Dict]) -> float:
    """
    ラウンド別の安定性を計算 (0-100)

    各マップでの KD が安定しているほど高スコア
    """
    if len(player_stats_per_map) <= 1:
        return 75.0  # データ不足の場合は中程度

    kds = []
    for map_info in player_stats_per_map:
        player = map_info["player"]
        kds.append(player.kd_ratio)

    # KD の分散が小さいほど安定している
    variance = statistics.variance(kds) if len(kds) > 1 else 0.0

    # 分散 0 = 100点、分散が大きいほど減点
    # 分散 2.0 = 0点
    consistency = max(0.0, 100.0 - (variance * 25))
    return min(100.0, consistency)


def calculate_player_rating(
    player_name: str, series: MatchSeries
) -> Optional[PlayerRating]:
    """
    プレイヤーのレーティングを計算

    Args:
        player_name: プレイヤー名
        series: シリーズデータ

    Returns:
        PlayerRating または None (プレイヤーが見つからない場合)
    """
    # 全マップからこのプレイヤーを抽出
    player_stats_per_map = []
    team_name = None

    for map_data in series.maps:
        player = map_data.get_player_stat(player_name)
        if player:
            player_stats_per_map.append(
                {
                    "player": player,
                    "map": map_data,
                }
            )
            team_name = player.team

    if not player_stats_per_map or not team_name:
        return None

    # 集計統計
    total_kills = sum(m["player"].kills for m in player_stats_per_map)
    total_deaths = sum(m["player"].deaths for m in player_stats_per_map)
    maps_played = len(player_stats_per_map)

    kd_ratio = total_kills / max(1, total_deaths)

    # 各スコアを計算
    kd_score = _calculate_kd_score(kd_ratio)
    map_contribution = _calculate_map_contribution(player_stats_per_map, series)
    threat_score = _calculate_threat_score(
        PlayerStat(
            name=player_name, team=team_name, kills=total_kills, deaths=total_deaths
        ),
        series,
        [m["map"] for m in player_stats_per_map],
    )
    consistency_score = _calculate_consistency_score(player_stats_per_map)

    # 重み付け平均でレーティングを算出
    # KD: 30%, マップ勝率: 25%, 脅威度: 30%, 安定性: 15%
    weighted_score = (
        kd_score * 0.30
        + map_contribution * 100 * 0.25
        + threat_score * 0.30
        + consistency_score * 0.15
    )

    # 0-100 のスコアを 1000-3000 のレーティングに変換
    # weighted_score が 50 のとき 1500、100 のとき 3000
    rating = 1500 + (weighted_score - 50) * 30

    return PlayerRating(
        name=player_name,
        team=team_name,
        rating=rating,
        kd_ratio=kd_ratio,
        map_win_contribution=map_contribution,
        threat_score=threat_score,
        consistency_score=consistency_score,
        kills=total_kills,
        deaths=total_deaths,
        maps_played=maps_played,
    )


def calculate_all_player_ratings(series: MatchSeries) -> List[PlayerRating]:
    """
    全プレイヤーのレーティングを計算
    """
    ratings = []
    all_players = series.get_all_players()

    for player_name in all_players.keys():
        rating = calculate_player_rating(player_name, series)
        if rating:
            ratings.append(rating)

    # レーティング順でソート
    ratings.sort(key=lambda x: x.rating, reverse=True)
    return ratings


def calculate_all_player_stats(series: MatchSeries) -> List[PlayerStat]:
    """
    全プレイヤーのスタッツを計算
    """
    stats = []
    all_players = series.get_all_players()

    for player_name in all_players.keys():
        for map_data in series.maps:
            stat = map_data.get_player_stat(player_name)
        if stat:
            stats.append(stat)

    # レーティング順でソート
    stats.sort(key=lambda x: x.name, reverse=True)
    return stats


def calculate_team_rating(
    team_name: str, series: MatchSeries, player_ratings: List[PlayerRating]
) -> TeamRating:
    """
    チームのレーティングを計算
    """
    team_player_ratings = [r for r in player_ratings if r.team == team_name]

    # チームレーティング = プレイヤーレーティングの平均（重み付け: キル数）
    if not team_player_ratings:
        return TeamRating(
            team=team_name, rating=1500.0, player_ratings=[], wins=0, losses=0
        )

    total_kills = sum(r.kills for r in team_player_ratings)
    weighted_rating = (
        sum(r.rating * r.kills for r in team_player_ratings) / max(1, total_kills)
        if total_kills > 0
        else sum(r.rating for r in team_player_ratings) / len(team_player_ratings)
    )

    # 勝敗数
    wins = sum(1 for m in series.maps if m.winner == team_name)
    losses = series.total_maps - wins

    return TeamRating(
        team=team_name,
        rating=weighted_rating,
        player_ratings=team_player_ratings,
        wins=wins,
        losses=losses,
    )


def print_ratings_summary(series: MatchSeries) -> None:
    """レーティング結果をコンソール出力"""
    player_ratings = calculate_all_player_ratings(series)

    print("\n=== Player Ratings ===")
    print(
        f"{'Name':<15} {'Team':<15} {'Rating':>8} {'K-D':>8} {'KD Ratio':>10} {'Threat':>8} {'Consistency':>12}"
    )
    print("=" * 100)

    for rating in player_ratings:
        print(
            f"{rating.name:<15} {rating.team:<15} {rating.rating:>8.1f} "
            f"{rating.kills}-{rating.deaths:>5} {rating.kd_ratio:>10.2f} "
            f"{rating.threat_score:>8.1f} {rating.consistency_score:>12.1f}"
        )

    print("\n=== Team Ratings ===")
    team1_rating = calculate_team_rating(series.team1, series, player_ratings)
    team2_rating = calculate_team_rating(series.team2, series, player_ratings)

    print(team1_rating)
    print(team2_rating)


if __name__ == "__main__":
    from match_analyzer import load_series_json
    from pathlib import Path

    test_file = Path(
        "series_Ghost_Champions_vs_とうやまゲーミング_0-2_20260913_141446.json"
    )
    if test_file.exists():
        series = load_series_json(str(test_file))
        print_ratings_summary(series)
    else:
        print(f"Test file not found: {test_file}")
