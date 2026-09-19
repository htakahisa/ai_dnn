"""
試合データ（JSON）を解析し、構造化データに変換するモジュール。

入力: SeriesResult JSON
出力: MatchSeries, Map, PlayerStat などの構造化オブジェクト
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from party_presets import normalize_team_names


@dataclass
class PlayerStat:
    """プレイヤーの単一マップでの成績"""

    name: str
    team: str
    kills: int
    deaths: int
    role: str = ""
    gunfights_participated: int = 0
    gunfights_won: int = 0
    gunfights_lost: int = 0
    gunfights_draw: int = 0
    one_v_one_participated: int = 0
    one_v_one_won: int = 0
    one_v_one_lost: int = 0
    one_v_one_draw: int = 0
    assists: int = 0
    covers: int = 0
    first_kills: int = 0
    first_deaths: int = 0
    preaim_angle_sum: float = 0.0
    preaim_angle_count: int = 0

    @property
    def kd_ratio(self) -> float:
        """キル/デス比率。デス0の場合はキル数をそのまま返す"""
        return float(self.kills) if self.deaths == 0 else self.kills / self.deaths

    @property
    def kda_string(self) -> str:
        """K-D 表記"""
        return f"{self.kills}-{self.deaths}"


@dataclass
class Map:
    """1マップ分の試合データ"""

    number: int
    seed: int
    team1: str
    team2: str
    score1: int  # Attacker wins
    score2: int  # Defender wins
    winner: str
    initial_attacker: str
    overtime: bool
    player_stats: List[PlayerStat] = field(default_factory=list)
    gunfights: List[Dict[str, Any]] = field(default_factory=list)
    assist_events: List[Dict[str, Any]] = field(default_factory=list)
    cover_events: List[Dict[str, Any]] = field(default_factory=list)
    attacker_side_stats: Dict[str, Any] = field(default_factory=dict)
    defender_side_stats: Dict[str, Any] = field(default_factory=dict)
    round_records: List[Dict[str, Any]] = field(default_factory=list)
    replay_frames: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_rounds(self) -> int:
        """総ラウンド数"""
        return self.score1 + self.score2

    @property
    def attacker_won(self) -> bool:
        """Attacker側が勝ったか"""
        return self.winner == self.initial_attacker

    def get_team_stats(self, team_name: str) -> List[PlayerStat]:
        """チーム名でそのマップのプレイヤーをフィルタ"""
        return [p for p in self.player_stats if p.team == team_name]

    def get_player_stat(self, player_name: str) -> Optional[PlayerStat]:
        """プレイヤー名で統計を取得"""
        for p in self.player_stats:
            if p.name == player_name:
                return p
        return None


@dataclass
class MatchSeries:
    """シリーズ全体の情報"""

    team1: str
    team2: str
    maps_to_win: int
    team1_wins: int
    team2_wins: int
    winner: str
    loser: str
    maps: List[Map] = field(default_factory=list)
    total_rounds: int = 0

    @property
    def series_winner(self) -> str:
        """シリーズの勝者"""
        return self.winner

    @property
    def series_loser(self) -> str:
        """シリーズの敗者"""
        return self.loser

    @property
    def total_maps(self) -> int:
        """プレイされたマップ総数"""
        return len(self.maps)

    def get_all_players(self) -> Dict[str, PlayerStat]:
        """全マップを通した全プレイヤーの集計統計を返す (team + name -> PlayerStat)"""
        player_totals: Dict[tuple, PlayerStat] = {}

        for map_data in self.maps:
            for player in map_data.player_stats:
                key = (player.team, player.name)
                if key not in player_totals:
                    player_totals[key] = PlayerStat(
                        name=player.name,
                        team=player.team,
                        kills=0,
                        deaths=0,
                    )
                player_totals[key].kills += player.kills
                player_totals[key].deaths += player.deaths
                player_totals[
                    key
                ].gunfights_participated += player.gunfights_participated
                player_totals[key].gunfights_won += player.gunfights_won
                player_totals[key].gunfights_lost += player.gunfights_lost
                player_totals[key].gunfights_draw += player.gunfights_draw
                player_totals[key].one_v_one_participated += player.one_v_one_participated
                player_totals[key].one_v_one_won += player.one_v_one_won
                player_totals[key].one_v_one_lost += player.one_v_one_lost
                player_totals[key].one_v_one_draw += player.one_v_one_draw
                player_totals[key].assists += player.assists
                player_totals[key].covers += player.covers
                player_totals[key].first_kills += player.first_kills
                player_totals[key].first_deaths += player.first_deaths
                player_totals[key].preaim_angle_sum += player.preaim_angle_sum
                player_totals[key].preaim_angle_count += player.preaim_angle_count
                if not player_totals[key].role:
                    player_totals[key].role = player.role

        # シンプルな名前キーを返す (同名プレイヤーがいない前提)
        result = {}
        for (team, name), stat in player_totals.items():
            result[name] = stat

        return result

    def get_team_players(self, team_name: str) -> Dict[str, PlayerStat]:
        """特定チームの全プレイヤー集計統計"""
        all_players = self.get_all_players()
        return {
            name: stat for name, stat in all_players.items() if stat.team == team_name
        }

    def get_map_by_number(self, map_number: int) -> Optional[Map]:
        """マップ番号で取得"""
        for map_data in self.maps:
            if map_data.number == map_number:
                return map_data
        return None


def load_series_json(json_path: str) -> MatchSeries:
    """JSON ファイルから MatchSeries を構築"""
    with open(json_path, encoding="utf-8") as f:
        data = normalize_team_names(json.load(f))

    series = MatchSeries(
        team1=data.get("team1", ""),
        team2=data.get("team2", ""),
        maps_to_win=data.get("maps_to_win", 0),
        team1_wins=data.get("team1_wins", 0),
        team2_wins=data.get("team2_wins", 0),
        winner=data.get("winner", ""),
        loser=data.get("loser", ""),
        total_rounds=int(data.get("total_rounds", 0) or 0),
    )

    # 各マップをパース
    for map_data in data.get("maps", []):
        player_stats = [
            PlayerStat(
                name=p["name"],
                team=p["team"],
                kills=p["kills"],
                deaths=p["deaths"],
                role=p.get("role", ""),
                gunfights_participated=p.get("gunfights_participated", 0),
                gunfights_won=p.get("gunfights_won", 0),
                gunfights_lost=p.get("gunfights_lost", 0),
                gunfights_draw=p.get("gunfights_draw", 0),
                one_v_one_participated=p.get("one_v_one_participated", 0),
                one_v_one_won=p.get("one_v_one_won", 0),
                one_v_one_lost=p.get("one_v_one_lost", 0),
                one_v_one_draw=p.get("one_v_one_draw", 0),
                assists=p.get("assists", 0),
                covers=p.get("covers", 0),
                first_kills=p.get("first_kills", 0),
                first_deaths=p.get("first_deaths", 0),
                preaim_angle_sum=p.get("preaim_angle_sum", 0.0),
                preaim_angle_count=p.get("preaim_angle_count", 0),
            )
            for p in map_data.get("player_stats", [])
        ]

        map_obj = Map(
            number=map_data["number"],
            seed=map_data["seed"],
            team1=map_data["team1"],
            team2=map_data["team2"],
            score1=map_data["score1"],
            score2=map_data["score2"],
            winner=map_data["winner"],
            initial_attacker=map_data["initial_attacker"],
            overtime=map_data.get("overtime", False),
            player_stats=player_stats,
            gunfights=map_data.get("gunfights", []),
            assist_events=map_data.get("assist_events", []),
            cover_events=map_data.get("cover_events", []),
            attacker_side_stats=map_data.get("attacker_side_stats", {}),
            defender_side_stats=map_data.get("defender_side_stats", {}),
            round_records=map_data.get("round_records", []),
            replay_frames=map_data.get("replay_frames", []),
        )

        series.maps.append(map_obj)

    if not series.total_rounds:
        series.total_rounds = sum(map_data.total_rounds for map_data in series.maps)

    return series


def print_series_summary(series: MatchSeries) -> None:
    """シリーズの基本情報をコンソール出力（デバッグ用）"""
    print(f"\n=== {series.team1} vs {series.team2} ===")
    print(
        f"Result: {series.winner} wins {series.series_winner == series.team1 and series.team1_wins or series.team2_wins} - {series.series_loser == series.team1 and series.team1_wins or series.team2_wins}"
    )
    print(f"Total Maps: {series.total_maps}")
    print()

    for map_data in series.maps:
        print(f"Map {map_data.number}: {map_data.team1} vs {map_data.team2}")
        print(
            f"  Score: {map_data.score1} - {map_data.score2} ({map_data.total_rounds} rounds)"
        )
        print(f"  Winner: {map_data.winner}")
        print(f"  Initial Attacker: {map_data.initial_attacker}")
        print()

        for team in [map_data.team1, map_data.team2]:
            team_players = map_data.get_team_stats(team)
            print(f"  {team}:")
            for player in team_players:
                print(f"    {player.name}: {player.kda_string}")
        print()

    print("=== Series Overall Stats ===")
    all_players = series.get_all_players()
    for name, stat in sorted(
        all_players.items(), key=lambda x: x[1].kills, reverse=True
    ):
        print(
            f"{stat.team:15} {name:15} K:{stat.kills:2} D:{stat.deaths:2} KD:{stat.kd_ratio:5.2f}"
        )


if __name__ == "__main__":
    # テスト用：ファイルが存在するかチェック
    test_file = Path(
        "series_Ghost_Champions_vs_とうやまゲーミング_0-2_20260913_141446.json"
    )
    if test_file.exists():
        series = load_series_json(str(test_file))
        print_series_summary(series)
    else:
        print(f"Test file not found: {test_file}")
