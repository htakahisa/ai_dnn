"""
試合内のより細かい戦術データを表現するデータモデル。

- Gunfight: 撃ち合いの記録
- Assist: アシストの記録
- Cover: カバーの記録
- PlayerMapAdvancedStats: マップごとのプレイヤー詳細統計
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal


@dataclass
class Gunfight:
    """撃ち合いの記録

    複数人参加時は、全参加プレイヤーが同じ Gunfight インスタンスを参照する。
    """

    gunfight_id: int
    map_number: int
    timestamp: int  # ゲーム内ティック
    result: Literal["attacker_win", "defender_win", "draw"]

    # 参加プレイヤー
    attacker_players: List[str]  # Attacker チーム参加者
    defender_players: List[str]  # Defender チーム参加者

    # 結果（プレイヤーごと）
    # {プレイヤー名: "win" / "loss" / "draw"}
    individual_results: Dict[str, Literal["win", "loss", "draw"]] = field(
        default_factory=dict
    )

    @property
    def attacker_count(self) -> int:
        return len(self.attacker_players)

    @property
    def defender_count(self) -> int:
        return len(self.defender_players)


@dataclass
class Assist:
    """アシストの記録

    判定: 自分がダメージ/フラッシュ/リコンを与えた敵を、
          味方が10tick以内に倒した場合
    """

    map_number: int
    assister: str
    victim: str  # ダメージを受けた敵プレイヤー
    killer: str  # 敵を倒した味方プレイヤー
    tick_difference: int  # ダメージ～キルまでの経過tick（最大10）
    method: Literal["damage", "flash", "recon"]

    def __str__(self) -> str:
        return f"{self.assister} → {self.victim} (killed by {self.killer}, method: {self.method})"


@dataclass
class Cover:
    """カバーの記録

    判定: 味方がやられた10tick以内に、その相手を自分が倒した場合
    """

    map_number: int
    coverer: str  # カバーした人
    ally: str  # やられた味方
    enemy_killed: str  # 敵を倒した
    tick_difference: int  # 味方死亡～敵撃破までの経過tick（最大10）

    def __str__(self) -> str:
        return f"{self.coverer} covered {self.ally} by killing {self.enemy_killed}"


@dataclass
class PlayerMapAdvancedStats:
    """プレイヤーの1マップ内詳細統計"""

    player_name: str
    team: str
    map_number: int

    # 基本
    kills: int
    deaths: int

    # ロール
    role: str  # "フラッシュ", "スモーカー", "シーカー", "タイガー"

    # 撃ち合い関連
    gunfights_participated: int  # 参加した撃ち合い数
    gunfights_won: int  # 勝った撃ち合い数
    gunfights_lost: int  # 負けた撃ち合い数
    gunfights_draw: int  # ドロー

    # アシスト・カバー
    assists: int  # 与えたアシスト数
    covers: int  # 与えたカバー数

    @property
    def kd_ratio(self) -> float:
        return self.kills / max(1, self.deaths)

    @property
    def gunfight_win_rate(self) -> float:
        total = self.gunfights_won + self.gunfights_lost + self.gunfights_draw
        if total == 0:
            return 0.0
        return self.gunfights_won / total

    def __str__(self) -> str:
        return (
            f"{self.player_name:15} K:{self.kills} D:{self.deaths} "
            f"Gunfights:{self.gunfights_won}-{self.gunfights_lost} "
            f"Assist:{self.assists} Cover:{self.covers}"
        )
