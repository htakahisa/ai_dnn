from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Dict, Iterable, List, Optional, Tuple

from character_stats import CharacterStats, get_stats

# ============================================================
# 基本型
# ============================================================

Position = Tuple[int, int]


class Team(Enum):
    """
    プレイヤーが所属するチーム。
    数値の0、1を直接使わないことで、コードを読みやすくする。
    """

    ATTACKER = "attacker"
    DEFENDER = "defender"


class PlayerAction(Enum):
    """
    将来、プレイヤーごとの行動を記録するときに使用する。

    現時点ですべて使わなくても問題ない。
    """

    WAIT = "wait"
    MOVE = "move"
    SHOOT = "shoot"
    PLANT = "plant"
    DEFUSE = "defuse"


# ============================================================
# 敵の最終確認情報
# ============================================================


@dataclass
class LastSeenInfo:
    """
    ある敵を最後に確認した情報。

    enemy_id:
        確認した敵の固有ID。

    position:
        最後に確認した位置。

    tick:
        最後に確認したTick。
    """

    enemy_id: str
    position: Position
    tick: int


# ============================================================
# 試合中プレイヤー
# ============================================================


@dataclass
class Player:
    """
    試合中のプレイヤーを表すクラス。

    CharacterStats:
        キャラクター固有の変化しない能力値。

    Player:
        HP、位置、生死、設置進行など、
        試合中に変化する状態。
    """

    # ----------------------------
    # 固定情報
    # ----------------------------

    player_id: str
    stats: CharacterStats
    team: Team
    spawn_pos: Position

    max_hp: int = 100

    # ----------------------------
    # 試合中に変化する情報
    # ----------------------------

    pos: Position = field(init=False)
    hp: int = field(init=False)
    alive: bool = field(init=False)

    has_spike: bool = False

    plant_progress: int = 0
    defuse_progress: int = 0

    is_planting: bool = False
    is_defusing: bool = False

    current_action: PlayerAction = PlayerAction.WAIT

    kills: int = 0
    deaths: int = 0
    damage_dealt: int = 0
    damage_taken: int = 0

    # 敵IDごとの最終確認情報
    last_seen_enemies: Dict[str, LastSeenInfo] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        生成直後に初期状態を設定する。
        """

        if not self.player_id:
            raise ValueError("player_id must not be empty")

        if self.max_hp <= 0:
            raise ValueError("max_hp must be greater than 0")

        self.pos = self.spawn_pos
        self.hp = self.max_hp
        self.alive = True

    # ========================================================
    # 能力値への短縮アクセス
    # ========================================================

    @property
    def name(self) -> str:
        return self.stats.name

    @property
    def hit_pct(self) -> float:
        return self.stats.hit_pct

    @property
    def hs_pct(self) -> float:
        return self.stats.hs_pct

    @property
    def dodge_pct(self) -> float:
        return self.stats.dodge_pct

    @property
    def reaction(self) -> float:
        return self.stats.reaction

    @property
    def iq(self) -> float:
        return self.stats.iq

    @property
    def role(self) -> str:
        return self.stats.role

    @property
    def influence(self) -> float:
        return self.stats.influence

    # ========================================================
    # ラウンド管理
    # ========================================================

    def reset_for_round(
        self,
        spawn_pos: Optional[Position] = None,
        keep_score: bool = True,
    ) -> None:
        """
        新しいラウンド用に状態をリセットする。

        spawn_pos:
            指定した場合は新しいスポーン位置を使用する。

        keep_score:
            Trueならkills、deathsなどの試合スコアを保持する。
            Falseなら完全初期化する。
        """

        if spawn_pos is not None:
            self.spawn_pos = spawn_pos

        self.pos = self.spawn_pos
        self.hp = self.max_hp
        self.alive = True

        self.has_spike = False

        self.plant_progress = 0
        self.defuse_progress = 0

        self.is_planting = False
        self.is_defusing = False

        self.current_action = PlayerAction.WAIT

        self.last_seen_enemies.clear()

        if not keep_score:
            self.kills = 0
            self.deaths = 0
            self.damage_dealt = 0
            self.damage_taken = 0

    # ========================================================
    # 移動
    # ========================================================

    def move_to(self, new_pos: Position) -> None:
        """
        プレイヤーを指定位置へ移動する。

        壁判定や他プレイヤーとの衝突判定は、
        Environment側で行ってから呼び出す。
        """

        if not self.alive:
            return

        self.pos = new_pos
        self.current_action = PlayerAction.MOVE

        # 移動したら設置・解除は中断
        self.stop_planting()
        self.stop_defusing(reset_progress=False)

    # ========================================================
    # ダメージ・死亡
    # ========================================================

    def take_damage(
        self,
        amount: int,
        attacker: Optional[Player] = None,
    ) -> int:
        """
        ダメージを受ける。

        戻り値:
            実際に受けたダメージ量。
        """

        if not self.alive:
            return 0

        if amount <= 0:
            return 0

        actual_damage = min(amount, self.hp)

        self.hp -= actual_damage
        self.damage_taken += actual_damage

        if attacker is not None:
            attacker.damage_dealt += actual_damage

        if self.hp <= 0:
            self.die(killer=attacker)

        return actual_damage

    def die(self, killer: Optional[Player] = None) -> None:
        """
        プレイヤーを死亡状態にする。
        """

        if not self.alive:
            return

        self.hp = 0
        self.alive = False
        self.deaths += 1

        self.is_planting = False
        self.is_defusing = False
        self.current_action = PlayerAction.WAIT

        if killer is not None and killer is not self:
            killer.kills += 1

    def heal(self, amount: int) -> int:
        """
        HPを回復する。

        戻り値:
            実際の回復量。
        """

        if not self.alive:
            return 0

        if amount <= 0:
            return 0

        old_hp = self.hp
        self.hp = min(self.max_hp, self.hp + amount)

        return self.hp - old_hp

    # ========================================================
    # スパイク設置
    # ========================================================

    def start_planting(self) -> bool:
        """
        設置状態を開始する。

        実際に設置可能な場所かどうかはEnvironment側で判定する。
        """

        if not self.alive:
            return False

        if not self.has_spike:
            return False

        self.is_planting = True
        self.is_defusing = False
        self.current_action = PlayerAction.PLANT

        return True

    def add_plant_progress(
        self,
        required_ticks: int,
    ) -> bool:
        """
        設置進行を1増やす。

        戻り値:
            設置が完了した場合True。
        """

        if not self.alive:
            return False

        if not self.is_planting:
            return False

        if not self.has_spike:
            return False

        self.plant_progress += 1

        if self.plant_progress >= required_ticks:
            self.plant_progress = required_ticks
            self.is_planting = False
            self.has_spike = False
            return True

        return False

    def stop_planting(
        self,
        reset_progress: bool = True,
    ) -> None:
        """
        設置を中断する。
        """

        self.is_planting = False

        if reset_progress:
            self.plant_progress = 0

    # ========================================================
    # スパイク解除
    # ========================================================

    def start_defusing(self) -> bool:
        """
        解除状態を開始する。

        スパイクとの距離判定はEnvironment側で行う。
        """

        if not self.alive:
            return False

        self.is_defusing = True
        self.is_planting = False
        self.current_action = PlayerAction.DEFUSE

        return True

    def add_defuse_progress(
        self,
        required_ticks: int,
    ) -> bool:
        """
        解除進行を1増やす。

        戻り値:
            解除が完了した場合True。
        """

        if not self.alive:
            return False

        if not self.is_defusing:
            return False

        self.defuse_progress += 1

        if self.defuse_progress >= required_ticks:
            self.defuse_progress = required_ticks
            self.is_defusing = False
            return True

        return False

    def stop_defusing(
        self,
        reset_progress: bool = False,
    ) -> None:
        """
        解除を中断する。

        reset_progress=Falseなら、途中までの解除進行を保持する。
        Valorant方式に近付ける場合は、
        中間地点のチェックポイント処理を別途追加できる。
        """

        self.is_defusing = False

        if reset_progress:
            self.defuse_progress = 0

    # ========================================================
    # 視認情報
    # ========================================================

    def remember_enemy(
        self,
        enemy: Player,
        tick: int,
    ) -> None:
        """
        敵の現在位置を最終確認情報として保存する。
        """

        if not enemy.alive:
            return

        self.last_seen_enemies[enemy.player_id] = LastSeenInfo(
            enemy_id=enemy.player_id,
            position=enemy.pos,
            tick=tick,
        )

    def forget_enemy(self, enemy_id: str) -> None:
        """
        指定した敵の最終確認情報を削除する。
        """

        self.last_seen_enemies.pop(enemy_id, None)

    def forget_old_enemies(
        self,
        current_tick: int,
        memory_ticks: int,
    ) -> None:
        """
        一定Tickより古い視認情報を削除する。
        """

        expired_ids = [
            enemy_id
            for enemy_id, info in self.last_seen_enemies.items()
            if current_tick - info.tick > memory_ticks
        ]

        for enemy_id in expired_ids:
            self.last_seen_enemies.pop(enemy_id, None)

    def get_last_seen(
        self,
        enemy_id: str,
    ) -> Optional[LastSeenInfo]:
        """
        指定した敵の最終確認情報を取得する。
        """

        return self.last_seen_enemies.get(enemy_id)

    def most_recent_last_seen(self) -> Optional[LastSeenInfo]:
        """
        最も最近確認した敵の情報を返す。
        """

        if not self.last_seen_enemies:
            return None

        return max(
            self.last_seen_enemies.values(),
            key=lambda info: info.tick,
        )

    # ========================================================
    # 状態確認
    # ========================================================

    def is_enemy_of(self, other: Player) -> bool:
        return self.team is not other.team

    def can_act(self) -> bool:
        return self.alive

    def can_shoot(self) -> bool:
        """
        解除中でも撃ち返せる仕様なので、
        生存していれば射撃可能。

        射撃した瞬間に解除を中断する処理は、
        Combat側で行う。
        """

        return self.alive

    def __str__(self) -> str:
        return (
            f"{self.player_id}"
            f"({self.name}, {self.team.value}, "
            f"HP={self.hp}, pos={self.pos}, alive={self.alive})"
        )


# ============================================================
# プレイヤー一覧管理
# ============================================================


@dataclass
class PlayerRoster:
    """
    試合に参加する全プレイヤーを管理する。

    1vs1、1vs2、5vs5で同じクラスを使える。
    """

    players: List[Player]
    rng: random.Random = field(
        default_factory=random.Random,
        repr=False,
    )

    def __post_init__(self) -> None:
        player_ids = [player.player_id for player in self.players]

        if len(player_ids) != len(set(player_ids)):
            raise ValueError("player_id must be unique")

    # ========================================================
    # 基本取得
    # ========================================================

    def get_player(
        self,
        player_id: str,
    ) -> Optional[Player]:
        for player in self.players:
            if player.player_id == player_id:
                return player

        return None

    def require_player(
        self,
        player_id: str,
    ) -> Player:
        """
        必ず存在する前提で取得する。
        存在しない場合は例外を出す。
        """

        player = self.get_player(player_id)

        if player is None:
            raise KeyError(f"Unknown player_id: {player_id}")

        return player

    # ========================================================
    # チーム取得
    # ========================================================

    def team_players(
        self,
        team: Team,
        alive_only: bool = False,
    ) -> List[Player]:
        result = [player for player in self.players if player.team is team]

        if alive_only:
            result = [player for player in result if player.alive]

        return result

    @property
    def attackers(self) -> List[Player]:
        return self.team_players(Team.ATTACKER)

    @property
    def defenders(self) -> List[Player]:
        return self.team_players(Team.DEFENDER)

    @property
    def alive_attackers(self) -> List[Player]:
        return self.team_players(
            Team.ATTACKER,
            alive_only=True,
        )

    @property
    def alive_defenders(self) -> List[Player]:
        return self.team_players(
            Team.DEFENDER,
            alive_only=True,
        )

    @property
    def alive_players(self) -> List[Player]:
        return [player for player in self.players if player.alive]

    # ========================================================
    # 敵取得
    # ========================================================

    def enemies_of(
        self,
        player: Player,
        alive_only: bool = True,
    ) -> List[Player]:
        enemies = [other for other in self.players if other.team is not player.team]

        if alive_only:
            enemies = [enemy for enemy in enemies if enemy.alive]

        return enemies

    # ========================================================
    # マップ位置
    # ========================================================

    def player_at(
        self,
        pos: Position,
        alive_only: bool = True,
    ) -> Optional[Player]:
        """
        指定位置にいるプレイヤーを返す。
        1マス1人仕様を想定。
        """

        for player in self.players:
            if alive_only and not player.alive:
                continue

            if player.pos == pos:
                return player

        return None

    def occupied_positions(
        self,
        exclude: Optional[Player] = None,
    ) -> set[Position]:
        """
        生存プレイヤーがいるマスを返す。
        """

        return {
            player.pos
            for player in self.players
            if player.alive and player is not exclude
        }

    # ========================================================
    # ラウンド終了判定
    # ========================================================

    def attackers_eliminated(self) -> bool:
        return len(self.alive_attackers) == 0

    def defenders_eliminated(self) -> bool:
        return len(self.alive_defenders) == 0

    # ========================================================
    # スパイク
    # ========================================================

    def spike_carrier(self) -> Optional[Player]:
        for player in self.attackers:
            if player.alive and player.has_spike:
                return player

        return None

    def assign_spike(
        self,
        player: Player,
    ) -> None:
        """
        指定アタッカーにスパイクを渡す。
        他のプレイヤーからはスパイクを外す。
        """

        if player.team is not Team.ATTACKER:
            raise ValueError("Only an attacker can carry the spike")

        for attacker in self.attackers:
            attacker.has_spike = False

        player.has_spike = True

    # ========================================================
    # 射撃順
    # ========================================================

    def random_shooting_order(
        self,
        shooters: Optional[Iterable[Player]] = None,
    ) -> List[Player]:
        """
        現在使用する射撃順。

        全員の反応速度を同じとして扱い、
        同じTick内ではランダム順になる。
        """

        if shooters is None:
            order = self.alive_players.copy()
        else:
            order = [player for player in shooters if player.alive]

        self.rng.shuffle(order)
        return order

    def reaction_shooting_order(
        self,
        shooters: Optional[Iterable[Player]] = None,
    ) -> List[Player]:
        """
        将来使用する反応速度順。

        reactionが高いプレイヤーから先に行動する。
        同値の場合はランダム順。

        ポケモンの素早さに近い方式。
        """

        if shooters is None:
            candidates = self.alive_players.copy()
        else:
            candidates = [player for player in shooters if player.alive]

        # 先にシャッフルすることで、
        # 同じreaction値のプレイヤーはランダム順になる。
        self.rng.shuffle(candidates)

        candidates.sort(
            key=lambda player: player.reaction,
            reverse=True,
        )

        return candidates

    def shooting_order(
        self,
        shooters: Optional[Iterable[Player]] = None,
        use_reaction: bool = False,
    ) -> List[Player]:
        """
        射撃順を返す共通メソッド。

        現在:
            use_reaction=False
            全員ランダム順。

        将来:
            use_reaction=True
            reactionが高い順。
        """

        if use_reaction:
            return self.reaction_shooting_order(shooters)

        return self.random_shooting_order(shooters)

    # ========================================================
    # ラウンドリセット
    # ========================================================

    def reset_round(
        self,
        spawn_positions: Optional[Dict[str, Position]] = None,
        keep_score: bool = True,
    ) -> None:
        """
        全プレイヤーをラウンド初期状態へ戻す。

        spawn_positions:
            {
                "attacker_1": (23, 18),
                "defender_1": (2, 25),
                "defender_2": (4, 24),
            }
        """

        for player in self.players:
            new_spawn = None

            if spawn_positions is not None:
                new_spawn = spawn_positions.get(player.player_id)

            player.reset_for_round(
                spawn_pos=new_spawn,
                keep_score=keep_score,
            )

    # ========================================================
    # 表示
    # ========================================================

    def summary(self) -> str:
        lines = []

        for player in self.players:
            lines.append(str(player))

        return "\n".join(lines)


# ============================================================
# CharacterStatsからPlayerを生成する補助関数
# ============================================================


def create_player(
    player_id: str,
    character_name: str,
    team: Team,
    spawn_pos: Position,
    max_hp: int = 100,
) -> Player:
    """
    キャラクター名からPlayerを生成する。

    例:
        create_player(
            player_id="defender_1",
            character_name="Leo",
            team=Team.DEFENDER,
            spawn_pos=(2, 25),
        )
    """

    stats = get_stats(character_name)

    if stats is None:
        raise ValueError(f"Character not found: {character_name}")

    return Player(
        player_id=player_id,
        stats=stats,
        team=team,
        spawn_pos=spawn_pos,
        max_hp=max_hp,
    )


# ============================================================
# 使用例
# ============================================================


def create_example_roster(
    seed: Optional[int] = None,
) -> PlayerRoster:
    """
    1vs2のテスト用Rosterを作る。
    """

    rng = random.Random(seed)

    attacker = create_player(
        player_id="attacker_1",
        character_name="Aspas",
        team=Team.ATTACKER,
        spawn_pos=(23, 18),
    )

    defender_1 = create_player(
        player_id="defender_1",
        character_name="Leo",
        team=Team.DEFENDER,
        spawn_pos=(2, 25),
    )

    defender_2 = create_player(
        player_id="defender_2",
        character_name="Chronicle",
        team=Team.DEFENDER,
        spawn_pos=(4, 25),
    )

    roster = PlayerRoster(
        players=[
            attacker,
            defender_1,
            defender_2,
        ],
        rng=rng,
    )

    roster.assign_spike(attacker)

    return roster


if __name__ == "__main__":
    roster = create_example_roster(seed=42)

    print("=== Players ===")
    print(roster.summary())

    print()
    print("=== Random shooting order ===")

    # 現在の仕様：
    # 全員同じ反応速度としてランダム処理
    for player in roster.shooting_order(use_reaction=False):
        print(player.player_id)

    print()
    print("=== Reaction shooting order ===")

    # 将来の仕様：
    # reactionの高い順、同値はランダム
    for player in roster.shooting_order(use_reaction=True):
        print(
            player.player_id,
            player.reaction,
        )
