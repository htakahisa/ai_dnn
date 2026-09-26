"""戦術シミュレーションツール - リテイクシーンのシミュレートとリプレイ保存

主な機能:
1. リテイクシーンの初期状態を設定（爆弾設置後の位置、残り時間など）
2. tickごとにプレイヤーの行動をシミュレート
3. 全フレームをJSONに保存し、既存のリプレイビューアで再生可能
4. 様々な初期配置でシミュレーションを実行して戦術を検証
"""

from __future__ import annotations
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Dict, Optional

from game_core import (
    ROUND_DURATION_TICKS,
    SPIKE_DETONATION_TICKS,
    TICK_TIME,
    Character,
    ULTIMATE_COSTS,
    ULTIMATE_NAMES,
    get_character_combat_stats,
)
from run_game import VisualFPSBattle
from battle_logic import BattleLogicMixin
from map_data import NEW_MAZE_STR

SIMULATION_OUTPUT_DIR = Path("tactical_simulations")
SIMULATION_OUTPUT_DIR.mkdir(exist_ok=True)


def get_character_resource_profile(name: str) -> dict[str, Any]:
    """Return the character's normal ability and ultimate limits for the editor."""
    role = get_character_combat_stats(name).get("role", "フラッシュ")
    ability, max_charges = {
        "フラッシュ": ("FLASH", 1),
        "スモーカー": ("SMOKE", 1),
        "シーカー": ("RECON", 2),
        "タイガー": ("HUNT", 0),
    }.get(role, ("FLASH", 1))
    return {
        "ability": ability,
        "max_charges": max_charges,
        "ultimate": ULTIMATE_NAMES.get(role, "TUNNEL"),
        "ultimate_cost": ULTIMATE_COSTS.get(role, 5),
    }


@dataclass
class RetakeScenario:
    """リテイクシーンのシナリオ定義"""

    scenario_name: str
    description: str
    # 爆弾設置位置
    planted_pos: tuple[int, int]
    # 残り爆弾起動時間（tick）
    detonate_timer: int = SPIKE_DETONATION_TICKS
    # ラウンド残り時間（tick）
    round_timer: int = 30
    # 攻撃側（防衛側）の初期配置
    attackers: List[dict[str, Any]] = field(default_factory=list)
    # 守備側（リテイク側）の初期配置
    defenders: List[dict[str, Any]] = field(default_factory=list)
    # 追加のオブジェクト（スモークなど）
    initial_smokes: List[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SimulationResult:
    """シミュレーション結果"""

    scenario: RetakeScenario
    winner: str  # "attackers" or "defenders"
    total_ticks: int
    reason: str = "unknown"
    replay_frames: List[dict[str, Any]] = field(default_factory=list)
    round_records: List[dict[str, Any]] = field(default_factory=list)
    player_stats: List[dict[str, Any]] = field(default_factory=list)


class TacticalSimulator(VisualFPSBattle):
    """戦術シミュレーター本体 - VisualFPSBattleを継承してリテイクシミュレーション専用に調整"""

    def __init__(
        self,
        scenario: RetakeScenario,
        attacker_ai_name: str = "touyama_gaming_v2",
        defender_ai_name: str = "touyama_gaming_v2",
        custom_roster: list = None,
    ):
        # NEW_MAZE_STRのパース結果をデバッグ表示
        lines = [
            line.strip()
            for line in NEW_MAZE_STR.strip("\n").split("\n")
            if line.strip()
        ]
        print(f"Parsed NEW_MAZE_STR: height={len(lines)}, width={len(lines[0])}")
        for i, line in enumerate(lines):
            print(f"Line {i}: length={len(line)}")

        # 指定されたAIをビルド
        from run_game import _build_team_ai

        attacker_ai = _build_team_ai(attacker_ai_name)
        defender_ai = _build_team_ai(defender_ai_name)
        self.scenario = scenario
        self._normalize_fixed_rosters(attacker_ai_name, defender_ai_name)

        super().__init__(
            maze_str=NEW_MAZE_STR,
            initial_attacker_team_ai=attacker_ai,
            initial_defender_team_ai=defender_ai,
            headless=True,  # UIを無効化してヘッドレス実行
        )
        # The parent starts defender setup in init_round(). This retake scenario
        # starts with a planted spike, so the live setup state must be finished.
        self.defender_setup_phase.finish()

        self.simulation_active = True

        # リテイクシナリオに基づいてゲーム状態を上書き
        self.current_round = 1
        self.battle_tick = 0
        self.round_timer = scenario.round_timer
        self.detonate_timer = scenario.detonate_timer
        self.attacker_wins = 0
        self.defender_wins = 0
        self.is_planted = True
        self.planted_pos = scenario.planted_pos
        self.target_plant_pos = scenario.planted_pos
        self.spike_pos = scenario.planted_pos
        # 爆弾は既に設置されているので、誰も保持していない状態に
        self.spike_holder = None
        self.round_over = False
        self.smokes = scenario.initial_smokes.copy()
        self.flash_projectiles = []
        self.recon_projectiles = []
        self.flash_bursts = []
        self.recon_bursts = []
        self.monitor_drones = []
        self.escape_portals = []
        self.tunnel_bursts = []
        self.ultimate_trails = []
        self.available_orbs = []
        self.match_over = False
        self.replay_frames = []  # リプレイフレームを初期化

        # キャラクターを初期化して親クラスのcharsに設定
        self.chars = self._init_characters()

        # 親クラスの_record_replay_frameで必要なプロパティを追加
        self.round_start_tick = 0
        self.analytics_tracker.round_history = []
        # 勝者フラグ
        self.winner = None
        self.result_reason = "unknown"
        self.total_ticks = 0

    def _normalize_fixed_rosters(self, attacker_ai_name: str, defender_ai_name: str):
        """キャラクター名を自動的に補完（どのAIでも任意のロスターを使用可能）"""
        # どのAIでも配置したプレイヤー数だけ使用可能。名前が未設定の場合は自動生成
        for team, ai_name, players in (
            ("A", attacker_ai_name, self.scenario.attackers),
            ("D", defender_ai_name, self.scenario.defenders),
        ):
            # 名前が未設定のプレイヤーには自動的に名前を割り当て
            for index, player in enumerate(players):
                if "name" not in player or not player["name"]:
                    player["name"] = f"{team}_player_{index}"

    def _init_characters(self) -> List[Character]:
        """シナリオからキャラクターを初期化"""
        chars = []
        # 攻撃側を追加
        for a_data in self.scenario.attackers:
            char = self._create_character(a_data, team="A")
            chars.append(char)
        # 守備側を追加
        for d_data in self.scenario.defenders:
            char = self._create_character(d_data, team="D")
            chars.append(char)
        return chars

    def _create_character(self, data: dict, team: str) -> Character:
        """Characterオブジェクトを生成"""
        name = data["name"]
        pos = data["pos"]
        facing = data.get("facing", "N")

        char = Character(
            name=name,
            team=team,
            pos=list(pos),
            text_color="#ffffff" if team == "A" else "#000000",
            bg_color="#0066ff" if team == "A" else "#00cc00",
        )
        char.facing = facing
        # 親クラスで必要なプロパティを全て設定
        char.is_alive = True
        char.blind_remaining = 0
        char.los_revealed = False
        ultimate_points = int(data.get("ultimate_points", 0))
        if not 0 <= ultimate_points <= char.ultimate_cost:
            raise ValueError(f"{name}: ウルトポイントは0～{char.ultimate_cost}で指定してください")
        char.ultimate_points = ultimate_points
        if "ability_charges" in data:
            charge_field = {
                "SMOKE": "smoke_charges",
                "FLASH": "flash_charges",
                "RECON": "recon_charges",
            }.get(char.ability_name)
            max_charges = getattr(char, charge_field) if charge_field else 0
            charges = int(data["ability_charges"])
            if not 0 <= charges <= max_charges:
                raise ValueError(f"{name}: アビリティ残数は0～{max_charges}で指定してください")
            if charge_field:
                setattr(char, charge_field, charges)
        char.orb_collect_timer = 0
        # キャラクターにbase_iqが存在しない場合のフォールバック
        if not hasattr(char, "base_iq"):
            char.base_iq = 100.0
        char.iq = char.base_iq
        return char

    def _apply_player_actions(self, action_handlers: Dict[str, Callable]):
        """各プレイヤーの行動を適用（action_handlersで各行動を制御）"""
        for char in self.chars:
            if not char.is_alive:
                continue
            # 全方向のハンドラーを実行（手動操作時には個別のキーで実行する設計）
            for direction in ["west", "east", "north", "south"]:
                handler_key = f"{char.team}_{char.name}_{direction}"
                handler = action_handlers.get(handler_key)
                if handler:
                    handler(char, self)

    def init_round(self):
        # The parent ends every headless round by immediately starting another one.
        # A tactical simulation contains exactly one round, so preserve its end state.
        if getattr(self, "simulation_active", False) and self.round_over:
            self.current_round -= 1  # check_match_winner incremented it before this call
            self.match_over = True
            self.simulation_active = False
            self._final_frame_recorded = True  # check_match_winner captured the terminal frame
            return
        super().init_round()

    def step(self, action_handlers: Dict[str, Callable]) -> bool:
        """Advance one tick; return False once the scenario's round has ended."""
        if not self.simulation_active or self.round_over:
            return False

        self._final_frame_recorded = False
        self._record_replay_frame()
        self._apply_player_actions(action_handlers)
        self.ultimate_trails = [
            trail
            for trail in getattr(self, "ultimate_trails", [])
            if trail["remaining_ticks"] > 1
        ]
        for trail in self.ultimate_trails:
            trail["remaining_ticks"] -= 1
        self._build_occupancy_counts()
        try:
            for char in self._move_order():
                if char.is_alive:
                    old_pos = tuple(char.pos)
                    old_ultimate_points = char.ultimate_points
                    self.move_character(char)
                    new_pos = tuple(char.pos)
                    if (
                        char.ultimate_name == "RAID"
                        and char.ultimate_points < old_ultimate_points
                        and new_pos != old_pos
                    ):
                        self.ultimate_trails.append(
                            {
                                "start": old_pos,
                                "end": new_pos,
                                "team": char.team,
                                "remaining_ticks": 2,
                            }
                        )
        finally:
            self._clear_occupancy_counts()

        # process_battle advances battle_tick and the active round/spike timer.
        self.process_battle()
        self.total_ticks += 1
        if self.round_over:
            self.winner = "attackers" if self.attacker_wins else "defenders"
            chars = getattr(self, "chars", [])
            alive_attackers = any(c.is_alive for c in chars if c.team == "A")
            alive_defenders = any(c.is_alive for c in chars if c.team == "D")
            is_planted = getattr(self, "is_planted", False)
            if getattr(self, "is_defused", False):
                self.result_reason = "defused"
            elif is_planted and getattr(self, "detonate_timer", 1) <= 0:
                self.result_reason = "detonated"
            elif is_planted and chars and not alive_defenders:
                self.result_reason = "defender_wipe"
            elif not is_planted and getattr(self, "round_timer", 1) <= 0:
                self.result_reason = "time_expired"
            elif not is_planted and chars and not alive_attackers:
                self.result_reason = "attacker_wipe"
            elif not is_planted and chars and not alive_defenders:
                self.result_reason = "defender_wipe"
            else:
                self.result_reason = "round_end"
            self.simulation_active = False
        return self.simulation_active

    def result(self) -> SimulationResult:
        """Capture the final state and build a result for automatic or stepped play."""
        if not getattr(self, "_final_frame_recorded", False):
            self._record_replay_frame()
            self._final_frame_recorded = True
        return SimulationResult(
            scenario=self.scenario,
            winner=self.winner,
            total_ticks=self.total_ticks,
            reason=self.result_reason,
            replay_frames=self.replay_frames,
            player_stats=self._generate_player_stats(),
        )

    def run(
        self,
        action_handlers: Dict[str, Callable],
        max_ticks: int = 200,
        on_tick: Optional[Callable[["TacticalSimulator"], None]] = None,
    ) -> SimulationResult:
        """シミュレーションを実行"""
        print(f"シミュレーション開始: {self.scenario.scenario_name}")

        while self.simulation_active and self.total_ticks < max_ticks:
            self.step(action_handlers)
            if on_tick is not None:
                on_tick(self)
        return self.result()

    def _generate_player_stats(self) -> List[dict]:
        """プレイヤーの統計を生成"""
        stats = []
        for c in self.chars:
            stats.append(
                {
                    "name": c.name,
                    "team": "attackers" if c.team == "A" else "defenders",
                    "survived": c.is_alive,
                    "final_hp": c.hp,
                }
            )
        return stats

    def save_result(self, result: SimulationResult):
        """シミュレーション結果をJSONファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sim_{self.scenario.scenario_name}_{timestamp}.json"
        filepath = SIMULATION_OUTPUT_DIR / filename

        # シナリオと結果を辞書に変換して保存
        output = {
            "scenario": asdict(result.scenario),
            "winner": result.winner,
            "reason": result.reason,
            "total_ticks": result.total_ticks,
            "replay_frames": result.replay_frames,
            "round_records": result.round_records,
            "player_stats": result.player_stats,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"シミュレーション結果を保存しました: {filepath}")
        return filepath


# サンプルシナリオ：Aサイトのリテイク
def create_sample_retake_scenario() -> RetakeScenario:
    """サンプルのリテイクシナリオを作成"""
    return RetakeScenario(
        scenario_name="retake_a_site_basic",
        description="Aサイトでの基本的なリテイクシチュエーション",
        planted_pos=(15, 10),  # マップ上の爆弾設置位置
        detonate_timer=40,  # 爆弾が40ticks後に爆発
        round_timer=60,
        # 攻撃側（爆弾を守る側）の初期配置
        attackers=[
            {"name": "タイガー", "pos": (14, 9), "facing": "NW"},
            {"name": "スモーカー", "pos": (16, 11), "facing": "SE"},
        ],
        # 守備側（リテイクする側）の初期配置
        defenders=[
            {"name": "フラッシュ", "pos": (12, 8), "facing": "E"},
            {"name": "シーカー", "pos": (13, 12), "facing": "NE"},
        ],
        # 初期スモーク
        initial_smokes=[
            {"cells": [(13, 9), (14, 9), (13, 10), (14, 10)], "remaining_ticks": 15}
        ],
    )


# サンプルの行動ハンドラー：直線的に移動する例
def create_sample_action_handlers() -> Dict[str, Callable]:
    """サンプルの行動ハンドラーを作成"""
    handlers = {}

    # 守備側フラッシュ：東に移動し続ける
    def flash_east_move(char, simulator):
        char.x += 0.1  # 1tickあたりの移動量

    handlers["D_フラッシュ"] = flash_east_move

    # 攻撃側タイガー：北西に移動し続ける
    def tiger_nw_move(char, simulator):
        char.x -= 0.05
        char.y -= 0.05

    handlers["A_タイガー"] = tiger_nw_move

    return handlers


def main():
    """サンプルの実行"""
    # シナリオ作成
    scenario = create_sample_retake_scenario()
    # シミュレーター初期化
    simulator = TacticalSimulator(scenario)
    # 行動ハンドラー作成
    action_handlers = create_sample_action_handlers()
    # シミュレーション実行
    result = simulator.run(action_handlers, max_ticks=100)
    # 結果保存
    simulator.save_result(result)


if __name__ == "__main__":
    main()
