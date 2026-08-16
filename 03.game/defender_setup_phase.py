"""Defender Setup Phase controller.

ラウンド開始前の Defender Setup Phase を管理する。

役割
----
- Setup Phase の開始 / 終了
- 残りTickの管理
- AttackerをSetup中に固定
- Defenderの移動先がSetupマップ上で許可されているか判定
- Setup中は通常戦闘を開始しないための状態フラグを提供

このファイル単体では battle_logic.py を直接変更しない。
battle_logic.py / run_game.py 側からこのクラスを呼び出して使用する。

Setupマップ
-----------
map_data_defender_setup.py

0 = Setup中に進入可能
1 = Setup中に進入禁止

通常ラウンド開始後はこの制限を使用しない。
"""

from __future__ import annotations

from dataclasses import dataclass

from map_data_defender_setup import (
    DEFENDER_SETUP_TICKS,
    get_setup_mask,
    is_setup_position_allowed,
)


@dataclass
class DefenderSetupState:
    """現在のSetup Phase状態。"""

    active: bool = False
    ticks_elapsed: int = 0
    ticks_remaining: int = 0


class DefenderSetupPhase:
    """Defender Setup Phase の状態管理。"""

    def __init__(self, setup_ticks: int = DEFENDER_SETUP_TICKS):
        self.setup_ticks = max(0, int(setup_ticks))
        self.state = DefenderSetupState()

        # 起動時に一度マップを検証・キャッシュする。
        self.setup_map = get_setup_mask()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return bool(self.state.active)

    @property
    def ticks_elapsed(self) -> int:
        return int(self.state.ticks_elapsed)

    @property
    def ticks_remaining(self) -> int:
        return int(self.state.ticks_remaining)

    @property
    def finished(self) -> bool:
        return not self.active

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """新しいラウンドのSetup Phaseを開始する。"""
        if self.setup_ticks <= 0:
            self.state = DefenderSetupState(
                active=False,
                ticks_elapsed=0,
                ticks_remaining=0,
            )
            return

        self.state = DefenderSetupState(
            active=True,
            ticks_elapsed=0,
            ticks_remaining=self.setup_ticks,
        )

    def reset(self) -> None:
        """ラウンドリセット時にSetup状態を初期化する。"""
        self.start()

    def finish(self) -> None:
        """Setup Phaseを即終了する。"""
        self.state.active = False
        self.state.ticks_remaining = 0

    def advance_tick(self) -> bool:
        """Setupを1Tick進める。

        Returns
        -------
        bool
            この呼び出しによってSetup Phaseが終了した場合 True。
        """
        if not self.active:
            return False

        self.state.ticks_elapsed += 1
        self.state.ticks_remaining = max(
            0,
            self.setup_ticks - self.state.ticks_elapsed,
        )

        if self.state.ticks_remaining <= 0:
            self.finish()
            return True

        return False

    # ------------------------------------------------------------------
    # Team rules
    # ------------------------------------------------------------------

    def team_can_move(self, team: str) -> bool:
        """Setup中にそのTeamが移動可能か。"""
        if not self.active:
            return True

        # Setup中はDefenderのみ移動可能。
        return str(team).upper() == "D"

    def attacker_is_frozen(self) -> bool:
        return self.active

    def defender_can_move_to(self, row: int, col: int) -> bool:
        """Defenderの移動先がSetup中に許可されているか。"""
        if not self.active:
            return True

        return is_setup_position_allowed(row, col)

    def can_move_to(self, team: str, row: int, col: int) -> bool:
        """Team込みのSetup移動判定。"""
        if not self.active:
            return True

        team = str(team).upper()

        if team == "A":
            return False

        if team != "D":
            return False

        return self.defender_can_move_to(row, col)

    # ------------------------------------------------------------------
    # Combat / objective rules
    # ------------------------------------------------------------------

    def combat_enabled(self) -> bool:
        """Setup中は射撃・ダメージ処理を止めるためFalse。"""
        return not self.active

    def abilities_enabled(self) -> bool:
        """Setup中はAbility使用禁止。"""
        return not self.active

    def plant_enabled(self) -> bool:
        """Setup中はPlant禁止。"""
        return not self.active

    def defuse_enabled(self) -> bool:
        """Setup中はDefuse禁止。"""
        return not self.active

    def round_timer_enabled(self) -> bool:
        """Setup中は通常ラウンド時間を進めない。"""
        return not self.active

    def spike_timer_enabled(self) -> bool:
        """Setup中はSpike関連タイマーを進めない。"""
        return not self.active

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def phase_label(self) -> str:
        if not self.active:
            return "LIVE"

        return f"DEFENDER SETUP {self.ticks_remaining}"

    def snapshot(self) -> dict:
        """UI / debug / logging用の状態辞書。"""
        return {
            "active": self.active,
            "ticks_elapsed": self.ticks_elapsed,
            "ticks_remaining": self.ticks_remaining,
            "setup_ticks": int(self.setup_ticks),
            "phase": self.phase_label(),
        }


if __name__ == "__main__":
    phase = DefenderSetupPhase()

    phase.start()
    print("START:", phase.snapshot())

    while phase.active:
        ended = phase.advance_tick()

        if phase.ticks_remaining in {15, 10, 5, 1, 0}:
            print(phase.snapshot())

        if ended:
            print("SETUP END -> LIVE")
