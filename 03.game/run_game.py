"""Game entry point and VisualFPSBattle composition.

Keep this file as the executable entry point. The implementation is split into
five focused modules beside it.
"""

import random
import tkinter as tk
import numpy as np


from learning_attacker_ai_v2 import LearningAttackerAIv2Controller
from controllers import (
    DefaultAttackerController,
    DefaultDefenderController,
    UserInputController,
)
from learning_defender import (
    LearningDefenderController,
    LearningDefenderAllAIController,
)
from learning_attacker import LearningAttackerController
from learning_attacker_multi import LearningAttackerMultiController
from attacker_v3.multi_role_attacker_controller import MultiRoleAttackerController
from defender_v3.multi_role_defender_controller import MultiRoleDefenderController
from policy_attacker_controller import PolicyAttackerController
from policy_ppo_attacker_controller import PolicyPPOAttackerController
from policy_defender_controller import PolicyDefenderController
from touyama_v1.touyama_defender_controller import TouyamaDefenderController
from touyama_v1.touyama_attacker_controller import TouyamaAttackerController

from touyama_v2.tv2_touyama_defender_controller import Tv2TouyamaDefenderController
from touyama_v2.tv2_touyama_attacker_controller import Tv2TouyamaAttackerController

from ghost_champions_v1_macro import GhostChampionsV1AttackerController, GhostChampionsV1DefenderController
from map_data import NEW_MAZE_STR
from roster_select import RosterSelectScreen
from team_ai import DualRoleTeamAI
from fnatic_v1_rules import FnaticV1AttackerController, FnaticV1DefenderController

from game_core import (
    Character,
    get_all_character_names,
    get_character_combat_stats,
    SIDE_PANEL_WIDTH,
    COMBO_BANNER_HEIGHT,
    TICK_TIME,
    SMOKE_DURATION_TICKS,
    ROUND_DURATION_TICKS,
    SPIKE_DETONATION_TICKS,
)
from combo_awakening import ComboAwakeningMixin
from abilities_los import AbilityLosMixin
from battle_logic import BattleLogicMixin
from analytics.combat_tracker import CombatTracker
from rendering_ui import RenderingUIMixin
from defender_setup_phase import DefenderSetupPhase
from map_data_defender_setup import validate_against_map

ATTACKER_AI_V2_MODEL_PATH = "attacker_ai_v2_data/dqn_attacker_ai_v2_best.pt"
FNATIC_V1_ATTACKER_MODEL_PATH = "policy_fnatic_attacker_dagger_final.pt"
FNATIC_V1_DEFENDER_MODEL_PATH = "policy_fnatic_defender_dagger_final.pt"

# Fnatic v2:
# Attacker = PPO強化学習
# Defender = 現在のDAgger模倣学習
FNATIC_V2_ATTACKER_MODEL_PATH = (
    "ppo_attacker_checkpoints/"
    "policy_fnatic_attacker_ppo_best.pt"
)
FNATIC_V2_DEFENDER_MODEL_PATH = "policy_fnatic_defender_dagger_final.pt"


def _build_team_ai(key):
    normalized = str(key or "default").strip().lower()

    if normalized == "fnatic_v1":
        return DualRoleTeamAI(
            name="Fnatic v1",
            attacker_factory=FnaticV1AttackerController,
            defender_factory=FnaticV1DefenderController,
        )

    if normalized == "fnatic_legacy_v1":
        return DualRoleTeamAI(
            name="Fnatic v1",
            attacker_factory=lambda: PolicyAttackerController(
                model_path=FNATIC_V1_ATTACKER_MODEL_PATH,
                device="auto",
            ),
            defender_factory=lambda: PolicyDefenderController(
                model_path=FNATIC_V1_DEFENDER_MODEL_PATH,
                device="auto",
            ),
        )

    if normalized in {
        "fnatic_v2",
        "fnatic_2",
        "fnatic2",
        "fnatic v2",
    }:
        return DualRoleTeamAI(
            name="Fnatic v2",
            attacker_factory=lambda: PolicyPPOAttackerController(
                model_path=FNATIC_V2_ATTACKER_MODEL_PATH,
                device="auto",
            ),
            defender_factory=lambda: PolicyDefenderController(
                model_path=FNATIC_V2_DEFENDER_MODEL_PATH,
                device="auto",
            ),
        )

    if normalized == "toru_ai_v3.1":
        return DualRoleTeamAI(
            name="Toru AI v3.1",
            attacker_factory=lambda: MultiRoleAttackerController(),
            defender_factory=lambda: MultiRoleDefenderController(),
        )

    if normalized == "touyama_gaming_v1":
        return DualRoleTeamAI(
            name="Touyama Gaming v1",
            attacker_factory=lambda: TouyamaAttackerController(),
            defender_factory=lambda: TouyamaDefenderController(),
        )
    
    if normalized == "touyama_gaming_v2":
        return DualRoleTeamAI(
            name="Touyama Gaming v2",
            attacker_factory=lambda: Tv2TouyamaAttackerController(),
            defender_factory=lambda: Tv2TouyamaDefenderController(),
        )

    if normalized in {
        "ghost_champions_v1",
        "ghost_champions",
        "gc_v1",
        "gc",
    }:
        return DualRoleTeamAI(
            name="Ghost Champions v1",
            attacker_factory=lambda: GhostChampionsV1AttackerController(
                greedy=True,
            ),
            defender_factory=lambda: GhostChampionsV1DefenderController(
                greedy=True,
            ),
        )

    if normalized == "learning_v1":
        return DualRoleTeamAI(
            name="AI v1",
            attacker_factory=lambda: LearningAttackerController(
                model_path=ATTACKER_MODEL_PATH,
                greedy=True,
            ),
            defender_factory=lambda: LearningDefenderAllAIController(
                model_path="dqn_defender_combined_best.pt",
            ),
        )

    if normalized == "default":
        return DualRoleTeamAI(
            name="ロジック",
            attacker_factory=lambda: DefaultAttackerController(),
            defender_factory=lambda: DefaultDefenderController(),
        )

    if normalized == "user":
        # 人間操作はIQ知覚補正でラップしない。
        # RenderingUIMixin は UserInputController を直接認識して
        # クリック操作を有効化するため、ラップされると操作不能になる。
        team_ai = DualRoleTeamAI(
            name="ユーザー操作",
            attacker_factory=lambda: UserInputController(),
            defender_factory=lambda: UserInputController(),
        )
        if hasattr(team_ai, "use_iq_perception"):
            team_ai.use_iq_perception = False
        return team_ai

    raise ValueError(f"不明なTeam AIです: {key}")

class VisualFPSBattle(
    ComboAwakeningMixin,
    AbilityLosMixin,
    BattleLogicMixin,
    RenderingUIMixin,
):

    SIDE_SWAP_MENTAL_RECOVERY = 0.045
    MAP_END_MENTAL_RECOVERY = 0.24

    def _refresh_active_controllers(self):
        self.attacker_controller = (
            self.current_attacker_team_ai.get_attacker_controller()
        )
        self.defender_controller = (
            self.current_defender_team_ai.get_defender_controller()
        )

        for team_ai in (
            self.current_attacker_team_ai,
            self.current_defender_team_ai,
        ):
            team_ai.bind_game(self)

    def __init__(
        self,
        maze_str,
        initial_attacker_team_ai,
        initial_defender_team_ai,
        headless=False,
        attacker_roster=None,
        defender_roster=None,
        spike_holder_name=None,
        defender_spike_holder_name=None,
        attacker_igl_name=None,
        defender_igl_name=None,
        attacker_team_name=None,
        defender_team_name=None,
        disable_side_swap=False,
        series_context=None,
    ):
        self.maze_str = maze_str
        self.headless = headless
        self.disable_side_swap = disable_side_swap
        self.series_context = dict(series_context or {})
        saved_mental_fatigue = self.series_context.get("mental_fatigue", {})
        self.player_mental_fatigue = {
            str(name): max(-0.75, min(0.75, float(value)))
            for name, value in dict(saved_mental_fatigue or {}).items()
        }
        self.team_round_loss_streak = {}
        self.attacker_roster = list(attacker_roster) if attacker_roster else None
        self.defender_roster = list(defender_roster) if defender_roster else None
        self.spike_holder_name = spike_holder_name
        self.defender_spike_holder_name = defender_spike_holder_name
        self.attacker_igl_name = attacker_igl_name
        self.defender_igl_name = defender_igl_name
        self.attacker_team_name = attacker_team_name or "ATTACKERS"
        self.defender_team_name = defender_team_name or "DEFENDERS"

        lines = [
            line.strip() for line in maze_str.strip("\n").split("\n") if line.strip()
        ]
        self.height, self.width = len(lines), len(lines[0])
        self.grid = np.array(
            [[int(ch) for ch in line] for line in lines], dtype=np.int32
        )
        self.cell_size = 24

        self.initial_attacker_team_ai = initial_attacker_team_ai
        self.initial_defender_team_ai = initial_defender_team_ai
        self.current_attacker_team_ai = initial_attacker_team_ai
        self.current_defender_team_ai = initial_defender_team_ai

        self.attacker_controller = None
        self.defender_controller = None
        self.active_user_team = None

        self.attacker_wins = 0
        self.defender_wins = 0
        self.current_round = 1
        self.sides_swapped = False
        self.overtime = False
        self.last_overtime_swap_round = 0
        self.battle_tick = 0
        self.match_stats = {}
        self.analytics_tracker = CombatTracker()
        self.map_offset_x = SIDE_PANEL_WIDTH
        self.map_pixel_width = self.width * self.cell_size
        self.map_pixel_height = self.height * self.cell_size
        self.ability_area_height = 110

        if not self.headless:
            self.root = tk.Tk()
            self.root.title("Attacker vs Defender")
            self.canvas = tk.Canvas(
                self.root,
                width=self.map_pixel_width + SIDE_PANEL_WIDTH * 2,
                height=self.map_pixel_height
                + self.ability_area_height
                + COMBO_BANNER_HEIGHT,
                bg="#10141c",
                highlightthickness=0,
            )
            self.canvas.pack(fill="both", expand=True)
            self.root.minsize(
                self.map_pixel_width + SIDE_PANEL_WIDTH * 2,
                self.map_pixel_height
                + self.ability_area_height
                + COMBO_BANNER_HEIGHT
                + 30,
            )
            self.canvas.bind("<Button-1>", self.on_canvas_click)
            self.label = tk.Label(self.root, text="Round 1 Start", font=("Arial", 10))
            self.label.pack()

        self.match_over = False
        # Store actual states so replay is exact and does not depend on
        # rerunning controller/model randomness from a seed.
        self.replay_frames = []

        # set_game()呼び出し前に実際のコントローラーインスタンスを
        # self.attacker_controller / self.defender_controllerへ反映させる必要がある。
        # (以前はここが self.attacker_controller=None のままset_game判定をしていたため、
        # TouyamaAttackerController.set_game() が一度も呼ばれずAI側のサイト選択上書きが
        # 常にスキップされていた。)
        self._refresh_active_controllers()

        if hasattr(self.attacker_controller, "set_game"):
            self.attacker_controller.set_game(self)

        if hasattr(self.defender_controller, "set_game"):
            self.defender_controller.set_game(self)

        setup_errors = validate_against_map(self.maze_str)
        if setup_errors:
            raise ValueError(
                "Defender Setup map mismatch: " + "; ".join(setup_errors)
            )
        self.defender_setup_phase = DefenderSetupPhase()

        self.init_round()

    def _record_replay_frame(self):
        """Append a JSON-safe snapshot of the current match state."""
        def pos(value):
            return list(map(int, value)) if value is not None else None

        chars = []
        for char in getattr(self, "chars", []):
            visible_to = ["A", "D"]
            if char.is_alive:
                visible_to = []
                for viewer_team in ("A", "D"):
                    if char.team == viewer_team or self.is_visible_to_team(
                        char, viewer_team
                    ):
                        visible_to.append(viewer_team)
            chars.append({
                "name": str(char.name),
                "display_name": str(getattr(char, "display_name", char.name)),
                "team": str(char.team),
                "pos": pos(char.pos),
                "hp": float(getattr(char, "hp", 0)),
                "alive": bool(getattr(char, "is_alive", False)),
                "facing": str(getattr(char, "facing", "")),
                "has_spike": bool(getattr(char, "has_spike", False)),
                "blind": int(getattr(char, "blind_remaining", 0)),
                "revealed": bool(getattr(char, "los_revealed", False)),
                "ultimate": str(getattr(char, "ultimate_name", "")),
                "ultimate_points": int(getattr(char, "ultimate_points", 0)),
                "ultimate_cost": int(getattr(char, "ultimate_cost", 0)),
                "orb_collect_timer": int(getattr(char, "orb_collect_timer", 0)),
                # Per-team visibility is stored for fog-of-war replay views.
                # Own players are always visible to their own team.
                "visible_to": visible_to,
            })

        def projectile(item):
            return {
                "path": [list(map(int, cell)) for cell in item.get("path", [])],
                "progress": int(item.get("progress", 0)),
            }

        self.replay_frames.append({
            "round": int(getattr(self, "current_round", 0)),
            "tick": int(getattr(self, "battle_tick", 0)),
            "setup": bool(getattr(self, "in_defender_setup_phase", False)),
            "setup_ticks_remaining": int(getattr(self, "defender_setup_ticks_remaining", 0)),
            "round_timer": int(getattr(self, "round_timer", 0)),
            "detonate_timer": int(getattr(self, "detonate_timer", 0)),
            "attacker_wins": int(getattr(self, "attacker_wins", 0)),
            "defender_wins": int(getattr(self, "defender_wins", 0)),
            "planted": bool(getattr(self, "is_planted", False)),
            "round_over": bool(getattr(self, "round_over", False)),
            "target_plant_pos": pos(getattr(self, "target_plant_pos", None)),
            "planted_pos": pos(getattr(self, "planted_pos", None)),
            "spike_pos": pos(getattr(self, "spike_pos", None)),
            "available_orbs": [
                list(map(int, cell))
                for cell in sorted(getattr(self, "available_orbs", set()))
            ],
            "chars": chars,
            "smokes": [
                {"cells": [list(map(int, cell)) for cell in smoke.get("cells", [])],
                 "remaining_ticks": int(smoke.get("remaining_ticks", smoke.get("remaining", 0)))}
                for smoke in getattr(self, "smokes", [])
            ],
            "flash_projectiles": [projectile(item) for item in getattr(self, "flash_projectiles", [])],
            "recon_projectiles": [projectile(item) for item in getattr(self, "recon_projectiles", [])],
            "flash_bursts": [
                {"pos": pos(item.get("pos")), "remaining_ticks": int(item.get("remaining_ticks", 0))}
                for item in getattr(self, "flash_bursts", [])
            ],
            "recon_bursts": [
                {"cells": [list(map(int, cell)) for cell in item.get("cells", [])],
                 "remaining_ticks": int(item.get("remaining_ticks", 0))}
                for item in getattr(self, "recon_bursts", [])
            ],
            "monitor_drones": [
                {
                    "name": str(drone.name),
                    "team": str(drone.team),
                    "pos": pos(drone.pos),
                    "hp": int(drone.hp),
                    "target": drone.target_name,
                }
                for drone in getattr(self, "monitor_drones", [])
                if drone.is_alive
            ],
            "tunnel_bursts": [
                {
                    "cells": [list(map(int, cell)) for cell in item.get("cells", [])],
                    "remaining_ticks": int(item.get("remaining_ticks", 0)),
                }
                for item in getattr(self, "tunnel_bursts", [])
            ],
        })

    def _apply_igl_iq_bonus(self):
        """IGL本人を含む全員へ補正し、最終IQを0～300へ制限する。"""
        IQ_MIN = 0.0
        IQ_MAX = 300.0

        for team, igl_name in (
            ("A", self.attacker_igl_name),
            ("D", self.defender_igl_name),
        ):
            members = [char for char in self.chars if char.team == team]
            igl = next(
                (char for char in members if char.name == igl_name),
                None,
            )

            pre_igl_iq = {}
            for char in members:
                char.is_igl = bool(igl and char.name == igl.name)
                value = float(getattr(char, "iq", getattr(char, "base_iq", 100.0)))
                value = max(IQ_MIN, value)
                pre_igl_iq[id(char)] = value
                char.iq = value
                char.effective_iq = value

            if igl is not None:
                multiplier = max(0.0, pre_igl_iq[id(igl)] / 100.0)
                for char in members:
                    value = pre_igl_iq[id(char)] * multiplier
                    value = min(IQ_MAX, max(IQ_MIN, value))
                    char.iq = value
                    char.effective_iq = value

            total_influence = sum(
                max(0.0, float(getattr(char, "influence", 0.0))) for char in members
            )
            penalty = max(0.0, (total_influence - 300.0) / 10.0)

            for char in members:
                value = float(char.effective_iq) - penalty
                value = min(IQ_MAX, max(IQ_MIN, value))
                char.iq = value
                char.effective_iq = value

    def _swap_sides(self):
        """編成・IGL・スパイク担当・スコアをチームごと攻守交換する。"""
        self.attacker_roster, self.defender_roster = (
            self.defender_roster,
            self.attacker_roster,
        )
        self.attacker_team_name, self.defender_team_name = (
            self.defender_team_name,
            self.attacker_team_name,
        )
        self.attacker_igl_name, self.defender_igl_name = (
            self.defender_igl_name,
            self.attacker_igl_name,
        )
        self.spike_holder_name, self.defender_spike_holder_name = (
            self.defender_spike_holder_name,
            self.spike_holder_name,
        )

        # スコアは所属チームに追従させる。
        self.attacker_wins, self.defender_wins = (
            self.defender_wins,
            self.attacker_wins,
        )
        self.current_attacker_team_ai, self.current_defender_team_ai = (
            self.current_defender_team_ai,
            self.current_attacker_team_ai,
        )

        self._recover_bad_mental_state(self.SIDE_SWAP_MENTAL_RECOVERY)

        self._refresh_active_controllers()

    def _recover_bad_mental_state(self, amount):
        """Move only positive mental pressure toward neutral by ``amount``."""
        amount = max(0.0, float(amount))
        for key, value in list(self.player_mental_fatigue.items()):
            value = float(value)
            if value > 0.0:
                self.player_mental_fatigue[key] = max(0.0, value - amount)

    def _swap_sides_if_needed(self):
        """通常戦は13R開始時、OTは毎ラウンド開始時に攻守を交代する。

        disable_side_swap=True の場合はスワップを一切行わない
        （学習データ収集時、A/D固定でロジックの実力をそのまま計測したい場合用）。
        """
        if self.disable_side_swap:
            return

        if not self.sides_swapped and self.current_round >= 13:
            self._swap_sides()
            self.sides_swapped = True
            return

        # 12-12の次に始まる25Rから、OT中は毎ラウンド交代する。
        if (
            self.overtime
            and self.current_round >= 25
            and self.last_overtime_swap_round != self.current_round
        ):
            self._swap_sides()
            self.last_overtime_swap_round = self.current_round

    @staticmethod
    def _mental_player_key(name, fallback_team=""):
        # Player identity must survive attacker/defender side swaps.
        return str(name)

    def _recover_mental_at_map_end(self):
        """Give struggling players a substantial between-map reset."""
        self._recover_bad_mental_state(self.MAP_END_MENTAL_RECOVERY)

    def _series_pressure_for_side(self, side):
        prefix = "attacker" if side == "A" else "defender"
        maps_won = int(self.series_context.get(f"{prefix}_maps_won", 0))
        maps_lost = int(self.series_context.get(f"{prefix}_maps_lost", 0))
        maps_played = int(self.series_context.get("maps_played", 0))
        maps_to_win = int(self.series_context.get("maps_to_win", 1))

        deficit = max(0, maps_lost - maps_won)
        pressure = 0.10 * deficit
        pressure += 0.035 * max(0, maps_played - 1)

        # BO5相当ではMap3以降の長期シリーズ負荷を少し強める。
        if maps_to_win >= 3 and maps_played >= 2:
            pressure += 0.03 * (maps_played - 1)
        return min(0.45, pressure)

    def _long_map_pressure(self):
        completed_rounds = max(0, int(self.current_round) - 1)
        pressure = 0.0
        if completed_rounds >= 20:
            pressure += (completed_rounds - 19) * 0.012
        if completed_rounds >= 26:
            pressure += (completed_rounds - 25) * 0.018
        return min(0.40, pressure)

    def _map_score_pressure(self, side):
        """Pressure from the current map score, independent of mentality."""
        if side == "A":
            own_score = int(self.attacker_wins)
            opponent_score = int(self.defender_wins)
        else:
            own_score = int(self.defender_wins)
            opponent_score = int(self.attacker_wins)

        deficit = max(0, opponent_score - own_score)
        # A small deficit is normal variance. From 3 rounds behind onward,
        # pressure grows quickly enough for an 0-8/1-9 map to be felt.
        return min(0.75, 0.12 * max(0, deficit - 2))

    def _mental_pressure_for_player(self, name, side):
        stats = get_character_combat_stats(name)
        form_variance = float(stats.get("form_variance", 0.0))
        if form_variance <= 0.0:
            return 0.0

        mental = max(0.0, min(20.0, float(stats.get("mental", 5.0))))
        vulnerability = 1.0 - mental / 10.0
        key = self._mental_player_key(name, side)

        accumulated = float(self.player_mental_fatigue.get(key, 0.0))
        situational = (
            self._series_pressure_for_side(side)
            + self._long_map_pressure()
            + self._map_score_pressure(side)
        ) * vulnerability
        return max(-0.75, min(0.75, accumulated + situational))

    def _record_round_mental_result(self, winning_side):
        losing_side = "D" if winning_side == "A" else "A"

        winner_key = next(
            (
                str(getattr(char.name, "team_id", winning_side))
                for char in self.chars
                if char.team == winning_side
            ),
            winning_side,
        )
        loser_key = next(
            (
                str(getattr(char.name, "team_id", losing_side))
                for char in self.chars
                if char.team == losing_side
            ),
            losing_side,
        )

        self.team_round_loss_streak[winner_key] = 0
        loser_streak = int(self.team_round_loss_streak.get(loser_key, 0)) + 1
        self.team_round_loss_streak[loser_key] = loser_streak

        # 1敗目は小さく、2～4連敗から明確に増える。
        base_damage = min(
            0.18,
            0.025 + 0.035 * max(0, loser_streak - 1),
        )

        for char in self.chars:
            stats = get_character_combat_stats(char.name)
            form_variance = float(stats.get("form_variance", 0.0))
            key = self._mental_player_key(char.name, char.team)

            if form_variance <= 0.0:
                self.player_mental_fatigue[key] = 0.0
                continue

            mental = max(
                0.0,
                min(20.0, float(stats.get("mental", 5.0))),
            )
            vulnerability = 1.0 - mental / 10.0
            current = float(self.player_mental_fatigue.get(key, 0.0))

            if char.team == losing_side:
                current += base_damage * vulnerability
            else:
                current -= 0.055

            self.player_mental_fatigue[key] = max(
                -0.75,
                min(0.75, current),
            )

    def init_round(self):
        self._swap_sides_if_needed()
        self.round_over = False
        if self.analytics_tracker is not None:
            self.analytics_tracker.begin_round()

        area_3 = list(zip(*np.where(self.grid == 3)))
        area_4 = list(zip(*np.where(self.grid == 4)))

        spike_holder_index = random.randint(0, len(area_3) - 1) if area_3 else -1
        configured_name = None
        if self.attacker_roster and area_3:
            configured_holder = self.spike_holder_name
            configured_name = (
                str(configured_holder)
                if configured_holder is not None
                else None
            )
            roster_index = next(
                (
                    index
                    for index, roster_name in enumerate(self.attacker_roster)
                    if str(roster_name) == configured_name
                ),
                None,
            )
            if roster_index is not None and roster_index < len(area_3):
                spike_holder_index = roster_index
            elif configured_name is not None:
                # A configured holder must never silently become random just
                # because a TeamPlayerKey/string or side-swap representation
                # differs. Keep the result deterministic until the mismatch
                # is corrected by the caller.
                spike_holder_index = 0
                print(
                    "[SPIKE][WARN] configured holder not found in attacker "
                    f"roster: {configured_name!r}; using roster index 0"
                )

        self.chars = []
        for i, pos in enumerate(area_3):
            has_spike = i == spike_holder_index
            if self.attacker_roster and i < len(self.attacker_roster):
                name = self.attacker_roster[i]
            else:
                name = f"Att{i+1}"
            saved = self.match_stats.setdefault(name, {"kills": 0, "deaths": 0})
            self.chars.append(
                Character(
                    name,
                    "A",
                    pos,
                    "white",
                    "#c0392b",
                    has_spike=has_spike,
                    kills=saved["kills"],
                    deaths=saved["deaths"],
                    ultimate_points=saved.get("ultimate_points", 0),
                    mental_pressure=self._mental_pressure_for_player(name, "A"),
                )
            )

        # Enforce the configured holder by player identity after characters
        # are created. This is deliberately name-based so side swaps and
        # TeamPlayerKey/string representations cannot redirect the spike to
        # another roster slot.
        if self.chars and self.attacker_roster and configured_name is not None:
            attackers = [char for char in self.chars if char.team == "A"]
            configured_char = next(
                (
                    char for char in attackers
                    if str(char.name) == configured_name
                ),
                None,
            )
            if configured_char is not None:
                for char in attackers:
                    char.has_spike = char is configured_char
        if self.defender_roster is None:
            registered_names = get_all_character_names()
            attacker_names = set(self.attacker_roster or [])
            defender_pool = [
                name for name in registered_names if name not in attacker_names
            ]
            if len(defender_pool) < len(area_4):
                defender_pool = list(registered_names)

            if defender_pool:
                if len(defender_pool) >= len(area_4):
                    self.defender_roster = random.sample(defender_pool, len(area_4))
                else:
                    self.defender_roster = [
                        random.choice(defender_pool) for _ in area_4
                    ]
            else:
                self.defender_roster = [f"Def{i+1}" for i in range(len(area_4))]

        defender_names = self.defender_roster

        for i, pos in enumerate(area_4):
            name = defender_names[i]
            saved = self.match_stats.setdefault(name, {"kills": 0, "deaths": 0})
            self.chars.append(
                Character(
                    name,
                    "D",
                    pos,
                    "white",
                    "#27ae60",
                    kills=saved["kills"],
                    deaths=saved["deaths"],
                    ultimate_points=saved.get("ultimate_points", 0),
                    mental_pressure=self._mental_pressure_for_player(name, "D"),
                )
            )

        # IQを含むコンボ補正を先に適用し、その後でIGL倍率を計算する。
        self._apply_player_combos()
        self._apply_igl_iq_bonus()
        if self.analytics_tracker is not None:
            self.analytics_tracker.register_players(
                self.chars,
                {"A": self.attacker_team_name, "D": self.defender_team_name},
            )
        # Setup後の配置をAnalyticsに記録する。スポーン直後ではなく、
        # LIVE開始から10Tick後にbattle_logic.loopがスナップショットする。
        self._analytics_initial_defender_positions = None
        self._analytics_post_setup_ticks = 0
        self.announcement_queue = []
        self.combo_announcement_index = 0
        self.combo_announcement_ticks_left = 0
        for combo_announcement in self.active_player_combos:
            item = dict(combo_announcement)
            item.setdefault("type", "combo")
            self._enqueue_announcement(item)

        plants = list(zip(*np.where(self.grid == 2)))
        self.target_plant_pos = random.choice(plants) if plants else None

        self.spike_pos = None
        self.is_planted = False
        self.planted_pos = None
        self.round_timer = ROUND_DURATION_TICKS
        self.detonate_timer = SPIKE_DETONATION_TICKS
        self.is_defused = False
        self.active_defuser_name = None
        self.last_engagements = []
        self.last_shot = None
        self.last_shots = []
        self.battle_tick = 0
        self.smokes = []
        self.flash_projectiles = []
        self.recon_projectiles = []
        self.flash_bursts = []
        self.recon_bursts = []
        self.monitor_drones = []
        self.monitor_drone_serial = 0
        self.tunnel_bursts = []
        self.available_orbs = set(zip(*np.where(self.grid == 5)))
        self.ability_mode = None
        self.ultimate_mode = None

        if hasattr(self.defender_controller, "reset_round"):
            self.defender_controller.reset_round()

        if hasattr(self.attacker_controller, "reset_round"):
            self.attacker_controller.reset_round()

        # 毎ラウンド、通常戦闘より先にDefender Setup Phaseを開始する。
        self.defender_setup_phase.start()
        if self.analytics_tracker is not None:
            self.analytics_tracker.begin_round_tactics(
                self.width,
                plants,
                self.chars,
            )
            self.analytics_tracker._target_plant_pos = self.target_plant_pos

    @property
    def in_defender_setup_phase(self):
        return bool(self.defender_setup_phase.active)

    @property
    def defender_setup_ticks_remaining(self):
        return int(self.defender_setup_phase.ticks_remaining)

    def run(self):
        if self.headless:
            self.run_headless_loop()
        else:
            self.draw()
            self.root.after(TICK_TIME, self.loop)
            self.root.mainloop()


if __name__ == "__main__":

    def start_match(
        attacker_roster,
        defender_roster,
        spike_holder_name=None,
        defender_spike_holder_name=None,
        initial_attacker_team_ai_key="default",
        initial_defender_team_ai_key="default",
        attacker_igl_name=None,
        defender_igl_name=None,
        attacker_team_name=None,
        defender_team_name=None,
    ):
        initial_attacker_team_ai = _build_team_ai(initial_attacker_team_ai_key)
        initial_defender_team_ai = _build_team_ai(initial_defender_team_ai_key)

        game = VisualFPSBattle(
            NEW_MAZE_STR,
            initial_attacker_team_ai,
            initial_defender_team_ai,
            headless=False,
            attacker_roster=attacker_roster,
            defender_roster=defender_roster,
            spike_holder_name=spike_holder_name,
            defender_spike_holder_name=defender_spike_holder_name,
            attacker_igl_name=attacker_igl_name,
            defender_igl_name=defender_igl_name,
            attacker_team_name=attacker_team_name,
            defender_team_name=defender_team_name,
        )
        game.run()

    roster_screen = RosterSelectScreen(on_confirm=start_match)
    roster_screen.run()
