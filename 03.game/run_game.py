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
from map_data import NEW_MAZE_STR
from roster_select import RosterSelectScreen
from team_ai import DualRoleTeamAI

from game_core import (
    Character,
    get_all_character_names,
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
from rendering_ui import RenderingUIMixin

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

    if normalized == "toru_ai_v3":
        return DualRoleTeamAI(
            name="Toru AI v3",
            attacker_factory=lambda: MultiRoleAttackerController(),
            defender_factory=lambda: MultiRoleDefenderController(),
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
        return DualRoleTeamAI(
            name="ユーザー操作",
            attacker_factory=lambda: UserInputController(),
            defender_factory=lambda: UserInputController(),
        )

    raise ValueError(f"不明なTeam AIです: {key}")


def _build_attacker_controller(key):
    normalized = str(key or "default").strip().lower()

    if normalized in {"default", "logic"}:
        return DefaultAttackerController()

    if normalized == "user":
        return UserInputController()

    if normalized in {
        "fnatic_v1",
        "fnatic",
    }:
        return PolicyAttackerController(
            model_path=FNATIC_V1_ATTACKER_MODEL_PATH,
            device="auto",
        )

    if normalized in {
        "fnatic_v2",
        "fnatic_2",
        "fnatic2",
        "fnatic v2",
    }:
        return PolicyPPOAttackerController(
            model_path=FNATIC_V2_ATTACKER_MODEL_PATH,
            device="auto",
        )

    if normalized in {
        "learning",
        "learning_multi",
        "learning_v1",
        "ai_v1",
        "v1",
        "aiv1",
    }:
        return LearningAttackerController(
            model_path=ATTACKER_MODEL_PATH,
            greedy=True,
        )

    if normalized in {
        "toru_attacker_v3",
    }:
        return MultiRoleAttackerController()

    if normalized in {
        "learning_v4",
        "ai_v4",
        "v4",
        "aiv4",
    }:
        return LearningAttackerAIv4Controller(
            model_path=ATTACKER_AI_V4_MODEL_PATH,
            greedy=True,
            enable_abilities=False,
            verbose=False,
        )

    raise ValueError(f"不明なAttackerコントローラーです: {key}")


def _build_defender_controller(key):
    if key == "default":
        return DefaultDefenderController()
    if key == "user":
        return UserInputController()

    if key == "toru_defender_v3":
        return MultiRoleDefenderController()

    # "learning_all" またはそれ以外は統合学習済みAIをデフォルトとする
    return LearningDefenderAllAIController(model_path="dqn_defender_combined_best.pt")


class VisualFPSBattle(
    ComboAwakeningMixin,
    AbilityLosMixin,
    BattleLogicMixin,
    RenderingUIMixin,
):

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
        disable_side_swap=False,
    ):
        self.maze_str = maze_str
        self.headless = headless
        self.disable_side_swap = disable_side_swap
        self.attacker_roster = list(attacker_roster) if attacker_roster else None
        self.defender_roster = list(defender_roster) if defender_roster else None
        self.spike_holder_name = spike_holder_name
        self.defender_spike_holder_name = defender_spike_holder_name
        self.attacker_igl_name = attacker_igl_name
        self.defender_igl_name = defender_igl_name

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

        if hasattr(self.attacker_controller, "set_game"):
            self.attacker_controller.set_game(self)

        if hasattr(self.defender_controller, "set_game"):
            self.defender_controller.set_game(self)

        self._refresh_active_controllers()

        self.init_round()

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

        self._refresh_active_controllers()

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

    def init_round(self):
        self._swap_sides_if_needed()
        self.round_over = False

        area_3 = list(zip(*np.where(self.grid == 3)))
        area_4 = list(zip(*np.where(self.grid == 4)))

        spike_holder_index = random.randint(0, len(area_3) - 1) if area_3 else -1
        if self.attacker_roster and self.spike_holder_name in self.attacker_roster:
            spike_holder_index = self.attacker_roster.index(self.spike_holder_name)

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
                )
            )
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
                )
            )

        # IQを含むコンボ補正を先に適用し、その後でIGL倍率を計算する。
        self._apply_player_combos()
        self._apply_igl_iq_bonus()
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
        self.ability_mode = None

        if hasattr(self.defender_controller, "reset_round"):
            self.defender_controller.reset_round()

        if hasattr(self.attacker_controller, "reset_round"):
            self.attacker_controller.reset_round()

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
        )
        game.run()

    roster_screen = RosterSelectScreen(on_confirm=start_match)
    roster_screen.run()
