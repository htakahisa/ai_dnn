"""Game entry point and VisualFPSBattle composition.

Keep this file as the executable entry point. The implementation is split into
five focused modules beside it.
"""

import random
import tkinter as tk
import numpy as np

from controllers import DefaultAttackerController, DefaultDefenderController, UserInputController
from learning_defender import LearningDefenderController, LearningDefenderAllAIController
from learning_attacker import LearningAttackerController
from map_data import NEW_MAZE_STR
from roster_select import RosterSelectScreen

from game_core import (
    Character, get_all_character_names, SIDE_PANEL_WIDTH,
    COMBO_BANNER_HEIGHT, TICK_TIME, SMOKE_DURATION_TICKS,
    ROUND_DURATION_TICKS, SPIKE_DETONATION_TICKS,
)
from combo_awakening import ComboAwakeningMixin
from abilities_los import AbilityLosMixin
from battle_logic import BattleLogicMixin
from rendering_ui import RenderingUIMixin


# 💡追加: 内部キー -> コントローラー生成関数
def _build_attacker_controller(key):
    if key == "default":
        return DefaultAttackerController()
    if key == "user":
        return UserInputController()
    # "learning" またはそれ以外は学習済みAIをデフォルトとする
    return LearningAttackerController(model_path="dqn_attacker_combined_best.pt")


def _build_defender_controller(key):
    if key == "default":
        return DefaultDefenderController()
    if key == "user":
        return UserInputController()
    # "learning_all" またはそれ以外は統合学習済みAIをデフォルトとする
    return LearningDefenderAllAIController(model_path="dqn_defender_combined_best.pt")


class VisualFPSBattle(
    ComboAwakeningMixin,
    AbilityLosMixin,
    BattleLogicMixin,
    RenderingUIMixin,
):
    def __init__(
        self,
        maze_str,
        attacker_controller,
        defender_controller,
        headless=False,
        attacker_roster=None,
        defender_roster=None,
        spike_holder_name=None,
        defender_spike_holder_name=None,
        attacker_igl_name=None,
        defender_igl_name=None,
    ):
        self.maze_str = maze_str
        self.headless = headless
        self.attacker_roster = list(attacker_roster) if attacker_roster else None
        self.defender_roster = list(defender_roster) if defender_roster else None
        self.spike_holder_name = spike_holder_name
        self.defender_spike_holder_name = defender_spike_holder_name
        self.attacker_igl_name = attacker_igl_name
        self.defender_igl_name = defender_igl_name

        lines = [line.strip() for line in maze_str.strip("\n").split("\n") if line.strip()]
        self.height, self.width = len(lines), len(lines[0])
        self.grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
        self.cell_size = 24

        self.attacker_controller = attacker_controller
        self.defender_controller = defender_controller
        self.active_user_team = None

        self.attacker_wins = 0
        self.defender_wins = 0
        self.current_round = 1
        self.sides_swapped = False
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
                height=self.map_pixel_height + self.ability_area_height + COMBO_BANNER_HEIGHT,
                bg="#10141c",
                highlightthickness=0,
            )
            self.canvas.pack(fill="both", expand=True)
            self.root.minsize(self.map_pixel_width + SIDE_PANEL_WIDTH * 2, self.map_pixel_height + self.ability_area_height + COMBO_BANNER_HEIGHT + 30)
            self.canvas.bind("<Button-1>", self.on_canvas_click)
            self.label = tk.Label(self.root, text="Round 1 Start", font=("Arial", 10))
            self.label.pack()

        self.match_over = False
        self.init_round()



    def _apply_igl_iq_bonus(self):
        """コンボ後IQを使ってIGL補正を行い、最後に影響度ペナルティを適用する。

        適用順:
            1. 素のIQ
            2. プレイヤーコンボによるIQ加算
            3. コンボ後のIGL IQを参照した味方IQ倍率
            4. 影響度超過によるチーム全員のIQ低下

        IGL本人にはIGL倍率を掛けない。
        ただし、IGL本人が受けたコンボIQ上昇はそのまま保持される。
        """
        for team, igl_name in (("A", self.attacker_igl_name), ("D", self.defender_igl_name)):
            members = [char for char in self.chars if char.team == team]
            igl = next((char for char in members if char.name == igl_name), None)

            # _apply_player_combos() 後の char.iq が、IGL計算前のIQ。
            pre_igl_iq = {}
            for char in members:
                char.is_igl = bool(igl and char.name == igl.name)
                current_iq = max(
                    0.0,
                    float(getattr(char, "iq", getattr(char, "base_iq", 100.0))),
                )
                pre_igl_iq[id(char)] = current_iq
                char.effective_iq = current_iq
                char.iq = current_iq

            if igl is not None:
                # IGL自身がコンボでIQ上昇していれば、その上昇後の値を参照する。
                igl_iq_after_combo = pre_igl_iq[id(igl)]
                multiplier = max(0.0, igl_iq_after_combo / 100.0)

                for char in members:
                    if char is igl:
                        continue

                    # 味方自身のコンボIQも保持した状態で倍率を掛ける。
                    adjusted_iq = max(0.0, pre_igl_iq[id(char)] * multiplier)
                    char.effective_iq = adjusted_iq
                    char.iq = adjusted_iq

            total_influence = sum(
                max(0.0, float(getattr(char, "influence", 0.0)))
                for char in members
            )
            influence_iq_penalty = max(0.0, (total_influence - 300.0) / 10.0)

            if influence_iq_penalty > 0.0:
                for char in members:
                    adjusted_iq = max(
                        0.0,
                        float(char.effective_iq) - influence_iq_penalty,
                    )
                    char.effective_iq = adjusted_iq
                    char.iq = adjusted_iq

    def _swap_sides_if_needed(self):
        """13ラウンド目の開始直前に攻守を一度だけ交代する。

        編成、IGL、次に攻撃側になった時のスパイク所持者、
        および画面上のサイド別スコアを入れ替える。

        コントローラーはサイドに紐づけたままにするため、
        交代後のAttackerにはAttacker用AIモデル、
        DefenderにはDefender用AIモデルが自動的に使われる。
        """
        if self.sides_swapped or self.current_round < 13:
            return

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

        # スコアはチームに追従させる。
        self.attacker_wins, self.defender_wins = (
            self.defender_wins,
            self.attacker_wins,
        )

        self.sides_swapped = True

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
            has_spike = (i == spike_holder_index)
            if self.attacker_roster and i < len(self.attacker_roster):
                name = self.attacker_roster[i]
            else:
                name = f"Att{i+1}"
            saved = self.match_stats.setdefault(name, {"kills": 0, "deaths": 0})
            self.chars.append(Character(name, "A", pos, "white", "#c0392b", has_spike=has_spike,
                                        kills=saved["kills"], deaths=saved["deaths"]))
        if self.defender_roster is None:
            registered_names = get_all_character_names()
            attacker_names = set(self.attacker_roster or [])
            defender_pool = [name for name in registered_names if name not in attacker_names]
            if len(defender_pool) < len(area_4):
                defender_pool = list(registered_names)

            if defender_pool:
                if len(defender_pool) >= len(area_4):
                    self.defender_roster = random.sample(defender_pool, len(area_4))
                else:
                    self.defender_roster = [random.choice(defender_pool) for _ in area_4]
            else:
                self.defender_roster = [f"Def{i+1}" for i in range(len(area_4))]

        defender_names = self.defender_roster

        for i, pos in enumerate(area_4):
            name = defender_names[i]
            saved = self.match_stats.setdefault(name, {"kills": 0, "deaths": 0})
            self.chars.append(Character(name, "D", pos, "white", "#27ae60",
                                        kills=saved["kills"], deaths=saved["deaths"]))

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
        attacker_ctrl_key="learning",
        defender_ctrl_key="learning_all",
        attacker_igl_name=None,
        defender_igl_name=None,
    ):
        """編成画面で決定された両チームを使って試合を開始する。"""
        att_ctrl = _build_attacker_controller(attacker_ctrl_key)
        def_ctrl = _build_defender_controller(defender_ctrl_key)

        game = VisualFPSBattle(
            NEW_MAZE_STR,
            att_ctrl,
            def_ctrl,
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