"""Game entry point and VisualFPSBattle composition."""

import random
import tkinter as tk
import numpy as np

from controllers import DefaultAttackerController, DefaultDefenderController, UserInputController
from learning_defender import LearningDefenderController, LearningDefenderAllAIController
from learning_attacker import LearningAttackerController
from learning_attacker_multi import LearningAttackerMultiController
from learning_attacker_ability import LearningAttackerAbilityController
from attacker_v2.multi_role_attacker_controller import MultiRoleAttackerController
from learning_attacker_guard import LearningAttackerGuardController
from map_data import NEW_MAZE_STR
from game_core import Character
from abilities import AbilityMixin
from battle_logic import BattleLogicMixin
from rendering_ui import RenderingUIMixin


class VisualFPSBattle(AbilityMixin, BattleLogicMixin, RenderingUIMixin):
    WINNING_ROUNDS = 13
    TICK_TIME = 100

    def __init__(self, maze_str, attacker_controller, defender_controller, headless=False):
        self.maze_str = maze_str
        self.headless = headless  # 【新機能】画面を描画しない設定

        lines = [line.strip() for line in maze_str.strip("\n").split("\n") if line.strip()]
        self.height, self.width = len(lines), len(lines[0])
        self.grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
        self.cell_size = 18

        self.attacker_controller = attacker_controller
        self.defender_controller = defender_controller

        self.attacker_wins = 0
        self.defender_wins = 0
        self.current_round = 1

        # 💡 打ち合い（射撃）の勝敗カウンター
        self.attacker_gunfight_wins = 0
        self.defender_gunfight_wins = 0

        # 画面非表示（headless）モードの時はTkinterを立ち上げない
        if not self.headless:
            self.root = tk.Tk()
            self.root.title("Attacker vs Defender")
            self.canvas = tk.Canvas(self.root, width=self.width*self.cell_size, height=self.height*self.cell_size)
            self.canvas.pack()
            self.canvas.bind("<Button-1>", self.on_canvas_click)
            self.canvas.bind("<Button-3>", self.on_canvas_right_click)
            self.label = tk.Label(self.root, text="Round 1 Start", font=("Arial", 10))
            self.label.pack()

        self.match_over = False
        self.init_round()

    def init_round(self):
        self.round_over = False

        area_3 = list(zip(*np.where(self.grid == 3)))
        area_4 = list(zip(*np.where(self.grid == 4)))

        spike_holder_index = random.randint(0, len(area_3) - 1) if area_3 else -1
        self.chars = []
        for i, pos in enumerate(area_3):
            has_spike = (i == spike_holder_index)
            self.chars.append(Character(f"Att{i+1}", "A", pos, "white", "#c0392b", has_spike=has_spike))

        for i, pos in enumerate(area_4):
            self.chars.append(Character(f"Def{i+1}", "D", pos, "white", "#27ae60"))

        plants = list(zip(*np.where(self.grid == 2)))
        self.target_plant_pos = random.choice(plants) if plants else None

        self.spike_pos = None
        self.is_planted = False
        self.planted_pos = None
        self.round_timer = 90
        self.detonate_timer = 45
        self.is_defused = False
        self.last_engagements = []

        if hasattr(self.defender_controller, "reset_round"):
            self.defender_controller.reset_round()

        if hasattr(self.attacker_controller, "reset_round"):
            self.attacker_controller.reset_round()

        # 💡アビリティ関連の状態初期化と割り当て
        self.smokes = []
        self.flash_projectiles = []
        self.recon_projectiles = []
        self.flash_bursts = []
        self.recon_bursts = []
        self.assign_abilities()

    def run(self):
        if self.headless:
            self.run_headless_loop()
        else:
            self.draw()
            self.root.after(self.TICK_TIME, self.loop)
            self.root.mainloop()


if __name__ == "__main__":

    att_ctrl = MultiRoleAttackerController(
        carry_model_path="attacker_v2/data/attacker_carry_data/dqn_attacker_carry_best_by_eval.pt",
        escort_model_path="attacker_v2/data/attacker_escort_data/dqn_attacker_escort_best_by_eval.pt",
        retrieve_model_path="attacker_v2/data/attacker_retrieve_data/dqn_attacker_retrieve_best_by_eval.pt",
        guard_model_path="attacker_v2/data/attacker_guard_data/dqn_attacker_guard_best_by_eval.pt",
        greedy=False,
    )
    def_ctrl = LearningDefenderAllAIController(model_path="dqn_defender_combined_best.pt")

    total_att_gunfight_wins = 0
    total_def_gunfight_wins = 0

    game = VisualFPSBattle(NEW_MAZE_STR, att_ctrl, def_ctrl, headless=False)
    game.run()
    total_att_gunfight_wins += game.attacker_gunfight_wins
    total_def_gunfight_wins += game.defender_gunfight_wins

    total_gunfights = total_att_gunfight_wins + total_def_gunfight_wins
    att_win_rate = (total_att_gunfight_wins / total_gunfights * 100) if total_gunfights > 0 else 0
    def_win_rate = (total_def_gunfight_wins / total_gunfights * 100) if total_gunfights > 0 else 0

    print("\n" + "=" * 55)
    print("【 射撃（打ち合い）対戦成績・勝率 】")
    print(f"総打ち合い回数 : {total_gunfights} 回")
    print(f"Attacker win  : {total_att_gunfight_wins} (勝率: {att_win_rate:.1f}%)")
    print(f"Defender win  : {total_def_gunfight_wins} (勝率: {def_win_rate:.1f}%)")
    print("=" * 55)