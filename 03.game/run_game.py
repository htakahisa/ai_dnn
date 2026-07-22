"""Game entry point and VisualFPSBattle composition.

Keep this file as the executable entry point. The implementation is split into
five focused modules beside it.
"""

import random
import tkinter as tk
import numpy as np

from controllers import DefaultAttackerController, DefaultDefenderController, UserInputController
from learning_defender import LearningDefenderController, LearningDefenderAllAIController
from map_data import NEW_MAZE_STR
from roster_select import RosterSelectScreen

from game_core import (
    Character, get_all_character_names, SIDE_PANEL_WIDTH,
    COMBO_BANNER_HEIGHT, TICK_TIME, SMOKE_DURATION_SECONDS,
)
from combo_awakening import ComboAwakeningMixin
from abilities_los import AbilityLosMixin
from battle_logic import BattleLogicMixin
from rendering_ui import RenderingUIMixin


class VisualFPSBattle(
    ComboAwakeningMixin,
    AbilityLosMixin,
    BattleLogicMixin,
    RenderingUIMixin,
):
    def __init__(self, maze_str, attacker_controller, defender_controller, headless=False, attacker_roster=None, spike_holder_name=None):
        self.maze_str = maze_str
        self.headless = headless # 【新機能】画面を描画しない設定
        # 💡編成画面で選択されたアタッカー5人の名前リスト（Noneなら Att1, Att2... のデフォルト名になる）
        self.attacker_roster = attacker_roster
        self.spike_holder_name = spike_holder_name
        self.defender_roster = None  # 試合開始時に一度だけ決定し、全ラウンドで固定
        
        lines = [line.strip() for line in maze_str.strip("\n").split("\n") if line.strip()]
        self.height, self.width = len(lines), len(lines[0])
        self.grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
        self.cell_size = 24
        
        self.attacker_controller = attacker_controller
        self.defender_controller = defender_controller
        
        self.attacker_wins = 0
        self.defender_wins = 0
        self.current_round = 1
        self.battle_tick = 0
        self.match_stats = {}  # 名前 -> {kills, deaths}。ラウンドをまたいで保持
        self.map_offset_x = SIDE_PANEL_WIDTH
        self.map_pixel_width = self.width * self.cell_size
        self.map_pixel_height = self.height * self.cell_size
        self.ability_area_height = 110
        
        # 画面非表示（headless）モードの時はTkinterを立ち上げない
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
            self.canvas.bind("<Button-1>", self.on_canvas_click)   # 💡追加
            self.label = tk.Label(self.root, text="Round 1 Start", font=("Arial", 10))
            self.label.pack()
        
        self.match_over = False
        self.init_round()


    def init_round(self):
        self.round_over = False
        
        area_3 = list(zip(*np.where(self.grid == 3)))
        area_4 = list(zip(*np.where(self.grid == 4)))
        
        spike_holder_index = random.randint(0, len(area_3) - 1) if area_3 else -1
        if self.attacker_roster and self.spike_holder_name in self.attacker_roster:
            spike_holder_index = self.attacker_roster.index(self.spike_holder_name)
        
        self.chars = []
        for i, pos in enumerate(area_3):
            has_spike = (i == spike_holder_index)
            # 💡編成画面でロスターが選ばれていればその名前を使い、なければデフォルト名(Att1, Att2...)
            if self.attacker_roster and i < len(self.attacker_roster):
                name = self.attacker_roster[i]
            else:
                name = f"Att{i+1}"
            saved = self.match_stats.setdefault(name, {"kills": 0, "deaths": 0})
            self.chars.append(Character(name, "A", pos, "white", "#c0392b", has_spike=has_spike,
                                        kills=saved["kills"], deaths=saved["deaths"]))
        # ディフェンダー編成は試合開始時に一度だけランダム決定し、以後のラウンドで固定する。
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

        # ロール補正を含む基礎能力の初期化後、チーム編成によるコンボ補正を適用する。
        # Character はラウンドごとに作り直されるため、補正が累積することはない。
        self._apply_player_combos()
        # コンボと覚醒イベントを同じ告知キューで順番に表示する。
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
        self.round_timer = 90          
        self.detonate_timer = 45       
        self.is_defused = False        
        self.active_defuser_name = None
        self.last_engagements = []
        self.last_shot = None
        self.last_shots = []
        self.battle_tick = 0
        self.smokes = []  # {cells:set[(r,c)], remaining_seconds:float, owner:str}
        self.flash_projectiles = []  # {owner, team, path, progress, ticks_alive}
        self.recon_projectiles = []  # {owner, team, path, progress}
        self.flash_bursts = []  # {pos, remaining_seconds}
        self.recon_bursts = []  # {cells, remaining_seconds}
        self.ability_mode = None
        
        # ディフェンダーコントローラの内部状態(サイト割り当て等)をリセットする
        if hasattr(self.defender_controller, "reset_round"):
            self.defender_controller.reset_round()
            
        # 💡追加：アタッカー側も同様にリセット(UserInputController用)
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
    # =========================================================================
    # ⚙️ モード切り替えスイッチ
    # =========================================================================
    # TRUE : 画面なしでバックグラウンド超高速計算（学習やテスト用）
    # FALSE: 画面ありでいつものプレイ（人間が観戦する用）
    LEARNING_MODE = True # AIモデルを使うのでTrueに

    def start_match(roster, spike_holder_name=None):
        """💡編成画面で「決定」が押されたときに呼ばれる。rosterは選ばれたキャラ名5人分のリスト"""
        #att_ctrl = DefaultAttackerController()
        att_ctrl = UserInputController()

        if LEARNING_MODE:
            # 新しい統合モデルでテストしたい場合
            def_ctrl = LearningDefenderAllAIController(model_path="dqn_defender_combined_best.pt")

            # 迷路探索のみモデル
            #def_ctrl = LearningDefenderController(model_path="dqn_gridworld_fixedmap.pt")

            # 動きを確認したいので headless=False で可視化する
            game = VisualFPSBattle(NEW_MAZE_STR, att_ctrl, def_ctrl, headless=False, attacker_roster=roster, spike_holder_name=spike_holder_name)
        else:
            def_ctrl = DefaultDefenderController()
            game = VisualFPSBattle(NEW_MAZE_STR, att_ctrl, def_ctrl, headless=False, attacker_roster=roster, spike_holder_name=spike_holder_name)

        game.run()

    # 💡まずキャラクター編成画面を開き、「決定」が押されたら start_match が呼ばれて試合開始
    roster_screen = RosterSelectScreen(on_confirm=start_match)
    roster_screen.run()