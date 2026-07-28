import random
import tkinter as tk
import numpy as np

# 分離した操作用クラスをインポート
from controllers import DefaultAttackerController, DefaultDefenderController, UserInputController
from learning_defender import LearningDefenderController, LearningDefenderAllAIController
from learning_attacker import LearningAttackerController
from learning_attacker_multi import LearningAttackerMultiController
from learning_attacker_ability import LearningAttackerAbilityController
from attacker_v2.multi_role_attacker_controller import MultiRoleAttackerController
from learning_attacker_guard import LearningAttackerGuardController
from train_attacker_ability import DEFUSE_REQUIRED
from map_data import NEW_MAZE_STR
from game_core import Character
from abilities import AbilityMixin

WINNING_ROUNDS = 13
TICK_TIME = 100


class VisualFPSBattle(AbilityMixin):
    def __init__(self, maze_str, attacker_controller, defender_controller, headless=False):
        self.maze_str = maze_str
        self.headless = headless # 【新機能】画面を描画しない設定
        
        lines = [line.strip() for line in maze_str.strip("\n").split("\n") if line.strip()]
        self.height, self.width = len(lines), len(lines[0])
        self.grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
        self.cell_size = 18
        
        self.attacker_controller = attacker_controller
        self.defender_controller = defender_controller
        
        self.attacker_wins = 0
        self.defender_wins = 0
        self.current_round = 1
        
        # 画面非表示（headless）モードの時はTkinterを立ち上げない
        if not self.headless:
            self.root = tk.Tk()
            self.root.title("Attacker vs Defender")
            self.canvas = tk.Canvas(self.root, width=self.width*self.cell_size, height=self.height*self.cell_size)
            self.canvas.pack()
            self.canvas.bind("<Button-1>", self.on_canvas_click)   # 💡追加
            self.canvas.bind("<Button-3>", self.on_canvas_right_click)   # 💡追加: 右クリックでアビリティ発動
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
        
        
        #self.chars = []
        #if area_3:
        #    spawn_pos = random.choice(area_3)
        #    self.chars.append(Character("Att1", "A", spawn_pos, "white", "#c0392b", has_spike=True))


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
        
        # ディフェンダーコントローラの内部状態(サイト割り当て等)をリセットする
        if hasattr(self.defender_controller, "reset_round"):
            self.defender_controller.reset_round()
            
        # 💡追加：アタッカー側も同様にリセット(UserInputController用)
        if hasattr(self.attacker_controller, "reset_round"):
            self.attacker_controller.reset_round()

        # 💡追加: アビリティ関連の状態初期化と割り当て
        self.smokes = []
        self.flash_projectiles = []
        self.recon_projectiles = []
        self.flash_bursts = []
        self.recon_bursts = []
        self.assign_abilities()

    def move_character(self, char):
        r, c = char.pos
        
        is_ai_attacker = isinstance(
            self.attacker_controller,
            (LearningAttackerController, LearningAttackerMultiController, LearningAttackerAbilityController, MultiRoleAttackerController)
        )
        
        # ---------------------------------------------------------------------
        # 💡 【修正】アタッカーのPlant自動処理の条件を厳密化
        # ---------------------------------------------------------------------
        if char.team == "A" and char.has_spike and not is_ai_attacker:
            is_user_controlled = isinstance(self.attacker_controller, UserInputController)

            if is_user_controlled:
                # 💡 ユーザー操作時：2のマスならどこでも、そこで止まればplant開始
                on_plant_site = (self.grid[r, c] == 2)
            else:
                # 💡 AI操作時：従来通りtarget_plant_posに到達した時のみ
                on_plant_site = self.target_plant_pos and list(char.pos) == list(self.target_plant_pos)

            if on_plant_site:
                char.plant_timer += 1
                if char.plant_timer >= 4:
                    self.is_planted = True
                    self.planted_pos = (r, c)
                    char.has_spike = False
                    char.plant_timer = 0
                return  # プラント中は移動処理を行わずその場に留まる
            else:
                char.plant_timer = 0

        # ---------------------------------------------------------------------
        # AIにどう動くか（または解除するか）を聞く
        # ---------------------------------------------------------------------
        # 💡 プラント状態に応じた適切なターゲット座標の確定
        if char.team == "A" and char.has_spike and self.target_plant_pos is None:
            # 盤面から2（サイト）の座標を探し、現在地から一番近いものを仮のターゲットにする
            site_coords = np.argwhere(self.grid == 2)
            if len(site_coords) > 0:
                dists = np.abs(site_coords[:, 0] - char.pos[0]) + np.abs(site_coords[:, 1] - char.pos[1])
                nearest_idx = np.argmin(dists)
                self.target_plant_pos = tuple(site_coords[nearest_idx])

        if self.is_planted:
            site_r = float(self.planted_pos[0]) if self.planted_pos else 0.0
            site_c = float(self.planted_pos[1]) if self.planted_pos else 0.0
        else:
            site_r = float(self.target_plant_pos[0]) if self.target_plant_pos else 0.0
            site_c = float(self.target_plant_pos[1]) if self.target_plant_pos else 0.0

        defender_defuse_info = {
            d.name: (d.defuse_timer, DEFUSE_REQUIRED)  # 6 = DEFUSE_REQUIRED (learning_attacker_multi.py の DEFUSE_REQUIRED と一致させる)
            for d in self.chars
            if d.team == "D" and d.is_alive
        }

        game_state = {
            "grid": self.grid,
            "spike_pos": self.spike_pos,
            "is_planted": self.is_planted,
            "planted_pos": self.planted_pos,
            "target_plant_pos": self.target_plant_pos,
            "chars": self.chars,
            "spotted_info": self.get_spotted_info() if not self.is_planted else {
                'spotted': 1.0,
                'site_r': site_r,
                'site_c': site_c
            },
            "defender_defuse_info": defender_defuse_info,
            "detonate_timer": self.detonate_timer,
        }

        if char.team == "A":
            result = self.attacker_controller.decide_move(char, game_state)
            if isinstance(result, tuple) and len(result) == 2:
                next_pos, action_type = result
            else:
                next_pos, action_type = result, "MOVE"
        else:
            # ディフェンダー側：コントローラーによって戻り値の数が異なるため自動判別
            result = self.defender_controller.decide_move(char, game_state)
            if isinstance(result, tuple) and len(result) == 2:
                # LearningDefenderAllAIController などのアクションタイプ付きの戻り値
                next_pos, action_type = result
            else:
                # DefaultDefenderController などの座標のみの戻り値
                next_pos = result
                action_type = "MOVE"

        # ---------------------------------------------------------------------
        #  アクションタイプに応じたシステム処理 (修正版)
        # ---------------------------------------------------------------------
        if action_type == "PLANT":
            if char.team == "A" and char.has_spike:
                r, c = char.pos
                on_site = self.grid[r, c] == 2   # 💡サイト内であればどこでも設置可能
                if on_site:
                    self.is_planted = True
                    self.planted_pos = tuple(char.pos)
                    char.has_spike = False
            return
        elif action_type == "ABILITY":
            char.plant_timer = 0     # 💡追加
            char.defuse_timer = 0    # 💡追加
            # 💡追加: next_pos はここでは発動先セル(target_cell)として扱う
            self.use_ability(char, tuple(next_pos))
            return
        elif action_type == "DEFUSE":
            if self.is_planted and self.planted_pos and char.team == "D":
                dist = max(abs(self.planted_pos[0] - r), abs(self.planted_pos[1] - c))
                if dist <= 1:
                    char.defuse_timer += 1
                    if char.defuse_timer >= 6:
                        self.is_defused = True
                    return  # 💡 解除時はここで完全に処理を終了させ、下の移動処理に流さない
            char.defuse_timer = 0

        else:
            char.plant_timer = 0
            char.defuse_timer = 0
            moved = False
            # 💡 【修正】tuple を判定に追加し、AIからの戻り値を確実に受け取る
            if isinstance(next_pos, (list, tuple, np.ndarray)) and len(next_pos) == 2:
                if self.grid[int(next_pos[0]), int(next_pos[1])] != 1:
                    char.pos = list(next_pos)
                    moved = True

    def check_line_of_sight(self, p1, p2):
        x0, y0, x1, y1 = p1.pos[1], p1.pos[0], p2.pos[1], p2.pos[0]
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        curr_x, curr_y = x0, y0
        line_cells = []
        while True:
            if self.grid[curr_y, curr_x] == 1: return False
            line_cells.append((curr_y, curr_x))
            if curr_x == x1 and curr_y == y1: break
            e2 = 2 * err
            if e2 >= dy: err += dy; curr_x += sx
            if e2 <= dx: err += dx; curr_y += sy
        # 💡追加: スモークによる視線遮断判定
        return self._smoke_allows_line(line_cells, self._smoke_cells())
            
    def get_user_controllers(self):
        """ユーザー操作のコントローラーとそのチームのペアを返す"""
        result = []
        if isinstance(self.attacker_controller, UserInputController):
            result.append((self.attacker_controller, "A"))
        if isinstance(self.defender_controller, UserInputController):
            result.append((self.defender_controller, "D"))
        return result

    def on_canvas_click(self, event):
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        if not (0 <= r < self.height and 0 <= c < self.width):
            return

        user_controllers = self.get_user_controllers()
        if not user_controllers:
            return

        for ctrl, team in user_controllers:
            ctrl.handle_click(r, c, self.grid, self.chars, team)

        self.draw()

    def on_canvas_right_click(self, event):
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        if not (0 <= r < self.height and 0 <= c < self.width):
            return

        user_controllers = self.get_user_controllers()
        if not user_controllers:
            return

        for ctrl, team in user_controllers:
            if hasattr(ctrl, "handle_right_click"):
                ctrl.handle_right_click(r, c, self.grid, self.chars, team)

        self.draw()

    def get_spotted_info(self):
        spike_holder = next((c for c in self.chars if c.is_alive and c.team == "A" and c.has_spike), None)
        if spike_holder is None:
            return {'spotted': 0.0, 'site_r': 0.0, 'site_c': 0.0}

        for d in self.chars:
            if d.is_alive and d.team == "D" and self.check_line_of_sight(d, spike_holder):
                return {'spotted': 1.0, 'site_r': float(spike_holder.pos[0]), 'site_c': float(spike_holder.pos[1])}

        return {'spotted': 0.0, 'site_r': 0.0, 'site_c': 0.0}


    def check_match_winner(self):
        if self.attacker_wins >= WINNING_ROUNDS:
            if not self.headless:
                self.label.config(text=f"🏆 MATCH OVER: Attacker WINS! ({self.attacker_wins} - {self.defender_wins})", fg="#c0392b", font=("Arial", 12, "bold"))
            print(f"MATCH OVER: 💥 Attacker WINS! ({self.attacker_wins} - {self.defender_wins})")
            self.match_over = True
        elif self.defender_wins >= WINNING_ROUNDS:
            if not self.headless:
                self.label.config(text=f"🏆 MATCH OVER: Defender WINS! ({self.defender_wins} - {self.attacker_wins})", fg="#27ae60", font=("Arial", 12, "bold"))
            print(f"MATCH OVER: 🛡️ Defender WINS! ({self.defender_wins} - {self.attacker_wins})")
            self.match_over = True
        else:
            self.current_round += 1
            if not self.headless:
                # プレイ用の時は2秒ディレイをかける
                self.root.after(2000, self.init_next_round_delayed)
            else:
                # 学習用の時はディレイなしで即時次ラウンドへ
                self.init_round()

    def init_next_round_delayed(self):
        self.init_round()
        self.loop()

    def process_battle(self):
        # 💡追加: アビリティ効果(投射物飛行・着弾・持続時間)の進行
        self._advance_ability_effects()

        # ---------------------------------------------------------------------
        # 💡 【追加】落ちているスパイクを生存しているアタッカーが踏んだら拾い上げる
        # ---------------------------------------------------------------------
        if self.spike_pos is not None:
            for c in self.chars:
                if c.is_alive and c.team == "A" and tuple(c.pos) == self.spike_pos:
                    c.has_spike = True
                    self.spike_pos = None  # マップ上からドロップ状態を解除
                    break  # 1人が拾えば十分なのでループを抜ける

        # ---------------------------------------------------------------------
        # 以下は既存の処理（死亡したスパイク持ちのドロップ処理など）
        # ---------------------------------------------------------------------
        for c in self.chars:
            if not c.is_alive and c.has_spike:
                self.spike_pos = tuple(c.pos)
                c.has_spike = False
        
        self.last_engagements = []
        alive = [c for c in self.chars if c.is_alive]
        engagements = [(alive[i], alive[j]) for i in range(len(alive)) for j in range(i + 1, len(alive)) 
                       if alive[i].team != alive[j].team and self.check_line_of_sight(alive[i], alive[j])]
        
        random.shuffle(engagements)
        # process_battle 内、交戦解決ループの target 決定部分を置き換え
        for c1, c2 in engagements:
            if not c1.is_alive or not c2.is_alive: continue
            self.last_engagements.append((c1, c2))

            c1_blind = c1.blind_remaining > 0
            c2_blind = c2.blind_remaining > 0
            c1_busy = (c1.plant_timer > 0) or (c1.defuse_timer > 0)
            c2_busy = (c2.plant_timer > 0) or (c2.defuse_timer > 0)

            if c1_blind and not c2_blind:
                target = c1
            elif c2_blind and not c1_blind:
                target = c2
            elif c1_busy and not c2_busy:
                target = c1
            elif c2_busy and not c1_busy:
                target = c2
            else:
                target = c2 if random.random() < 0.5 else c1

            target.is_alive = False
            target.just_died = True
            if target.has_spike:
                self.spike_pos = tuple(target.pos)
                target.has_spike = False

        alive_A = any(c.is_alive for c in self.chars if c.team == "A")
        alive_D = any(c.is_alive for c in self.chars if c.team == "D")

        score_text = f" [Score: Att {self.attacker_wins} - {self.defender_wins} Def]"
        
        if self.is_defused:
            self.defender_wins += 1
            if not self.headless: self.label.config(text=f"⚙️ Spike Defused! Defender WIN Round {self.current_round}! {score_text}", fg="green")
            self.round_over = True
            self.check_match_winner()
            
        elif self.is_planted:
            self.detonate_timer -= 1
            if self.detonate_timer <= 0:
                self.attacker_wins += 1
                if not self.headless: self.label.config(text=f"💥 Spike Detonated! Attacker WIN Round {self.current_round}! {score_text}", fg="red")
                self.round_over = True
                self.check_match_winner()
            elif not alive_D:
                self.attacker_wins += 1
                if not self.headless: self.label.config(text=f"🏆 Defender Annihilated! Attacker WIN Round {self.current_round}! {score_text}", fg="#c0392b")
                self.round_over = True
                self.check_match_winner()
            elif not alive_A:
                if not self.headless:
                    max_defuse = max([c.defuse_timer for c in self.chars if c.team == "D" and c.is_alive] + [0])
                    defuse_str = f" (Defusing: {max_defuse}/6)" if max_defuse > 0 else ""
                    self.label.config(text=f"💀 Attacker Eliminated! Defuse the Spike! {self.detonate_timer}s{defuse_str} | R{self.current_round}{score_text}", fg="#27ae60")
            else:
                if not self.headless:
                    max_defuse = max([c.defuse_timer for c in self.chars if c.team == "D" and c.is_alive] + [0])
                    defuse_str = f" (Defusing: {max_defuse}/6)" if max_defuse > 0 else ""
                    self.label.config(text=f"🔥 Spike Planted! Detonation in {self.detonate_timer}s{defuse_str} | R{self.current_round}{score_text}", fg="red")
                
        else:
            self.round_timer -= 1
            if self.round_timer <= 0:
                self.defender_wins += 1
                if not self.headless: self.label.config(text=f"⏰ Time Expired! Defender WIN Round {self.current_round}! {score_text}", fg="#27ae60")
                self.round_over = True
                self.check_match_winner()
            elif not alive_A:
                self.defender_wins += 1
                if not self.headless: self.label.config(text=f"🏆 Attacker Annihilated! Defender WIN Round {self.current_round}! {score_text}", fg="#27ae60")
                self.round_over = True
                self.check_match_winner()
            elif not alive_D:
                self.attacker_wins += 1
                if not self.headless: self.label.config(text=f"🏆 Defender Annihilated! Attacker WIN Round {self.current_round}! {score_text}", fg="#c0392b")
                self.round_over = True
                self.check_match_winner()
            else:
                if not self.headless:
                    site_side = "Left Side" if self.target_plant_pos and self.target_plant_pos[1] < self.width // 2 else "Right Side"
                    self.label.config(text=f"⚔️ Round {self.current_round} (Attacking {site_side}) | Ends in {self.round_timer}s | {score_text}", fg="black")

    def draw(self):
        if self.headless: return  # 描画スキップ
        self.canvas.delete("all")
        color_map = {"0":"white", "1":"#34495e", "2":"#fff9c4", "3":"#ffcccc", "4":"#ccffcc"}
        
        for r in range(self.height):
            for c in range(self.width):
                color = color_map.get(str(self.grid[r, c]), "white")
                self.canvas.create_rectangle(c*self.cell_size, r*self.cell_size, (c+1)*self.cell_size, (r+1)*self.cell_size, fill=color, outline="#eee")
        
        # -----------------------------------------------------------------
        # 💡 [追加] アビリティの描画処理
        # -----------------------------------------------------------------
        # 1. スモークの描画（半透明風のグレーで範囲を描画）
        for smoke in self.smokes:
            for r, c in smoke["cells"]:
                self.canvas.create_rectangle(
                    c * self.cell_size, r * self.cell_size,
                    (c + 1) * self.cell_size, (r + 1) * self.cell_size,
                    fill="#7f8c8d", outline="#95a5a6", stipple="gray50"
                )

        # 2. リコン着弾範囲の描画（水色枠）
        for burst in self.recon_bursts:
            for r, c in burst["cells"]:
                self.canvas.create_rectangle(
                    c * self.cell_size, r * self.cell_size,
                    (c + 1) * self.cell_size, (r + 1) * self.cell_size,
                    fill="#3498db", outline="#2980b9", stipple="gray25"
                )

        # 3. フラッシュ爆発（黄色の大円）
        for burst in self.flash_bursts:
            r, c = burst["pos"]
            cx, cy = (c + 0.5) * self.cell_size, (r + 0.5) * self.cell_size
            rad = self.cell_size * 0.8
            self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill="#f1c40f", outline="#f39c12")

        # 4. フラッシュ弾（飛翔中の小さな黄色円）
        for p in self.flash_projectiles:
            if p["path"] and p["progress"] < len(p["path"]):
                r, c = p["path"][p["progress"]]
                cx, cy = (c + 0.5) * self.cell_size, (r + 0.5) * self.cell_size
                rad = self.cell_size * 0.3
                self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill="#f1c40f", outline="black")

        # 5. リコン弾（飛翔中の小さな青色円）
        for p in self.recon_projectiles:
            if p["path"] and p["progress"] < len(p["path"]):
                r, c = p["path"][p["progress"]]
                cx, cy = (c + 0.5) * self.cell_size, (r + 0.5) * self.cell_size
                rad = self.cell_size * 0.3
                self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill="#2980b9", outline="black")
        # -----------------------------------------------------------------


        if not self.is_planted and self.target_plant_pos:
            tr, tc = self.target_plant_pos
            self.canvas.create_rectangle(tc*self.cell_size, tr*self.cell_size, (tc+1)*self.cell_size, (tr+1)*self.cell_size, fill="#f39c12", outline="#d35400")

        if self.is_planted and self.planted_pos:
            pr, pc = self.planted_pos
            self.canvas.create_oval(pc*self.cell_size+2, pr*self.cell_size+2, (pc+1)*self.cell_size-2, (pr+1)*self.cell_size-2, fill="red", outline="")
        
        if self.spike_pos:
            sr, sc = self.spike_pos
            self.canvas.create_oval(sc*self.cell_size+2, sr*self.cell_size+2, (sc+1)*self.cell_size-2, (sr+1)*self.cell_size-2, fill="black", outline="")

        for c1, c2 in self.last_engagements:
            self.canvas.create_line((c1.pos[1]+0.5)*self.cell_size, (c1.pos[0]+0.5)*self.cell_size, 
                                    (c2.pos[1]+0.5)*self.cell_size, (c2.pos[0]+0.5)*self.cell_size, fill="red", width=1)
        
        # ユーザー操作コントローラーの現在の選択中キャラ名を集める
        selected_names = {ctrl.selected_char for ctrl, _ in self.get_user_controllers() if ctrl.selected_char is not None}

        
        for c in self.chars:
            if not c.is_alive and not c.just_died: continue
            row, col = c.pos
            cx, cy = (col+0.5)*self.cell_size, (row+0.5)*self.cell_size
            if c.just_died:
                self.canvas.create_text(cx, cy, text="X", fill="orange", font=("Arial", 12, "bold"))
                c.just_died = False
            else:
                bg = "#2980b9" if (getattr(c, 'defuse_timer', 0) > 0 and self.is_planted) else c.bg_color
               # 選択中は黄色い枠線をつける
                outline_color = "yellow" if c.name in selected_names else ""
                outline_width = 3 if c.name in selected_names else 1
                self.canvas.create_oval(col*self.cell_size, row*self.cell_size, (col+1)*self.cell_size, (row+1)*self.cell_size, fill=bg, outline=outline_color, width=outline_width)
                
                if c.has_spike:
                    self.canvas.create_oval(col*self.cell_size+1, row*self.cell_size+1, (col+1)*self.cell_size-1, (row+1)*self.cell_size-1, fill="black", outline="")
                    self.canvas.create_text(cx, cy, text=c.name, fill="yellow", font=("Arial", 6, "bold"))
                else:
                    self.canvas.create_text(cx, cy, text=c.name, fill=c.text_color, font=("Arial", 6, "bold"))

    def loop(self):
        if not self.round_over and not self.match_over:
            for c in self.chars:
                if c.is_alive: self.move_character(c)
            self.process_battle()
            self.draw()
            self.root.after(TICK_TIME, self.loop)

    def run_headless_loop(self):
        """【AI学習用】画面を描画せず、限界速度でシミュレーションを回す"""
        print("💡 Headless Mode: シミュレーションをバックグラウンドで高速実行中...")
        while not self.match_over:
            if not self.round_over:
                for c in self.chars:
                    if c.is_alive: self.move_character(c)
                self.process_battle()
            # round_over 時の初期化は check_match_winner 内で自動処理されます

    def run(self):
        if self.headless:
            self.run_headless_loop()
        else:
            self.draw()
            self.root.after(TICK_TIME, self.loop)
            self.root.mainloop()

if __name__ == "__main__":

    
    #att_ctrl = DefaultAttackerController()
    #att_ctrl = LearningAttackerMultiController(model_path="dqn_attacker_multi_best_by_eval.pt", greedy=True)
    #att_ctrl = UserInputController()
    #att_ctrl = LearningAttackerAbilityController(model_path="data_temp/attacker_ability_data/dqn_attacker_ability_ep6000.pt", greedy=False)
    att_ctrl = MultiRoleAttackerController(
        carry_model_path="attacker_v2/data/attacker_carry_data/dqn_attacker_carry_best_by_eval.pt",
        escort_model_path="attacker_v2/data/attacker_escort_data/dqn_attacker_escort_best_by_eval.pt",
        retrieve_model_path="attacker_v2/data/attacker_retrieve_data/dqn_attacker_retrieve_best_by_eval.pt",
        guard_model_path="attacker_v2/data/attacker_guard_data/dqn_attacker_guard_best_by_eval.pt",
        greedy=False,
    )

    # 新しい統合モデルでテストしたい場合
    def_ctrl = LearningDefenderAllAIController(model_path="dqn_defender_combined_best.pt")
    #def_ctrl = UserInputController()
    
    # 動きを確認したいので headless=False で可視化する
    game = VisualFPSBattle(NEW_MAZE_STR, att_ctrl, def_ctrl, headless=False)

        
    game.run()