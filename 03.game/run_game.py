import random
import tkinter as tk
import numpy as np

# 分離した操作用クラスをインポート
from controllers import DefaultAttackerController, DefaultDefenderController
from learning_defender import LearningDefenderController, LearningDefenderAllAIController
from map_data import NEW_MAZE_STR

WINNING_ROUNDS = 5
TICK_TIME = 100


class Character:
    def __init__(self, name, team, pos, text_color, bg_color, has_spike=False):
        self.name = name
        self.team = team
        self.pos = list(pos)
        self.text_color = text_color
        self.bg_color = bg_color    
        self.is_alive = True
        self.just_died = False
        self.has_spike = has_spike
        self.plant_timer = 0
        self.defuse_timer = 0 

class VisualFPSBattle:
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
        
        # ディフェンダーコントローラの内部状態(サイト割り当て等)をリセットする
        if hasattr(self.defender_controller, "reset_round"):
            self.defender_controller.reset_round()

    def move_character(self, char):
        r, c = char.pos
        
        # ---------------------------------------------------------------------
        # 💡 【修正】アタッカーのPlant自動処理の条件を厳密化
        # ---------------------------------------------------------------------
        # 単に grid == 2 ではなく、アタッカーが目指している target_plant_pos に到達した時のみタイマーを進める
        if char.team == "A" and char.has_spike and self.target_plant_pos:
            if list(char.pos) == list(self.target_plant_pos):
                char.plant_timer += 1
                if char.plant_timer >= 4:
                    self.is_planted = True
                    self.planted_pos = (r, c)  
                    char.has_spike = False
                return  # プラント中は移動処理を行わずその場に留まる
            else:
                # ターゲットサイトに向かう途中なら、交戦などで蓄積したタイマーをリセットして移動を許可する
                char.plant_timer = 0

        # ---------------------------------------------------------------------
        # AIにどう動くか（または解除するか）を聞く
        # ---------------------------------------------------------------------
        # 💡 プラント状態に応じた適切なターゲット座標の確定
        if self.is_planted:
            site_r = float(self.planted_pos[0]) if self.planted_pos else 0.0
            site_c = float(self.planted_pos[1]) if self.planted_pos else 0.0
        else:
            site_r = float(self.target_plant_pos[0]) if self.target_plant_pos else 0.0
            site_c = float(self.target_plant_pos[1]) if self.target_plant_pos else 0.0

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
            }
        }

        if char.team == "A":
            # アタッカー側（位置だけ返す想定）
            next_pos = self.attacker_controller.decide_move(char, game_state)
            action_type = "MOVE"
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
        if action_type == "DEFUSE":
            if self.is_planted and self.planted_pos and char.team == "D":
                dist = max(abs(self.planted_pos[0] - r), abs(self.planted_pos[1] - c))
                if dist <= 1:
                    char.defuse_timer += 1
                    if char.defuse_timer >= 6:
                        self.is_defused = True
                    return  # 💡 解除時はここで完全に処理を終了させ、下の移動処理に流さない
            char.defuse_timer = 0

        else:
            # MOVE アクションの処理
            char.defuse_timer = 0  
            # 💡 インデックス参照エラーを防ぐため、next_pos が有効な2次元座標であることを保証
            if isinstance(next_pos, (list, np.ndarray)) and len(next_pos) == 2:
                if self.grid[next_pos[0], next_pos[1]] != 1:
                    char.pos = list(next_pos)

    def check_line_of_sight(self, p1, p2):
        x0, y0, x1, y1 = p1.pos[1], p1.pos[0], p2.pos[1], p2.pos[0]
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        curr_x, curr_y = x0, y0
        while True:
            if self.grid[curr_y, curr_x] == 1: return False
            if curr_x == x1 and curr_y == y1: return True
            e2 = 2 * err
            if e2 >= dy: err += dy; curr_x += sx
            if e2 <= dx: err += dx; curr_y += sy

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
            print(f"MATCH OVER: Attacker WINS! ({self.attacker_wins} - {self.defender_wins})")
            self.match_over = True
        elif self.defender_wins >= WINNING_ROUNDS:
            if not self.headless:
                self.label.config(text=f"🏆 MATCH OVER: Defender WINS! ({self.defender_wins} - {self.attacker_wins})", fg="#27ae60", font=("Arial", 12, "bold"))
            print(f"MATCH OVER: Defender WINS! ({self.defender_wins} - {self.attacker_wins})")
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
        for c1, c2 in engagements:
            if not c1.is_alive or not c2.is_alive: continue
            self.last_engagements.append((c1, c2))
            
            c1_busy = (c1.plant_timer > 0) or (c1.defuse_timer > 0)
            c2_busy = (c2.plant_timer > 0) or (c2.defuse_timer > 0)
            
            if c1_busy and not c2_busy:
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
        
        for c in self.chars:
            if not c.is_alive and not c.just_died: continue
            row, col = c.pos
            cx, cy = (col+0.5)*self.cell_size, (row+0.5)*self.cell_size
            if c.just_died:
                self.canvas.create_text(cx, cy, text="X", fill="orange", font=("Arial", 12, "bold"))
                c.just_died = False
            else:
                bg = "#2980b9" if (getattr(c, 'defuse_timer', 0) > 0 and self.is_planted) else c.bg_color
                self.canvas.create_oval(col*self.cell_size, row*self.cell_size, (col+1)*self.cell_size, (row+1)*self.cell_size, fill=bg)
                
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
    # =========================================================================
    # ⚙️ モード切り替えスイッチ
    # =========================================================================
    # TRUE : 画面なしでバックグラウンド超高速計算（学習やテスト用）
    # FALSE: 画面ありでいつものプレイ（人間が観戦する用）
    LEARNING_MODE = True # AIモデルを使うのでTrueに
    
    att_ctrl = DefaultAttackerController()
    
    if LEARNING_MODE:
        # 新しい統合モデルでテストしたい場合
        def_ctrl = LearningDefenderAllAIController(model_path="dqn_defender_combined_best.pt")
    
        # 迷路探索のみモデル
        #def_ctrl = LearningDefenderController(model_path="dqn_gridworld_fixedmap.pt")
        
        # 動きを確認したいので headless=False で可視化する
        game = VisualFPSBattle(NEW_MAZE_STR, att_ctrl, def_ctrl, headless=False)
    else:
        def_ctrl = DefaultDefenderController()
        game = VisualFPSBattle(NEW_MAZE_STR, att_ctrl, def_ctrl, headless=False)
        
    game.run()