import random
import tkinter as tk
import numpy as np

# 分離した操作用クラスをインポート
from controllers import DefaultAttackerController, DefaultDefenderController

# 新しいマップデータ
NEW_MAZE_STR = """
11111111111111111111111111111111111111111111
11111111111111111100000111111111111111111111
11111111111111111000000004000000000000404001
11111111111111111001111110111110111111000111
11111140000040000011111110111110000111002221
11111101111101111011111110111111000111002121
10001101011101111000000000000000000111002121
10022001010001110011100000000000000000002121
10022101010000110111100011111110001111002121
11121101000000000001100011111000001111002121
10001101000001110001100011111000001111002221
10000000000001110011100011111100011111110111
10000000111001110001100001100000111111000111
10001100111111110001111000001100111111110111
11101111111111110000000000111110000000000111
11100001111100000001100000111110101111110111
11100000001100110111100000000110111111110111
11101111001100111110000111110110111100000111
11111111000000111110000111110000000001110001
11111111001111111110000111111111111111110001
11111111001111111110000111111111111111110001
11111111100000000000000111111111011111110111
11111111100000000000000000000000000000000111
11111111111111111033333111111111111111111111
11111111111111111111111111111111111111111111
11111111111111111111111111111111111111111111
"""

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
    def __init__(self, maze_str, attacker_controller, defender_controller):
        lines = [line.strip() for line in maze_str.strip("\n").split("\n") if line.strip()]
        self.height, self.width = len(lines), len(lines[0])
        self.grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
        self.cell_size = 12 
        
        self.attacker_controller = attacker_controller
        self.defender_controller = defender_controller
        
        area_3 = list(zip(*np.where(self.grid == 3)))
        area_4 = list(zip(*np.where(self.grid == 4)))
        
        spike_holder_index = random.randint(0, len(area_3) - 1) if area_3 else -1
        
        self.chars = []
        for i, pos in enumerate(area_3):
            has_spike = (i == spike_holder_index)
            self.chars.append(Character(f"Att{i+1}", "A", pos, "white", "#c0392b", has_spike=has_spike))
        for i, pos in enumerate(area_4):
            self.chars.append(Character(f"Def{i+1}", "D", pos, "white", "#27ae60"))
            
        # 【新機能】開始時にマップ上のすべての設置可能マスから、今ラウンドの目的地をランダムで1つ決定
        plants = list(zip(*np.where(self.grid == 2)))
        self.target_plant_pos = random.choice(plants) if plants else None
            
        self.spike_pos = None          
        self.is_planted = False        
        self.planted_pos = None        
        self.round_timer = 90          
        self.detonate_timer = 45       
        self.is_defused = False        
        self.last_engagements = []
        
        self.root = tk.Tk()
        self.root.title("Attacker vs Defender")
        self.canvas = tk.Canvas(self.root, width=self.width*self.cell_size, height=self.height*self.cell_size)
        self.canvas.pack()
        self.label = tk.Label(self.root, text="Round Start", font=("Arial", 10))
        self.label.pack()
        self.game_over = False

    def move_character(self, char):
        r, c = char.pos
        if char.has_spike and self.grid[r, c] == 2:
            char.plant_timer += 1
            if char.plant_timer >= 4:
                self.is_planted = True
                self.planted_pos = (r, c)  
                char.has_spike = False
            return

        if self.is_planted and self.planted_pos and char.team == "D":
            dist = max(abs(self.planted_pos[0] - r), abs(self.planted_pos[1] - c))
            if dist <= 1:
                char.defuse_timer += 1
                if char.defuse_timer >= 6:
                    self.is_defused = True
                return
            else:
                char.defuse_timer = 0 

        # ゲーム状態にターゲット位置を追加してコントローラーに渡す
        game_state = {
            "grid": self.grid,
            "spike_pos": self.spike_pos,
            "is_planted": self.is_planted,
            "planted_pos": self.planted_pos,
            "target_plant_pos": self.target_plant_pos, # 【追加】
            "chars": self.chars
        }

        if char.team == "A":
            next_pos = self.attacker_controller.decide_move(char, game_state)
        else:
            next_pos = self.defender_controller.decide_move(char, game_state)
            
        if self.grid[next_pos[0], next_pos[1]] != 1:
            char.pos = list(next_pos)

        if char.team == "A":
            if self.spike_pos and char.pos == list(self.spike_pos):
                char.has_spike = True
                self.spike_pos = None
            if char.has_spike and self.grid[char.pos[0], char.pos[1]] == 2:
                char.plant_timer = 1

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

    def process_battle(self):
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

        if self.is_defused:
            self.label.config(text="⚙️ Spike Defused! Defender WIN!", fg="green")
            self.game_over = True
            
        elif self.is_planted:
            self.detonate_timer -= 1
            if self.detonate_timer <= 0:
                self.label.config(text="💥 Spike Detonated! Attacker WIN!", fg="red")
                self.game_over = True
            elif not alive_D:
                self.label.config(text="🏆 Defender Annihilated! Attacker WIN!", fg="#c0392b")
                self.game_over = True
            elif not alive_A:
                max_defuse = max([c.defuse_timer for c in self.chars if c.team == "D" and c.is_alive] + [0])
                defuse_str = f" (Defusing: {max_defuse}/6)" if max_defuse > 0 else ""
                self.label.config(text=f"💀 Attacker Eliminated! Defuse the Spike! {self.detonate_timer}s{defuse_str}", fg="#27ae60")
            else:
                max_defuse = max([c.defuse_timer for c in self.chars if c.team == "D" and c.is_alive] + [0])
                defuse_str = f" (Defusing: {max_defuse}/6)" if max_defuse > 0 else ""
                self.label.config(text=f"🔥 Spike Planted! Detonation in {self.detonate_timer}s{defuse_str}", fg="red")
                
        else:
            self.round_timer -= 1
            
            if self.round_timer <= 0:
                self.label.config(text="⏰ Time Expired! Defender WIN!", fg="#27ae60")
                self.game_over = True
            elif not alive_A:
                self.label.config(text="🏆 Defender (Green) WIN!", fg="#27ae60")
                self.game_over = True
            elif not alive_D:
                self.label.config(text="🏆 Attacker (Red) WIN!", fg="#c0392b")
                self.game_over = True
            else:
                # ターゲットしている方向（左側・右側）をテキストで少しわかりやすく可視化
                site_side = "Left Side" if self.target_plant_pos and self.target_plant_pos[1] < self.width // 2 else "Right Side"
                self.label.config(text=f"⚔️ Attacking {site_side} | Round Ends in {self.round_timer}s", fg="black")

    def draw(self):
        self.canvas.delete("all")
        color_map = {"0":"white", "1":"#34495e", "2":"#fff9c4", "3":"#ffcccc", "4":"#ccffcc"}
        
        for r in range(self.height):
            for c in range(self.width):
                color = color_map.get(str(self.grid[r, c]), "white")
                self.canvas.create_rectangle(c*self.cell_size, r*self.cell_size, (c+1)*self.cell_size, (r+1)*self.cell_size, fill=color, outline="#eee")
        
        # 【機能追加】今ラウンドのターゲットとなっている設置マスを、少し濃い黄色（オレンジ枠）で強調表示
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
        if not self.game_over:
            for c in self.chars:
                if c.is_alive: self.move_character(c)
            self.process_battle()
            self.draw()
            self.root.after(300, self.loop)

    def run(self):
        self.draw()
        self.root.after(300, self.loop)
        self.root.mainloop()

if __name__ == "__main__":
    att_ctrl = DefaultAttackerController()
    def_ctrl = DefaultDefenderController()
    VisualFPSBattle(NEW_MAZE_STR, attacker_controller=att_ctrl, defender_controller=def_ctrl).run()