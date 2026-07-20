import random
import tkinter as tk
import numpy as np

# 分離した操作用クラスをインポート
from controllers import DefaultAttackerController, DefaultDefenderController, UserInputController
from learning_defender import LearningDefenderController, LearningDefenderAllAIController
from map_data import NEW_MAZE_STR
# 💡追加：試合開始前のキャラクター編成画面
from roster_select import RosterSelectScreen

WINNING_ROUNDS = 13
TICK_TIME = 500

MAX_HP = 100
BODY_DAMAGE = 40
HEADSHOT_DAMAGE = 160
SHOOT_INTERVAL_TICKS = 1
SIDE_PANEL_WIDTH = 260
DEFUSE_REQUIRED_TICKS = 12  # 0.5秒/tick × 12 = 6秒（従来より3秒延長）
SMOKE_DURATION_TICKS = 30  # 15秒
MOVING_ACCURACY = 0.50
MOVING_TARGET_HIT_MULTIPLIER = 0.70

# character_stats.py が持つキー名に差があっても読み込めるようにする。
try:
    import character_stats as _character_stats
except ImportError:
    _character_stats = None


def _clamp_rate(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    # 50 のような百分率表記にも対応
    if value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))


def get_character_combat_stats(name):
    """キャラクター定義から命中率・弾除け率・HS率を取得する。未定義時は既定値。"""
    defaults = {"accuracy": 0.50, "dodge_rate": 0.10, "hs_rate": 0.20}
    if _character_stats is None:
        return defaults

    raw = None
    for attr in ("CHARACTER_STATS", "character_stats", "characters", "CHARACTERS"):
        table = getattr(_character_stats, attr, None)
        if isinstance(table, dict) and name in table:
            raw = table[name]
            break

    if raw is None:
        # 現在の character_stats.py の正式な取得関数に対応
        for getter_name in ("get_by_name", "get_stats"):
            getter = getattr(_character_stats, getter_name, None)
            if callable(getter):
                try:
                    raw = getter(name)
                except Exception:
                    raw = None
                if raw is not None:
                    break

    if raw is None:
        return defaults
    if not isinstance(raw, dict):
        raw = vars(raw) if hasattr(raw, "__dict__") else {}

    def pick(keys, default):
        for key in keys:
            if key in raw:
                return _clamp_rate(raw[key], default)
        return default

    return {
        "accuracy": pick(("accuracy", "aim", "hit_rate", "hit_pct", "命中率"), defaults["accuracy"]),
        "dodge_rate": pick(("dodge_rate", "dodge", "dodge_pct", "evasion", "弾除け率"), defaults["dodge_rate"]),
        "hs_rate": pick(("hs_rate", "hs", "hs_pct", "headshot_rate", "HS", "HS%"), defaults["hs_rate"]),
    }


def get_all_character_names():
    """character_stats.py に登録された全キャラクター名を取得する。"""
    if _character_stats is None:
        return []

    all_names = getattr(_character_stats, "all_names", None)
    if callable(all_names):
        try:
            names = all_names()
            if names:
                return [str(name) for name in names]
        except Exception:
            pass

    all_characters = getattr(_character_stats, "all_characters", None)
    if callable(all_characters):
        try:
            result = []
            for char in all_characters():
                if isinstance(char, dict):
                    name = char.get("name")
                else:
                    name = getattr(char, "name", None)
                if name:
                    result.append(str(name))
            if result:
                return result
        except Exception:
            pass

    for attr in ("CHARACTER_TABLE", "CHARACTER_STATS", "character_stats", "characters", "CHARACTERS"):
        table = getattr(_character_stats, attr, None)
        if isinstance(table, dict):
            return [str(name) for name in table.keys()]
        if isinstance(table, (list, tuple)):
            result = []
            for char in table:
                if isinstance(char, dict):
                    name = char.get("name")
                else:
                    name = getattr(char, "name", None)
                if name:
                    result.append(str(name))
            if result:
                return result
    return []


class Character:
    def __init__(self, name, team, pos, text_color, bg_color, has_spike=False, kills=0, deaths=0):
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
        self.max_hp = MAX_HP
        self.hp = MAX_HP
        self.kills = kills
        self.deaths = deaths

        stats = get_character_combat_stats(name)
        self.accuracy = stats["accuracy"]
        self.dodge_rate = stats["dodge_rate"]
        self.hs_rate = stats["hs_rate"]
        self.moved_this_tick = False
        self.smoke_charges = 1

class VisualFPSBattle:
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
        self.ability_area_height = 92
        
        # 画面非表示（headless）モードの時はTkinterを立ち上げない
        if not self.headless:
            self.root = tk.Tk()
            self.root.title("Attacker vs Defender")
            self.canvas = tk.Canvas(
                self.root,
                width=self.map_pixel_width + SIDE_PANEL_WIDTH * 2,
                height=self.map_pixel_height + self.ability_area_height,
                bg="#10141c",
                highlightthickness=0,
            )
            self.canvas.pack(fill="both", expand=True)
            self.root.minsize(self.map_pixel_width + SIDE_PANEL_WIDTH * 2, self.map_pixel_height + self.ability_area_height + 30)
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
            
        plants = list(zip(*np.where(self.grid == 2)))
        self.target_plant_pos = random.choice(plants) if plants else None
            
        self.spike_pos = None          
        self.is_planted = False        
        self.planted_pos = None        
        self.round_timer = 90          
        self.detonate_timer = 45       
        self.is_defused = False        
        self.last_engagements = []
        self.last_shot = None
        self.last_shots = []
        self.battle_tick = 0
        self.smokes = []  # {cells:set[(r,c)], remaining:int, owner:str}
        self.ability_mode = None
        
        # ディフェンダーコントローラの内部状態(サイト割り当て等)をリセットする
        if hasattr(self.defender_controller, "reset_round"):
            self.defender_controller.reset_round()
            
        # 💡追加：アタッカー側も同様にリセット(UserInputController用)
        if hasattr(self.attacker_controller, "reset_round"):
            self.attacker_controller.reset_round()

    def move_character(self, char):
        r, c = char.pos
        old_pos = tuple(char.pos)
        char.moved_this_tick = False
        
        # ---------------------------------------------------------------------
        # 💡 【修正】アタッカーのPlant自動処理の条件を厳密化
        # ---------------------------------------------------------------------
        # 単に grid == 2 ではなく、アタッカーが目指している target_plant_pos に到達した時のみタイマーを進める
        if char.team == "A" and char.has_spike:
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
                    # 設置完了後も plant_timer が残ると、射撃処理で永久に
                    # 「設置中」と判定されるため必ずリセットする。
                    char.plant_timer = 0
                return  # プラント中は移動処理を行わずその場に留まる
            else:
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
                    # 解除完了は射撃解決後に判定する。これにより最終解除tickでも、
                    # 射線が通る攻撃側は先に射撃でき、解除者が倒れた場合は解除失敗になる。
                    return  # 解除時はここで処理を終了し、その場に留まる
            char.defuse_timer = 0

        else:
            # MOVE アクションの処理
            char.defuse_timer = 0  
            # 💡 インデックス参照エラーを防ぐため、next_pos が有効な2次元座標であることを保証
            if isinstance(next_pos, (list, np.ndarray)) and len(next_pos) == 2:
                if self.grid[next_pos[0], next_pos[1]] != 1:
                    char.pos = list(next_pos)

        char.moved_this_tick = tuple(char.pos) != old_pos

    def _smoke_cells(self):
        cells = set()
        for smoke in self.smokes:
            cells.update(smoke["cells"])
        return cells

    def check_line_of_sight(self, p1, p2):
        """壁またはスモークを横切る射線を遮断する。"""
        x0, y0, x1, y1 = p1.pos[1], p1.pos[0], p2.pos[1], p2.pos[0]
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        curr_x, curr_y = x0, y0
        smoke_cells = self._smoke_cells()
        first = True
        while True:
            if self.grid[curr_y, curr_x] == 1:
                return False
            # 自分がいる開始マスは判定せず、それ以降にスモークがあれば遮断
            if not first and (curr_y, curr_x) in smoke_cells:
                return False
            if curr_x == x1 and curr_y == y1:
                return True
            first = False
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                curr_x += sx
            if e2 <= dx:
                err += dx
                curr_y += sy

    def get_viewer_team(self):
        controllers = self.get_user_controllers()
        return controllers[0][1] if controllers else None

    def is_visible_to_team(self, target, viewer_team):
        if viewer_team is None or target.team == viewer_team:
            return True
        return any(ally.is_alive and ally.team == viewer_team and self.check_line_of_sight(ally, target)
                   for ally in self.chars)

    def get_user_controllers(self):
        """ユーザー操作のコントローラーとそのチームのペアを返す"""
        result = []
        if isinstance(self.attacker_controller, UserInputController):
            result.append((self.attacker_controller, "A"))
        if isinstance(self.defender_controller, UserInputController):
            result.append((self.defender_controller, "D"))
        return result

    def on_canvas_click(self, event):
        # 下部アビリティパネルのSMOKEボタン。構え中に再クリックするとキャンセル。
        panel = self._ability_panel_bounds()
        if panel and panel[0] <= event.x <= panel[2] and panel[1] <= event.y <= panel[3]:
            selected = self._selected_user_character()
            same_smoke_is_armed = (
                selected is not None
                and self.ability_mode == ("SMOKE", selected.team, selected.name)
            )
            if same_smoke_is_armed:
                self.ability_mode = None
            elif selected and selected.smoke_charges > 0:
                self.ability_mode = ("SMOKE", selected.team, selected.name)
            self.draw()
            return

        map_x = event.x - self.map_offset_x
        if map_x < 0 or map_x >= self.map_pixel_width:
            return
        c = map_x // self.cell_size
        r = event.y // self.cell_size
        if not (0 <= r < self.height and 0 <= c < self.width):
            return

        if self.ability_mode and self.ability_mode[0] == "SMOKE":
            _, team, owner_name = self.ability_mode
            owner = next((ch for ch in self.chars if ch.name == owner_name and ch.is_alive), None)
            if owner and owner.team == team and owner.smoke_charges > 0 and self.grid[r, c] != 1:
                cells = {(rr, cc) for rr in range(r-1, r+2) for cc in range(c-1, c+2)
                         if 0 <= rr < self.height and 0 <= cc < self.width and self.grid[rr, cc] != 1}
                self.smokes.append({"cells": cells, "remaining": SMOKE_DURATION_TICKS, "owner": owner.name})
                owner.smoke_charges -= 1
            self.ability_mode = None
            self.draw()
            return

        for ctrl, team in self.get_user_controllers():
            ctrl.handle_click(r, c, self.grid, self.chars, team)
        self.draw()

    def _selected_user_character(self):
        for ctrl, team in self.get_user_controllers():
            if ctrl.selected_char:
                return next((ch for ch in self.chars if ch.name == ctrl.selected_char and ch.team == team and ch.is_alive), None)
        return None

    def _ability_panel_bounds(self):
        selected = self._selected_user_character()
        if not selected:
            return None
        w, h = 250, 62
        x1 = self.map_offset_x + (self.map_pixel_width - w) / 2
        y1 = self.map_pixel_height + 14
        return (x1, y1, x1+w, y1+h)

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

    def _kill_character(self, shooter, target):
        target.hp = 0
        target.is_alive = False
        target.just_died = True
        target.deaths += 1
        shooter.kills += 1
        self.match_stats.setdefault(target.name, {"kills": 0, "deaths": 0})["deaths"] = target.deaths
        self.match_stats.setdefault(shooter.name, {"kills": 0, "deaths": 0})["kills"] = shooter.kills
        if target.has_spike:
            self.spike_pos = tuple(target.pos)
            target.has_spike = False

    def _resolve_all_shots(self, engagements=None):
        """生存中の各キャラクターが、毎tick最大1回ずつ射撃する。

        各射手について、その瞬間に生存していて射線が通る敵を直接再検索する。
        解除中の敵が見えている場合は、その敵を最優先で狙う。
        射撃予定を全員分確定してからダメージを適用するため、同じtick中に
        倒されたキャラクターも、そのtick開始時に予定した射撃は実行できる。
        設置中・解除中のキャラクター自身は射撃できない。
        """
        if self.battle_tick % SHOOT_INTERVAL_TICKS != 0:
            self.last_shots = []
            self.last_shot = None
            return

        alive_at_tick_start = [c for c in self.chars if c.is_alive]
        shot_intents = []

        for shooter in alive_at_tick_start:
            busy = (shooter.plant_timer > 0) or (shooter.defuse_timer > 0)
            if busy:
                continue

            possible_targets = [
                target for target in alive_at_tick_start
                if target.team != shooter.team
                and self.check_line_of_sight(shooter, target)
            ]
            if not possible_targets:
                continue

            # 解除中の敵を最優先。複数いれば近い敵、その次にHPの低い敵を狙う。
            defusers = [
                target for target in possible_targets
                if self.is_planted and target.defuse_timer > 0
            ]
            target_pool = defusers if defusers else possible_targets
            target = min(
                target_pool,
                key=lambda t: (
                    max(abs(t.pos[0] - shooter.pos[0]), abs(t.pos[1] - shooter.pos[1])),
                    t.hp,
                    t.name,
                ),
            )

            shooter_accuracy = MOVING_ACCURACY if shooter.moved_this_tick else shooter.accuracy
            hit_chance = shooter_accuracy * (1.0 - target.dodge_rate)
            if target.moved_this_tick:
                hit_chance *= MOVING_TARGET_HIT_MULTIPLIER
            hit_chance = max(0.0, min(1.0, hit_chance))

            hit = random.random() < hit_chance
            headshot = hit and (random.random() < shooter.hs_rate)
            damage = (HEADSHOT_DAMAGE if headshot else BODY_DAMAGE) if hit else 0
            shot_intents.append({
                "shooter": shooter,
                "target": target,
                "hit": hit,
                "headshot": headshot,
                "damage": damage,
                "hit_chance": hit_chance,
            })

        for shot in shot_intents:
            target = shot["target"]
            if shot["damage"] <= 0 or not target.is_alive:
                continue
            target.hp = max(0, target.hp - shot["damage"])
            if target.hp <= 0:
                self._kill_character(shot["shooter"], target)

        self.last_shots = shot_intents
        self.last_shot = shot_intents[-1] if shot_intents else None

    def _resolve_defuse_completion(self):
        """射撃後に解除完了を確定する。生存している解除者だけが完了できる。"""
        if not self.is_planted or self.is_defused:
            return
        completed = [
            c for c in self.chars
            if c.is_alive and c.team == "D" and c.defuse_timer >= DEFUSE_REQUIRED_TICKS
        ]
        if completed:
            self.is_defused = True

    def process_battle(self):
        self.battle_tick += 1
        for smoke in self.smokes:
            smoke["remaining"] -= 1
        self.smokes = [smoke for smoke in self.smokes if smoke["remaining"] > 0]

        if self.spike_pos is not None:
            for c in self.chars:
                if c.is_alive and c.team == "A" and tuple(c.pos) == self.spike_pos:
                    c.has_spike = True
                    self.spike_pos = None
                    break

        for c in self.chars:
            if not c.is_alive and c.has_spike:
                self.spike_pos = tuple(c.pos)
                c.has_spike = False

        alive = [c for c in self.chars if c.is_alive]
        engagements = [
            (alive[i], alive[j])
            for i in range(len(alive))
            for j in range(i + 1, len(alive))
            if alive[i].team != alive[j].team and self.check_line_of_sight(alive[i], alive[j])
        ]
        self.last_engagements = engagements
        self.last_shot = None
        self._resolve_all_shots(engagements)
        # 射撃を解決してから解除完了を判定する。
        self._resolve_defuse_completion()

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
                    defuse_str = f" (Defusing: {max_defuse}/{DEFUSE_REQUIRED_TICKS})" if max_defuse > 0 else ""
                    self.label.config(text=f"💀 Attacker Eliminated! Defuse the Spike! {self.detonate_timer}s{defuse_str} | R{self.current_round}{score_text}", fg="#27ae60")
            elif not self.headless:
                max_defuse = max([c.defuse_timer for c in self.chars if c.team == "D" and c.is_alive] + [0])
                defuse_str = f" (Defusing: {max_defuse}/{DEFUSE_REQUIRED_TICKS})" if max_defuse > 0 else ""
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
            elif not self.headless:
                site_side = "Left Side" if self.target_plant_pos and self.target_plant_pos[1] < self.width // 2 else "Right Side"
                self.label.config(text=f"⚔️ Round {self.current_round} (Attacking {site_side}) | Ends in {self.round_timer}s | {score_text}", fg="black")

    def _map_x(self, x):
        return self.map_offset_x + x

    def _draw_team_panel(self, team, x0, title, accent):
        panel_w = SIDE_PANEL_WIDTH
        self.canvas.create_rectangle(x0, 0, x0 + panel_w, self.map_pixel_height + self.ability_area_height, fill="#111722", outline="#2a3444")
        self.canvas.create_rectangle(x0, 0, x0 + panel_w, 42, fill=accent, outline="")
        self.canvas.create_text(x0 + panel_w / 2, 21, text=title, fill="white", font=("Arial", 13, "bold"))
        chars = [c for c in self.chars if c.team == team][:5]
        viewer_team = self.get_viewer_team()
        row_h = 76
        for i, char in enumerate(chars):
            y = 54 + i * row_h
            row_fill = "#1b2432" if char.is_alive else "#17191e"
            self.canvas.create_rectangle(x0 + 10, y, x0 + panel_w - 10, y + 66, fill=row_fill, outline="#323e50")
            name_fill = "white" if char.is_alive else "#777b83"
            # 名前とK/Dは視認状態に関係なく常時表示する。
            display_name = char.name
            self.canvas.create_text(x0 + 20, y + 16, text=display_name, anchor="w", fill=name_fill, font=("Arial", 10, "bold"))
            kd_text = f"K {char.kills}  D {char.deaths}"
            self.canvas.create_text(x0 + panel_w - 18, y + 16, text=kd_text, anchor="e", fill="#d6dde8", font=("Arial", 9, "bold"))
            # 左右パネルでは、名前・K/D・HPを視認状態に関係なく常時表示する。
            # 視界判定で隠すのはマップ上の敵キャラクター本体だけ。
            hp_ratio = char.hp / char.max_hp if char.is_alive else 0.0
            bar_x1, bar_x2 = x0 + 20, x0 + panel_w - 20
            self.canvas.create_rectangle(bar_x1, y + 36, bar_x2, y + 50, fill="#343a45", outline="")
            self.canvas.create_rectangle(bar_x1, y + 36, bar_x1 + (bar_x2 - bar_x1) * hp_ratio, y + 50, fill=accent, outline="")
            hp_text = f"{char.hp}/{char.max_hp}" if char.is_alive else "DEAD"
            self.canvas.create_text((bar_x1 + bar_x2) / 2, y + 43, text=hp_text, fill="white", font=("Arial", 8, "bold"))

    def draw(self):
        if self.headless:
            return
        self.canvas.delete("all")
        self._draw_team_panel("A", 0, "ATTACKERS", "#c0392b")
        self._draw_team_panel("D", self.map_offset_x + self.map_pixel_width, "DEFENDERS", "#27ae60")

        color_map = {"0":"white", "1":"#34495e", "2":"#fff9c4", "3":"#ffcccc", "4":"#ccffcc"}
        for r in range(self.height):
            for c in range(self.width):
                x1 = self._map_x(c * self.cell_size)
                color = color_map.get(str(self.grid[r, c]), "white")
                self.canvas.create_rectangle(x1, r*self.cell_size, x1+self.cell_size, (r+1)*self.cell_size, fill=color, outline="#eee")

        # 煙らしく見えるように、半透明風の円を重ねて雲状に描画する。
        # Tkinter CanvasはRGBA非対応なのでstippleを使う。
        for smoke_index, smoke in enumerate(self.smokes):
            for sr, sc in smoke["cells"]:
                x1 = self._map_x(sc*self.cell_size)
                y1 = sr*self.cell_size
                pad = max(2, self.cell_size // 10)
                self.canvas.create_oval(
                    x1-pad, y1-pad, x1+self.cell_size+pad, y1+self.cell_size+pad,
                    fill="#d97706", outline="#f59e0b", width=1, stipple="gray50"
                )
                # 小さな煙の塊をずらして重ねる
                offsets = [
                    (-0.18, -0.12, 0.72), (0.20, -0.18, 0.66),
                    (-0.10, 0.22, 0.68), (0.24, 0.20, 0.62),
                ]
                for ox, oy, scale in offsets:
                    size = self.cell_size * scale
                    cx = x1 + self.cell_size * (0.5 + ox)
                    cy = y1 + self.cell_size * (0.5 + oy)
                    self.canvas.create_oval(
                        cx-size/2, cy-size/2, cx+size/2, cy+size/2,
                        fill="#f59e0b", outline="", stipple="gray50"
                    )

        if not self.is_planted and self.target_plant_pos:
            tr, tc = self.target_plant_pos
            x1 = self._map_x(tc*self.cell_size)
            self.canvas.create_rectangle(x1, tr*self.cell_size, x1+self.cell_size, (tr+1)*self.cell_size, fill="#f39c12", outline="#d35400")
        if self.is_planted and self.planted_pos:
            pr, pc = self.planted_pos
            x1 = self._map_x(pc*self.cell_size)
            self.canvas.create_oval(x1+2, pr*self.cell_size+2, x1+self.cell_size-2, (pr+1)*self.cell_size-2, fill="red", outline="")
        if self.spike_pos:
            sr, sc = self.spike_pos
            x1 = self._map_x(sc*self.cell_size)
            self.canvas.create_oval(x1+2, sr*self.cell_size+2, x1+self.cell_size-2, (sr+1)*self.cell_size-2, fill="black", outline="")

        if self.last_shot:
            shooter = self.last_shot["shooter"]
            target = self.last_shot["target"]
            self.canvas.create_line(
                self._map_x((shooter.pos[1]+0.5)*self.cell_size), (shooter.pos[0]+0.5)*self.cell_size,
                self._map_x((target.pos[1]+0.5)*self.cell_size), (target.pos[0]+0.5)*self.cell_size,
                fill="#ff3b30" if self.last_shot["hit"] else "#aab2bd", width=2, dash=() if self.last_shot["hit"] else (4, 3),
            )

        selected_names = {ctrl.selected_char for ctrl, _ in self.get_user_controllers() if ctrl.selected_char is not None}
        viewer_team = self.get_viewer_team()
        for char in self.chars:
            if not self.is_visible_to_team(char, viewer_team):
                continue
            if not char.is_alive and not char.just_died:
                continue
            row, col = char.pos
            x1 = self._map_x(col*self.cell_size)
            cx, cy = x1 + self.cell_size/2, (row+0.5)*self.cell_size
            if char.just_died:
                self.canvas.create_text(cx, cy, text="X", fill="orange", font=("Arial", 12, "bold"))
                char.just_died = False
                continue

            bg = "#2980b9" if (char.defuse_timer > 0 and self.is_planted) else char.bg_color
            outline_color = "yellow" if char.name in selected_names else ""
            outline_width = 3 if char.name in selected_names else 1
            self.canvas.create_oval(x1, row*self.cell_size, x1+self.cell_size, (row+1)*self.cell_size, fill=bg, outline=outline_color, width=outline_width)
            if char.has_spike:
                self.canvas.create_oval(x1+3, row*self.cell_size+3, x1+self.cell_size-3, (row+1)*self.cell_size-3, fill="black", outline="")

            # キャラクター上部の名前・HPパネル
            panel_w = max(56, min(110, 16 + len(char.name) * 7))
            panel_h = 23
            px1 = cx - panel_w / 2
            py2 = row*self.cell_size - 3
            py1 = py2 - panel_h
            self.canvas.create_rectangle(px1, py1, px1+panel_w, py2, fill="#101820", outline=char.bg_color, width=1)
            self.canvas.create_text(cx, py1+8, text=char.name, fill="yellow" if char.has_spike else "white", font=("Arial", 8, "bold"))
            hp_ratio = char.hp / char.max_hp
            self.canvas.create_rectangle(px1+4, py2-7, px1+panel_w-4, py2-3, fill="#3a404a", outline="")
            self.canvas.create_rectangle(px1+4, py2-7, px1+4+(panel_w-8)*hp_ratio, py2-3, fill=char.bg_color, outline="")

        # アビリティ専用の下部領域。マップとは完全に分離する。
        bottom_y = self.map_pixel_height
        self.canvas.create_rectangle(
            self.map_offset_x, bottom_y,
            self.map_offset_x + self.map_pixel_width, bottom_y + self.ability_area_height,
            fill="#0b1018", outline="#2a3444", width=2
        )
        self.canvas.create_text(
            self.map_offset_x + 18, bottom_y + 18,
            text="ABILITIES", anchor="w", fill="#8b98a9", font=("Arial", 10, "bold")
        )

        bounds = self._ability_panel_bounds()
        selected = self._selected_user_character()
        if bounds and selected:
            x1, y1, x2, y2 = bounds
            armed = self.ability_mode is not None
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#151c27", outline="#e67e22" if armed else "#536273", width=2)

            # 左側に煙のようなアイコンをCanvas図形で描く（環境依存の絵文字を使わない）。
            icon_cx = x1 + 34
            icon_cy = (y1 + y2) / 2
            smoke_fill = "#f39c12" if selected.smoke_charges > 0 else "#59616c"
            for ox, oy, radius in [(-9, 4, 8), (0, -3, 11), (10, 3, 9), (2, 8, 10)]:
                self.canvas.create_oval(
                    icon_cx + ox - radius, icon_cy + oy - radius,
                    icon_cx + ox + radius, icon_cy + oy + radius,
                    fill=smoke_fill, outline="#f8c471" if armed else "#7f8c8d", width=1
                )

            text_cx = (x1 + 58 + x2) / 2
            state = "構え中：再クリックでキャンセル" if armed else f"SMOKE  残り {selected.smoke_charges}"
            self.canvas.create_text(text_cx, y1+20, text=state, fill="#f5b041" if selected.smoke_charges else "#777", font=("Arial", 11, "bold"))
            help_text = "マップ上のマスを選択" if armed else "クリックしてスモークを構える"
            self.canvas.create_text(text_cx, y1+44, text=help_text, fill="white", font=("Arial", 9))

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