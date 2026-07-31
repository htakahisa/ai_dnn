"""Round progression, movement, shooting, spike flow, and win conditions."""

import random
import numpy as np

from controllers import UserInputController
from learning_attacker import LearningAttackerController
from learning_attacker_multi import LearningAttackerMultiController
from learning_attacker_ability import LearningAttackerAbilityController
from attacker_v2.multi_role_attacker_controller import MultiRoleAttackerController
from train_attacker_ability import DEFUSE_REQUIRED


class BattleLogicMixin:

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
            # 💡 【修正】tuple を判定に追加し、AIからの戻り値を確実に受け取る
            if isinstance(next_pos, (list, tuple, np.ndarray)) and len(next_pos) == 2:
                target_r, target_c = int(next_pos[0]), int(next_pos[1])

                # 1. 移動先が壁でないかをチェック
                if self.grid[target_r, target_c] != 1:

                    # 2. 移動先に他の生存しているキャラクターがいないかをチェック
                    is_occupied = False
                    for other_char in self.chars:
                        if other_char != char and other_char.is_alive:
                            if other_char.pos[0] == target_r and other_char.pos[1] == target_c:
                                is_occupied = True
                                break

                    # 3. 壁でもキャラクターでもなければ移動を実行
                    if not is_occupied:
                        char.pos = list(next_pos)

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

    def get_spotted_info(self):
        spike_holder = next((c for c in self.chars if c.is_alive and c.team == "A" and c.has_spike), None)
        if spike_holder is None:
            return {'spotted': 0.0, 'site_r': 0.0, 'site_c': 0.0}

        for d in self.chars:
            if d.is_alive and d.team == "D" and self.check_line_of_sight(d, spike_holder):
                return {'spotted': 1.0, 'site_r': float(spike_holder.pos[0]), 'site_c': float(spike_holder.pos[1])}

        return {'spotted': 0.0, 'site_r': 0.0, 'site_c': 0.0}

    def check_match_winner(self):
        if self.attacker_wins >= self.WINNING_ROUNDS:
            if not self.headless:
                self.label.config(text=f"🏆 MATCH OVER: Attacker WINS! ({self.attacker_wins} - {self.defender_wins})", fg="#c0392b", font=("Arial", 12, "bold"))
            print(f"MATCH OVER: 💥 Attacker WINS! ({self.attacker_wins} - {self.defender_wins})")
            self.match_over = True
        elif self.defender_wins >= self.WINNING_ROUNDS:
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

            # 💡 倒されたキャラの相手チーム（撃ち合い勝者）にカウントを入れる
            if target.team == "A":
                self.defender_gunfight_wins += 1
            else:
                self.attacker_gunfight_wins += 1

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

    def loop(self):
        if not self.round_over and not self.match_over:
            for c in self.chars:
                if c.is_alive: self.move_character(c)
            self.process_battle()
            self.draw()
            self.root.after(self.TICK_TIME, self.loop)

    def run_headless_loop(self):
        """【AI学習用】画面を描画せず、限界速度でシミュレーションを回す"""
        print("💡 Headless Mode: シミュレーションをバックグラウンドで高速実行中...")
        while not self.match_over:
            if not self.round_over:
                for c in self.chars:
                    if c.is_alive: self.move_character(c)
                self.process_battle()
            # round_over 時の初期化は check_match_winner 内で自動処理されます