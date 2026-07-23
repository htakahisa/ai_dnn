"""Round progression, movement, shooting, spike flow, and win conditions."""

import math
import random
import numpy as np

from controllers import UserInputController
from game_core import (
    TICK_TIME, PLANT_REQUIRED_TICKS, DEFUSE_REQUIRED_TICKS, ROUND_TRANSITION_TICKS,
    SHOOT_INTERVAL_TICKS, MOVING_ACCURACY, BLIND_ACCURACY_MULTIPLIER,
    REVEALED_DODGE_MULTIPLIER, MOVING_TARGET_HIT_MULTIPLIER,
    HEADSHOT_DAMAGE, BODY_DAMAGE, WINNING_ROUNDS,
)

class BattleLogicMixin:

    def move_character(self, char):
        r, c = char.pos
        old_pos = tuple(char.pos)
        char.moved_this_tick = False
        
        # ---------------------------------------------------------------------
        # プラント処理
        # ユーザー操作時は、プラントゾーンに入っただけでは開始しない。
        # 選択中キャラクターの下部UIに出るPLANTボタンから明示的に開始する。
        # AI操作時のみ、従来どおり目標サイト到達時に自動で開始する。
        # ---------------------------------------------------------------------
        if char.team == "A" and char.has_spike and not self.is_planted:
            is_user_controlled = isinstance(self.attacker_controller, UserInputController)
            on_plant_site = (self.grid[r, c] == 2) if is_user_controlled else (
                self.target_plant_pos and list(char.pos) == list(self.target_plant_pos)
            )
            should_plant = char.is_planting if is_user_controlled else bool(on_plant_site)

            if should_plant and on_plant_site:
                char.plant_timer += 1
                if char.plant_timer >= PLANT_REQUIRED_TICKS:
                    self.is_planted = True
                    self.planted_pos = (r, c)
                    char.has_spike = False
                    char.plant_timer = 0
                    char.is_planting = False
                return  # 設置中は移動・射撃を行わない
            if char.is_planting and not on_plant_site:
                char.is_planting = False
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
            # アタッカー側もコントローラーによって戻り値が異なる。
            # DefaultAttackerController: 座標のみ
            # LearningAttackerController: (座標, "MOVE"/"PLANT")
            result = self.attacker_controller.decide_move(char, game_state)
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[1], str)
            ):
                next_pos, action_type = result
            else:
                next_pos = result
                action_type = "MOVE"
        else:
            # ディフェンダー側も同様に自動判別する。
            result = self.defender_controller.decide_move(char, game_state)
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[1], str)
            ):
                next_pos, action_type = result
            else:
                next_pos = result
                action_type = "MOVE"

        # ---------------------------------------------------------------------
        #  アクションタイプに応じたシステム処理 (修正版)
        # ---------------------------------------------------------------------
        if action_type == "DEFUSE":
            if self.is_planted and self.planted_pos and char.team == "D":
                dist = max(abs(self.planted_pos[0] - r), abs(self.planted_pos[1] - c))
                if dist <= 1:
                    # 同時に解除できるのは一人だけ。担当者がいない時だけロックを取得する。
                    if self.active_defuser_name in (None, char.name):
                        self.active_defuser_name = char.name
                        char.defuse_timer += 1
                        # 解除完了は射撃解決後に判定する。最終解除tickでも射撃を先に解決する。
                        return
                    # 別のキャラクターが解除中なら、このキャラクターは解除を開始できない。
                    char.defuse_timer = 0
                    return
            if self.active_defuser_name == char.name:
                self.active_defuser_name = None
            char.defuse_timer = 0

        else:
            # MOVE アクションの処理。解除担当者が解除をやめたらロックを解放する。
            if self.active_defuser_name == char.name:
                self.active_defuser_name = None
            char.defuse_timer = 0  
            # 💡 インデックス参照エラーを防ぐため、next_pos が有効な2次元座標であることを保証
            if isinstance(next_pos, (list, np.ndarray)) and len(next_pos) == 2:
                nr, nc = int(next_pos[0]), int(next_pos[1])
                in_bounds = 0 <= nr < self.height and 0 <= nc < self.width
                occupied = any(
                    other is not char and other.is_alive and tuple(other.pos) == (nr, nc)
                    for other in self.chars
                )
                if in_bounds and self.grid[nr, nc] != 1 and not occupied:
                    char.pos = [nr, nc]

        char.moved_this_tick = tuple(char.pos) != old_pos


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
                # ラウンド間待機もリアルタイム秒ではなくTick数で管理する。
                self.round_transition_ticks_left = ROUND_TRANSITION_TICKS
                self._advance_round_transition()
            else:
                # 学習用は表示待機が不要なので即時次ラウンドへ。
                self.init_round()


    def _advance_round_transition(self):
        if self.match_over:
            return
        if self.round_transition_ticks_left <= 0:
            self.init_round()
            self.loop()
            return
        self.round_transition_ticks_left -= 1
        self.root.after(TICK_TIME, self._advance_round_transition)


    def _kill_character(self, shooter, target):
        target.hp = 0
        target.is_alive = False
        target.just_died = True
        target.deaths += 1
        shooter.kills += 1
        shooter.round_kills += 1
        self.match_stats.setdefault(target.name, {"kills": 0, "deaths": 0})["deaths"] = target.deaths
        self.match_stats.setdefault(shooter.name, {"kills": 0, "deaths": 0})["kills"] = shooter.kills
        target.is_planting = False
        target.plant_timer = 0
        if self.active_defuser_name == target.name:
            self.active_defuser_name = None
        if target.has_spike:
            self.spike_pos = tuple(target.pos)
            target.has_spike = False


    def _resolve_all_shots(self, engagements=None, current_los_revealed_names=None):
        """同Tickの射撃を反応速度が高い順に逐次処理する。

        Tick開始時に射撃予定者と標的を確定する。
        反応速度の高い射手から順に射撃し、同値の場合だけランダム順にする。
        自分の射撃順が来る前に死亡した射手は射撃できない。
        """
        if self.battle_tick % SHOOT_INTERVAL_TICKS != 0:
            self.last_shots = []
            self.last_shot = None
            return

        alive_at_tick_start = [c for c in self.chars if c.is_alive]
        shot_intents = []
        executed_shots = []

        if current_los_revealed_names is None:
            current_los_revealed_names = self._current_los_revealed_names()

        for shooter in alive_at_tick_start:
            if shooter.plant_timer > 0 or shooter.defuse_timer > 0:
                continue

            possible_targets = [
                target for target in alive_at_tick_start
                if target.team != shooter.team
                and self.check_line_of_sight(shooter, target)
            ]
            if not possible_targets:
                continue

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
            shot_intents.append({"shooter": shooter, "target": target})

        # シャッフル後に安定ソートすることで、同じ反応速度だけ順番がランダムになる。
        random.shuffle(shot_intents)
        shot_intents.sort(
            key=lambda intent: intent["shooter"].reaction,
            reverse=True,
        )

        for intent in shot_intents:
            shooter = intent["shooter"]
            target = intent["target"]

            if not shooter.is_alive:
                continue
            if not target.is_alive:
                continue

            shooter_accuracy = MOVING_ACCURACY if shooter.moved_this_tick else shooter.accuracy
            if shooter.blind_remaining > 0:
                shooter_accuracy *= BLIND_ACCURACY_MULTIPLIER

            effective_dodge = target.dodge_rate * (
                REVEALED_DODGE_MULTIPLIER
                if self._is_revealed_for_shot(target, current_los_revealed_names)
                else 1.0
            )
            hit_chance = shooter_accuracy * (1.0 - effective_dodge)
            if target.moved_this_tick:
                hit_chance *= MOVING_TARGET_HIT_MULTIPLIER
            hit_chance = max(0.0, min(1.0, hit_chance))

            hit = random.random() < hit_chance
            headshot = hit and random.random() < shooter.hs_rate
            damage = (HEADSHOT_DAMAGE if headshot else BODY_DAMAGE) if hit else 0

            shot = {
                "shooter": shooter,
                "target": target,
                "hit": hit,
                "headshot": headshot,
                "damage": damage,
                "hit_chance": hit_chance,
                "reaction": shooter.reaction,
            }
            executed_shots.append(shot)

            if damage > 0:
                target.hp = max(0, target.hp - damage)
                if target.hp <= 0:
                    self._kill_character(shooter, target)

        self.last_shots = executed_shots
        self.last_shot = executed_shots[-1] if executed_shots else None


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
            self.active_defuser_name = None


    def process_battle(self):
        self.battle_tick += 1
        # すべての持続効果をTick数で管理する。
        for char in self.chars:
            char.blind_remaining = max(0, char.blind_remaining - 1)
            char.reveal_remaining = max(0, char.reveal_remaining - 1)
        for burst in self.flash_bursts:
            burst["remaining_ticks"] -= 1
        self.flash_bursts = [burst for burst in self.flash_bursts if burst["remaining_ticks"] > 0]
        for burst in self.recon_bursts:
            burst["remaining_ticks"] -= 1
        self.recon_bursts = [burst for burst in self.recon_bursts if burst["remaining_ticks"] > 0]
        self._advance_flash_projectiles()
        self._advance_recon_projectiles()
        for smoke in self.smokes:
            smoke["remaining_ticks"] -= 1
        self.smokes = [smoke for smoke in self.smokes if smoke["remaining_ticks"] > 0]

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

        # 現在の射線状況は先に計算するが、射線リビール状態への反映は射撃後に行う。
        # そのため、初めて敵を視認したTickの射撃は通常の回避率で判定される。
        current_los_revealed_names = self._current_los_revealed_names()
        alive = [c for c in self.chars if c.is_alive]
        engagements = [
            (alive[i], alive[j])
            for i in range(len(alive))
            for j in range(i + 1, len(alive))
            if alive[i].team != alive[j].team and self.check_line_of_sight(alive[i], alive[j])
        ]
        self.last_engagements = engagements
        self.last_shot = None
        self._resolve_all_shots(engagements, current_los_revealed_names)

        # 射撃判定後に、現在の射線状況を次のTick用リビール状態として反映する。
        for char in self.chars:
            char.los_revealed = char.is_alive and char.name in current_los_revealed_names

        # 射撃結果で条件を満たした覚醒イベントを判定する。
        self._check_awakening_events()

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
                    defuse_str = f" (Defusing: {int(max_defuse)}/{DEFUSE_REQUIRED_TICKS} Tick)" if max_defuse > 0 else ""
                    self.label.config(text=f"💀 Attacker Eliminated! Defuse the Spike! {int(self.detonate_timer)} Tick{defuse_str} | R{self.current_round}{score_text}", fg="#27ae60")
            elif not self.headless:
                max_defuse = max([c.defuse_timer for c in self.chars if c.team == "D" and c.is_alive] + [0])
                defuse_str = f" (Defusing: {int(max_defuse)}/{DEFUSE_REQUIRED_TICKS} Tick)" if max_defuse > 0 else ""
                self.label.config(text=f"🔥 Spike Planted! Detonation in {int(self.detonate_timer)} Tick{defuse_str} | R{self.current_round}{score_text}", fg="red")
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
                self.label.config(text=f"⚔️ Round {self.current_round} (Attacking {site_side}) | Ends in {int(self.round_timer)} Tick | {score_text}", fg="black")


    def loop(self):
        if not self.round_over and not self.match_over:
            for c in self.chars:
                if c.is_alive: self.move_character(c)
            self.process_battle()
            self.draw()
            self._advance_combo_announcement()
            self.root.after(TICK_TIME, self.loop)


    def run_headless_loop(self):
        """【AI学習用】画面を描画せず、限界速度でシミュレーションを回す"""
        print("💡 Headless Mode: シミュレーションをバックグラウンドで高速実行中...")
        while not self.match_over:
            if not self.round_over:
                for c in self.chars:
                    if c.is_alive: self.move_character(c)
                self.process_battle()
                self._advance_combo_announcement()


