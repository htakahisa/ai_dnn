"""Round progression, movement, shooting, spike flow, and win conditions."""

import math
import random
import numpy as np

from controllers import UserInputController
from game_core import (
    TICK_TIME,
    PLANT_REQUIRED_TICKS,
    DEFUSE_REQUIRED_TICKS,
    ROUND_TRANSITION_TICKS,
    SHOOT_INTERVAL_TICKS,
    MOVING_ACCURACY,
    BLIND_ACCURACY_MULTIPLIER,
    REVEALED_DODGE_MULTIPLIER,
    MOVING_TARGET_HIT_MULTIPLIER,
    HEADSHOT_DAMAGE,
    BODY_DAMAGE,
    WINNING_ROUNDS,
    DEFUSE_REQUIRED,
)


class BattleLogicMixin:

    def _build_occupancy_counts(self):
        """現在生存中のキャラクター位置を数える。移動フェーズ中だけ使用する。"""
        counts = {}
        for char in self.chars:
            if not char.is_alive:
                continue
            pos = (int(char.pos[0]), int(char.pos[1]))
            counts[pos] = counts.get(pos, 0) + 1
        self._movement_occupancy_counts = counts

    def _clear_occupancy_counts(self):
        self._movement_occupancy_counts = None

    def _is_position_occupied(self, char, position, old_position):
        """既存のany走査と同じ判定を、位置カウントからO(1)で返す。"""
        counts = getattr(self, "_movement_occupancy_counts", None)
        if counts is None:
            return any(
                other is not char
                and other.is_alive
                and tuple(other.pos) == position
                for other in self.chars
            )

        occupied_count = int(counts.get(position, 0))
        if position == old_position:
            occupied_count -= 1
        return occupied_count > 0

    def _update_occupancy_after_move(self, old_position, new_position):
        counts = getattr(self, "_movement_occupancy_counts", None)
        if counts is None or old_position == new_position:
            return

        old_count = int(counts.get(old_position, 0))
        if old_count <= 1:
            counts.pop(old_position, None)
        else:
            counts[old_position] = old_count - 1

        counts[new_position] = int(counts.get(new_position, 0)) + 1

    def move_character(self, char):
        r, c = char.pos
        old_pos = tuple(char.pos)
        char.moved_this_tick = False

        # ---------------------------------------------------------------------
        # プラントは自動開始しない。
        # AIはコントローラーから "PLANT" を返した場合のみ設置する。
        # ユーザー操作はPLANTボタンで char.is_planting=True になった場合のみ設置する。
        # ---------------------------------------------------------------------

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

        defender_defuse_info = {
            d.name: (
                d.defuse_timer,
                DEFUSE_REQUIRED,
            )  # 6 = DEFUSE_REQUIRED (learning_attacker_multi.py の DEFUSE_REQUIRED と一致させる)
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
            "spotted_info": (
                self.get_spotted_info()
                if not self.is_planted
                else {"spotted": 1.0, "site_r": site_r, "site_c": site_c}
            ),
            "defender_defuse_info": defender_defuse_info,
            "detonate_timer": self.detonate_timer,
        }

        if char.team == "A":
            # アタッカー側のコントローラー戻り値パターン：
            # 1. 座標のみ: (move_pos)
            # 2. アビリティ: (move_pos, {"ability": "SMOKE"|"FLASH"|"RECON", "target": (r, c)})
            # 3. レガシー: (座標, "MOVE"/"PLANT") ← 既存モデル対応
            result = self.attacker_controller.decide_move(char, game_state)
            ability_payload = None

            if isinstance(result, tuple) and len(result) >= 2:
                next_pos = result[0]
                second_elem = result[1]

                # ケース1: 辞書型アビリティ（新しいcontrollers.py）
                if isinstance(second_elem, dict) and "ability" in second_elem:
                    ability_payload = second_elem
                    action_type = "ABILITY"
                # ケース2: 文字列型アクション（既存モデル）
                elif isinstance(second_elem, str):
                    action_type = second_elem
                    if len(result) >= 3:
                        ability_payload = result[2]
                else:
                    next_pos = result
                    action_type = "MOVE"
            else:
                next_pos = result
                action_type = "MOVE"

            # 手動操作ではPLANTボタンが char.is_planting を立てる。
            # サイトに入っただけでは開始せず、ボタン操作時だけPLANTとして処理する。
            if (
                isinstance(self.attacker_controller, UserInputController)
                and char.is_planting
            ):
                next_pos = char.pos
                action_type = "PLANT"

            # print(
            #     "[ATTACKER DEBUG]",
            #     "controller=",
            #     self.attacker_controller.__class__.__name__,
            #     "pos=",
            #     tuple(char.pos),
            #     "action=",
            #     action_type,
            #     "result=",
            #     result,
            # )
        else:
            # ディフェンダー側も同様に自動判別する。
            result = self.defender_controller.decide_move(char, game_state)
            ability_payload = None

            if isinstance(result, tuple) and len(result) >= 2:
                next_pos = result[0]
                second_elem = result[1]

                # ケース1: 辞書型アビリティ（新しいcontrollers.py）
                if isinstance(second_elem, dict) and "ability" in second_elem:
                    ability_payload = second_elem
                    action_type = "ABILITY"
                # ケース2: 文字列型アクション（既存モデル）
                elif isinstance(second_elem, str):
                    action_type = second_elem
                    if len(result) >= 3:
                        ability_payload = result[2]
                else:
                    next_pos = result
                    action_type = "MOVE"
            else:
                next_pos = result
                action_type = "MOVE"

        # ---------------------------------------------------------------------
        # アクションタイプに応じたシステム処理
        # ---------------------------------------------------------------------
        if action_type == "ABILITY":
            # アビリティ使用Tickは移動・設置・解除を行わない。
            char.is_planting = False
            char.plant_timer = 0
            if self.active_defuser_name == char.name:
                self.active_defuser_name = None
            char.defuse_timer = 0
            self.execute_ai_ability(char, ability_payload)
            char.moved_this_tick = False
            return

        if action_type == "PLANT":
            # PLANTは明示的に選択された場合だけ進行する。
            on_plant_site = (
                char.team == "A"
                and char.has_spike
                and not self.is_planted
                and 0 <= r < self.height
                and 0 <= c < self.width
                and self.grid[r, c] == 2
            )

            if on_plant_site:
                char.is_planting = True
                char.plant_timer += 1
                char.defuse_timer = 0
                if self.active_defuser_name == char.name:
                    self.active_defuser_name = None

                if char.plant_timer >= PLANT_REQUIRED_TICKS:
                    self.is_planted = True
                    self.planted_pos = (r, c)
                    self.spike_pos = None
                    char.has_spike = False
                    char.plant_timer = 0
                    char.is_planting = False

                # 設置中は移動・射撃を行わない。
                char.moved_this_tick = False
                return

            # 無効な場所でのPLANTは失敗し、設置進捗をリセットする。
            char.is_planting = False
            char.plant_timer = 0
            char.moved_this_tick = False
            return

        # PLANT以外を選んだ時点で設置を中断する。
        char.is_planting = False
        char.plant_timer = 0

        if action_type == "DEFUSE":
            if self.is_planted and self.planted_pos and char.team == "D":
                dist = max(abs(self.planted_pos[0] - r), abs(self.planted_pos[1] - c))
                if dist <= 1:
                    # 同時に解除できるのは一人だけ。
                    if self.active_defuser_name in (None, char.name):
                        self.active_defuser_name = char.name
                        char.defuse_timer += 1
                        # 解除完了は射撃解決後に判定する。
                        return
                    char.defuse_timer = 0
                    return

            if self.active_defuser_name == char.name:
                self.active_defuser_name = None
            char.defuse_timer = 0
            return

        # MOVE処理。解除担当者が解除をやめたらロックを解放する。
        if self.active_defuser_name == char.name:
            self.active_defuser_name = None
        char.defuse_timer = 0

        if isinstance(next_pos, (list, tuple, np.ndarray)) and len(next_pos) == 2:
            nr, nc = int(next_pos[0]), int(next_pos[1])
            in_bounds = 0 <= nr < self.height and 0 <= nc < self.width
            target_pos = (nr, nc)
            occupied = self._is_position_occupied(
                char,
                target_pos,
                old_pos,
            )
            is_wall = in_bounds and self.grid[nr, nc] == 1

            # print(
            #     "[MOVE DEBUG]",
            #     "from=",
            #     old_pos,
            #     "to=",
            #     (nr, nc),
            #     "in_bounds=",
            #     in_bounds,
            #     "wall=",
            #     is_wall,
            #     "occupied=",
            #     occupied,
            # )

            if in_bounds and not is_wall and not occupied:
                char.pos = [nr, nc]
                self._update_occupancy_after_move(
                    old_pos,
                    (nr, nc),
                )

        char.moved_this_tick = tuple(char.pos) != old_pos

    def get_spotted_info(self):
        spike_holder = next(
            (c for c in self.chars if c.is_alive and c.team == "A" and c.has_spike),
            None,
        )
        if spike_holder is None:
            return {"spotted": 0.0, "site_r": 0.0, "site_c": 0.0}

        for d in self.chars:
            if (
                d.is_alive
                and d.team == "D"
                and self.check_line_of_sight(d, spike_holder)
            ):
                return {
                    "spotted": 1.0,
                    "site_r": float(spike_holder.pos[0]),
                    "site_c": float(spike_holder.pos[1]),
                }

        return {"spotted": 0.0, "site_r": 0.0, "site_c": 0.0}

    def check_match_winner(self):
        """通常戦は13本先取、12-12以降は2点差が付くまで継続する。"""
        # 12-12に到達した瞬間からオーバータイムへ移行する。
        if (
            not self.overtime
            and self.attacker_wins == WINNING_ROUNDS - 1
            and self.defender_wins == WINNING_ROUNDS - 1
        ):
            self.overtime = True

        if self.overtime:
            match_finished = abs(self.attacker_wins - self.defender_wins) >= 2
        else:
            match_finished = (
                self.attacker_wins >= WINNING_ROUNDS
                or self.defender_wins >= WINNING_ROUNDS
            )

        if match_finished:
            attacker_won = self.attacker_wins > self.defender_wins
            winner_name = "Attacker" if attacker_won else "Defender"
            winner_score = self.attacker_wins if attacker_won else self.defender_wins
            loser_score = self.defender_wins if attacker_won else self.attacker_wins
            winner_color = "#c0392b" if attacker_won else "#27ae60"
            overtime_text = " [OVERTIME]" if self.overtime else ""

            if not self.headless:
                self.label.config(
                    text=(
                        f"🏆 MATCH OVER{overtime_text}: "
                        f"{winner_name} WINS! ({winner_score} - {loser_score})"
                    ),
                    fg=winner_color,
                    font=("Arial", 12, "bold"),
                )
            print(
                f"MATCH OVER{overtime_text}: "
                f"{winner_name} WINS! ({winner_score} - {loser_score})"
            )
            self.match_over = True
            return

        self.current_round += 1
        if not self.headless:
            self.round_transition_ticks_left = ROUND_TRANSITION_TICKS
            self._advance_round_transition()
        else:
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
        self.match_stats.setdefault(target.name, {"kills": 0, "deaths": 0})[
            "deaths"
        ] = target.deaths
        self.match_stats.setdefault(shooter.name, {"kills": 0, "deaths": 0})[
            "kills"
        ] = shooter.kills
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

            if engagements is not None:
                possible_targets = []
                for first, second in engagements:
                    if first is shooter and second.is_alive:
                        possible_targets.append(second)
                    elif second is shooter and first.is_alive:
                        possible_targets.append(first)
            else:
                possible_targets = [
                    target
                    for target in alive_at_tick_start
                    if target.team != shooter.team
                    and self.check_line_of_sight(shooter, target)
                ]
            if not possible_targets:
                continue

            defusers = [
                target
                for target in possible_targets
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

            shooter_accuracy = (
                MOVING_ACCURACY if shooter.moved_this_tick else shooter.accuracy
            )
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
            c
            for c in self.chars
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
        self.flash_bursts = [
            burst for burst in self.flash_bursts if burst["remaining_ticks"] > 0
        ]
        for burst in self.recon_bursts:
            burst["remaining_ticks"] -= 1
        self.recon_bursts = [
            burst for burst in self.recon_bursts if burst["remaining_ticks"] > 0
        ]
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
            if alive[i].team != alive[j].team
            and self.check_line_of_sight(alive[i], alive[j])
        ]
        self.last_engagements = engagements
        self.last_shot = None
        self._resolve_all_shots(engagements, current_los_revealed_names)

        # 射撃判定後に、現在の射線状況を次のTick用リビール状態として反映する。
        for char in self.chars:
            char.los_revealed = (
                char.is_alive and char.name in current_los_revealed_names
            )

        # 射撃結果で条件を満たした覚醒イベントを判定する。
        self._check_awakening_events()

        # 射撃を解決してから解除完了を判定する。
        self._resolve_defuse_completion()

        alive_A = any(c.is_alive for c in self.chars if c.team == "A")
        alive_D = any(c.is_alive for c in self.chars if c.team == "D")
        overtime_text = " [OT]" if self.overtime else ""
        score_text = f" [Score: Att {self.attacker_wins} - {self.defender_wins} Def]{overtime_text}"

        if self.is_defused:
            self.defender_wins += 1
            self._record_round_mental_result("D")
            if not self.headless:
                self.label.config(
                    text=f"⚙️ Spike Defused! Defender WIN Round {self.current_round}! {score_text}",
                    fg="green",
                )
            self.round_over = True
            self.check_match_winner()
        elif self.is_planted:
            self.detonate_timer -= 1
            if self.detonate_timer <= 0:
                self.attacker_wins += 1
                self._record_round_mental_result("A")
                if not self.headless:
                    self.label.config(
                        text=f"💥 Spike Detonated! Attacker WIN Round {self.current_round}! {score_text}",
                        fg="red",
                    )
                self.round_over = True
                self.check_match_winner()
            elif not alive_D:
                self.attacker_wins += 1
                self._record_round_mental_result("A")
                if not self.headless:
                    self.label.config(
                        text=f"🏆 Defender Annihilated! Attacker WIN Round {self.current_round}! {score_text}",
                        fg="#c0392b",
                    )
                self.round_over = True
                self.check_match_winner()
            elif not alive_A:
                if not self.headless:
                    max_defuse = max(
                        [
                            c.defuse_timer
                            for c in self.chars
                            if c.team == "D" and c.is_alive
                        ]
                        + [0]
                    )
                    defuse_str = (
                        f" (Defusing: {int(max_defuse)}/{DEFUSE_REQUIRED_TICKS} Tick)"
                        if max_defuse > 0
                        else ""
                    )
                    self.label.config(
                        text=f"💀 Attacker Eliminated! Defuse the Spike! {int(self.detonate_timer)} Tick{defuse_str} | R{self.current_round}{score_text}",
                        fg="#27ae60",
                    )
            elif not self.headless:
                max_defuse = max(
                    [c.defuse_timer for c in self.chars if c.team == "D" and c.is_alive]
                    + [0]
                )
                defuse_str = (
                    f" (Defusing: {int(max_defuse)}/{DEFUSE_REQUIRED_TICKS} Tick)"
                    if max_defuse > 0
                    else ""
                )
                self.label.config(
                    text=f"🔥 Spike Planted! Detonation in {int(self.detonate_timer)} Tick{defuse_str} | R{self.current_round}{score_text}",
                    fg="red",
                )
        else:
            self.round_timer -= 1
            if self.round_timer <= 0:
                self.defender_wins += 1
                self._record_round_mental_result("D")
                if not self.headless:
                    self.label.config(
                        text=f"⏰ Time Expired! Defender WIN Round {self.current_round}! {score_text}",
                        fg="#27ae60",
                    )
                self.round_over = True
                self.check_match_winner()
            elif not alive_A:
                self.defender_wins += 1
                self._record_round_mental_result("D")
                if not self.headless:
                    self.label.config(
                        text=f"🏆 Attacker Annihilated! Defender WIN Round {self.current_round}! {score_text}",
                        fg="#27ae60",
                    )
                self.round_over = True
                self.check_match_winner()
            elif not alive_D:
                self.attacker_wins += 1
                self._record_round_mental_result("A")
                if not self.headless:
                    self.label.config(
                        text=f"🏆 Defender Annihilated! Attacker WIN Round {self.current_round}! {score_text}",
                        fg="#c0392b",
                    )
                self.round_over = True
                self.check_match_winner()
            elif not self.headless:
                site_side = (
                    "Left Side"
                    if self.target_plant_pos
                    and self.target_plant_pos[1] < self.width // 2
                    else "Right Side"
                )
                self.label.config(
                    text=f"⚔️ Round {self.current_round} (Attacking {site_side}) | Ends in {int(self.round_timer)} Tick | {score_text}",
                    fg="black",
                )

    def _move_order(self):
        """スパイク保持者(carry)を最優先で処理する。
        先に動いた者勝ちの衝突判定のため、carryの移動先を先に確定させることで
        escort/retrieveが自然とそのマスを避けるようになる。"""
        carriers = [c for c in self.chars if c.is_alive and c.has_spike]
        others = [c for c in self.chars if c.is_alive and not c.has_spike]
        return carriers + others

    def loop(self):
        if not self.round_over and not self.match_over:
            self._build_occupancy_counts()
            try:
                for c in self._move_order():
                    if c.is_alive:
                        self.move_character(c)
            finally:
                self._clear_occupancy_counts()
            self.process_battle()
            self.draw()
            self._advance_combo_announcement()
            self.root.after(TICK_TIME, self.loop)

    def run_headless_loop(self):
        """【AI学習用】画面を描画せず、限界速度でシミュレーションを回す"""
        while not self.match_over:
            if not self.round_over:
                self._build_occupancy_counts()
                try:
                    for c in self._move_order():
                        if c.is_alive:
                            self.move_character(c)
                finally:
                    self._clear_occupancy_counts()
                self.process_battle()
                self._advance_combo_announcement()
