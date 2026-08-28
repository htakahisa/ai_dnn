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
    CLUTCH_ACE_BANNER_TICKS,
    EXPLOSION_DURATION_TICKS,
    FACING_VECTORS,
    SHOOTING_SITE_DIGREE,
)

class BattleLogicMixin:

    def _facing_from_delta(self, dr, dc, fallback):
        """移動delta(dr, dc)から4方向facingを判定する。斜め移動は存在しない前提。"""
        dr, dc = int(dr), int(dc)
        if dr == 0 and dc == 0:
            return fallback
        if dr != 0:
            return "N" if dr < 0 else "S"
        return "W" if dc < 0 else "E"

    def _facing_angle_diff(self, shooter, target):
        """射手のfacingと、射手→標的方向との角度差(度)を返す。"""
        dc = float(target.pos[1] - shooter.pos[1])
        dr = float(target.pos[0] - shooter.pos[0])
        dist = math.hypot(dc, dr)
        if dist == 0:
            return 0.0
        fx, fy = FACING_VECTORS[shooter.facing]
        dot = max(-1.0, min(1.0, (fx * dc + fy * dr) / dist))
        return math.degrees(math.acos(dot))

    def _facing_accuracy_multiplier(self, shooter, target):
        """正面100%～真横50%まで、角度差に応じて線形に精度を落とす。"""
        angle = self._facing_angle_diff(shooter, target)
        return 1.0 - min(SHOOTING_SITE_DIGREE, angle) / SHOOTING_SITE_DIGREE * 0.5

    def _facing_towards(self, from_pos, to_pos):
        """from_posからto_posへ最も近い8方向のfacingを返す。"""
        dc = float(to_pos[1] - from_pos[1])
        dr = float(to_pos[0] - from_pos[0])
        dist = math.hypot(dc, dr)
        if dist == 0:
            return None
        nx, ny = dc / dist, dr / dist
        best_dir, best_dot = None, -2.0
        for direction, (fx, fy) in FACING_VECTORS.items():
            dot = fx * nx + fy * ny
            if dot > best_dot:
                best_dot = dot
                best_dir = direction
        return best_dir

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

    def _finalize_movement_transition_state(self, char):
        """今Tickの移動結果から、停止・Smoke境界通過状態を確定する。"""
        char.stopped_after_move_this_tick = bool(
            getattr(char, "moved_last_tick", False)
            and not getattr(char, "moved_this_tick", False)
        )

        smoke_cells_now = self._smoke_cells()
        in_smoke_now = tuple(char.pos) in smoke_cells_now
        was_in_smoke = bool(getattr(char, "was_in_smoke_before_move", False))
        char.entered_smoke_this_tick = bool(not was_in_smoke and in_smoke_now)
        char.exited_smoke_this_tick = bool(was_in_smoke and not in_smoke_now)

    def _move_character_during_defender_setup(self, char):
        """Setup Phase中のDefender専用移動。

        コントローラーから移動先だけを受け取り、Ability / Plant / Defuseは
        一切実行しない。Setupマップで禁止されているセルにも入れない。
        """
        if char.team != "D" or not char.is_alive:
            return

        old_pos = tuple(char.pos)
        char.moved_last_tick = bool(getattr(char, "moved_this_tick", False))
        char.moved_this_tick = False
        char.stopped_after_move_this_tick = False
        char.entered_smoke_this_tick = False
        char.exited_smoke_this_tick = False
        char.was_in_smoke_before_move = False
        char.is_planting = False
        char.plant_timer = 0
        char.defuse_timer = 0

        game_state = {
            "grid": self.grid,
            "spike_pos": self.spike_pos,
            "is_planted": False,
            "planted_pos": None,
            "target_plant_pos": self.target_plant_pos,
            "chars": self.chars,
            "spotted_info": {"spotted": 0.0, "site_r": 0.0, "site_c": 0.0},
            "defender_defuse_info": {},
            "detonate_timer": self.detonate_timer,
            "round_timer": self.round_timer,
            "defender_setup_active": True,
            "defender_setup_ticks_remaining": self.defender_setup_phase.ticks_remaining,
        }

        # Setup Phase はラウンド前の自陣配置なので、IQ知覚補正を通さない。
        # IQAwareController を通すと perceived_char / perceived_state のコピー座標で
        # Setup BFSが計算され、実キャラの位置と食い違う。
        #
        # Setup中だけ inner_controller を直接呼び、
        # LIVE後は通常の self.defender_controller 経由へ戻す。
        setup_controller = self.defender_controller
        seen_controller_ids = set()

        while hasattr(setup_controller, "inner_controller"):
            controller_id = id(setup_controller)
            if controller_id in seen_controller_ids:
                break
            seen_controller_ids.add(controller_id)

            inner = getattr(setup_controller, "inner_controller", None)
            if inner is None or inner is setup_controller:
                break
            setup_controller = inner

        # IQ wrapper が通常時に inner.set_game(perceived_view) を呼ぶため、
        # Setup判断の直前に実ゲームへ戻しておく。
        if hasattr(setup_controller, "set_game"):
            setup_controller.set_game(self)

        result = setup_controller.decide_move(char, game_state)
        next_pos = result

        # Ability付き戻り値や legacy action でも、Setup中は座標部分だけ使う。
        if isinstance(result, tuple) and len(result) >= 1:
            next_pos = result[0]

        if isinstance(next_pos, (list, tuple, np.ndarray)) and len(next_pos) == 2:
            nr, nc = int(next_pos[0]), int(next_pos[1])
            in_bounds = 0 <= nr < self.height and 0 <= nc < self.width
            target_pos = (nr, nc)
            occupied = self._is_position_occupied(char, target_pos, old_pos)
            is_wall = in_bounds and self.grid[nr, nc] == 1
            setup_allowed = (
                in_bounds
                and self.defender_setup_phase.defender_can_move_to(nr, nc)
            )

            if in_bounds and not is_wall and not occupied and setup_allowed:
                new_facing = self._facing_from_delta(
                    nr - old_pos[0], nc - old_pos[1], char.facing
                )
                # print(
                #     "[SETUP FACING DEBUG]", char.name,
                #     "old_pos=", old_pos, "new_pos=", (nr, nc),
                #     "dr=", nr - old_pos[0], "dc=", nc - old_pos[1],
                #     "old_facing=", char.facing, "new_facing=", new_facing,
                # )
                char.facing = new_facing
                char.pos = [nr, nc]
                self._update_occupancy_after_move(old_pos, (nr, nc))

        char.moved_this_tick = tuple(char.pos) != old_pos

    def _run_defender_setup_tick(self):
        """Setup Phaseを1Tick処理する。Defenderだけが移動する。"""
        self._build_occupancy_counts()
        try:
            for char in self.chars:
                if char.is_alive and char.team == "D":
                    self._move_character_during_defender_setup(char)
        finally:
            self._clear_occupancy_counts()

        ended = self.defender_setup_phase.advance_tick()
        if ended and not self.headless:
            self.label.config(text=f"⚔️ Round {self.current_round} LIVE", fg="black")

    def move_character(self, char):
        r, c = char.pos
        old_pos = tuple(char.pos)

        # ---------------------------------------------------------------------
        # Rush対策用の1Tick状態
        # ---------------------------------------------------------------------
        # 前Tickに移動していたかを退避してから、今Tickの状態を初期化する。
        # 「移動→停止」は、前Tick moved=True かつ今Tick moved=False で判定する。
        char.moved_last_tick = bool(getattr(char, "moved_this_tick", False))
        char.moved_this_tick = False
        char.stopped_after_move_this_tick = False

        # Smoke境界通過も今Tick単位で記録する。
        smoke_cells_before_move = self._smoke_cells()
        char.was_in_smoke_before_move = old_pos in smoke_cells_before_move
        char.entered_smoke_this_tick = False
        char.exited_smoke_this_tick = False

        # 前Tickで被弾していれば、このTickだけ強制的に相手の方向を向く。
        char.facing_forced_this_tick = False
        if getattr(char, "forced_facing_next_tick", None):
            char.facing = char.forced_facing_next_tick
            char.facing_forced_this_tick = True
        char.forced_facing_next_tick = None

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
            # GC Macro Plant Commitment: pre-plant remaining round time.
            "round_timer": self.round_timer,
            "smoke_cells": self._smoke_cells(),
        }

        if char.team == "A":
            # アタッカー側のコントローラー戻り値パターン：
            # 1. 座標のみ: (move_pos)
            # 2. アビリティ: (move_pos, {"ability": "SMOKE"|"FLASH"|"RECON", "target": (r, c)})
            # 3. レガシー: (座標, "MOVE"/"PLANT") ← 既存モデル対応
            result = self.attacker_controller.decide_move(char, game_state)
            ability_payload = None
            facing_payload = None

            if isinstance(result, tuple) and len(result) >= 2:
                next_pos = result[0]
                second_elem = result[1]

                # ケース1: 辞書型アビリティ（新しいcontrollers.py）
                if isinstance(second_elem, dict) and "ability" in second_elem:
                    ability_payload = second_elem
                    action_type = "ABILITY"
                # ケース1.5: 辞書型「その場で向きだけ変える」（移動しない）
                elif isinstance(second_elem, dict) and "facing" in second_elem:
                    facing_payload = second_elem
                    action_type = "TURN"
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
            facing_payload = None

            if isinstance(result, tuple) and len(result) >= 2:
                next_pos = result[0]
                second_elem = result[1]

                # ケース1: 辞書型アビリティ（新しいcontrollers.py）
                if isinstance(second_elem, dict) and "ability" in second_elem:
                    ability_payload = second_elem
                    action_type = "ABILITY"
                # ケース1.5: 辞書型「移動先(現在地含む)+向き」を同時指定。
                # next_posが現在地と同じなら実質その場旋回、異なれば移動しつつ
                # facingを指定通りに固定する(移動方向への自動追従を上書きする)。
                # 通常のMOVE処理に相乗りさせるだけなので action_type は "MOVE"。
                elif isinstance(second_elem, dict) and "facing" in second_elem:
                    facing_payload = second_elem
                    action_type = "MOVE"
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
            self._finalize_movement_transition_state(char)
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
                self._finalize_movement_transition_state(char)
                return

            # 無効な場所でのPLANTは失敗し、設置進捗をリセットする。
            char.is_planting = False
            char.plant_timer = 0
            char.moved_this_tick = False
            self._finalize_movement_transition_state(char)
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
                        self._finalize_movement_transition_state(char)
                        return
                    char.defuse_timer = 0
                    self._finalize_movement_transition_state(char)
                    return

            if self.active_defuser_name == char.name:
                self.active_defuser_name = None
            char.defuse_timer = 0
            self._finalize_movement_transition_state(char)
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
                # facing_payloadで明示的にfacingが指定されていれば、移動方向とは
                # 無関係にそちらを優先する(例: 前進しながら後ろを向く、等)。
                # 指定が無ければ従来通り移動方向から自動計算する。
                explicit_facing = (facing_payload or {}).get("facing")
                if explicit_facing in FACING_VECTORS:
                    if not char.facing_forced_this_tick:
                        char.facing = explicit_facing
                elif not char.facing_forced_this_tick:
                    new_facing = self._facing_from_delta(
                        nr - old_pos[0], nc - old_pos[1], char.facing
                    )
                    # print(
                    #     "[FACING DEBUG]", char.name,
                    #     "old_pos=", old_pos, "new_pos=", (nr, nc),
                    #     "dr=", nr - old_pos[0], "dc=", nc - old_pos[1],
                    #     "old_facing=", char.facing, "new_facing=", new_facing,
                    #     "forced=", char.facing_forced_this_tick,
                    # )
                    char.facing = new_facing
                char.pos = [nr, nc]
                self._update_occupancy_after_move(
                    old_pos,
                    (nr, nc),
                )

        char.moved_this_tick = tuple(char.pos) != old_pos
        self._finalize_movement_transition_state(char)

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
            winner_name = self.attacker_team_name if attacker_won else self.defender_team_name
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
            banner_ticks = (
                CLUTCH_ACE_BANNER_TICKS if self.special_round_banner else 0
            )
            explosion_ticks = (
                EXPLOSION_DURATION_TICKS if self.explosion_effect else 0
            )
            extra_ticks = max(banner_ticks, explosion_ticks)
            self.round_transition_ticks_left = ROUND_TRANSITION_TICKS + extra_ticks
            self._advance_round_transition()
        else:
            self.init_round()

    def _advance_round_transition(self):
        if self.match_over:
            return
        if self.round_transition_ticks_left <= 0:
            self.special_round_banner = None
            self.explosion_effect = None
            self.init_round()
            self.loop()
            return
        if self.explosion_effect is not None:
            self.explosion_effect["ticks_elapsed"] = min(
                EXPLOSION_DURATION_TICKS,
                self.explosion_effect["ticks_elapsed"] + 1,
            )
        self.round_transition_ticks_left -= 1
        self.draw()
        self.root.after(TICK_TIME, self._advance_round_transition)

    def _ensure_round_tracking_state(self):
        """ラウンドが切り替わったら clutch/ace/爆発 用の状態をリセットする。"""
        round_key = getattr(self, "current_round", None)
        if getattr(self, "_round_tracking_round_key", None) != round_key:
            self._round_tracking_round_key = round_key
            self.clutch_watch = {}
            self.special_round_banner = None
            self.explosion_effect = None

    def _check_special_round_banner(self, winning_team):
        """ラウンド勝利チームに ACE / CLUTCH が発生していたかを判定する。"""
        self._ensure_round_tracking_state()
        self.special_round_banner = None

        enemy_team = "D" if winning_team == "A" else "A"
        enemy_total = sum(1 for c in self.chars if c.team == enemy_team)

        # ACE: 勝利チームの誰かが敵全員を単独で撃破
        ace_player = next(
            (
                c
                for c in self.chars
                if c.team == winning_team
                and enemy_total > 0
                and c.round_kills >= enemy_total
            ),
            None,
        )
        if ace_player is not None:
            self.special_round_banner = {
                "type": "ACE",
                "name": ace_player.display_name,
            }
            return

        # CLUTCH: 自チーム1人生存の状態からそのまま勝利
        clutch_name = self.clutch_watch.get(winning_team)
        if clutch_name:
            survivor = next(
                (
                    c
                    for c in self.chars
                    if c.team == winning_team and c.name == clutch_name
                ),
                None,
            )
            if survivor is not None and survivor.is_alive:
                self.special_round_banner = {
                    "type": "CLUTCH",
                    "name": survivor.display_name,
                }

    def _kill_character(self, shooter, target):
        target.hp = 0
        target.is_alive = False
        target.just_died = True
        target.deaths += 1
        shooter.kills += 1
        shooter.round_kills += 1

        # タイガー「ハンター」：敵を倒した瞬間にHPを50回復。
        # 最大HPを超えて回復しない。
        if (
            getattr(shooter, "role", None) == "タイガー"
            or getattr(shooter, "ability_name", None) == "HUNT"
        ):
            max_hp = float(getattr(shooter, "max_hp", 100))
            shooter.hp = min(max_hp, float(shooter.hp) + 50.0)

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

        # 1人生存になった瞬間を記録しておき、ラウンド終了時にクラッチ判定へ使う。
        self._ensure_round_tracking_state()
        for watch_team in ("A", "D"):
            alive_members = [
                c for c in self.chars if c.team == watch_team and c.is_alive
            ]
            if len(alive_members) == 1 and watch_team not in self.clutch_watch:
                self.clutch_watch[watch_team] = alive_members[0].name

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
            # 視認できていても、射手と標的の間に別プレイヤーがいれば撃てない。
            possible_targets = [
                target
                for target in possible_targets
                if self.check_shot_line_of_sight(shooter, target)
            ]
            # 正面から左右x度を超える(背後含む)相手は視界外のため撃てない。
            possible_targets = [
                target
                for target in possible_targets
                if self._facing_angle_diff(shooter, target) <= SHOOTING_SITE_DIGREE
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
            # 正面からの角度差による補正(正面100%～真横50%)
            shooter_accuracy *= self._facing_accuracy_multiplier(shooter, target)

            # -----------------------------------------------------------------
            # Rush対策の射撃精度補正（すべて乗算）
            # -----------------------------------------------------------------
            # Smokeへ入ったTick: x0.75
            if getattr(shooter, "entered_smoke_this_tick", False):
                shooter_accuracy *= 0.75

            # Smokeから出たTick: x0.75
            if getattr(shooter, "exited_smoke_this_tick", False):
                shooter_accuracy *= 0.75

            # 前Tickに移動し、今Tick停止した最初のTick: x0.75
            if getattr(shooter, "stopped_after_move_this_tick", False):
                shooter_accuracy *= 0.75

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

            # 距離補正（ユークリッド距離）。
            # 基準: 1マス=2.00倍 / 15マス=1.00倍 / 40マス=0.75倍。
            # 指定点の間は線形補間し、40マス以遠は0.75倍で下限固定する。
            dr = float(target.pos[0] - shooter.pos[0])
            dc = float(target.pos[1] - shooter.pos[1])
            distance = math.hypot(dr, dc)
            if distance <= 1.0:
                distance_multiplier = 2.0
            elif distance <= 15.0:
                distance_multiplier = 2.0 + (1.0 - 2.0) * ((distance - 1.0) / 14.0)
            elif distance <= 40.0:
                distance_multiplier = 1.0 + (0.75 - 1.0) * ((distance - 15.0) / 25.0)
            else:
                distance_multiplier = 0.75
            hit_chance *= distance_multiplier

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

            # 命中・被弾を問わず、撃たれたら次のTickだけ相手の方向を強制的に向く。
            target.forced_facing_next_tick = self._facing_towards(
                target.pos, shooter.pos
            )

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
        self._ensure_round_tracking_state()
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
        score_text = (
            f" [Score: {self.attacker_team_name} {self.attacker_wins} - "
            f"{self.defender_wins} {self.defender_team_name}]{overtime_text}"
        )

        if self.is_defused:
            self.defender_wins += 1
            self._record_round_mental_result("D")
            self._check_special_round_banner("D")
            if not self.headless:
                self.label.config(
                    text=f"⚙️ Spike Defused! {self.defender_team_name} WIN Round {self.current_round}! {score_text}",
                    fg="green",
                )
            self.round_over = True
            self.check_match_winner()
        elif self.is_planted:
            self.detonate_timer -= 1
            if self.detonate_timer <= 0:
                self.attacker_wins += 1
                self._record_round_mental_result("A")
                self._check_special_round_banner("A")
                self.explosion_effect = {
                    "pos": self.planted_pos,
                    "ticks_elapsed": 0,
                }
                if not self.headless:
                    self.label.config(
                        text=f"💥 Spike Detonated! {self.attacker_team_name} WIN Round {self.current_round}! {score_text}",
                        fg="red",
                    )
                self.round_over = True
                self.check_match_winner()
            elif not alive_D:
                self.attacker_wins += 1
                self._record_round_mental_result("A")
                self._check_special_round_banner("A")
                if not self.headless:
                    self.label.config(
                        text=f"🏆 {self.defender_team_name} Annihilated! {self.attacker_team_name} WIN Round {self.current_round}! {score_text}",
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
                        text=f"💀 {self.attacker_team_name} Eliminated! Defuse the Spike! {int(self.detonate_timer)} Tick{defuse_str} | R{self.current_round}{score_text}",
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
                self._check_special_round_banner("D")
                if not self.headless:
                    self.label.config(
                        text=f"⏰ Time Expired! {self.defender_team_name} WIN Round {self.current_round}! {score_text}",
                        fg="#27ae60",
                    )
                self.round_over = True
                self.check_match_winner()
            elif not alive_A:
                self.defender_wins += 1
                self._record_round_mental_result("D")
                self._check_special_round_banner("D")
                if not self.headless:
                    self.label.config(
                        text=f"🏆 {self.attacker_team_name} Annihilated! {self.defender_team_name} WIN Round {self.current_round}! {score_text}",
                        fg="#27ae60",
                    )
                self.round_over = True
                self.check_match_winner()
            elif not alive_D:
                self.attacker_wins += 1
                self._record_round_mental_result("A")
                self._check_special_round_banner("A")
                if not self.headless:
                    self.label.config(
                        text=f"🏆 {self.defender_team_name} Annihilated! {self.attacker_team_name} WIN Round {self.current_round}! {score_text}",
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
            if self.defender_setup_phase.active:
                self._run_defender_setup_tick()
            else:
                self._build_occupancy_counts()
                try:
                    for c in self._move_order():
                        if c.is_alive:
                            self.move_character(c)
                finally:
                    self._clear_occupancy_counts()
                self.process_battle()
                self._advance_combo_announcement()

            self.draw()
            self.root.after(TICK_TIME, self.loop)

    def run_headless_loop(self):
        """【AI学習用】画面を描画せず、限界速度でシミュレーションを回す"""
        while not self.match_over:
            if not self.round_over:
                if self.defender_setup_phase.active:
                    self._run_defender_setup_tick()
                else:
                    self._build_occupancy_counts()
                    try:
                        for c in self._move_order():
                            if c.is_alive:
                                self.move_character(c)
                    finally:
                        self._clear_occupancy_counts()
                    self.process_battle()
                    self._advance_combo_announcement()
