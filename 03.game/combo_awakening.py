"""Player combo, awakening, and announcement-queue behavior."""

import random
from game_core import (
    PLAYER_COMBOS,
    AWAKENING_EVENTS,
    COMBO_DISPLAY_TICKS,
    SMOKE_DURATION_TICKS,
    _character_stats,
    _clamp_rate,
    _canonical_combo_stat_key,
    _apply_combo_bonus,
    _normalize_hs_rate,
)

# duration_ticks未指定の覚醒は999Tickとして扱う（ラウンド内では実質切れない＝既存互換）。
_DEFAULT_AWAKENING_DURATION_TICKS = 999

# 覚醒発動前にスナップショットし、Tick切れ時に元へ戻す対象の属性。
_AWAKENING_SNAPSHOT_KEYS = (
    "accuracy",
    "hs_rate",
    "dodge_rate",
    "reaction",
    "iq",
    "effective_iq",
    "mental",
    "move_steps_per_tick",
    "max_hp",
    "role",
    "ability_name",
    "hunter_active",
    "smoke_charges",
    "flash_charges",
    "recon_charges",
    "display_name",
    "sees_through_smoke",
    "abilities_per_round",
    "ultimate_cost",
    "start_round_with_full_ult",
)


def _snapshot_character_awakening_state(char):
    return {key: getattr(char, key, None) for key in _AWAKENING_SNAPSHOT_KEYS}


def _restore_character_awakening_state(char, snapshot):
    for key, value in snapshot.items():
        setattr(char, key, value)


class ComboAwakeningMixin:
    def _apply_player_combos(self):
        """同じチームに必要メンバーが全員いるコンボを発動する。"""
        self.active_player_combos = []
        for team in ("A", "D"):
            team_chars = [char for char in self.chars if char.team == team]

            for combo in PLAYER_COMBOS:
                if not isinstance(combo, dict):
                    continue
                chars_by_name = {}
                for char in team_chars:
                    chars_by_name[char.name] = char
                    display_name = getattr(char, "display_name", None)
                    if display_name:
                        chars_by_name[display_name] = char
                team_names = set(chars_by_name)
                combo_name = str(combo.get("name", "名称未設定コンボ"))
                required_players = tuple(str(name) for name in combo.get("players", ()))
                if not required_players or not set(required_players).issubset(
                    team_names
                ):
                    continue

                common_bonuses = combo.get("bonuses", {})
                per_player_bonuses = combo.get("player_bonuses", {})
                renames = combo.get("renames", {})
                affected = []

                for player_name in required_players:
                    char = chars_by_name[player_name]
                    if isinstance(common_bonuses, dict):
                        for stat_key, value in common_bonuses.items():
                            _apply_combo_bonus(char, stat_key, value)
                    own_bonuses = (
                        per_player_bonuses.get(player_name, {})
                        if isinstance(per_player_bonuses, dict)
                        else {}
                    )
                    if isinstance(own_bonuses, dict):
                        for stat_key, value in own_bonuses.items():
                            _apply_combo_bonus(char, stat_key, value)
                    if isinstance(renames, dict) and player_name in renames:
                        char.display_name = str(renames[player_name])
                    if combo_name not in char.active_combos:
                        char.active_combos.append(combo_name)
                    # Carnal Lust Syndicateの特殊効果フラグを設定
                    if combo.get("carnal_lust_syndicate_effect", False):
                        char.carnal_lust_syndicate_active = True
                    # AsunaとZekkenの特殊効果：相手チームのランダムなプレイヤーのaccuracyを10%下げる
                    if combo.get("asuna_zekken_debuff_effect", False):
                        # 相手チームを判定
                        opponent_team = "D" if team == "A" else "A"
                        # 相手チームの生存プレイヤーを取得
                        opponent_chars = [
                            c
                            for c in self.chars
                            if c.team == opponent_team and c.is_alive
                        ]
                        if opponent_chars:
                            # ランダムに1人選択
                            random_opponent = random.choice(opponent_chars)
                            # baseのaccuracyから10%削減（毎ラウンドbaseがリセットされるため、過去のデバフは自動的にクリアされる）
                            from game_core import _clamp_rate

                            new_accuracy = (
                                random_opponent.base_accuracy_before_condition - 0.1
                            )
                            random_opponent.accuracy = _clamp_rate(
                                new_accuracy, random_opponent.accuracy
                            )
                            # condition_modifierを再適用して正規化
                            condition_multiplier = 1.0 + getattr(
                                random_opponent, "condition_modifier", 0.0
                            )
                            random_opponent.accuracy *= condition_multiplier
                    affected.append(player_name)

                # AsunaとZekkenの特殊効果：相手チームのランダムなプレイヤーのaccuracyを10%下げる
                if combo.get("asuna_zekken_debuff_effect", False):
                    # 相手チームの生存プレイヤーを取得
                    opponent_team = "D" if team == "A" else "A"
                    opponent_chars = [
                        char
                        for char in self.chars
                        if char.team == opponent_team and char.is_alive
                    ]
                    if opponent_chars:
                        # ランダムに1人選択
                        random_opponent = random.choice(opponent_chars)
                        # accuracyを10%（0.1）減少させる
                        from game_core import _clamp_rate

                        random_opponent.accuracy = _clamp_rate(
                            random_opponent.accuracy - 0.1, random_opponent.accuracy
                        )
                        # デバフを記録して次のラウンドでリセットできるようにする
                        if not hasattr(self, "asuna_zekken_debuffed_this_round"):
                            self.asuna_zekken_debuffed_this_round = []
                        self.asuna_zekken_debuffed_this_round.append(
                            random_opponent.name
                        )

                self.active_player_combos.append(
                    {
                        "type": "combo",
                        "name": combo_name,
                        "team": team,
                        "players": tuple(affected),
                        "display_players": tuple(
                            chars_by_name[n].display_name for n in affected
                        ),
                        "effect_text": str(
                            combo.get("effect_text") or "相手ランダム1人の命中率-10%"
                        ),
                    }
                )

    def _describe_bonuses(self, common_bonuses, per_player_bonuses=None):
        labels = {
            "accuracy": "Hit%",
            "hs_rate": "HS%",
            "dodge_rate": "回避率",
            "reaction": "反応速度",
            "iq": "IQ",
            "max_hp": "最大HP",
            "condition_bonus": "調子補正",
        }
        parts = []
        if isinstance(common_bonuses, dict):
            for key, value in common_bonuses.items():
                canonical = _canonical_combo_stat_key(key)
                if canonical:
                    amount = float(value)
                    if (
                        canonical
                        in ("accuracy", "hs_rate", "dodge_rate", "condition_bonus")
                        and abs(amount) <= 1
                    ):
                        shown = amount * 100
                    else:
                        shown = amount
                    parts.append(f"{labels.get(canonical, canonical)} {shown:+g}")
        if isinstance(per_player_bonuses, dict) and per_player_bonuses:
            parts.append("個別補正あり")
        return " / ".join(parts) if parts else "特殊効果"

    def _enqueue_announcement(self, announcement):
        """コンボ・覚醒イベントの告知を表示待ちキューへ追加する。"""
        if not isinstance(announcement, dict):
            return
        if not hasattr(self, "announcement_queue"):
            self.announcement_queue = []
        self.announcement_queue.append(announcement)

        # 現在何も表示していない場合は、追加された告知を直ちに表示開始する。
        if self.combo_announcement_index >= len(self.announcement_queue) - 1:
            self.combo_announcement_index = len(self.announcement_queue) - 1
            self.combo_announcement_ticks_left = COMBO_DISPLAY_TICKS

    def _advance_combo_announcement(self):
        """現在の告知を1Tick進め、終了したら次の告知へ移る。"""
        if not getattr(self, "announcement_queue", None):
            return
        if self.combo_announcement_index >= len(self.announcement_queue):
            return
        self.combo_announcement_ticks_left -= 1
        if self.combo_announcement_ticks_left <= 0:
            self.combo_announcement_index += 1
            if self.combo_announcement_index < len(self.announcement_queue):
                self.combo_announcement_ticks_left = COMBO_DISPLAY_TICKS

    def _get_character_preset(self, name):
        if _character_stats is None:
            return None
        getter = getattr(_character_stats, "get_by_name", None)
        raw = getter(name) if callable(getter) else None
        if raw is None:
            table = getattr(_character_stats, "CHARACTER_TABLE", {})
            raw = table.get(name) if isinstance(table, dict) else None
        return raw

    def _apply_awakening_preset(self, char, preset_name):
        raw = self._get_character_preset(preset_name)
        if raw is None:
            return
        data = vars(raw) if hasattr(raw, "__dict__") else raw
        char.accuracy = _clamp_rate(
            data.get("hit_pct", data.get("accuracy")), char.accuracy
        )
        char.hs_rate = _normalize_hs_rate(
            data.get("hs_pct", data.get("hs_rate")), char.hs_rate
        )
        char.dodge_rate = _clamp_rate(
            data.get("dodge_pct", data.get("dodge_rate")), char.dodge_rate
        )
        try:
            char.iq = float(data.get("iq", data.get("IQ", char.iq)))
        except (TypeError, ValueError):
            pass
        try:
            char.reaction = float(
                data.get(
                    "reaction",
                    data.get("reaction_speed", data.get("反応速度", char.reaction)),
                )
            )
        except (TypeError, ValueError):
            pass
        role = str(data.get("role", char.role))
        char.role = role
        char.ability_name = {
            "フラッシュ": "FLASH",
            "スモーカー": "SMOKE",
            "シーカー": "RECON",
            "タイガー": "HUNT",
        }.get(role, char.ability_name)
        char.hunter_active = role == "タイガー"
        char.smoke_charges = 1 if char.ability_name == "SMOKE" else 0
        char.flash_charges = 1 if char.ability_name == "FLASH" else 0
        char.recon_charges = 1 if char.ability_name == "RECON" else 0

    def _awakening_condition_met(self, event, char, event_name):
        """覚醒イベントの発動条件を判定する。

        対応条件:
        - all_allies_dead:
            覚醒者以外の味方が全員死亡
        - hp_at_or_below:
            覚醒者のHPが condition_value 以下
        - kills_at_least:
            覚醒者のラウンド内キル数が condition_value 以上
        - specific_player_dead:
            condition_player で指定したプレイヤーが死亡
        - specific_player_killed:
            condition_player で指定したプレイヤーを覚醒者が倒した
        - team_kills_at_least:
            覚醒者のチームのラウンド内合計キル数が condition_value 以上
        - enemy_count_at_or_below:
            生存している敵人数が condition_value 以下
        - whathappend
            覚醒者がラウンドでキルした時、味方より敵の人数が多い場合
        - smoke_thrown
            このTickに誰かがスモークを使用した(味方・敵問わず)
        - permanent:
            常時発動する覚醒イベント（まだトリガーされていなければ常にtrue）
        """
        condition = str(event.get("condition", "")).strip()
        value = event.get("condition_value")

        if condition == "permanent":
            # 常時発動する覚醒イベントは、まだトリガーされていなければ常にtrue
            return event_name not in char.active_awakenings
        if condition == "all_allies_dead":
            allies = [c for c in self.chars if c.team == char.team and c is not char]
            return (
                char.is_alive and bool(allies) and all(not c.is_alive for c in allies)
            )
        if condition == "hp_at_or_below":
            try:
                return char.is_alive and char.hp <= float(value)
            except (TypeError, ValueError):
                return False

        if condition == "kills_at_least":
            try:
                return char.is_alive and int(getattr(char, "round_kills", 0)) >= int(
                    value
                )
            except (TypeError, ValueError):
                return False

        if condition == "specific_player_dead":
            target_name = str(
                event.get("condition_player") or event.get("condition_value") or ""
            ).strip()

            if not target_name or not char.is_alive:
                return False

            return any(
                not candidate.is_alive
                and getattr(candidate, "base_name", getattr(candidate, "name", ""))
                == target_name
                for candidate in self.chars
            )

        if condition == "specific_player_killed":
            target_name = str(
                event.get("condition_player") or event.get("condition_value") or ""
            ).strip()

            if not target_name or not char.is_alive:
                return False

            # 推奨形式:
            # char.round_killed_players = ["Lohen", ...]
            killed_players = getattr(char, "round_killed_players", ())
            if target_name in killed_players:
                return True

            # 互換用:
            # char.killed_players = ["Lohen", ...]
            killed_players = getattr(char, "killed_players", ())
            return target_name in killed_players

        if condition == "team_kills_at_least":
            try:
                required_kills = int(value)
            except (TypeError, ValueError):
                return False

            team_kills = sum(
                int(getattr(candidate, "round_kills", 0))
                for candidate in self.chars
                if candidate.team == char.team
            )
            return char.is_alive and team_kills >= required_kills

        if condition == "enemy_count_at_or_below":
            try:
                maximum_enemies = int(value)
            except (TypeError, ValueError):
                return False

            alive_enemies = sum(
                1
                for candidate in self.chars
                if candidate.team != char.team and candidate.is_alive
            )
            return char.is_alive and alive_enemies <= maximum_enemies

        if condition in {"overtime", "ot"}:
            # battle_logic.py が12-12到達時に self.overtime=True にする。
            # OT中のラウンドでは、生存中の覚醒対象に対して成立する。
            return char.is_alive and bool(getattr(self, "overtime", False))

        if condition == "whathappend":
            try:
                alive_ally = sum(
                    1
                    for candidate in self.chars
                    if candidate.team == char.team and candidate.is_alive
                )
                alive_enemies = sum(
                    1
                    for candidate in self.chars
                    if candidate.team != char.team and candidate.is_alive
                )
                return (
                    char.is_alive
                    and int(getattr(char, "round_kills", 0)) >= 1
                    and alive_ally < alive_enemies
                )
            except (TypeError, ValueError):
                return False

        if condition == "smoke_thrown":
            # このTickに誰か(味方・敵問わず)がスモークを使用した瞬間に発動する。
            return char.is_alive and bool(
                getattr(self, "smoke_thrown_this_tick", False)
            )

        if condition == "iron_will_triggered":
            # このTickに「気合の鉢巻」(HP1で耐える効果)が発動した瞬間に発動する。
            return char.is_alive and bool(
                getattr(char, "iron_will_triggered_this_tick", False)
            )

        if condition == "own_charge_depleted":
            # 自身のロール対応アビリティのチャージが0の間、発動し続ける。
            charge_attr = {
                "SMOKE": "smoke_charges",
                "FLASH": "flash_charges",
                "RECON": "recon_charges",
            }.get(char.ability_name)
            if charge_attr is None:
                return False
            return char.is_alive and getattr(char, charge_attr, 0) <= 0

        if condition == "enemy_in_straight_line":
            # 自身のマスから上下左右方向(直線)へcondition_value以内の敵がいれば発動。
            try:
                max_distance = int(value)
            except (TypeError, ValueError):
                max_distance = 3
            if not char.is_alive:
                return False
            for other in self.chars:
                if other.team == char.team or not other.is_alive:
                    continue
                dr = other.pos[0] - char.pos[0]
                dc = other.pos[1] - char.pos[1]
                in_line = (dr == 0 and 0 < abs(dc) <= max_distance) or (
                    dc == 0 and 0 < abs(dr) <= max_distance
                )
                if in_line and self.check_line_of_sight(char, other):
                    return True
            return False

        if condition == "escapefromthebattle":
            # このTickに射手/標的として交戦し、なおかつ生存している場合に発動。
            # (命中・回避・被弾いずれでも「交戦した」とみなす。倒されていれば False)
            if not char.is_alive:
                return False
            last_shots = getattr(self, "last_shots", None) or []
            return any(
                shot.get("shooter") is char or shot.get("target") is char
                for shot in last_shots
            )

        # end
        return False

    def _handle_awakening_kill(self, killer):
        """キルをトリガーにした覚醒イベントを処理する。"""
        for event in AWAKENING_EVENTS:
            if not isinstance(event, dict) or event.get("condition") != "on_kill":
                continue
            if str(event.get("player", "")) != str(getattr(killer, "base_name", "")):
                continue

            enemies = [
                char
                for char in self.chars
                if char.team != killer.team and char.is_alive
            ]
            if not enemies:
                continue
            effect = event.get("kill_effect")
            if effect == "random_enemy_stop":
                target = random.choice(enemies)
                target.movement_disabled_remaining = max(
                    target.movement_disabled_remaining,
                    int(event.get("duration_ticks", 5)),
                )
            elif effect == "nearest_enemy_reveal":
                target = min(
                    enemies,
                    key=lambda char: (
                        max(
                            abs(char.pos[0] - killer.pos[0]),
                            abs(char.pos[1] - killer.pos[1]),
                        ),
                        char.name,
                    ),
                )
                target.reveal_remaining = max(
                    target.reveal_remaining,
                    int(event.get("duration_ticks", 5)) + 1,
                )
            self._enqueue_triggered_awakening_announcement(event, killer)

    def _handle_awakening_round_win(self, winning_team):
        """ラウンド勝利をトリガーにした覚醒イベントを処理する。"""
        winners = [char for char in self.chars if char.team == winning_team]
        for event in AWAKENING_EVENTS:
            if not isinstance(event, dict) or event.get("condition") != "on_round_win":
                continue
            if not any(
                str(getattr(char, "base_name", "")) == str(event.get("player", ""))
                for char in winners
            ):
                continue
            enemies = [char for char in self.chars if char.team != winning_team]
            if (
                not enemies
                or event.get("round_win_effect") != "random_enemy_mental_down"
            ):
                continue
            target = random.choice(enemies)
            amount = float(event.get("mental_delta", -1))
            debuffs = getattr(self, "awakening_mental_debuffs", None)
            if debuffs is None:
                self.awakening_mental_debuffs = {}
                debuffs = self.awakening_mental_debuffs
            debuffs[target.base_name] = max(
                0.0,
                float(debuffs.get(target.base_name, 0.0)) - min(0.0, amount),
            )
            target.mental = max(0.0, target.mental + amount)
            self._enqueue_triggered_awakening_announcement(
                event,
                next(
                    char
                    for char in winners
                    if str(getattr(char, "base_name", ""))
                    == str(event.get("player", ""))
                ),
            )

    def _enqueue_triggered_awakening_announcement(self, event, char):
        """条件判定を経由しない覚醒効果も、通常の覚醒パネルへ表示する。"""
        enqueue = getattr(self, "_enqueue_announcement", None)
        if not callable(enqueue):
            return
        enqueue(
            {
                "type": "awakening",
                "name": str(event.get("name", "名称未設定の覚醒")),
                "team": char.team,
                "players": (char.base_name,),
                "display_players": (char.display_name,),
                "effect_text": str(event.get("effect_text", "特殊効果")),
            }
        )

    def _apply_round_awakening_state(self):
        """ラウンド再生成後に、覚醒イベントの引継ぎ状態を反映する。"""
        debuffs = getattr(self, "awakening_mental_debuffs", {})
        for char in self.chars:
            char.mental = max(
                0.0,
                char.mental - float(debuffs.get(char.base_name, 0.0)),
            )

    def _check_awakening_events(self):
        for event in AWAKENING_EVENTS:
            if not isinstance(event, dict):
                continue
            player = str(event.get("player", ""))
            event_name = str(event.get("name", "名称未設定の覚醒"))
            char = next((c for c in self.chars if c.base_name == player), None)
            if char is None:
                continue

            if event_name in char.active_awakenings:
                # refreshable指定があれば、発動中でも条件を再度満たした時点で
                # タイマーをdurationへ再セットする（スナップショット・効果は再適用しない）。
                if event.get("refreshable") and self._awakening_condition_met(
                    event, char, event_name
                ):
                    try:
                        refreshed_duration = int(
                            event.get(
                                "duration_ticks", _DEFAULT_AWAKENING_DURATION_TICKS
                            )
                        )
                    except (TypeError, ValueError):
                        refreshed_duration = _DEFAULT_AWAKENING_DURATION_TICKS
                    char.active_awakenings[event_name]["ticks_remaining"] = max(
                        1, refreshed_duration
                    )
                continue

            if not self._awakening_condition_met(event, char, event_name):
                continue

            snapshot = _snapshot_character_awakening_state(char)

            preset = event.get("transform_to")
            if preset:
                self._apply_awakening_preset(char, str(preset))
            new_role = event.get("role")
            if new_role:
                char.role = str(new_role)

                role_to_ability = {
                    "フラッシュ": "FLASH",
                    "スモーカー": "SMOKE",
                    "シーカー": "RECON",
                    "タイガー": "HUNT",
                }

                char.ability_name = role_to_ability.get(
                    char.role,
                    char.ability_name,
                )
                char.hunter_active = char.role == "タイガー"

                char.smoke_charges = 1 if char.ability_name == "SMOKE" else 0
                char.flash_charges = 1 if char.ability_name == "FLASH" else 0
                char.recon_charges = 1 if char.ability_name == "RECON" else 0
            bonuses = event.get("bonuses", {})
            if isinstance(bonuses, dict):
                for key, value in bonuses.items():
                    _apply_combo_bonus(char, key, value)
            if event.get("grants_smoke_vision"):
                char.sees_through_smoke = True
            # Stormfrontのアビリティ数増加処理
            if event.get("abilities_per_round"):
                char.abilities_per_round = event["abilities_per_round"]
                # アビリティチャージを増やす
                if char.ability_name == "SMOKE":
                    char.smoke_charges = event["abilities_per_round"]
                elif char.ability_name == "FLASH":
                    char.flash_charges = event["abilities_per_round"]
                elif char.ability_name == "RECON":
                    char.recon_charges = event["abilities_per_round"]
            # Stormfrontのウルトコスト減少処理
            if event.get("ult_cost_reduction"):
                char.ultimate_cost = max(
                    1, char.ultimate_cost - event["ult_cost_reduction"]
                )
            if event.get("rename"):
                char.display_name = str(event["rename"])

            try:
                duration = int(
                    event.get("duration_ticks", _DEFAULT_AWAKENING_DURATION_TICKS)
                )
            except (TypeError, ValueError):
                duration = _DEFAULT_AWAKENING_DURATION_TICKS
            char.active_awakenings[event_name] = {
                "ticks_remaining": max(1, duration),
                "snapshot": snapshot,
            }
            char.awakening_activation_counts[event_name] = (
                char.awakening_activation_counts.get(event_name, 0) + 1
            )
            char.active_awakening = event_name

            # 覚醒した瞬間に、プレイヤーコンボと同じ上部パネルへ表示する。
            # 同一Tickに複数人が覚醒した場合も、追加された順に3Tickずつ表示される。
            effect_text = str(
                event.get("effect_text") or self._describe_bonuses(bonuses)
            )
            self._enqueue_announcement(
                {
                    "type": "awakening",
                    "name": event_name,
                    "team": char.team,
                    "players": (char.base_name,),
                    "display_players": (char.display_name,),
                    "effect_text": effect_text,
                }
            )

    def _apply_round_start_effects(self):
        """毎ラウンド開始時に、キャラクター固有のラウンド開始効果を適用する。"""
        for char in self.chars:
            if not char.is_alive:
                continue
            # Deepの毎ラウンドウルト満タン効果
            if getattr(char, "start_round_with_full_ult", False):
                char.ultimate_points = char.ultimate_cost
            # Stormfrontの毎ラウンドアビリティ2つ所持効果
            abilities_per_round = getattr(char, "abilities_per_round", 1)
            if char.ability_name == "SMOKE":
                char.smoke_charges = abilities_per_round
            elif char.ability_name == "FLASH":
                char.flash_charges = abilities_per_round
            elif char.ability_name == "RECON":
                char.recon_charges = abilities_per_round

    def _maybe_trigger_leap_awakening(self, shooter, target):
        """leap_on_kill指定の覚醒が有効な撃破時、相手の位置へ移動し
        キャラ自身のアビリティとは別枠でその場にスモークを焚く。
        1ラウンドにつき1度だけ発動する。"""
        active = getattr(shooter, "active_awakenings", None)
        if not active or not shooter.is_alive:
            return

        used_names = getattr(shooter, "_leap_awakenings_used", None)
        if used_names is None:
            used_names = set()
            shooter._leap_awakenings_used = used_names

        for event in AWAKENING_EVENTS:
            if not isinstance(event, dict) or not event.get("leap_on_kill"):
                continue
            event_name = str(event.get("name", ""))
            if event_name not in active or event_name in used_names:
                continue

            used_names.add(event_name)
            impact = (int(target.pos[0]), int(target.pos[1]))
            shooter.pos = [impact[0], impact[1]]
            cells = {
                (rr, cc)
                for rr in range(impact[0] - 1, impact[0] + 2)
                for cc in range(impact[1] - 1, impact[1] + 2)
                if 0 <= rr < self.height
                and 0 <= cc < self.width
                and self.grid[rr, cc] != 1
            }
            self.smokes.append(
                {
                    "cells": cells,
                    "remaining_ticks": SMOKE_DURATION_TICKS,
                    "owner": shooter.name,
                }
            )

    def _advance_timed_awakenings(self):
        """Tick制限のある覚醒を1Tick進め、切れたら発動前の状態へ戻す。"""
        for char in self.chars:
            active = getattr(char, "active_awakenings", None)
            if not active:
                continue
            expired_names = []
            for event_name, state in active.items():
                state["ticks_remaining"] -= 1
                if state["ticks_remaining"] <= 0:
                    expired_names.append(event_name)
            for event_name in expired_names:
                state = active.pop(event_name)
                _restore_character_awakening_state(char, state["snapshot"])
                event_def = next(
                    (
                        e
                        for e in AWAKENING_EVENTS
                        if isinstance(e, dict) and str(e.get("name")) == event_name
                    ),
                    None,
                )
                if event_def and event_def.get("recharge_on_expire"):
                    charge_attr = {
                        "SMOKE": "smoke_charges",
                        "FLASH": "flash_charges",
                        "RECON": "recon_charges",
                    }.get(char.ability_name)
                    if charge_attr:
                        setattr(
                            char,
                            charge_attr,
                            min(1, getattr(char, charge_attr, 0) + 1),
                        )
            char.active_awakening = next(iter(active), None)
