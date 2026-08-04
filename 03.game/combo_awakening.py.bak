"""Player combo, awakening, and announcement-queue behavior."""

from game_core import (
    PLAYER_COMBOS, AWAKENING_EVENTS, COMBO_DISPLAY_TICKS,
    _character_stats, _clamp_rate, _canonical_combo_stat_key, _apply_combo_bonus,
)

class ComboAwakeningMixin:
    def _apply_player_combos(self):
        """同じチームに必要メンバーが全員いるコンボを発動する。"""
        self.active_player_combos = []
        for team in ("A", "D"):
            team_chars = [char for char in self.chars if char.team == team]
            chars_by_name = {char.name: char for char in team_chars}
            team_names = set(chars_by_name)

            for combo in PLAYER_COMBOS:
                if not isinstance(combo, dict):
                    continue
                combo_name = str(combo.get("name", "名称未設定コンボ"))
                required_players = tuple(str(name) for name in combo.get("players", ()))
                if not required_players or not set(required_players).issubset(team_names):
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
                    own_bonuses = per_player_bonuses.get(player_name, {}) if isinstance(per_player_bonuses, dict) else {}
                    if isinstance(own_bonuses, dict):
                        for stat_key, value in own_bonuses.items():
                            _apply_combo_bonus(char, stat_key, value)
                    if isinstance(renames, dict) and player_name in renames:
                        char.display_name = str(renames[player_name])
                    if combo_name not in char.active_combos:
                        char.active_combos.append(combo_name)
                    affected.append(player_name)

                self.active_player_combos.append({
                    "type": "combo",
                    "name": combo_name,
                    "team": team,
                    "players": tuple(affected),
                    "display_players": tuple(chars_by_name[n].display_name for n in affected),
                    "effect_text": str(combo.get("effect_text") or self._describe_bonuses(common_bonuses, per_player_bonuses)),
                })


    def _describe_bonuses(self, common_bonuses, per_player_bonuses=None):
        labels = {
            "accuracy": "Hit%",
            "hs_rate": "HS%",
            "dodge_rate": "回避率",
            "reaction": "反応速度",
            "iq": "IQ",
            "max_hp": "最大HP",
        }
        parts = []
        if isinstance(common_bonuses, dict):
            for key, value in common_bonuses.items():
                canonical = _canonical_combo_stat_key(key)
                if canonical:
                    amount = float(value)
                    if canonical in ("accuracy", "hs_rate", "dodge_rate") and abs(amount) <= 1:
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
        char.accuracy = _clamp_rate(data.get("hit_pct", data.get("accuracy")), char.accuracy)
        char.hs_rate = _clamp_rate(data.get("hs_pct", data.get("hs_rate")), char.hs_rate)
        char.dodge_rate = _clamp_rate(data.get("dodge_pct", data.get("dodge_rate")), char.dodge_rate)
        try:
            char.iq = float(data.get("iq", data.get("IQ", char.iq)))
        except (TypeError, ValueError):
            pass
        try:
            char.reaction = float(
                data.get("reaction", data.get("reaction_speed", data.get("反応速度", char.reaction)))
            )
        except (TypeError, ValueError):
            pass
        role = str(data.get("role", char.role))
        char.role = role
        char.ability_name = {"フラッシュ":"FLASH", "スモーカー":"SMOKE", "シーカー":"RECON", "タイガー":"HUNT"}.get(role, char.ability_name)
        char.hunter_active = role == "タイガー"
        char.smoke_charges = 1 if char.ability_name == "SMOKE" else 0
        char.flash_charges = 1 if char.ability_name == "FLASH" else 0
        char.recon_charges = 1 if char.ability_name == "RECON" else 0


    def _awakening_condition_met(self, event, char):
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
        """
        condition = str(event.get("condition", "")).strip()
        value = event.get("condition_value")

        if condition == "all_allies_dead":
            allies = [
                c for c in self.chars
                if c.team == char.team and c is not char
            ]
            return (
                char.is_alive
                and bool(allies)
                and all(not c.is_alive for c in allies)
            )

        if condition == "hp_at_or_below":
            try:
                return char.is_alive and char.hp <= float(value)
            except (TypeError, ValueError):
                return False

        if condition == "kills_at_least":
            try:
                return (
                    char.is_alive
                    and int(getattr(char, "round_kills", 0)) >= int(value)
                )
            except (TypeError, ValueError):
                return False

        if condition == "specific_player_dead":
            target_name = str(
                event.get("condition_player")
                or event.get("condition_value")
                or ""
            ).strip()

            if not target_name or not char.is_alive:
                return False

            return any(
                not candidate.is_alive
                and getattr(candidate, "base_name", getattr(candidate, "name", "")) == target_name
                for candidate in self.chars
            )

        if condition == "specific_player_killed":
            target_name = str(
                event.get("condition_player")
                or event.get("condition_value")
                or ""
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

        return False


    def _check_awakening_events(self):
        for event in AWAKENING_EVENTS:
            if not isinstance(event, dict):
                continue
            player = str(event.get("player", ""))
            event_name = str(event.get("name", "名称未設定の覚醒"))
            char = next((c for c in self.chars if c.base_name == player), None)
            if char is None or event_name in char.triggered_awakening_events:
                continue
            if not self._awakening_condition_met(event, char):
                continue
            preset = event.get("transform_to")
            if preset:
                self._apply_awakening_preset(char, str(preset))
            bonuses = event.get("bonuses", {})
            if isinstance(bonuses, dict):
                for key, value in bonuses.items():
                    _apply_combo_bonus(char, key, value)
            if event.get("rename"):
                char.display_name = str(event["rename"])
            char.active_awakening = event_name
            char.triggered_awakening_events.add(event_name)

            # 覚醒した瞬間に、プレイヤーコンボと同じ上部パネルへ表示する。
            # 同一Tickに複数人が覚醒した場合も、追加された順に3Tickずつ表示される。
            effect_text = str(event.get("effect_text") or self._describe_bonuses(bonuses))
            self._enqueue_announcement({
                "type": "awakening",
                "name": event_name,
                "team": char.team,
                "players": (char.base_name,),
                "display_players": (char.display_name,),
                "effect_text": effect_text,
            })
