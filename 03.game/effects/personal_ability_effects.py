"""設定済みの個人能力だけに発動演出を付ける。パスは03.game基準。"""

import json
from pathlib import Path

PERSONAL_ABILITY_EFFECTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "personal_ability_effects.json"
)


def load_personal_ability_effects():
    try:
        settings = json.loads(PERSONAL_ABILITY_EFFECTS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return settings if isinstance(settings, dict) else {}


def effect_duration_ms(effect):
    try:
        return max(0, int(effect.get("duration_ms", 2000)))
    except (TypeError, ValueError):
        return 2000


class PersonalAbilityEffectsMixin:
    def _queue_personal_ability_effect(self, char, ability_name):
        """各能力の発動境界から呼ぶ共通入口。未設定・高速処理では表示しない。"""
        if getattr(self, "fast_forward_mode", None):
            return
        settings = load_personal_ability_effects()
        character_settings = settings.get(getattr(char, "base_name", char.name), {})
        if not isinstance(character_settings, dict):
            return
        effect = character_settings.get(ability_name)
        if not isinstance(effect, dict) or effect_duration_ms(effect) == 0:
            return
        if self.headless:
            visual_sink = getattr(self, "network_visual_event_sink", None)
            if callable(visual_sink):
                visual_sink({
                    "type": "ABILITY", "name": str(char.name),
                    "ability_name": ability_name, "effect": effect,
                })
                sound_sink = getattr(self, "_emit_network_sound_event", None)
                if callable(sound_sink):
                    sound_sink(effect.get("sound"))
            return
        banner = {"type": "ABILITY", "name": char.name, "ability_name": ability_name,
                  "effect": effect, "animation_elapsed_ms": 0, "sound_started": False}
        if getattr(self, "personal_ability_effect", None) is None:
            self.personal_ability_effect = banner
        else:
            if not hasattr(self, "personal_ability_effect_queue"):
                self.personal_ability_effect_queue = []
            self.personal_ability_effect_queue.append(banner)

    def _prepare_personal_ability_effects(self):
        """通常tickの前に発動を確定し、演出がある場合はtickをまだ進めない。"""
        if self.headless or getattr(self, "fast_forward_mode", None):
            return False
        if getattr(self, "personal_ability_effect", None):
            return True
        from omoko_v1.ov1_ability_cornered_omoko import OMOKO_CORNERED_NAME
        for char in self.chars:
            if char.name != "おもこ":
                continue
            was_triggered = char.omoko_cornered_triggered
            if self._omoko_cornered_active(char) and not was_triggered:
                self._queue_personal_ability_effect(char, OMOKO_CORNERED_NAME)
        return bool(getattr(self, "personal_ability_effect", None))

    def _advance_personal_ability_effect(self):
        self._begin_match_callback()
        if getattr(self, "paused", False) or self.match_over:
            return
        banner = getattr(self, "personal_ability_effect", None)
        if banner is None:
            self.loop()
            return
        elapsed = banner["animation_elapsed_ms"]
        duration = effect_duration_ms(banner["effect"])
        if elapsed >= duration:
            queue = getattr(self, "personal_ability_effect_queue", [])
            self.personal_ability_effect = queue.pop(0) if queue else None
            if self.personal_ability_effect is not None:
                self._advance_personal_ability_effect()
            else:
                self.loop()
            return
        if not banner["sound_started"]:
            self._play_personal_ability_effect_sound(banner["effect"])
            banner["sound_started"] = True
        self.draw()
        self._update_playback_controls()
        delay = min(50, duration - elapsed)
        banner["animation_elapsed_ms"] += delay
        self._schedule_match_callback(self._advance_personal_ability_effect, delay)
