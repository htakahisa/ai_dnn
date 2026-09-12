"""player_combos.pyを学習環境でも実戦と同じ形で適用する。"""

try:
    from player_combos import COMBOS
except ImportError:
    from game_core import PLAYER_COMBOS as COMBOS


def build_combo_bonuses(roster):
    roster = {str(name) for name in roster}
    bonuses = {name: {} for name in roster}
    for combo in COMBOS:
        if not isinstance(combo, dict):
            continue
        players = tuple(str(name) for name in combo.get("players", ()))
        if not players or not set(players).issubset(roster):
            continue
        common = combo.get("bonuses", {})
        per_player = combo.get("player_bonuses", {})
        for name in players:
            merged = bonuses[name]
            if isinstance(common, dict):
                for key, value in common.items():
                    merged[key] = merged.get(key, 0.0) + float(value)
            own = per_player.get(name, {}) if isinstance(per_player, dict) else {}
            if isinstance(own, dict):
                for key, value in own.items():
                    merged[key] = merged.get(key, 0.0) + float(value)
    return bonuses

