"""Player abilities and visible status, with explicit unknown-state masks."""
import math
import numpy as np
from game_core import get_character_combat_stats, ULTIMATE_NAMES, ULTIMATE_COSTS, RAMP_ELECTRIC_TICKS

ABILITIES = ("RECON", "FLASH", "SMOKE", "RAMP", "DANCE", "ASH", "HUNT", "UNKNOWN")
ULTIMATES = ("MONITOR", "TUNNEL", "ESCAPE", "NEON", "SERENADE", "BALEMOON", "RAID", "UNKNOWN")
ROLE_ABILITIES = {"タイガー": "HUNT", "スモーカー": "SMOKE", "シーカー": "RECON",
                  "フラッシュ": "FLASH", "エンジニア": "RAMP", "アイドル": "DANCE", "コントラクター": "ASH"}
CHARGES = ("recon_charges", "flash_charges", "smoke_charges", "ramp_charges", "dance_charges", "ash_charges")
BASE_STATS = (("shield_hp", 100), ("shield_piercer", 1), ("shield_crash", 100),
              ("erosion_curse", 10), ("fate_loom", 10), ("ultimate_cost", 10))
LIVE_STATS = (("hp", 100), ("max_hp", 100), ("shield_hp", 100), ("max_shield_hp", 100),
              ("shield_piercer", 1), ("shield_crash", 100), ("erosion_curse", 10), ("fate_loom", 10),
              *((key, 3) for key in CHARGES), ("ultimate_points", 10), ("ultimate_cost", 10),
              ("blind_remaining", 15), ("reveal_remaining", 15), ("electric_remaining", 5),
              ("life_contract_remaining", 10), ("ability_seal_remaining", 10),
              ("fate_loom_remaining", 10), ("movement_disabled_remaining", 10),
              ("contract_max_hp_lost", 100), ("fate_max_hp_lost", 100),
              ("move_steps_per_tick", 3), ("hunter_active", 1), ("iron_will_charges", 1),
              ("accuracy", 2), ("hs_rate", 2), ("dodge_rate", 1), ("reaction", 200),
              ("effective_iq", 200), ("sees_through_smoke", 1),
              ("shield_abilities_enabled", 1))
UNIT_FIELDS = ("present", "is_self", "alive", *("ability_" + key for key in ABILITIES),
               *("ultimate_" + key for key in ULTIMATES), *("base_" + key for key, _ in BASE_STATS),
               "status_known", *(key for key, _ in LIVE_STATS), "awakening_active",
               "can_move", "can_use_ability", "can_use_ultimate")
SELF_FIELDS = ("can_move", "can_use_ability", "can_use_ultimate", "can_turn", "ability_sealed",
               "rooted", "action_frozen", "escape_channeling", "ability_seal_remaining",
               "fate_loom_remaining", "movement_disabled_remaining", "electric_remaining",
               "escape_remaining", "shield_abilities_enabled")
UNIT_WIDTH = len(UNIT_FIELDS)
UNIT_BLOCKS = (("player_abilities_status", 10 * UNIT_WIDTH), ("self_capabilities", len(SELF_FIELDS)))


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.
    return max(0., min(1e6, value)) if math.isfinite(value) else 0.


def action_capabilities(char, state):
    """Engine status permissions for the next action phase, before geometry.

    RAMP's final displayed tick already permits ordinary movement. Awakening
    freeze and an ESCAPE channel skip the entire controller action phase.
    """
    alive = bool(getattr(char, "is_alive", False))
    portals = [p for p in state.get("ally_escape_portals", ()) if p.get("owner") == char.name]
    channeling = bool(portals) or char.name in state.get("visible_escape_channel_owners", ())
    escape_remaining = max((_number(p.get("remaining_ticks", 0)) for p in portals), default=0.)
    frozen = _number(getattr(char, "movement_disabled_remaining", 0)) > 0
    phase_allowed = alive and not frozen and not channeling
    sealed = _number(getattr(char, "ability_seal_remaining", 0)) > 0
    rooted = _number(getattr(char, "fate_loom_remaining", 0)) > 0
    electric = _number(getattr(char, "electric_remaining", 0)) > 0
    applied = getattr(char, "electric_applied_tick", None)
    electric_blocks = electric and (applied is None or
        state.get("battle_tick", 0) + 1 < applied + RAMP_ELECTRIC_TICKS)
    can_move = phase_allowed and not rooted and not electric_blocks
    casting_allowed = phase_allowed and not sealed
    ability = getattr(char, "ability_name", "UNKNOWN")
    can_ability = casting_allowed and ability in ABILITIES[:-2] and _number(
        getattr(char, ability.lower() + "_charges", 0)) > 0
    ultimate = getattr(char, "ultimate_name", "UNKNOWN")
    can_ultimate = (casting_allowed and ultimate in ULTIMATES[:-1]
        and ultimate != "SERENADE"
        and _number(getattr(char, "ultimate_points", 0)) >= _number(getattr(char, "ultimate_cost", 1))
        and (ultimate not in ("RAID", "ESCAPE") or can_move))
    if ultimate == "SERENADE":
        can_ultimate = (not alive and not sealed and not state.get("round_over", False)
            and not state.get("match_over", False)
            and all(any(unit.team == team and unit.is_alive for unit in state.get("chars", ()))
                    for team in ("A", "D"))
            and _number(getattr(char, "ultimate_points", 0)) >= _number(getattr(char, "ultimate_cost", 1)))
    return dict(can_move=bool(can_move), can_use_ability=bool(can_ability),
                can_use_ultimate=bool(can_ultimate), can_act=bool(phase_allowed),
                can_turn=bool(phase_allowed and not getattr(char, "facing_forced_this_tick", False)),
                ability_sealed=bool(sealed), rooted=bool(rooted), action_frozen=bool(frozen),
                escape_channeling=bool(channeling), escape_remaining=escape_remaining,
                casting_allowed=bool(casting_allowed))


def self_capabilities(char, state):
    values = action_capabilities(char, state)
    for key in ("ability_seal_remaining", "fate_loom_remaining", "movement_disabled_remaining",
                "electric_remaining"):
        values[key] = _number(getattr(char, key, 0)) / 10
    values["escape_remaining"] /= 10
    values["shield_abilities_enabled"] = bool(getattr(char, "shield_abilities_enabled", True))
    return np.array([values[key] for key in SELF_FIELDS], dtype=np.float32)


def _slots(char, state, team):
    units = {str(unit.name): unit for unit in state.get("chars", ())
             if (unit.team == char.team) == (team == "ally")}
    entries = {}
    for public in state.get(team + "_roster", ()):
        entry = dict(public) if isinstance(public, dict) else dict(name=str(public), base_name=public)
        name = str(entry.get("name", entry.get("base_name", "")))
        if not entry.get("stats"):
            entry["stats"] = get_character_combat_stats(entry.get("base_name", name))
        entries[name] = entry
    # Missing runtime units keep their roster slot; never infer their live state.
    for name, unit in units.items():
        if name not in entries:
            base_name = getattr(unit, "base_name", unit.name)
            entries[name] = dict(name=name, stats=get_character_combat_stats(base_name))
    for name in sorted(entries)[:5]:
        yield name, entries[name], units.get(name)


def player_abilities_status(char, state):
    """Five allies then five enemies, sorted by runtime name in each team.

    Base traits/types are public roster information. Live shield, charges,
    debuffs, and awakening data are zero with status_known=0 for unseen enemies.
    Nothing is read from hidden live attributes to build their static features.
    """
    result = np.zeros((2, 5, UNIT_WIDTH), dtype=np.float32)
    for team_index, team in enumerate(("ally", "enemy")):
        for row, (name, entry, unit) in zip(result[team_index], _slots(char, state, team)):
            stats = entry.get("stats", {})
            role = entry.get("role", stats.get("role", "UNKNOWN"))
            known = unit is not None and (team == "ally" or bool(getattr(unit, "position_known", False)))
            ability = (getattr(unit, "ability_name", "UNKNOWN") if known else
                       entry.get("ability_name", ROLE_ABILITIES.get(role, "UNKNOWN")))
            ultimate = (getattr(unit, "ultimate_name", "UNKNOWN") if known else
                        entry.get("ultimate_name", ULTIMATE_NAMES.get(role, "UNKNOWN")))
            values = [1., name == str(char.name) and team == "ally",
                      bool(getattr(unit, "is_alive", False)) if unit is not None else 0.]
            values += [ability == key if ability in ABILITIES else key == "UNKNOWN" for key in ABILITIES]
            values += [ultimate == key if ultimate in ULTIMATES else key == "UNKNOWN" for key in ULTIMATES]
            values += [_number(entry.get(key, ULTIMATE_COSTS.get(role, 0)) if key == "ultimate_cost"
                               else stats.get(key, 0)) / scale for key, scale in BASE_STATS]
            values += [known]
            if known:
                values += [_number(getattr(unit, key, 1 if key in ("move_steps_per_tick", "shield_abilities_enabled") else 0)) / scale
                           for key, scale in LIVE_STATS]
                capabilities = action_capabilities(unit, state)
                values += [bool(getattr(unit, "active_awakenings", None) or getattr(unit, "active_awakening", None)),
                           capabilities["can_move"], capabilities["can_use_ability"], capabilities["can_use_ultimate"]]
            else:
                values += [0.] * (len(LIVE_STATS) + 4)
            row[:] = values
    return result.reshape(-1)
