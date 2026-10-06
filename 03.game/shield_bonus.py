"""Shared shield bonus keys and operation validation for combat and descriptions."""

import math


SHIELD_STAT_ALIASES = {
    "shield_hp": "shield_hp",
    "shield": "shield_hp",
    "シールド": "shield_hp",
    "シールドhp": "shield_hp",
    "shield_piercer": "shield_piercer",
    "シールドピアサー": "shield_piercer",
    "shield_crash": "shield_crash",
    "shield_crush": "shield_crash",
    "シールドクラッシュ": "shield_crash",
}


def shield_stat_key(key):
    return SHIELD_STAT_ALIASES.get(str(key).strip().lower().replace(" ", ""))


def parse_shield_bonus(stat, value):
    """Return (add/set, value), or None for an invalid shield operation.

    Numeric shorthand adds HP; {"set": HP} replaces the capacity/damage.
    Piercing is a boolean setting, never a numeric increment.
    """
    if stat not in ("shield_hp", "shield_crash", "shield_piercer"):
        return None
    operation = "add"
    if isinstance(value, dict):
        if len(value) != 1:
            return None
        operation, value = next(iter(value.items()))
        if operation not in ("add", "set"):
            return None
    elif stat == "shield_piercer":
        operation = "set"
    if stat == "shield_piercer":
        return ("set", value) if operation == "set" and isinstance(value, bool) else None
    if isinstance(value, bool):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return (operation, amount) if math.isfinite(amount) else None
