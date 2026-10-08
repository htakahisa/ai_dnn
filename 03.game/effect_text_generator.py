"""コンボ・覚醒イベントの表示テキスト自動生成。"""

from numbers import Real

from shield_bonus import parse_shield_bonus, shield_stat_key

RATE_STATS = {"accuracy", "hs_rate", "dodge_rate"}

STAT_LABELS = {
    "accuracy": "Hit率",
    "hs_rate": "HS率",
    "dodge_rate": "回避率",
    "max_hp": "最大HP",
    "reaction": "反応速度",
    "mental": "メンタル",
    "condition_bonus": "調子補正",
    "調子補正": "調子補正",
    "shield_hp": "シールド",
    "shield_piercer": "シールドピアサー",
    "shield_crash": "シールドクラッシュ",
    "erosion_curse": "摩耗の呪い",
    "fate_loom": "運命の織機",
}

CONDITION_LABELS = {
    "all_allies_dead": "自分以外の味方が全滅",
    "hp_at_or_below": "HP{condition_value}以下",
    "kills_at_least": "{condition_value}キル達成",
    "specific_player_dead": "{condition_player}が死亡",
    "specific_player_killed": "{condition_player}を撃破",
    "team_kills_at_least": "チーム合計{condition_value}キル達成",
    "enemy_count_at_or_below": "生存している敵が{condition_value}人以下",
    "whathappend": "急になにー!?",
}


def _number_text(value):
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def format_stat_bonus(stat_name, value):
    shield_stat = shield_stat_key(stat_name)
    if shield_stat:
        parsed = parse_shield_bonus(shield_stat, value)
        if parsed is None:
            return ""
        operation, amount = parsed
        label = STAT_LABELS[shield_stat]
        if shield_stat == "shield_piercer":
            return f"{label} {'有効化' if amount else '無効化'}"
        if operation == "set":
            return f"{label}を{_number_text(max(0, amount))}HPに設定"
        return f"{label} {amount:+g}HP"
    label = STAT_LABELS.get(stat_name, stat_name)
    if stat_name in ("erosion_curse", "fate_loom", "摩耗の呪い", "運命の織機"):
        return f"{label} {float(value):+g}tick"

    if stat_name in ("condition_bonus", "調子補正"):
        return f"調子補正 {float(value) * 100:+g}ポイント"

    if stat_name in RATE_STATS:
        amount = round(float(value) * 100)
        sign = "+" if amount > 0 else ""
        return f"{label} {sign}{amount}ポイント"

    sign = "+" if float(value) > 0 else ""
    return f"{label} {sign}{_number_text(value)}"


def format_bonus_lines(bonuses):
    if not bonuses:
        return []

    lines = []
    for stat_name, value in bonuses.items():
        if isinstance(value, Real) or shield_stat_key(stat_name):
            line = format_stat_bonus(stat_name, value)
            if line:
                lines.append(line)
    return lines


def format_awakening_condition(event):
    condition = event.get("condition", "")
    template = CONDITION_LABELS.get(condition)

    if template is None:
        return condition or "条件達成"

    try:
        return template.format(**event)
    except KeyError:
        return condition


def generate_combo_effect_text(combo):
    manual = combo.get("effect_text")
    if manual:
        return str(manual)

    sections = []

    common_lines = format_bonus_lines(combo.get("bonuses"))
    if common_lines:
        sections.append("参加者全員\n" + "\n".join(common_lines))

    for player_name, bonuses in (combo.get("player_bonuses") or {}).items():
        lines = format_bonus_lines(bonuses)
        if lines:
            sections.append(f"{player_name}\n" + "\n".join(lines))

    for old_name, new_name in (combo.get("renames") or {}).items():
        sections.append(f"{old_name}の表示名を「{new_name}」に変更")

    return "\n\n".join(sections) if sections else "能力補正なし"


def generate_awakening_effect_text(event):
    manual = event.get("effect_text")
    if manual:
        return str(manual)

    sections = [format_awakening_condition(event)]

    if event.get("rename"):
        sections.append(f"「{event['rename']}」へ覚醒")

    if event.get("transform_to"):
        sections.append(f"{event['transform_to']}の能力へ変化")

    bonus_lines = format_bonus_lines(event.get("bonuses"))
    if bonus_lines:
        sections.append("\n".join(bonus_lines))

    return "\n\n".join(sections)


def apply_combo_effect_texts(combos):
    for combo in combos:
        combo.setdefault("effect_text", generate_combo_effect_text(combo))


def apply_awakening_effect_texts(events):
    for event in events:
        event.setdefault("effect_text", generate_awakening_effect_text(event))
