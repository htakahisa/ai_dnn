"""コンボ・覚醒イベントの表示テキスト自動生成。"""

from numbers import Real

RATE_STATS = {"accuracy", "hs_rate", "dodge_rate"}

STAT_LABELS = {
    "accuracy": "Hit率",
    "hs_rate": "HS率",
    "dodge_rate": "回避率",
    "max_hp": "最大HP",
    "reaction": "反応速度",
}

CONDITION_LABELS = {
    "all_allies_dead": "自分以外の味方が全滅",
    "hp_at_or_below": "HP{condition_value}以下",
    "kills_at_least": "{condition_value}キル達成",
    "specific_player_dead": "{condition_player}が死亡",
    "specific_player_killed": "{condition_player}を撃破",
    "team_kills_at_least": "チーム合計{condition_value}キル達成",
    "enemy_count_at_or_below": "生存している敵が{condition_value}人以下",
}


def _number_text(value):
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def format_stat_bonus(stat_name, value):
    label = STAT_LABELS.get(stat_name, stat_name)

    if stat_name in RATE_STATS:
        amount = round(float(value) * 100)
        sign = "+" if amount > 0 else ""
        return f"{label} {sign}{amount}ポイント"

    sign = "+" if float(value) > 0 else ""
    return f"{label} {sign}{_number_text(value)}"


def format_bonus_lines(bonuses):
    if not bonuses:
        return []

    return [
        format_stat_bonus(stat_name, value)
        for stat_name, value in bonuses.items()
        if isinstance(value, Real)
    ]


def format_awakening_condition(event):
    condition = event.get("condition", "")
    template = CONDITION_LABELS.get(condition)

    if template is None:
        return condition or "条件達成"

    try:
        return template.format(**event)
    except KeyError:
        return condition


def _compact_bonus_text(bonuses):
    """コンボ告知用に、能力補正を短い1行へ変換します。"""
    return " / ".join(format_bonus_lines(bonuses))


def generate_combo_effect_text(combo):
    """コンボ効果を必ず1行で生成します。

    既存のコンボ告知欄は高さが1行分なので、改行を入れると
    ATTACKERS表示やマップへ重なります。
    """
    manual = combo.get("effect_text")
    if manual:
        # 手動文にも改行が含まれていたら、表示崩れ防止のため1行化します。
        return " / ".join(str(manual).splitlines())

    sections = []

    common_text = _compact_bonus_text(combo.get("bonuses"))
    if common_text:
        sections.append(common_text)

    for player_name, bonuses in (combo.get("player_bonuses") or {}).items():
        bonus_text = _compact_bonus_text(bonuses)
        if bonus_text:
            sections.append(f"{player_name}: {bonus_text}")

    for old_name, new_name in (combo.get("renames") or {}).items():
        sections.append(f"{old_name}→{new_name}")

    return " / ".join(sections) if sections else "能力補正なし"


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