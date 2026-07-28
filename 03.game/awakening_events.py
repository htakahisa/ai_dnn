"""覚醒イベント定義。

effect_text は任意です。
省略した場合、覚醒条件 / rename / transform_to / bonuses から自動生成されます。
"""

from effect_text_generator import apply_awakening_effect_texts

AWAKENING_EVENTS = [
    {
        "name": "最後には彼がいる",
        "player": "Leo",
        "condition": "all_allies_dead",
        "rename": "-ラスボス-Leo",
        "bonuses": {"accuracy": 0.15, "hs_rate": 0.1, "dodge_rate": 0.1},
    },
    {
        "name": "彼を起こしてはいけない",
        "player": "Demon1",
        "condition": "hp_at_or_below",
        "condition_value": 30,
        "rename": "-地獄の王者-Demon1",
        "bonuses": {"accuracy": 0.1, "hs_rate": 0.15, "reaction": 35},
    },
    {
        "name": "楽になれると思うなよ",
        "player": "Lohen",
        "condition": "kills_at_least",
        "condition_value": 3,
        "rename": "-戦闘狂-Lohen",
        "bonuses": {"accuracy": 0.35, "reaction": 20},
    },
    {
        "name": "邪魔者の排除",
        "player": "Arlecchino",
        "condition": "specific_player_dead",
        "condition_player": "Lohen",
        "rename": "-お父様-Arlecchino",
        "bonuses": {"hs_rate": 0.2, "dodge_rate": 0.25},
    },
    {
        "name": "怪物が目覚めた",
        "player": "Chronicle",
        "condition": "specific_player_dead",
        "condition_player": "nAts",
        "rename": "-王子様-Chronicle",
        "bonuses": {"hs_rate": 0.55, "reaction": 45, "accuracy": 0.3, "iq": -100},
    },
    {
        "name": "現実に目が覚めた",
        "player": "nAts",
        "condition": "specific_player_dead",
        "condition_player": "Chronicle",
        "rename": "-裏切り者-nAts",
        "bonuses": {"hs_rate": 0.25, "reaction": 35, "accuracy": 0.6, "iq": -100},
    },
    # {
    # "name": "気に入ってくれると嬉しいな",
    # "player": "Furina",
    # "condition": "all_allies_dead",
    # "condition_player": "Furina",
    # "rename": "-鏡の中の僕-Focalors",
    # "bonuses": {
    # "hs_rate": 0.35,
    # "reaction": 50,
    # "accuracy": 0.15,
    # "dodge_rate": 0.3,
    # },
    # },
    {
        "name": "",
        "player": "Nanasaki",
        "condition": "all_allies_dead",
        "condition_player": "Furina",
        "rename": "-鏡の中の僕-Focalors",
        "bonuses": {
            "hs_rate": 0.35,
            "reaction": 50,
            "accuracy": 0.1,
            "dodge_rate": 0.2,
        },
    },
]


# 既存の表示処理との互換性を保つため、読み込み時にeffect_textを補完します。
apply_awakening_effect_texts(AWAKENING_EVENTS)
