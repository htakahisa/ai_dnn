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
        "bonuses": {"accuracy": 0.25, "hs_rate": 0.25, "dodge_rate": 0.15},
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
        "bonuses": {"accuracy": 0.35, "hs_rate": 0.05, "reaction": 40},
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
        "bonuses": {"hs_rate": 0.55, "reaction": 45, "accuracy": 0.35, "iq": -100},
    },
    {
        "name": "現実に目が覚めた",
        "player": "nAts",
        "condition": "specific_player_dead",
        "condition_player": "Chronicle",
        "rename": "-裏切り者-nAts",
        "bonuses": {"hs_rate": 0.25, "reaction": 35, "accuracy": 0.65, "iq": -100},
    },
    {
        "name": "気に入ってくれると嬉しいな",
        "player": "Furina",
        "condition": "overtime",
        "rename": "-鏡の中の僕-Focalors",
        "bonuses": {
            "hs_rate": 0.5,
            "reaction": 50,
            "accuracy": 0.15,
            "dodge_rate": 0.45,
        },
    },
    {
        "name": "クロネコのいたずら",
        "player": "Nanasaki",
        "condition": "hp_at_or_below",
        "condition_value": 30,
        "rename": "-クロネコ-Nanasaki",
        "bonuses": {"accuracy": 0.3, "hs_rate": 0.1, "reaction": 40},
    },
    {
        "name": "老い耄れを見たら生き残りと思え",
        "player": "Ethan",
        "condition": "overtime",
        "rename": "-老兵-Ethan",
        "bonuses": {"accuracy": 0.1, "hs_rate": 0.3, "reaction": 40},
    },
    {
        "name": "夜の始まり",
        "player": "Derke",
        "condition": "overtime",
        "rename": "-ミッドナイトの化け物-Derke",
        "bonuses": {"hs_rate": 0.3, "reaction": 50},
        "role": "フラッシュ",
    },
    {
        "name": "もうあきた。",
        "player": "Canezerra",
        "condition": "overtime",
        "rename": "-問題児-Canezerra",
        "bonuses": {"hs_rate": 0.4, "reaction": 40},
    },
    {
        "name": "ハゲが二人",
        "player": "Sayonara",
        "condition": "specific_player_dead",
        "condition_player": "Derke",
        "rename": "-トキシック-Sayonara",
        "bonuses": {"hs_rate": 0.3, "accuracy": 0.3, "dodge_rate": 0.2},
    },
    {
        "name": "倒したで",
        "player": "Tortlilyan",
        "condition": "kills_at_least",
        "condition_value": 1,
        "rename": "-調子にのる-とうやま",
        "bonuses": {"hs_rate": 0.20, "dodge_rate": 0.20},
    },
    {
        "name": "急になにー!?",
        "player": "Tortlilyan",
        "condition": "whathappend",
        "condition_value": 1,
        "rename": "-焦る-とうやま",
        "bonuses": {
            "hs_rate": 0.77,
            "dodge_rate": 0.77,
            "accuracy": 0.07,
            "reaction": 77,
        },
    },
]


# 既存の表示処理との互換性を保つため、読み込み時にeffect_textを補完します。
apply_awakening_effect_texts(AWAKENING_EVENTS)
