"""プレイヤーコンボ定義。

effect_text は任意です。
省略した場合、bonuses / player_bonuses / renames から自動生成されます。
率系の値は 0.10 = 10ポイントです。
"""

from effect_text_generator import apply_combo_effect_texts

COMBOS = [
    {
        "name": "天才と悪魔",
        "players": ("jawgemo", "Demon1"),
        "bonuses": {"hs_rate": 0.1},
        "renames": {},
    },
    {
        "name": "ソウルフレンド",
        "players": ("F0rsakeN", "Jinggg"),
        "bonuses": {"accuracy": 0.1},
        "renames": {},
    },
    {
        "name": "期待の新人",
        "players": ("Leo", "Chronicle"),
        "bonuses": {"accuracy": 0.1},
        "renames": {},
    },
    {
        "name": "犬猿の仲",
        "players": ("Aspas", "Demon1"),
        "bonuses": {"hs_rate": 0.1},
        "renames": {},
    },
    {
        "name": "世界一の名門",
        "players": ("Alfajer", "Boaster", "Chronicle", "Derke", "Leo"),
        "bonuses": {"hs_rate": 0.05, "accuracy": 0.4, "iq": 5},
        "renames": {},
    },
    {
        "name": "悪魔率いる天才集団",
        "players": ("Demon1", "jawgemo", "Ethan", "Boostio", "C0M"),
        "bonuses": {"reaction": 35, "accuracy": 0.15, "iq": 15},
        "renames": {},
    },
    {
        "name": "ベテランより奇人に",
        "players": ("Ethan", "Boostio"),
        "player_bonuses": {"Ethan": {"accuracy": 0.1}, "Boostio": {"iq": 50}},
        "renames": {},
    },
    {
        "name": "ジャパンホープ",
        "players": ("Laz", "Dep", "Meiy", "SugarZ3ro"),
        "bonuses": {"hs_rate": 0.25},
        "renames": {},
    },
    {
        "name": "レイキャビック戦士",
        "players": ("Laz", "Dep", "SugarZ3ro"),
        "bonuses": {"hs_rate": 0.1, "accuracy": 0.1, "dodge_rate": 0.1},
        "renames": {},
    },
    {
        "name": "クイーンズギャンビット",
        "players": ("nAts", "Chronicle", "Lar0k"),
        "bonuses": {"hs_rate": 0.15, "iq": 15},
        "renames": {},
    },
    {
        "name": "クイーンズフラワー",
        "players": ("leaf", "Lar0k"),
        "bonuses": {"accuracy": 0.15, "reaction": 15},
        "renames": {},
    },
    {
        "name": "フラワーギャンビット",
        "players": ("leaf", "Chronicle", "nAts"),
        "bonuses": {"accuracy": 0.15, "iq": 15},
        "renames": {},
    },
    {
        "name": "黒い破片飛び散る夜空",
        "players": ("Meteor", "Nanasaki"),
        "bonuses": {"hs_rate": 0.25, "accuracy": 0.3, "reaction": 20},
        "renames": {},
    },
    {
        "name": "赤い果実迫りゆく森林",
        "players": ("Zest", "Nanasaki"),
        "bonuses": {"hs_rate": 0.25, "reaction": 50},
        "renames": {},
    },
    {
        "name": "白き珊瑚浮かれ揺られ海洋",
        "players": ("WoohyuN", "Nanasaki"),
        "bonuses": {"hs_rate": 0.25, "dodge_rate": 0.3, "reaction": 20},
        "renames": {},
    },
    {
        "name": "隕石の流血事件",
        "players": ("Meteor", "Zest"),
        "bonuses": {"hs_rate": 0.2, "accuracy": 0.1},
        "renames": {},
    },
    {
        "name": "クソガキ老兵",
        "players": ("Canezerra", "Ethan"),
        "player_bonuses": {
            "Canezerra": {"hs_rate": 0.1, "reaction": 20},
            "Ethan": {"accuracy": 0.1, "dodge_rate": 0.2},
        },
        "renames": {},
    },
    {
        "name": "クロネコのドラゴンテイル",
        "players": ("vo0kashu", "Nanasaki"),
        "bonuses": {"hs_rate": 0.15, "reaction": 35},
        "renames": {},
    },
    {
        "name": "生意気なドラゴンテイル",
        "players": ("Canezerra", "Nanasaki"),
        "bonuses": {"accuracy": 0.15, "hs_rate": 0.25, "reaction": 10},
        "renames": {},
    },
    {
        "name": "北欧仕立てのドラゴンテイル",
        "players": ("Derke", "Nanasaki"),
        "bonuses": {"dodge_rate": 0.15, "accuracy": 0.25, "reaction": 10},
        "renames": {},
    },
    {
        "name": "かつて犠牲になった儚い花",
        "players": ("Lar0k", "leaf", "Chronicle", "nAts", "Sayonara"),
        "bonuses": {"accuracy": 0.15, "hs_rate": 0.15, "iq": 10, "reaction": 10},
        "renames": {},
    },
    {
        "name": "モーツァルト～アイネクライネ～",
        "players": ("FNS", "crashies", "cNed", "soulcas", "trexx"),
        "player_bonuses": {
            "FNS": {"dodge_rate": 0.1, "accuracy": 0.1, "reaction": 10},
            "crashies": {"accuracy": 0.15, "reaction": 15},
            "cNed": {"hs_rate": 0.15, "dodge_rate": 0.15},
            "crashies": {"dodge_rate": 0.3},
            "soulcas": {"accuracy": 0.3},
        },
        "renames": {},
    },
    {
        "name": "夜逃げした裏切者たち",
        "players": ("Derke", "Sayonara"),
        "player_bonuses": {
            "Derke": {"dodge_rate": 0.1, "accuracy": 0.1, "reaction": 10},
            "Sayonara": {"accuracy": 0.15, "reaction": 15},
        },
        "renames": {},
    },
    {
        "name": "何者である必要もないこの夜に踊る",
        "players": ("Derke", "Sayonara", "Lar0k", "marteen", "something"),
        "bonuses": {"dodge_rate": 0.4, "iq": 30},
        "renames": {},
    },
    {
        "name": "モーツァルト～アイネクライネ～",
        "players": ("FNS", "crashies", "cNed", "soulcas", "trexx"),
        "player_bonuses": {
            "FNS": {"iq": 30},
            "crashies": {"accuracy": 0.15, "reaction": 15},
            "cNed": {"hs_rate": 0.15, "dodge_rate": 0.15},
            "crashies": {"dodge_rate": 0.3},
            "soulcas": {"accuracy": 0.3},
        },
        "renames": {},
    },
    {
        "name": "ヴェートーベン～エリーゼのために～",
        "players": ("FNS", "crashies"),
        "player_bonuses": {
            "FNS": {"iq": 20},
            "crashies": {"accuracy": 0.1, "hs_rate": 0.1},
        },
        "renames": {},
    },
    {
        "name": "サリエリ～ファルスタッフ～",
        "players": ("FNS", "cNed"),
        "player_bonuses": {
            "FNS": {"iq": 20},
            "cNed": {"dodge_rate": 0.1, "hs_rate": 0.1},
        },
        "renames": {},
    },
    {
        "name": "バッハ～G線上のアリア～",
        "players": ("FNS", "soulcas"),
        "player_bonuses": {
            "FNS": {"iq": 20},
            "soulcas": {"hs_rate": 0.2},
        },
        "renames": {},
    },
    {
        "name": "ラヴェル～ボレロ～",
        "players": ("FNS", "trexx"),
        "player_bonuses": {
            "FNS": {"iq": 20},
            "trexx": {"hs_rate": 0.2},
        },
        "renames": {},
    },
    {
        "name": "糖質制限は忍びのたしなみでして",
        "players": ("IbarakiNinja", "SugarZ3ro"),
        "player_bonuses": {
            "IbarakiNinja": {"dodge_rate": 0.1, "accuracy": 0.1, "reaction": 10}
        },
        "renames": {},
    },
    {
        "name": "偏食家と美食家の違い",
        "players": ("IbarakiNinja", "Laz"),
        "player_bonuses": {
            "IbarakiNinja": {"accuracy": 0.1, "reaction": 10},
            "Laz": {"accuracy": 0.1, "hs_rate": 0.1},
        },
        "renames": {},
    },
    {
        "name": "天才との別れ",
        "players": ("Leo", "Sayonara"),
        "bonuses": {"accuracy": 0.15},
        "renames": {},
    },
    {
        "name": "天才と秀才",
        "players": ("Leo", "C0M"),
        "bonuses": {"accuracy": 0.15},
        "renames": {},
    },
    {
        "name": "フラッシュバックするトラウマ",
        "players": ("Flashback", "Demon1", "Leo"),
        "bonuses": {"accuracy": 0.2},
        "renames": {},
    },
    {
        "name": "暗闇でフェアウェル",
        "players": ("Derke", "Sayonara"),
        "bonuses": {"accuracy": -0.1, "iq": 50},
        "renames": {},
    },
    {
        "name": "誰なん君たち",
        "players": ("Tortlilyan", "まーやまくん", "おもこ"),
        "bonuses": {"accuracy": 0.2},
        "renames": {},
    },
    {
        "name": "花と剣",
        "players": ("Furina", "Lohen"),
        "player_bonuses": {
            "Lohen": {"hs_rate": 0.2, "accuracy": 0.2, "dodge_rate": -0.1}
        },
        "renames": {},
    },
    {
        "name": "戦況は傾いている",
        "players": ("Furina", "Kachina"),
        "player_bonuses": {"Kachina": {"reaction": 20, "hs_rate": 0.2, "iq": -10}},
        "renames": {},
    },
    {
        "name": "意外な関係値",
        "players": ("Furina", "Jean"),
        "player_bonuses": {"Furina": {"dodge_rate": 0.1}, "Jean": {"hs_rate": 0.2}},
        "renames": {},
    },
    {
        "name": "紫と黄色",
        "players": ("Lisa", "Jean"),
        "bonuses": {"hs_rate": 0.15},
        "renames": {},
    },
    {
        "name": "寒色の三角関係",
        "players": ("Lisa", "Lohen", "Furina"),
        "player_bonuses": {
            "Furina": {"hs_rate": 0.1},
            "Lisa": {"dodge_rate": 0.25},
            "Lohen": {"hs_rate": 0.25},
        },
        "renames": {},
    },
    {
        "name": "同じ青い花を見よう",
        "players": ("Arlecchino", "Lohen"),
        "bonuses": {"accuracy": 0.1, "hs_rate": 0.1, "dodge_rate": -0.05},
        "renames": {},
    },
    {
        "name": "生殺与奪の水と炎",
        "players": ("Arlecchino", "Furina"),
        "player_bonuses": {
            "Arlecchino": {"accuracy": 0.2, "hs_rate": 0.2, "dodge_rate": -0.1}
        },
        "renames": {},
    },
    {
        "name": "花と鯨",
        "players": ("Furina", "Tartaglia"),
        "player_bonuses": {"Furina": {"dodge_rate": 0.2, "iq": 10}},
        "renames": {},
    },
    {
        "name": "剣と鯨",
        "players": ("Lohen", "Tartaglia"),
        "player_bonuses": {"Lohen": {"hs_rate": 0.3, "reaction": 30}},
        "renames": {},
    },
    {
        "name": "圧倒的なスナイパー",
        "players": ("ZMJKK", "something", "t3xture"),
        "bonuses": {"accuracy": 0.15, "reaction": 15},
        "renames": {},
    },
    {
        "name": "工場現場",
        "players": ("IbarakiNinja", "Brawk"),
        "bonuses": {"accuracy": 0.2, "reaction": 5},
        "renames": {},
    },
    {
        "name": "VisionStrikers",
        "players": ("stax", "Mako", "Buzz", "Rb", "Zest"),
        "bonuses": {
            "accuracy": 0.45,
            "reaction": 75,
            "dodge_rate": 0.15,
            "iq": -20,
            "hs_rate": 0.45,
        },
        "renames": {},
    },
    {
        "name": "スキューバカンカンチュー",
        "players": ("skuba", "ZMJKK", "CHICHOO"),
        "bonuses": {"accuracy": 0.2, "reaction": 10},
        "renames": {},
    },
    {
        "name": "ライチューモッギー",
        "players": ("Lysoar", "CHICHOO", "Smoggy"),
        "bonuses": {"accuracy": 0.2, "reaction": 10},
        "renames": {},
    },
    {
        "name": "問題児",
        "players": ("Canezerra", "Rossy", "Sayonara"),
        "bonuses": {"accuracy": 0.2, "reaction": 10},
        "renames": {},
    },
]


# 既存の表示処理との互換性を保つため、読み込み時にeffect_textを補完します。
apply_combo_effect_texts(COMBOS)
