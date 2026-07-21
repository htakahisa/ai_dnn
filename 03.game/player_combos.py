"""プレイヤーコンボ定義。

COMBOS に辞書を追加するだけで、新しいコンボを登録できます。

主なキー:
- name: コンボ名
- players: 同じチームに必要な内部キャラクター名
- bonuses: コンボ参加者全員への加算補正
- player_bonuses: 特定の参加者だけに与える加算補正
- renames: 発動中の表示名変更（内部名は変化しない）
- effect_text: ラウンド開始時の告知パネルに表示する効果文

率系の値は 0.10 = 10ポイントです。
代表的な能力キー:
- accuracy: Hit率
- hs_rate: HS率
- dodge_rate: 回避率
- max_hp: 最大HP
- reaction : 反応速度
"""

COMBOS = [
    {
        "name": "天才と悪魔",
        "players": ("jawgemo", "Demon1"),
        "bonuses": {
            "hs_rate": 0.10,
        },
        "renames": {},
        "effect_text": "jawgemoとDemon1のHS% +10ポイント",
    },
    {
        "name": "ソウルフレンド",
        "players": ("F0rsakeN", "Jinggg"),
        "bonuses": {
            "accuracy": 0.05,
        },
        "renames": {},
        "effect_text": "F0rsakeNとJingggのHit率 +5ポイント",
    },
    {
        "name": "期待の新人",
        "players": ("Leo", "Chronicle"),
        "bonuses": {
            "accuracy": 0.05,
        },
        "renames": {},
        "effect_text": "LeoとChronicleのHit率 +5ポイント",
    },
    {
        "name": "犬猿の仲",
        "players": ("Aspas", "Demon1"),
        "bonuses": {
            "hs_rate": 0.10,
        },
        "renames": {},
        "effect_text": "AspasとDemon1のHS% +10ポイント",
    },
    {
        "name": "世界一の名門",
        "players": ("Alfajer", "Boaster", "Chronicle", "Derke", "Leo"),
        "bonuses": {
            "hs_rate": 0.10,
            "accuracy": 0.10,
        },
        "renames": {},
        "effect_text": "参加者全員のHit率・HS% +10ポイント",
    },
    {
        "name": "ジャパンホープ",
        "players": ("Laz", "Dep", "Meiy", "SugarZ3ro"),
        "bonuses": {
            "hs_rate": 0.25,
        },
        "renames": {},
        "effect_text": "参加者全員のHS% +20ポイント",
    },
    {
        "name": "天才との別れ",
        "players": ("Leo", "Sayonara"),
        "bonuses": {
            "accuracy": 0.10,
        },
        "renames": {},
        "effect_text": "LeoとSayonaraのHit率 +10ポイント",
    },
    {
        "name": "天才と秀才",
        "players": ("Leo", "C0M"),
        "bonuses": {
            "accuracy": 0.10,
        },
        "renames": {},
        "effect_text": "LeoとC0MのHit率 +10ポイント",
    },
    {
        "name": "フラッシュバックするトラウマ",
        "players": ("Flashback", "Demon1", "Leo"),
        "bonuses": {
            "accuracy": 0.15,
        },
        "renames": {},
        "effect_text": "参加者全員のHit率 +15ポイント",
    },
    {
        "name": "誰なん君たち",
        "players": ("Tortlilyan", "まーやまくん", "おもこ"),
        "bonuses": {
            "accuracy": 0.15,
        },
        "renames": {},
        "effect_text": "参加者全員のHit率 +15ポイント",
    },
    {
        "name": "花と剣",
        "players": ("Furina", "Lohen"),
        "bonuses": {
            "reaction": 15,
        },
        "renames": {},
        "effect_text": "FurinaとLohenの反応速度 +15ポイント",
    },
    {
        "name": "戦況は傾いている",
        "players": ("Furina", "Kachina"),
        "bonuses": {
            "hs_rate": 0.15,
        },
        "renames": {},
        "effect_text": "FurinaとKachinaのHS% +15ポイント",
    },
    {
        "name": "紫と黄色",
        "players": ("Lisa", "Jean"),
        "bonuses": {
            "hs_rate": 0.05,
        },
        "renames": {},
        "effect_text": "LisaとJeanのHS% +5ポイント",
    },
    {
        "name": "問題がふたつ",
        "players": ("Lisa", "Lohen"),
        "bonuses": {
            "reaction": 5,
        },
        "renames": {},
        "effect_text": "LisaとLohenの反応速度 +5ポイント",
    },
    {
        "name": "寒色の三角関係",
        "players": ("Lisa", "Lohen", "Furina"),
        "bonuses": {
            "hs_rate": 0.15,
            "dodge_rate": 0.15,
        },
        "renames": {},
        "effect_text": "参加者全員のHS%・回避率 +15ポイント",
    },
    {
        "name": "圧倒的なスナイパー",
        "players": ("ZMJKK", "something", "t3xture"),
        "bonuses": {
            "accuracy": 0.15,
            "reaction": 10,
        },
        "renames": {},
        "effect_text": "参加者全員のHit率 +15ポイント、反応速度 +10ポイント",
    },
    {
        "name": "工場現場",
        "players": ("IbarakiNinja", "Brawk"),
        "bonuses": {
            "accuracy": 0.20,
            "reaction": 5,
        },
        "renames": {},
        "effect_text": "IbarakiNinjaとBrawkのHit率 +20ポイント、反応速度 +5ポイント",
    },
    {
        "name": "VisionStrikers",
        "players": ("Stax", "Mako", "Buzz", "Rb"),
        "bonuses": {
            "accuracy": 0.20,
            "reaction": 10,
            "dodge_rate": 0.15,
        },
        "renames": {},
        "effect_text": "参加者全員のHit率 +20ポイント、反応速度 +10ポイント、回避率 + 15ポイント",
    },
    {
        "name": "スキューバカンカンチュー",
        "players": ("skuba", "ZMJKK", "CHICHOO"),
        "bonuses": {
            "accuracy": 0.20,
            "reaction": 10,
        },
        "renames": {},
        "effect_text": "参加者全員のHit率 +20ポイント、反応速度 +10ポイント",
    },
    
]