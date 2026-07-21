"""覚醒イベント定義。

AWAKENING_EVENTS に辞書を追加すると量産できます。

主なキー:
- name: イベント名
- player: 覚醒対象の内部キャラクター名
- condition: 条件名
    - all_allies_dead: 自分以外の味方が全滅し、自分が生存
    - hp_at_or_below: HPが condition_value 以下
    - kills_at_least: キル数が condition_value 以上
- rename: 覚醒後の画面表示名
- transform_to: character_stats内の別キャラクター定義へ能力を置換
- bonuses: 追加で加算する能力補正
- effect_text: 告知・ログ用の説明文
"""

AWAKENING_EVENTS = [
    {
        "name": "最後には彼がいる",
        "player": "Leo",
        "condition": "all_allies_dead",
        "rename": "-ラスボス-Leo",
        # 最新ExcelのLeoを基準に、覚醒後は Hit 90% / HS 55% / 回避 30% になる。
        "bonuses": {"accuracy": 0.07, "hs_rate": 0.28, "dodge_rate": 0.08},
        "effect_text": "味方全滅、ラスボスLeoへ覚醒",
    },
    {
        "name": "彼を起こしてはいけない",
        "player": "Demon1",
        "condition": "hp_at_or_below",
        "condition_value": 30,
        "rename": "-地獄の王者-Demon1",
        "bonuses": {"accuracy": 0.15,"hs_rate": 0.25},           
        "effect_text": "体力が30以下、-地獄の王者-Demon1へ覚醒",
    },
    {
        "name": "楽になれると思うなよ",
        "player": "Lohen",
        "condition": "kills_at_least",
        "condition_value": 3,
        "rename": "-戦闘狂-Lohen",
        "bonuses": {"accuracy": 0.15,"reaction": 15},           
        "effect_text": "3キル突破、-戦闘狂-Lohenへ覚醒",
    },

]