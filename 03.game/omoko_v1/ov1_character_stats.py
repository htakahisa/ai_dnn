"""omoko_v1/ov1_character_stats.py

キャラクター実データ(CharacterStats定義・数値)は character_stats.py を
正とし、ここでは二重管理しない。このファイルはomoko_v1固定チームの
ロースター順(ROSTER_ORDER)のみを保持し、それ以外は character_stats.py を
そのまま再エクスポートする。
"""

from character_stats import (
    CharacterStats,
    CHARACTER_TABLE,
    CHARACTER_STATS,
    character_stats,
    get_by_name,
    get_stats,
    all_characters,
    all_names,
)

ROSTER_ORDER = ["いぬさん", "ねこさん", "おもこ", "ひつじさん", "とりさん"]
