"""omoko_v1 固定チームのロスター設定。

並び順は唯一の正本である party_presets.py の ``players`` を使う。
"""

from party_presets import get_preset


_PRESET = get_preset("Omoko Gaming")
if _PRESET is None:
    raise RuntimeError("party_presets.py に Omoko Gaming がありません")

ROSTER_ORDER = tuple(_PRESET.players)
