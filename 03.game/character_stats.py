"""キャラクターのステータス一覧。元データ: ai_dnnキャラ一覧.xlsx の上段表。"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CharacterStats:
    name: str
    hs_pct: float
    dodge_pct: float
    iq: int
    hit_pct: float
    power: float


CHARACTER_TABLE: List[CharacterStats] = [
    CharacterStats(name='Chronicle', hs_pct=0.35, dodge_pct=0.2, iq=80, hit_pct=0.7, power=197.5),
    CharacterStats(name='Demon1', hs_pct=0.45, dodge_pct=0.11, iq=80, hit_pct=0.69, power=193.2),
    CharacterStats(name='something', hs_pct=0.35, dodge_pct=0.25, iq=70, hit_pct=0.77, power=211.6),
    CharacterStats(name='Leo', hs_pct=0.25, dodge_pct=0.15, iq=150, hit_pct=0.8, power=220.5),
    CharacterStats(name='Alfajer', hs_pct=0.45, dodge_pct=0.13, iq=65, hit_pct=0.8, power=204),
    CharacterStats(name='Derke', hs_pct=0.35, dodge_pct=0.2, iq=75, hit_pct=0.7, power=195),
    CharacterStats(name='Boaster', hs_pct=0.2, dodge_pct=0.1, iq=165, hit_pct=0.55, power=178),
    CharacterStats(name='Aspas', hs_pct=0.4, dodge_pct=0.3, iq=80, hit_pct=0.75, power=231.5),
    CharacterStats(name='F0rsakeN', hs_pct=0.25, dodge_pct=0.15, iq=125, hit_pct=0.65, power=188.5),
    CharacterStats(name='Jinggg', hs_pct=0.3, dodge_pct=0.25, iq=68, hit_pct=0.69, power=192.7),
    CharacterStats(name='D4v41', hs_pct=0.35, dodge_pct=0.15, iq=70, hit_pct=0.69, power=181.2),
    CharacterStats(name='Sato', hs_pct=0.4, dodge_pct=0.2, iq=67, hit_pct=0.67, power=194.6),
    CharacterStats(name='jawgemo', hs_pct=0.28, dodge_pct=0.29, iq=89, hit_pct=0.65, power=203),
    CharacterStats(name='valyn', hs_pct=0.25, dodge_pct=0.13, iq=135, hit_pct=0.6, power=183),
    CharacterStats(name='Ethan', hs_pct=0.25, dodge_pct=0.2, iq=140, hit_pct=0.65, power=206),
    CharacterStats(name='wo0t', hs_pct=0.4, dodge_pct=0.15, iq=71, hit_pct=0.74, power=195.7),
    CharacterStats(name='kaajak', hs_pct=0.35, dodge_pct=0.15, iq=89, hit_pct=0.76, power=199.8),
    CharacterStats(name='Meiy', hs_pct=0.35, dodge_pct=0.23, iq=65, hit_pct=0.71, power=197.3),
    CharacterStats(name='Verno', hs_pct=0.35, dodge_pct=0.16, iq=86, hit_pct=0.67, power=188.6),
    CharacterStats(name='Sayonara', hs_pct=0.33, dodge_pct=0.2, iq=89, hit_pct=0.68, power=196.4),
    CharacterStats(name='Jamppi', hs_pct=0.25, dodge_pct=0.16, iq=125, hit_pct=0.69, power=195.7),
    CharacterStats(name='Boostio', hs_pct=0.28, dodge_pct=0.17, iq=128, hit_pct=0.69, power=203.7),
    CharacterStats(name='味方が全滅したLeo', hs_pct=0.45, dodge_pct=0.3, iq=170, hit_pct=0.87, power=299.6),
    CharacterStats(name='チャンピオンズのEthan', hs_pct=0.3, dodge_pct=0.25, iq=145, hit_pct=0.83, power=249.4),
    CharacterStats(name='Primmie', hs_pct=0.45, dodge_pct=0.25, iq=59, hit_pct=0.75, power=218.5),
    CharacterStats(name='Demon1（坊主）', hs_pct=0.51, dodge_pct=0.2, iq=89, hit_pct=0.79, power=237.7),
    CharacterStats(name='Tortlilyan', hs_pct=0.2, dodge_pct=0.31, iq=157, hit_pct=0.69, power=234.2),
    CharacterStats(name='まーやまくん', hs_pct=0.45, dodge_pct=0.23, iq=79, hit_pct=0.7, power=218),
    CharacterStats(name='おもこ', hs_pct=0.15, dodge_pct=0.51, iq=55, hit_pct=0.8, power=230),
    CharacterStats(name='Meteor', hs_pct=0.36, dodge_pct=0.3, iq=73, hit_pct=0.76, power=223.3),
    CharacterStats(name='Laz', hs_pct=0.4, dodge_pct=0.15, iq=70, hit_pct=0.74, power=195.2),
    CharacterStats(name='ZMJKK', hs_pct=0.3, dodge_pct=0.25, iq=55, hit_pct=0.71, power=188.8),
    CharacterStats(name='Brawk', hs_pct=0.3, dodge_pct=0.15, iq=79, hit_pct=0.67, power=175.6),
    CharacterStats(name='HYUNMIN', hs_pct=0.37, dodge_pct=0.2, iq=76, hit_pct=0.76, power=206.3),
    CharacterStats(name='Flashback', hs_pct=0.33, dodge_pct=0.24, iq=79, hit_pct=0.78, power=212.4),
    CharacterStats(name='Mako', hs_pct=0.28, dodge_pct=0.2, iq=120, hit_pct=0.7, power=207),
    CharacterStats(name='t3xture', hs_pct=0.33, dodge_pct=0.17, iq=89, hit_pct=0.77, power=202.1),
]

_BY_ID: Dict[int, CharacterStats] = {i: c for i, c in enumerate(CHARACTER_TABLE)}
_BY_NAME: Dict[str, CharacterStats] = {c.name: c for c in CHARACTER_TABLE}


def get_by_id(char_id: int) -> Optional[CharacterStats]:
    return _BY_ID.get(char_id)


def get_by_name(name: str) -> Optional[CharacterStats]:
    return _BY_NAME.get(name)


def all_characters() -> List[CharacterStats]:
    return list(CHARACTER_TABLE)


def all_names() -> List[str]:
    return [c.name for c in CHARACTER_TABLE]


if __name__ == "__main__":
    print(f"キャラクター数: {len(CHARACTER_TABLE)}")
    for c in CHARACTER_TABLE:
        print(c)