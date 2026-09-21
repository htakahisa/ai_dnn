"""キャラクター選択画面で使用するパーティープリセット。

PARTY_PRESETS に TeamPreset を追加すれば、選択画面のプルダウンへ
自動的に表示されます。

players:
    編成する5人。並び順もそのまま出撃順になります。
igl:
    その編成のIGL。
spike_holder:
    Attackerとして適用した場合のスパイク所持者。
    Defenderとして適用する場合は無視されます。
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class TeamPreset:
    name: str
    players: Tuple[str, ...]
    igl: str
    spike_holder: Optional[str] = None
    description: str = ""
    short_name: str = ""

    def validate(self, available_names, expected_size=5):
        """現在使用可能なキャラクターに対してプリセットを検証する。"""
        available = set(available_names)
        errors = []

        if len(self.players) != expected_size:
            errors.append(
                f"メンバー数が{len(self.players)}人です"
                f"（必要人数: {expected_size}人）"
            )

        if len(set(self.players)) != len(self.players):
            errors.append("同じ選手が編成内で重複しています")

        missing = [name for name in self.players if name not in available]
        if missing:
            errors.append("未解禁または未登録の選手: " + ", ".join(missing))

        if self.igl not in self.players:
            errors.append(f"IGLの{self.igl}がメンバーに含まれていません")

        if self.spike_holder is not None and self.spike_holder not in self.players:
            errors.append(
                f"スパイク所持者の{self.spike_holder}が" "メンバーに含まれていません"
            )

        return errors


PARTY_PRESETS: Dict[str, TeamPreset] = {
    "Leo And Friends": TeamPreset(
        name="Leo And Friends",
        short_name="LAF",
        players=("Demon1", "Leo", "jawgemo", "Flashback", "Aspas"),
        igl="Leo",
        spike_holder="jawgemo",
        description="高い個人戦闘力とLeoのIGL性能を軸にした編成",
    ),
    "Fnatic2023": TeamPreset(
        name="Fnatic2023",
        short_name="FNC",
        players=("Leo", "Boaster", "Derke", "Chronicle", "Alfajer"),
        igl="Boaster",
        spike_holder="Derke",
        description="IQ、個人能力共に高水準。それぞれが自分の仕事をこなす、2023年で1番強かったチーム",
    ),
    "EG2023": TeamPreset(
        name="EG2023",
        short_name="EG",
        players=("Demon1", "Ethan", "jawgemo", "Boostio", "C0M"),
        igl="Boostio",
        spike_holder="jawgemo",
        description="2023 Champions優勝メンバーEG構成",
    ),
    "Furina Classic": TeamPreset(
        name="Furina Classic",
        short_name="FRC",
        players=("Furina", "Lisa", "Lohen", "Jean", "Arlecchino"),
        igl="Furina",
        spike_holder="Lohen",
        description="FurinaをIGL兼コンボ中核にした最もクラシックな編成",
    ),
    "Furina Tartaglia": TeamPreset(
        name="Furina Tartaglia",
        short_name="FRT",
        players=("Furina", "Lisa", "Lohen", "Tartaglia", "Arlecchino"),
        igl="Furina",
        spike_holder="Lohen",
        description="クラシックなフリーナパのジンをタルタリヤに変更したアレンジ",
    ),
    "Vision Strikers": TeamPreset(
        name="Vision Strikers",
        short_name="VS",
        players=("Mako", "stax", "Buzz", "Rb", "Zest"),
        igl="stax",
        spike_holder="Buzz",
        description="プレイヤーコンボにより圧倒的な戦闘力を誇る怪物集団",
    ),
    "Japan All-Stars": TeamPreset(
        name="Japan All-Stars",
        short_name="JAS",
        players=("Laz", "SugarZ3ro", "Meiy", "Dep", "IbarakiNinja"),
        igl="SugarZ3ro",
        spike_holder="Meiy",
        description="プレイヤーコンボで全員が底上げ強化、安定の日本オールスターズ",
    ),
    "Queen's Flower Gambit": TeamPreset(
        name="Queen's Flower Gambit",
        short_name="QFG",
        players=("leaf", "nAts", "Lar0k", "Chronicle", "Sayonara"),
        igl="nAts",
        spike_holder="Lar0k",
        description="圧倒的な補完性能、元祖2Flash構成",
    ),
    "Dragon Tail": TeamPreset(
        name="Dragon Tail",
        short_name="DRT",
        players=("Nanasaki", "Ethan", "Derke", "Canezerra", "vo0kashu"),
        igl="Ethan",
        spike_holder="Derke",
        description="ハイレベルハイスペックのドリームチーム",
    ),
    "Blood Moon": TeamPreset(
        name="Blood Moon",
        short_name="BM",
        players=("Nanasaki", "Meteor", "WoohyuN", "Zest", "Ethan"),
        igl="Ethan",
        spike_holder="WoohyuN",
        description="HS%連発の殺意に満ちた化け物達に、世界一のIGL、Ethanを添える",
    ),
    "radiantdancer": TeamPreset(
        name="radiantdancer",
        short_name="RD",
        players=("Sayonara", "Lar0k", "Derke", "marteen", "something"),
        igl="Sayonara",
        spike_holder="Derke",
        description="個性の激しい自由気ままなエゴイスト集団",
    ),
    "Eine Kleine": TeamPreset(
        name="Eine Kleine",
        short_name="EK",
        players=("FNS", "crashies", "cNed", "soulcas", "trexx"),
        igl="FNS",
        spike_holder="cNed",
        description="昔からの選手たちが集う知の巨人集団",
    ),
    "Touyama Gaming": TeamPreset(
        name="Touyama Gaming",
        short_name="TYG",
        players=("夢の街", "いぐるん", "ろびぃな", "Tortlilyan", "えんぺん"),
        igl="えんぺん",
        spike_holder="ろびぃな",
        description="とうやまのおうちゲーミングチーム",
    ),
    "Omoko Gaming": TeamPreset(
        name="Omoko Gaming",
        short_name="OMG",
        players=("おもこ", "いぬさん", "ねこさん", "とりさん", "ひつじさん"),
        igl="ひつじさん",
        spike_holder="ねこさん",
        description="おもこが集めた動物たちのチーム。かわいい。かわいい。かわいい。",
    ),
    "BBL": TeamPreset(
        name="BBL",
        short_name="BBL",
        players=("lovers rock", "Loita", "Lar0k", "Rosé", "Crewn"),
        igl="Rosé",
        spike_holder="Lar0k",
        description="Lar0kにすべてを捧げるワンマンチーム",
    ),
    "Team Elites": TeamPreset(
        name="Team Elites",
        short_name="TE",
        players=("Tortlilyan", "まーやまくん", "おもこ", "Demon1", "Aspas"),
        igl="Tortlilyan",
        spike_holder="おもこ",
        description="個人技の高い選手を集めたグッドスタッフチーム",
    ),
    "Ghost Champions": TeamPreset(
        name="Ghost Champions",
        short_name="GC",
        players=("Xdll", "SyouTa", "Absol", "eKo", "SugarZ3ro"),
        igl="SugarZ3ro",
        spike_holder="Absol",
        description="個人技とマクロのバランス",
    ),
    "Carnal Lust Syndicate": TeamPreset(
        name="Carnal Lust Syndicate",
        short_name="CLLS",
        players=("koldamenta", "Rossy", "CHICHOO", "Smoggy", "SereNa"),
        igl="SereNa",
        spike_holder="SereNa",
        description="圧倒的なミクロ、瞬発的なフィジカル",
    ),
    "SUPES": TeamPreset(
        name="SUPES",
        short_name="SPS",
        players=("A-Train", "Deep", "Homelander", "Stormfront", "Blacknoir"),
        igl="Homelander",
        spike_holder="A-Train",
        description="勝つためなら手段を選ばない、堕落した5人のヒーロー",
    ),
    "Alien Rex": TeamPreset(
        name="Alien Rex",
        short_name="ARX",
        players=("alecks", "mindfreak", "Wo0t", "something", "eggseterr"),
        igl="mindfreak",
        spike_holder="Wo0t",
        description="開始時、somethingがcgrsに変化",
    ),
}

# Backward-compatible team-name migration. Historical ratings, brackets, and
# model metadata may still contain the former spelling.
PRESET_NAME_ALIASES = {
    "VisionStrikers": "Vision Strikers",
    "ブラッドムーン": "Blood Moon",
    "とうやまゲーミング": "Touyama Gaming",
    "ドラゴンテイル": "Dragon Tail",
    "ブラッドムウン": "Blood Moon",
    "Leo軸": "Leo And Friends",
    "フリーナクラシック": "Furina Classic",
    "フリーナタルタリヤ": "Furina Tartaglia",
    "日本代表": "Japan All-Stars",
    "クイーンズフラワーギャンビット": "Queen's Flower Gambit",
    "アイネクライネ": "Eine Kleine",
    "個人能力パ": "Team Elites",
}


def canonical_preset_name(name):
    return PRESET_NAME_ALIASES.get(str(name), str(name))


def all_preset_names():
    return list(PARTY_PRESETS.keys())


def get_preset(name):
    return PARTY_PRESETS.get(canonical_preset_name(name))


def get_team_short_name(name):
    """Display-only abbreviation; never use as a team identity/key."""
    preset = get_preset(name)
    return (preset.short_name or preset.name) if preset else str(name)


def normalize_team_names(data):
    """Normalize legacy team names in JSON keys/values without touching files."""
    if isinstance(data, str):
        return canonical_preset_name(data)
    if isinstance(data, list):
        return [normalize_team_names(value) for value in data]
    if isinstance(data, dict):
        return {
            canonical_preset_name(key): normalize_team_names(value)
            for key, value in data.items()
        }
    return data
