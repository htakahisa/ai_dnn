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
    "Leo軸": TeamPreset(
        name="Leo軸",
        players=("Demon1", "Leo", "jawgemo", "Flashback", "Aspas"),
        igl="Leo",
        spike_holder="jawgemo",
        description="高い個人戦闘力とLeoのIGL性能を軸にした編成",
    ),
    "Fnatic2023": TeamPreset(
        name="Fnatic2023",
        players=("Leo", "Boaster", "Derke", "Chronicle", "Alfajer"),
        igl="Boaster",
        spike_holder="Derke",
        description="IQ、個人能力共に高水準。それぞれが自分の仕事をこなす、2023年で1番強かったチーム",
    ),
    "EG2023": TeamPreset(
        name="EG2023",
        players=("Demon1", "Ethan", "jawgemo", "Boostio", "C0M"),
        igl="Boostio",
        spike_holder="jawgemo",
        description="2023 Champions優勝メンバーEG構成",
    ),
    "フリーナクラシック": TeamPreset(
        name="フリーナクラシック",
        players=("Furina", "Lisa", "Lohen", "Jean", "Arlecchino"),
        igl="Furina",
        spike_holder="Lohen",
        description="FurinaをIGL兼コンボ中核にした最もクラシックな編成",
    ),
    "フリーナタルタリヤ": TeamPreset(
        name="フリーナタルタリヤ",
        players=("Furina", "Lisa", "Lohen", "Tartaglia", "Arlecchino"),
        igl="Furina",
        spike_holder="Lohen",
        description="クラシックなフリーナパのジンをタルタリヤに変更したアレンジ",
    ),
    "VisionStrikers": TeamPreset(
        name="VisionStrikers",
        players=("Mako", "stax", "Buzz", "Rb", "Zest"),
        igl="stax",
        spike_holder="Buzz",
        description="プレイヤーコンボにより圧倒的な戦闘力を誇る怪物集団",
    ),
    "日本代表": TeamPreset(
        name="日本代表",
        players=("Laz", "SugarZ3ro", "Meiy", "Dep", "IbarakiNinja"),
        igl="SugarZ3ro",
        spike_holder="Meiy",
        description="プレイヤーコンボで全員が底上げ強化、安定の日本オールスターズ",
    ),
    "クイーンズフラワーギャンビット": TeamPreset(
        name="クイーンズフラワーギャンビット",
        players=("leaf", "nAts", "Lar0k", "Chronicle", "Sayonara"),
        igl="nAts",
        spike_holder="Lar0k",
        description="圧倒的な補完性能、元祖2Flash構成",
    ),
    "ドラゴンテイル": TeamPreset(
        name="ドラゴンテイル",
        players=("Nanasaki", "Ethan", "Derke", "Canezerra", "vo0kashu"),
        igl="Ethan",
        spike_holder="Derke",
        description="ハイレベルハイスペックのドリームチーム",
    ),
    "ブラッドムーン": TeamPreset(
        name="ブラッドムーン",
        players=("Nanasaki", "Meteor", "WoohyuN", "Zest", "Ethan"),
        igl="Ethan",
        spike_holder="WoohyuN",
        description="HS%連発の殺意に満ちた化け物達に、世界一のIGL、Ethanを添える",
    ),
    "radiantdancer": TeamPreset(
        name="radiantdancer",
        players=("Sayonara", "Lar0k", "Derke", "marteen", "something"),
        igl="Sayonara",
        spike_holder="Derke",
        description="個性の激しい自由気ままなエゴイスト集団",
    ),
    "アイネクライネ": TeamPreset(
        name="アイネクライネ",
        players=("FNS", "crashies", "cNed", "soulcas", "trexx"),
        igl="FNS",
        spike_holder="cNed",
        description="昔からの選手たちが集う知の巨人集団",
    ),
    "とうやまゲーミング": TeamPreset(
        name="とうやまゲーミング",
        players=("ろびぃな", "えんぺん", "Tortlilyan", "いぐるん", "夢の街"),
        igl="えんぺん",
        spike_holder="ろびぃな",
        description="とうやまのおうちゲーミングチーム",
    ),
    "BBL": TeamPreset(
        name="BBL",
        players=("lovers rock", "Loita", "Lar0k", "Rosé", "Crewn"),
        igl="Rosé",
        spike_holder="Lar0k",
        description="Lar0kにすべてを捧げるワンマンチーム",
    ),
    "個人能力パ": TeamPreset(
        name="個人能力パ",
        players=("Tortlilyan", "まーやまくん", "おもこ", "Demon1", "Aspas"),
        igl="Tortlilyan",
        spike_holder="おもこ",
        description="個人技の高い選手を集めたグッドスタッフチーム",
    ),
    "Ghost Champions": TeamPreset(
        name="Ghost Champions",
        players=("Xdll", "SyouTa", "Absol", "eKo", "SugarZ3ro"),
        igl="SugarZ3ro",
        spike_holder="Absol",
        description="個人技とマクロのバランス",
    ),
}


def all_preset_names():
    return list(PARTY_PRESETS.keys())


def get_preset(name):
    return PARTY_PRESETS.get(name)
