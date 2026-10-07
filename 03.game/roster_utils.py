"""
ロースター構築ユーティリティ

問題点：
- attacker_roster=None のままだと "Att1"〜"Att5" という架空名になり、
  character_stats.py に登録がないためデフォルト値（弱い・全員フラッシュ役）になる
- defender は None のとき実在キャラ72人からランダム選出されるため、
  結果的にアタッカーだけ著しく弱くなっていた

このユーティリティは：
- 実在するキャラクター名から、指定した役割構成（例：フラッシュ1・シーカー2・スモーカー2）で
  ロースターを組む
- アタッカー側・ディフェンダー側で使用キャラが重複しないようにする
  （match_stats が名前だけで集計されるため、重複すると統計が混ざるのを防ぐ）
"""

import random
from collections import defaultdict

from game_core import get_all_character_names, get_character_combat_stats

# ability_name のマッピング（game_core.py の Character.__init__ 参照）
# フラッシュ → FLASH, スモーカー → SMOKE, シーカー → RECON, タイガー → HUNT(パッシブのみ)
DEFAULT_ROLE_COMPOSITION = {
    "フラッシュ": 1,
    "シーカー": 2,   # リコン
    "スモーカー": 2,  # スモーク
}


def roster_information(chars, viewer_team):
    """Public lineup information, independent of visibility and survival.

    Entries follow character/roster order and contain copied base stats, never
    live enemy objects, positions, HP, spike ownership or remaining charges.
    Runtime names distinguish duplicate players; base_name identifies their
    character definition (including season-specific player keys).
    """
    allies, enemies = [], []
    for char in chars:
        base_name = getattr(char, "base_name", char.name)
        stats = dict(get_character_combat_stats(base_name))
        entry = {
            "name": str(char.name),
            "base_name": str(base_name),
            "team": char.team,
            "role": getattr(char, "role", stats["role"]),
            "ability_name": getattr(char, "ability_name", None),
            "ultimate_name": getattr(char, "ultimate_name", None),
            "ultimate_cost": getattr(char, "ultimate_cost", None),
            "stats": stats,
        }
        (allies if char.team == viewer_team else enemies).append(entry)
    return {"ally_roster": allies, "enemy_roster": enemies}


def model_slots(players, trained_names):
    """Map actual teammates to distinct trained slots without renaming players.

    Original members keep their learned slot; replacements fill unused slots in
    roster order. Include dead allies so their slot stays reserved this round.
    """
    players = tuple(players)
    slots, used = {}, set()
    for player in players:
        name = str(getattr(player, "base_name", player.name))
        if name in trained_names:
            slot = trained_names.index(name)
            if slot not in used:
                slots[player.name] = slot
                used.add(slot)
    free = iter(slot for slot in range(len(trained_names)) if slot not in used)
    for player in players:
        if player.name not in slots:
            slots[player.name] = next(free)
    return slots


def _build_role_map():
    """役割名 → キャラ名リスト の辞書を作る"""
    role_map = defaultdict(list)
    for name in get_all_character_names():
        stats = get_character_combat_stats(name)
        role_map[stats["role"]].append(name)
    return role_map


def build_two_balanced_rosters(role_composition=None, seed=None):
    """
    指定した役割構成で、アタッカー用・ディフェンダー用のロースターを
    それぞれ組む（両者でキャラが重複しないようにする）。

    Args:
        role_composition: {"フラッシュ": 1, "シーカー": 2, "スモーカー": 2} のような辞書
        seed: 再現性が欲しい場合の乱数シード

    Returns:
        (attacker_roster, defender_roster) のタプル（それぞれ名前のリスト）
    """
    if role_composition is None:
        role_composition = DEFAULT_ROLE_COMPOSITION

    if seed is not None:
        random.seed(seed)

    role_map = _build_role_map()

    # 各役割に十分なキャラがいるかチェック（両チーム分必要なので×2）
    for role, count in role_composition.items():
        available = len(role_map.get(role, []))
        required = count * 2
        if available < required:
            raise ValueError(
                f"役割「{role}」のキャラが不足しています。"
                f"必要: {required}人（両チーム分）, 実在: {available}人"
            )

    used_names = set()

    def pick_roster():
        roster = []
        for role, count in role_composition.items():
            candidates = [n for n in role_map[role] if n not in used_names]
            picked = random.sample(candidates, count)
            roster.extend(picked)
            used_names.update(picked)
        random.shuffle(roster)
        return roster

    attacker_roster = pick_roster()
    defender_roster = pick_roster()

    return attacker_roster, defender_roster


if __name__ == "__main__":
    # 動作確認用
    att, defd = build_two_balanced_rosters()
    print("Attacker roster:", att)
    print("Defender roster:", defd)
