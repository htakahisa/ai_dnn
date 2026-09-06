"""touyama_v2/common_defender.py

Defender側(search / retake)のtrain_*.pyで重複していた touyama_v2固定チーム
定義・実効ステータス計算を集約したファイル。

【仕様統一について】
train_defender_search.py は accuracy/hs_rate/dodge_rateを0-100スケール
(例: 92)のまま保持していたが、命中判定式(hit_chance = accuracy * (1 - dodge)
を[0,1]にclip)は0-1スケールの確率を前提とした式だったため、実質的に
「ほぼ必中」になっていた。train_defender_retake.py は元々0-1スケールに
正しく変換していた。

本ファイルでは0-1スケール(retake方式)に統一する。これにより
train_defender_search.py側の学習内容(命中バランス)は変更される
(2026-08-11 合意: 将来的に実際の命中率に応じた確率で学習させる方針のため)。

また train_defender_retake.py の元実装は未定義の TOUYAMA_RAW_STATS を
参照しておりNameErrorになるバグがあった。本ファイルでは
character_stats_touyama.CHARACTER_TABLE (name -> .hit_pct 等の属性を持つ
オブジェクト)を正しく参照する形に統一している。

【自己完結ルールとの関係】common_rl.py / common_attacker.py と同様、
touyama_v2配下のtrain_*.py同士でのimportは許容する。run_game.py /
controllers.py / battle_logic.py / abilities_los.py 等は引き続き
一切importしない。
"""

from tv2_character_stats_touyama import TOUYAMA_ROSTER_ORDER

TOUYAMA_COMBO_NAME = "ふわんだりぃず"
TOUYAMA_COMBO_MEMBERS = {"ろびぃな", "えんぺん", "いぐるん"}
TOUYAMA_COMBO_BONUS = {"accuracy": 0.15, "hs_rate": 0.10, "dodge_rate": 0.20, "reaction": 30.0}

# タイガー固有パッシブ(game_core.Character準拠)
TIGER_ACCURACY_BONUS = 0.10
TIGER_HS_BONUS = 0.05

# ロール(日本語)→ ability名。HUNT(タイガー)はアビリティを持たず、
# own_ability_charge()相当の判定で常に0チャージ扱いになる。
TOUYAMA_ROLE_TO_ABILITY = {
    "フラッシュ": "FLASH",
    "スモーカー": "SMOKE",
    "シーカー": "RECON",
    "タイガー": "HUNT",
}


def compute_touyama_effective_stats(stats_table, roster_order=TOUYAMA_ROSTER_ORDER):
    """character_stats_touyama.py の生値(stats_table)に、常時発動する
    チームコンボ(ふわんだりぃず)とタイガーパッシブを適用した確定値を返す。
    accuracy/hs_rate/dodge_rateは0-1スケール(確率)で返す。

    stats_tableは character_stats_touyama.CHARACTER_TABLE を想定
    (name -> .hit_pct / .hs_pct / .dodge_pct / .reaction / .role 属性を持つこと。
    hit_pct等は0-100スケールの生値)。
    """
    effective = {}
    for name in roster_order:
        raw = stats_table[name]
        accuracy = float(raw.hit_pct)
        hs_rate = float(raw.hs_pct)
        dodge_rate = float(raw.dodge_pct)
        reaction = float(raw.reaction)

        if raw.role == "タイガー":
            accuracy += TIGER_ACCURACY_BONUS
            hs_rate += TIGER_HS_BONUS

        if name in TOUYAMA_COMBO_MEMBERS:
            accuracy += TOUYAMA_COMBO_BONUS["accuracy"]
            hs_rate += TOUYAMA_COMBO_BONUS["hs_rate"]
            dodge_rate += TOUYAMA_COMBO_BONUS["dodge_rate"]
            reaction += TOUYAMA_COMBO_BONUS["reaction"]

        effective[name] = {
            "accuracy": max(0.0, accuracy),
            "hs_rate": max(0.0, min(1.0, hs_rate)),
            "dodge_rate": max(0.0, min(1.0, dodge_rate)),
            "reaction": max(0.0, reaction),
            "role": raw.role,
            "ability": TOUYAMA_ROLE_TO_ABILITY[raw.role],
        }
    return effective


def print_effective_stats(effective_stats, label="Defender", roster_order=TOUYAMA_ROSTER_ORDER):
    print(f"[touyama_v2] 固定チーム({label}) 確定ステータス:")
    for name in roster_order:
        s = effective_stats[name]
        print(
            f"  {name}: acc={s['accuracy']:.2f} hs={s['hs_rate']:.2f} "
            f"dodge={s['dodge_rate']:.2f} reaction={s['reaction']:.0f} "
            f"ability={s['ability']}"
        )