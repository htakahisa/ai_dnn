"""touyama_v1/common_attacker.py

Attacker側(carry / escort / guard / retrieve)のtrain_*.pyで重複していた
touyama_v1固定チーム定義・実効ステータス計算を集約したファイル。

いずれのAttacker系train_*.pyでも完全に同一の実装だったため、計算結果・
学習内容への影響はない(純粋な重複除去)。

【自己完結ルールとの関係】common_rl.pyと同様、touyama_v1配下のtrain_*.py
同士でのimportは許容する。run_game.py / controllers.py / battle_logic.py /
abilities_los.py 等は引き続き一切importしない。
"""

TOUYAMA_ROSTER_ORDER = ["Tortlilyan", "いぐるん", "ろびぃな", "夢の街", "えんぺん"]
TOUYAMA_SPIKE_HOLDER = "ろびぃな"  # 通常ラウンド開始時の既定キャリア

TOUYAMA_COMBO_MEMBERS = {"ろびぃな", "えんぺん", "いぐるん"}
TOUYAMA_COMBO_BONUS = {
    "accuracy": 0.15,
    "hs_rate": 0.10,
    "dodge_rate": 0.20,
    "reaction": 30.0,
}

TOUYAMA_ROLE_TO_ABILITY = {
    "フラッシュ": "FLASH",
    "スモーカー": "SMOKE",
    "シーカー": "RECON",
    "タイガー": "HUNT",
}
TIGER_ACCURACY_BONUS = 0.10
TIGER_HS_BONUS = 0.05

# 敵(Defender)側の既定ステータス(当面ヒューリスティックのため簡易値のまま。
# carry / escort / guard / retrieve で共通の値)
DEFAULT_ACCURACY = 0.50
DEFAULT_DODGE = 0.12
DEFAULT_HS_RATE = 0.20
DEFAULT_REACTION = 100.0


def compute_touyama_effective_stats(stats_table, roster_order=TOUYAMA_ROSTER_ORDER):
    """character_stats_touyama.py の生値(stats_table)に、常時発動する
    チームコンボ(ふわんだりぃず)とタイガーパッシブを適用した確定値を返す。

    stats_tableは character_stats_touyama.CHARACTER_TABLE を想定
    (name -> オブジェクト。.hit_pct / .hs_pct / .dodge_pct / .reaction / .role
    属性を持つこと)。
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
            "ability": TOUYAMA_ROLE_TO_ABILITY[raw.role],
        }
    return effective


def print_effective_stats(effective_stats, label, roster_order=TOUYAMA_ROSTER_ORDER):
    """各train_*.pyがモジュール読み込み時に出していた確認ログと同一の出力。
    呼び出し側で `print_effective_stats(TOUYAMA_EFFECTIVE_STATS, "Attacker/carry")`
    のように使う。"""
    print(f"[touyama_v1] 固定チーム({label}) 確定ステータス:")
    for name in roster_order:
        s = effective_stats[name]
        print(
            f"  {name}: acc={s['accuracy']:.2f} hs={s['hs_rate']:.2f} "
            f"dodge={s['dodge_rate']:.2f} reaction={s['reaction']:.0f} "
            f"ability={s['ability']}"
        )