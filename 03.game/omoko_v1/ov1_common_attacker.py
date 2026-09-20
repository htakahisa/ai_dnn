"""omoko_v1/common_attacker.py

Attacker側(carry / escort / guard / retrieve)のtrain_*.pyで重複していた
omoko_v1固定チーム定義・実効ステータス計算を集約したファイル。

いずれのAttacker系train_*.pyでも完全に同一の実装だったため、計算結果・
学習内容への影響はない(純粋な重複除去)。

【自己完結ルールとの関係】common_rl.pyと同様、omoko_v1配下のtrain_*.py
同士でのimportは許容する。run_game.py / controllers.py / battle_logic.py /
abilities_los.py 等は引き続き一切importしない。
"""

from ov1_character_stats import ROSTER_ORDER
from player_combos import COMBOS
from party_presets import get_preset

_TEAM_PRESET = get_preset("Omoko Gaming")
if _TEAM_PRESET is None:
    raise RuntimeError("party_presets.py に「Omoko Gaming」プリセットが見つかりません。")

SPIKE_HOLDER = _TEAM_PRESET.spike_holder  # 通常ラウンド開始時の既定キャリア(party_presets.py準拠)

ROLE_TO_ABILITY = {
    "フラッシュ": "FLASH",
    "スモーカー": "SMOKE",
    "シーカー": "RECON",
    "タイガー": "HUNT",
}

# player_combos.py の表記ゆれ(キー名)を、この計算で扱う4指標へ正規化する。
# game_core._canonical_combo_stat_keyのうち、accuracy/hs_rate/dodge_rate/reaction
# に該当する部分のみを複製している(iq/mental/move_steps_per_tick等はここでは未使用)。
_STAT_KEY_ALIASES = {
    "accuracy": "accuracy", "hit": "accuracy", "hit%": "accuracy",
    "hit_pct": "accuracy", "hit_rate": "accuracy", "命中率": "accuracy",
    "hs": "hs_rate", "hs%": "hs_rate", "hs_pct": "hs_rate",
    "hs_rate": "hs_rate", "headshot_rate": "hs_rate", "ヘッドショット率": "hs_rate",
    "dodge": "dodge_rate", "dodge%": "dodge_rate", "dodge_pct": "dodge_rate",
    "dodge_rate": "dodge_rate", "回避率": "dodge_rate", "弾除け率": "dodge_rate",
    "reaction": "reaction", "reaction_speed": "reaction", "反応速度": "reaction",
}


def _apply_stat_bonus(stats, key, value):
    """game_core._apply_combo_bonusと同一の正規化ルール(率は絶対値>1.0なら
    ÷100して加算)で、accuracy/hs_rate/dodge_rate/reactionのみ加算する。"""
    canonical = _STAT_KEY_ALIASES.get(str(key).strip().lower().replace("％", "%").replace(" ", ""))
    if canonical is None:
        return
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return
    if canonical in ("accuracy", "hs_rate", "dodge_rate") and abs(amount) > 1.0:
        amount /= 100.0
    stats[canonical] = stats.get(canonical, 0.0) + amount

# 敵(Defender)側の既定ステータス(当面ヒューリスティックのため簡易値のまま。
# carry / escort / guard / retrieve で共通の値)
DEFAULT_ACCURACY = 0.50
DEFAULT_DODGE = 0.12
DEFAULT_HS_RATE = 0.20
DEFAULT_REACTION = 100.0


def compute_effective_stats(stats_table, roster_order=ROSTER_ORDER):
    """character_stats.py の生値(stats_table)に、player_combos.py上で
    roster_order全員がそのチーム内にいるコンボを自動判定して適用した確定値を返す。

    ハードコードされたコンボ内容を持たず、常にplayer_combos.COMBOSを参照する
    ため、コンボ側の変更(メンバー・数値)に追従して自動的に反映される。

    stats_tableは character_stats.CHARACTER_TABLE を想定
    (name -> オブジェクト。.hit_pct / .hs_pct / .dodge_pct / .reaction / .role
    属性を持つこと)。
    """
    roster_set = set(roster_order)
    effective = {
        name: {
            "accuracy": float(stats_table[name].hit_pct),
            "hs_rate": float(stats_table[name].hs_pct),
            "dodge_rate": float(stats_table[name].dodge_pct),
            "reaction": float(stats_table[name].reaction),
        }
        for name in roster_order
    }

    for combo in COMBOS:
        if not isinstance(combo, dict):
            continue
        required_players = set(combo.get("players", ()))
        # roster_order(このチーム)に、コンボ必要メンバーが全員含まれている場合のみ発動する
        # (game_core._apply_player_combosの「チーム内に全員揃っているか」判定と同一)。
        if not required_players or not required_players.issubset(roster_set):
            continue

        common_bonuses = combo.get("bonuses", {})
        per_player_bonuses = combo.get("player_bonuses", {})
        for name in required_players:
            if isinstance(common_bonuses, dict):
                for key, value in common_bonuses.items():
                    _apply_stat_bonus(effective[name], key, value)
            own_bonuses = per_player_bonuses.get(name, {}) if isinstance(per_player_bonuses, dict) else {}
            if isinstance(own_bonuses, dict):
                for key, value in own_bonuses.items():
                    _apply_stat_bonus(effective[name], key, value)

    for name in roster_order:
        stats = effective[name]
        # game_core._apply_combo_bonusと同一: accuracy/hs_rateは上限なし(0のみ下限)、
        # dodge_rateのみ0〜1へクランプする。
        stats["accuracy"] = max(0.0, stats["accuracy"])
        stats["hs_rate"] = max(0.0, stats["hs_rate"])
        stats["dodge_rate"] = max(0.0, min(1.0, stats["dodge_rate"]))
        stats["reaction"] = max(0.0, stats["reaction"])
        stats["ability"] = ROLE_TO_ABILITY[stats_table[name].role]

    return effective


def print_effective_stats(effective_stats, label, roster_order=ROSTER_ORDER):
    """各train_*.pyがモジュール読み込み時に出していた確認ログと同一の出力。
    呼び出し側で `print_effective_stats(EFFECTIVE_STATS, "Attacker/carry")`
    のように使う。"""
    print(f"[omoko_v1] 固定チーム({label}) 確定ステータス:")
    for name in roster_order:
        s = effective_stats[name]
        print(
            f"  {name}: acc={s['accuracy']:.2f} hs={s['hs_rate']:.2f} "
            f"dodge={s['dodge_rate']:.2f} reaction={s['reaction']:.0f} "
            f"ability={s['ability']}"
        )