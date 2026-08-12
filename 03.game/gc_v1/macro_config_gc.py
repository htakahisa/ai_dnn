"""Ghost Champions / Attacker Macro 戦術ウェイト設定。

このファイルだけ編集すれば、Macro学習時の
「どの戦術を多く経験させるか」と
「各ルート候補をどちら寄りに選ぶか」を調整できる。

============================================================================
1. 戦術出現ウェイト
============================================================================

GC_MACRO_STRATEGY_WEIGHTS

値の意味:
    1.0 = 基準
    2.0 = 約2倍選ばれやすい
    0.5 = 約半分
    0.0 = 学習時に選ばない

候補数やマップ上のマーカー数とは無関係。
戦術の追加・削除時だけ辞書を編集すればよい。

============================================================================
2. ルート候補バイアス
============================================================================

GC_MACRO_ROUTE_BIAS

候補数が何個あっても自動対応する可変式を想定。

値の意味:
    0.0  = 均等ランダム
    正数 = 短い / 直接的な候補を選びやすい
    負数 = 長い / 安全寄りの候補を選びやすい

目安:
    +0.5 / -0.5 = 少し偏る
    +1.0 / -1.0 = はっきり偏る
    +2.0 / -2.0 = かなり偏る
    +3.0 / -3.0 = 強く偏る

実装側では、候補をBFS距離等で並べたうえで、
このbias値から候補数に応じた重みを自動生成する。

============================================================================
3. 戦術内部の人数配分
============================================================================

GC_MACRO_GROUP_SIZES

Split / Default等での基本人数配分。
必要なら変更可能。

例:
    A_SPLIT = main 3 / support 2
    DEFAULT = A 2 / MID 1 / B 2

============================================================================
4. 切り替え判断しきい値
============================================================================

GC_MACRO_DECISION_THRESHOLDS

「相手が厚いのでRotate」
「情報が十分あるので空きサイトと判断」
などの判断しきい値。

後から学習モデル側に観測として渡すこともできるが、
最初は学習環境の条件生成や教師戦術の基準として使える。
"""


# ============================================================================
# 1. 戦術出現ウェイト
# ============================================================================

GC_MACRO_STRATEGY_WEIGHTS = {
    # 既存の基本戦術
    "A_RUSH": 0.6,
    "B_RUSH": 0.6,
    "MID_TO_B": 0.8,

    # 今回増やしたい戦術
    "A_SPLIT": 1.8,
    "B_SPLIT": 1.6,
    "DEFAULT": 2.0,

    # Fake / Rotate
    "FAKE_A_TO_B": 1.2,
    "FAKE_B_TO_A": 1.2,
    "ROTATE_A_TO_B": 1.4,
    "ROTATE_B_TO_A": 1.4,

    # 一度引いて同じサイトへ入り直す
    "REHIT_A": 0.8,
    "REHIT_B": 0.8,
}


# ============================================================================
# 2. ルート候補バイアス
# ============================================================================

GC_MACRO_ROUTE_BIAS = {
    # Split Entry
    # 正 = より直接的/短いEntry寄り
    # 負 = より遠回り/安全寄り
    "A_SPLIT_ENTRY": 0.0,
    "B_SPLIT_ENTRY": 0.0,

    # Rotate
    "A_TO_B_ROTATE": -0.2,
    "B_TO_A_ROTATE": -0.2,

    # Staging
    "A_STAGING": 0.0,
    "B_STAGING": 0.0,
    "MID_STAGING": 0.0,
    "NEUTRAL_STAGING": 0.0,

    # Reset/Re-hit
    "SAFE_RESET": -0.5,
}


# ============================================================================
# 3. 基本人員配分
# ============================================================================

GC_MACRO_GROUP_SIZES = {
    # main + support = 5
    "A_SPLIT": {
        "main": 3,
        "support": 2,
    },
    "B_SPLIT": {
        "main": 3,
        "support": 2,
    },

    # A + MID + B = 5
    "DEFAULT": {
        "A": 2,
        "MID": 1,
        "B": 2,
    },

    # Fake時の初動人数
    "FAKE_A_TO_B": {
        "fake": 3,
        "rotate": 2,
    },
    "FAKE_B_TO_A": {
        "fake": 3,
        "rotate": 2,
    },
}


# ============================================================================
# 4. 戦術切り替え判断
# ============================================================================

GC_MACRO_DECISION_THRESHOLDS = {
    # そのエリアの情報がどの程度新しければ「信頼できる」とみなすか。
    # 例: 0.70以上ならRotate判断材料として扱う。
    "INFO_CONFIDENCE_MIN": 0.70,

    # 攻略中サイトで確認された敵人数がこの値以上なら
    # Rotate候補を強く考慮する。
    "HEAVY_SITE_ENEMY_COUNT": 3,

    # 反対サイトで確認された敵人数がこの値以下なら
    # 空きサイト候補として扱う。
    "LIGHT_SITE_ENEMY_COUNT": 1,

    # Rotateを考慮する最低残り時間比率。
    # 0.30なら、ラウンド残り30%以上ある時のみ大きなRotateを許容。
    "ROTATE_MIN_TIME_RATIO": 0.30,

    # Re-hit時、一旦距離を取ったとみなす目安。
    "RESET_MIN_BFS_DISTANCE": 4,
}


# ============================================================================
# 5. 戦術別の途中切り替え可否
# ============================================================================

GC_MACRO_TRANSITIONS = {
    "A_RUSH": {
        "allow_rotate": True,
        "allow_rehit": True,
    },
    "B_RUSH": {
        "allow_rotate": True,
        "allow_rehit": True,
    },
    "MID_TO_B": {
        "allow_rotate": True,
        "allow_rehit": False,
    },
    "A_SPLIT": {
        "allow_rotate": True,
        "allow_rehit": True,
    },
    "B_SPLIT": {
        "allow_rotate": True,
        "allow_rehit": True,
    },
    "DEFAULT": {
        "allow_rotate": True,
        "allow_rehit": True,
    },
    "FAKE_A_TO_B": {
        "allow_rotate": True,
        "allow_rehit": False,
    },
    "FAKE_B_TO_A": {
        "allow_rotate": True,
        "allow_rehit": False,
    },
    "ROTATE_A_TO_B": {
        "allow_rotate": False,
        "allow_rehit": True,
    },
    "ROTATE_B_TO_A": {
        "allow_rotate": False,
        "allow_rehit": True,
    },
    "REHIT_A": {
        "allow_rotate": True,
        "allow_rehit": False,
    },
    "REHIT_B": {
        "allow_rotate": True,
        "allow_rehit": False,
    },
}


# ============================================================================
# ヘルパー
# ============================================================================

def normalized_strategy_weights():
    """戦術ウェイトを確率に正規化して返す。"""
    active = {
        name: max(0.0, float(weight))
        for name, weight in GC_MACRO_STRATEGY_WEIGHTS.items()
        if float(weight) > 0.0
    }

    total = sum(active.values())
    if total <= 0.0:
        raise ValueError("GC_MACRO_STRATEGY_WEIGHTS の合計が0です。")

    return {
        name: weight / total
        for name, weight in active.items()
    }


def validate_macro_config():
    """設定値の簡易検証。"""
    if not GC_MACRO_STRATEGY_WEIGHTS:
        raise ValueError("戦術ウェイトが空です。")

    if sum(max(0.0, float(v)) for v in GC_MACRO_STRATEGY_WEIGHTS.values()) <= 0.0:
        raise ValueError("有効な戦術ウェイトが1つもありません。")

    for strategy, groups in GC_MACRO_GROUP_SIZES.items():
        total = sum(int(v) for v in groups.values())
        if total != 5:
            raise ValueError(
                f"{strategy} の人数合計が5ではありません: {groups}"
            )

    info_min = float(GC_MACRO_DECISION_THRESHOLDS["INFO_CONFIDENCE_MIN"])
    if not 0.0 <= info_min <= 1.0:
        raise ValueError("INFO_CONFIDENCE_MIN は0～1にしてください。")

    time_ratio = float(GC_MACRO_DECISION_THRESHOLDS["ROTATE_MIN_TIME_RATIO"])
    if not 0.0 <= time_ratio <= 1.0:
        raise ValueError("ROTATE_MIN_TIME_RATIO は0～1にしてください。")

    return True


if __name__ == "__main__":
    validate_macro_config()

    print("[GC Macro Config] OK")
    print("\n[Strategy probabilities]")
    for name, probability in normalized_strategy_weights().items():
        print(f"  {name:16s}: {probability:.3%}")

    print("\n[Route bias]")
    for name, bias in GC_MACRO_ROUTE_BIAS.items():
        print(f"  {name:16s}: {float(bias):+.2f}")
