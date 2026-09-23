"""AI の判定結果を、画面用の固定文言へ変換する。"""

import pickle
from pathlib import Path


class AnalysisUnavailableError(RuntimeError):
    """AI による分析を完了できなかった。"""


# 判定条件はここに置かない。AI は各 bool と enum の値だけを返す。
AI_ANALYSIS_RULES = [
    {
        "variable": "entry_coordination_needed",
        "type": "bool",
        "description": "攻撃側の初期交戦やエントリー連携に改善余地がある",
        "text": "初期交戦の連携を見直しましょう。先頭役の進路と味方の支援タイミングを合わせる練習が有効です。",
    },
    {
        "variable": "first_deaths_issue",
        "type": "bool",
        "description": "先制デスがチームの課題になっている",
        "text": "先制デスが課題です。単独で交戦する場面を減らし、味方が支援できる位置で接敵しましょう。",
    },
    {
        "variable": "retake_coordination_needed",
        "type": "bool",
        "description": "防衛側のリテイク連携に改善余地がある",
        "text": "リテイク時の連携を見直しましょう。味方の到着とアビリティの使用を合わせて進入する練習が有効です。",
    },
    {
        "variable": "post_plant_defense_needed",
        "type": "bool",
        "description": "設置後の守り方に改善余地がある",
        "text": "設置後の守り方を見直しましょう。スパイクを守れる位置取りと、解除を止めるアビリティの使い方を練習してください。",
    },
    {
        "variable": "plant_opportunities_needed",
        "type": "bool",
        "description": "攻撃側の設置機会を増やす必要がある",
        "text": "設置の機会を増やすため、サイトへの進入とスパイク保持者の護衛を見直しましょう。",
    },
    {
        "variable": "team_role_imbalance",
        "type": "bool",
        "description": "特定選手への負担や成績の偏りがチームの課題になっている",
        "text": "一部の選手に負担が偏っています。役割分担と味方への支援を見直しましょう。",
    },
    {
        "variable": "duel_support_needed",
        "type": "bool",
        "description": "一対一の交戦が課題で、味方の支援が有効である",
        "text": "単独での交戦が課題です。味方と射線を合わせ、互いを支援できる位置を取りましょう。",
    },
    {
        "variable": "attack_variation_needed",
        "type": "bool",
        "description": "攻撃方法の偏りが読み合いの課題になっている",
        "text": "攻撃方法が偏っています。進入経路や攻撃タイミングの選択肢を増やしましょう。",
    },
    {
        "variable": "time_management_needed",
        "type": "bool",
        "description": "時間切れや設置の遅れが課題になっている",
        "text": "時間の使い方を見直しましょう。残り時間から逆算し、進入と設置を始めるタイミングを決めてください。",
    },
    {
        "variable": "site_focus",
        "type": "enum",
        "default": "none",
        "description": "攻撃するサイトの偏り。十分な根拠がなければ none",
        "enum_values": ["A", "B", "none"],
        "text_map": {
            "A": "Aサイトへの攻撃が偏っています。Bサイトへの攻撃も組み込み、防衛側の配置を揺さぶりましょう。",
            "B": "Bサイトへの攻撃が偏っています。Aサイトへの攻撃も組み込み、防衛側の配置を揺さぶりましょう。",
        },
    },
    {
        "variable": "ability_usage",
        "type": "enum",
        "default": "good",
        "description": "アビリティの使い方の評価",
        "enum_values": ["excellent", "good", "poor"],
        "text_map": {
            "excellent": "アビリティの使い方が非常に優れています。この調子で効果的な使用を継続しましょう。",
            "good": "アビリティの使い方は良好です。更なるタイミングの最適化でより高い効果が期待できます。",
            "poor": "アビリティの使い方に改善の余地があります。使用するタイミングと位置を見直しましょう。",
        },
    },
    {
        "variable": "macro_precision",
        "type": "enum",
        "default": "good",
        "description": "マクロ戦術の精度（戦術・サイト選択・設置後の判断）",
        "enum_values": ["excellent", "good", "poor"],
        "text_map": {
            "excellent": "マクロ戦術の精度が非常に高いです。",
            "good": "マクロ戦術は概ね良好です。",
            "poor": "マクロ戦術の改善余地があります。",
        },
    },
    {
        "variable": "micro_precision",
        "type": "enum",
        "default": "good",
        "description": "ミクロ戦術の精度（撃ち合い・1v1・初動判断）",
        "enum_values": ["excellent", "good", "poor"],
        "text_map": {
            "excellent": "ミクロ戦術の精度が非常に高いです。",
            "good": "ミクロ戦術は概ね良好です。",
            "poor": "ミクロ戦術の改善余地があります。",
        },
    },
]


MODEL_PATH = Path(__file__).resolve().parent / "models" / "analysis_models.pkl"


def _feature_vector(data, feature_names):
    """学習スクリプトと共有する、数値特徴量の取り出し口。"""
    aggregate = data.get("map_aggregate", {})
    rounds = data.get("round_features", [])
    players = data.get("player_stats", [])
    values = {
        f"map_{key}": value
        for key, value in aggregate.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    values["round_count"] = len(rounds)
    values["site_a_count"] = sum(r.get("site") == "A" for r in rounds)
    values["site_b_count"] = sum(r.get("site") == "B" for r in rounds)
    values["time_expired_count"] = sum(
        r.get("result") == "time_expired" for r in rounds
    )
    values["player_count"] = len(players)
    for key in ("kd", "firstd", "1v1_winrate"):
        numbers = [
            p.get(key, 0) for p in players if isinstance(p.get(key, 0), (int, float))
        ]
        values[f"player_{key}_mean"] = sum(numbers) / len(numbers) if numbers else 0.0
    return [[float(values.get(name, 0.0)) for name in feature_names]]


def _request_ai_decisions(data, team_name):
    """学習済みのローカルモデルから bool/enum の値を得る。"""
    if not MODEL_PATH.exists():
        raise AnalysisUnavailableError(
            f"学習済みモデルがありません: {MODEL_PATH}。先に教師データで学習してください。"
        )
    try:
        with MODEL_PATH.open("rb") as file:
            bundle = pickle.load(file)
        feature_names = bundle["feature_names"]
        models = bundle["models"]
        row = _feature_vector(data, feature_names)
        decisions = {}
        for rule in AI_ANALYSIS_RULES:
            model = models[rule["variable"]]
            probabilities = model.predict_proba(row)[0]
            classes = list(model.classes_)
            value = classes[
                max(range(len(probabilities)), key=probabilities.__getitem__)
            ]
            if rule["type"] == "bool":
                decisions[rule["variable"]] = bool(value)
            else:
                class_index = int(value)
                if class_index < 0 or class_index >= len(rule["enum_values"]):
                    raise ValueError(f"enum class index out of range: {class_index}")
                decisions[rule["variable"]] = rule["enum_values"][class_index]
        return decisions
    except (OSError, KeyError, ValueError, AttributeError, ImportError) as exc:
        raise AnalysisUnavailableError(
            f"ローカルモデルを読み込めませんでした: {exc}"
        ) from exc


def format_ai_decisions(decisions):
    """AI の型付き判定のみを許可し、固定の表示文へ変換する。"""
    if not isinstance(decisions, dict) or set(decisions) != {
        rule["variable"] for rule in AI_ANALYSIS_RULES
    }:
        raise AnalysisUnavailableError("AI の判定項目が一致しません。")

    results = []
    for rule in AI_ANALYSIS_RULES:
        value = decisions[rule["variable"]]
        if rule["type"] == "bool":
            if type(value) is not bool:
                raise AnalysisUnavailableError("AI の bool 判定が不正です。")
            if not value:
                # attack_variation_neededがfalseの場合、site_focusの値に応じてメッセージを追加
                if rule["variable"] == "attack_variation_needed":
                    site_focus_value = decisions["site_focus"]
                    if site_focus_value in ["A", "B"]:
                        site_text = f"{site_focus_value}サイト"
                        results.append(
                            {
                                "variable": "attack_variation_approved",
                                "value": False,
                                "text": f"{site_text}への攻撃が偏っていますが、相手の弱点を突いた結果であり問題ありません。",
                                "type": "info",
                                "severity": "info",
                            }
                        )
                continue
            output_text = rule["text"]
        else:
            if not isinstance(value, str) or value not in rule["enum_values"]:
                raise AnalysisUnavailableError("AI の enum 判定が不正です。")
            if value == "none":
                continue
            output_text = rule["text_map"][value]
        results.append(
            {
                "variable": rule["variable"],
                "value": value,
                "text": output_text,
                "type": rule["type"],
                "severity": "warning",
            }
        )
    return results


def analyze_match_data(data, team_name):
    """分析データを AI に渡し、改善点を {variable, value, text} 形式で返す。"""
    return format_ai_decisions(_request_ai_decisions(data, team_name))
