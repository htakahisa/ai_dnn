"""手動ラベル JSONL から、改善点判定用の XGBoost モデルを学習する。

実行例:
    py -3.13 analytics/train_analysis_models.py
"""

import argparse
import json
import pickle
import random
from pathlib import Path

try:
    from .analysis_branching import AI_ANALYSIS_RULES, _feature_vector
except ImportError:
    from analysis_branching import AI_ANALYSIS_RULES, _feature_vector


DERIVED_FEATURES = [
    "round_count",
    "site_a_count",
    "site_b_count",
    "time_expired_count",
    "player_count",
    "player_kd_mean",
    "player_firstd_mean",
    "player_1v1_winrate_mean",
]


def load_records(path):
    if not path.exists():
        raise ValueError(
            f"教師データがありません: {path}。"
            "試合詳細画面の「教師データを保存」を押してから再実行してください。"
        )
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not record.get("match_path") or not record.get("team_name"):
                    raise ValueError("match_path/team_name がありません")
                if not isinstance(record.get("features"), dict):
                    raise ValueError("features がありません")
                if not isinstance(record.get("labels"), dict):
                    raise ValueError("labels がありません")
                records.append(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"教師データが空です: {path}")
    return records


def feature_names_for(records):
    names = {
        f"map_{key}"
        for record in records
        for key, value in record["features"].get("map_aggregate", {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return sorted(names) + DERIVED_FEATURES


def group_split(records, test_size, seed):
    groups = sorted({record["match_path"] for record in records})
    if len(groups) < 2 or test_size <= 0:
        return list(range(len(records))), []
    random.Random(seed).shuffle(groups)
    test_count = max(1, round(len(groups) * test_size))
    test_groups = set(groups[:test_count])
    train = [i for i, record in enumerate(records) if record["match_path"] not in test_groups]
    test = [i for i, record in enumerate(records) if record["match_path"] in test_groups]
    return train, test


def train(args):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost がありません。requirements.txt の xgboost をインストールしてください。"
        ) from exc

    records = load_records(args.labels)
    feature_names = feature_names_for(records)
    matrix = [
        _feature_vector(record["features"], feature_names)[0] for record in records
    ]
    train_indices, test_indices = group_split(records, args.test_size, args.seed)
    models = {}
    for rule in AI_ANALYSIS_RULES:
        variable = rule["variable"]
        eligible_indices = list(range(len(records)))
        if variable == "ability_usage":
            eligible_indices = [
                index
                for index, record in enumerate(records)
                if record["features"].get("map_aggregate", {}).get(
                    "ability_data_available", 0
                )
            ]
            if not eligible_indices:
                raise ValueError(
                    "ability_usage を学習できる試合がありません。"
                    "アビリティ残数を記録した新しい試合データを追加してください。"
                )
        labels = [records[index]["labels"].get(variable) for index in eligible_indices]
        if any(value is None for value in labels):
            raise ValueError(f"ラベルが不足しています: {variable}")
        local_groups = sorted({records[index]["match_path"] for index in eligible_indices})
        random.Random(args.seed).shuffle(local_groups)
        local_test_count = max(1, round(len(local_groups) * args.test_size)) if local_groups and args.test_size > 0 else 0
        local_test_groups = set(local_groups[:local_test_count])
        train_indices = [index for index in eligible_indices if records[index]["match_path"] not in local_test_groups]
        test_indices = [index for index in eligible_indices if records[index]["match_path"] in local_test_groups]
        if rule["type"] == "bool":
            labels = [int(value) for value in labels]
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            allowed = set(rule["enum_values"])
            if any(value not in allowed for value in labels):
                raise ValueError(f"enum 値が不正です: {variable}")
            labels = [rule["enum_values"].index(value) for value in labels]
            objective = "multi:softprob"
            eval_metric = "mlogloss"
        label_by_index = dict(zip(eligible_indices, labels))
        if len(set(labels)) < 2:
            raise ValueError(
                f"{variable} はラベルが1種類しかありません。複数の結果を入力してください。"
            )
        train_classes = {label_by_index[i] for i in train_indices}
        if len(train_classes) < 2:
            raise ValueError(
                f"{variable} は学習用分割に複数クラスがありません。教師データを増やしてください。"
            )

        try:
            model = XGBClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                subsample=0.9,
                colsample_bytree=0.9,
                objective=objective,
                eval_metric=eval_metric,
                random_state=args.seed,
                n_jobs=args.n_jobs,
            )
        except ImportError as exc:
            raise RuntimeError(
                "XGBClassifier の利用には scikit-learn が必要です。"
                "仮想環境で `python -m pip install scikit-learn` を実行してください。"
            ) from exc
        model.fit(
            [matrix[i] for i in train_indices],
            [label_by_index[i] for i in train_indices],
        )
        if test_indices and len({label_by_index[i] for i in train_indices}) == len(set(labels)):
            predictions = model.predict([matrix[i] for i in test_indices])
            correct = sum(
                prediction == label_by_index[i]
                for prediction, i in zip(predictions, test_indices)
            )
            print(f"{variable}: holdout_accuracy={correct / len(test_indices):.3f}")
        models[variable] = model

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(
            {
                "version": 1,
                "feature_names": feature_names,
                "models": models,
                "record_count": len(records),
            },
            file,
        )
    print(f"保存しました: {args.output} ({len(records)} records)")


def main():
    parser = argparse.ArgumentParser(description="手動教師データから分析モデルを学習")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).resolve().parent / "training_labels.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "analysis_models.pkl",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    try:
        train(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
