"""original.json の特徴量と手動ラベルから分析用 XGBoost モデルを学習する。

実行例:
    py -3.13 analytics/train_analysis_models.py

JSONL からは match_path/team_name/labels を読み、保存済み features は使わない。
項目ごとにラベルと対応する実データがあるレコードを選ぶ。
"""

import argparse
import json
import pickle
import random
from pathlib import Path

try:
    from .analysis_branching import (
        AI_ANALYSIS_RULES,
        _feature_vector,
        analysis_data_available,
    )
    from .match_calculation import calculate_match_data_from_original
    from .map_analysis_data import record_map_index, team_features
except ImportError:
    from analysis_branching import AI_ANALYSIS_RULES, _feature_vector, analysis_data_available
    from match_calculation import calculate_match_data_from_original
    from map_analysis_data import record_map_index, team_features


SERIES_DATA_DIR = Path(__file__).resolve().parent.parent / "series_data"


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


def load_records(path, series_data_dir=SERIES_DATA_DIR):
    if not path.exists():
        raise ValueError(
            f"教師データがありません: {path}。"
            "試合詳細画面の「教師データを保存」を押してから再実行してください。"
        )
    records = []
    series_data_dir = series_data_dir.resolve()
    matches = {}
    map_features = {}
    legacy_count = 0
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not record.get("match_path") or not record.get("team_name"):
                    raise ValueError("match_path/team_name がありません")
                if not isinstance(record.get("labels"), dict):
                    raise ValueError("labels がありません")
                match_path = Path(record["match_path"])
                if match_path.name.endswith("_inference.json"):
                    match_path = match_path.with_name(
                        match_path.name[: -len("_inference.json")] + "_original.json"
                    )
                if not match_path.name.endswith("_original.json"):
                    raise ValueError("入力は _original.json を指定してください")
                original_path = (series_data_dir / match_path).resolve()
                if not original_path.is_relative_to(series_data_dir):
                    raise ValueError("match_path が series_data の外を指しています")
                if original_path not in matches:
                    if not original_path.is_file():
                        raise ValueError(f"original ファイルがありません: {original_path}")
                    with original_path.open("r", encoding="utf-8") as original_file:
                        matches[original_path] = json.load(original_file)
                original = matches[original_path]
                map_index = record_map_index(record, len(original.get("maps", [])))
                if map_index is None:
                    legacy_count += 1
                    continue
                key = (original_path, map_index)
                if key not in map_features:
                    map_features[key] = calculate_match_data_from_original(original, map_index)
                match = map_features[key]
                team = record["team_name"]
                if team not in match["map_aggregate"]:
                    raise ValueError(f"original にチームがありません: {team}")
                record["match_path"] = original_path.relative_to(series_data_dir).as_posix()
                record["map_index"] = map_index
                record["scope"] = "map"
                record["features"] = team_features(match, team)
                records.append(record)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if legacy_count:
        print(f"旧シリーズラベル {legacy_count} 件を除外しました。マップごとに再入力してください。")
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


def group_split(records, test_size, seed, labels=None):
    groups = sorted({record["match_path"] for record in records})
    if len(groups) < 2 or test_size <= 0:
        return list(range(len(records))), []
    random.Random(seed).shuffle(groups)
    test_count = min(len(groups) - 1, max(1, round(len(groups) * test_size)))
    test_groups = set()
    for group in groups:
        candidates = test_groups | {group}
        if labels is not None:
            remaining_classes = {
                labels[i]
                for i, record in enumerate(records)
                if record["match_path"] not in candidates
            }
            if remaining_classes != set(labels):
                continue
        test_groups = candidates
        if len(test_groups) >= test_count:
            break
    train = [i for i, record in enumerate(records) if record["match_path"] not in test_groups]
    test = [i for i, record in enumerate(records) if record["match_path"] in test_groups]
    return train, test


def eligible_record_indices(records, rule):
    return [
        index
        for index, record in enumerate(records)
        if record["labels"].get(rule["variable"]) is not None
        and analysis_data_available(record["features"], rule["variable"])
    ]


def train(args):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost がありません。requirements.txt の xgboost をインストールしてください。"
        ) from exc

    records = load_records(args.labels, args.series_data)
    feature_names = feature_names_for(records)
    matrix = [
        _feature_vector(record["features"], feature_names)[0] for record in records
    ]
    models = {}
    class_values = {}
    model_record_counts = {}
    skipped_models = {}
    for rule in AI_ANALYSIS_RULES:
        variable = rule["variable"]
        eligible_indices = eligible_record_indices(records, rule)
        if not eligible_indices:
            skipped_models[variable] = "ラベルと対応する実データが揃った試合がありません"
            print(f"{variable}: skipped ({skipped_models[variable]})")
            continue
        labels = [records[index]["labels"].get(variable) for index in eligible_indices]
        if rule["type"] == "bool":
            if any(type(value) is not bool for value in labels):
                raise ValueError(f"bool 値が不正です: {variable}")
            labels = [int(value) for value in labels]
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            allowed = set(rule["enum_values"])
            if any(value not in allowed for value in labels):
                raise ValueError(f"enum 値が不正です: {variable}")
            observed = [value for value in rule["enum_values"] if value in labels]
            class_values[variable] = observed
            labels = [observed.index(value) for value in labels]
            objective = "multi:softprob" if len(observed) > 2 else "binary:logistic"
            eval_metric = "mlogloss" if len(observed) > 2 else "logloss"
        if len(set(labels)) < 2:
            skipped_models[variable] = "ラベルが1種類しかありません"
            print(f"{variable}: skipped ({skipped_models[variable]})")
            continue
        local_train, local_test = group_split(
            [records[index] for index in eligible_indices],
            args.test_size,
            args.seed,
            labels,
        )
        train_indices = [eligible_indices[index] for index in local_train]
        test_indices = [eligible_indices[index] for index in local_test]
        label_by_index = dict(zip(eligible_indices, labels))
        print(
            f"{variable}: eligible={len(eligible_indices)} "
            f"train={len(train_indices)} holdout={len(test_indices)} "
            f"excluded={len(records) - len(eligible_indices)}"
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
        model_record_counts[variable] = {
            "eligible": len(eligible_indices),
            "train": len(train_indices),
            "holdout": len(test_indices),
        }

    if not models:
        raise ValueError("学習できる項目がありません。実データと複数種類のラベルを追加してください。")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(
            {
                "version": 3,
                "analysis_scope": "map",
                "feature_source": "original",
                "feature_names": feature_names,
                "models": models,
                "class_values": class_values,
                "model_record_counts": model_record_counts,
                "skipped_models": skipped_models,
                "record_count": len(records),
            },
            file,
        )
    print(f"保存しました: {args.output} ({len(records)} records)")


def main():
    parser = argparse.ArgumentParser(description="original と手動教師ラベルから分析モデルを学習")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).resolve().parent / "training_labels.jsonl",
    )
    parser.add_argument(
        "--series-data",
        type=Path,
        default=SERIES_DATA_DIR,
        help="入力の *_original.json がある試合データディレクトリ",
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
