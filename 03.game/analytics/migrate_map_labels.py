"""Preserve series labels and migrate unambiguous labels to individual maps.

Preview: python analytics/migrate_map_labels.py
Apply:   python analytics/migrate_map_labels.py --apply
No model training is performed.
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .map_analysis_data import load_original, record_map_index, team_features
    from .match_calculation import calculate_match_data_from_original
else:
    from map_analysis_data import load_original, record_map_index, team_features
    from match_calculation import calculate_match_data_from_original


def migrate_records(records, series_data_dir):
    active, pending = [], []
    sources, features = {}, {}
    for record in records:
        path = record["match_path"]
        if path not in sources:
            sources[path] = load_original(path, series_data_dir)
        canonical, original = sources[path]
        index = record_map_index(record, len(original["maps"]))
        if index is None:
            pending.append({
                **record, "match_path": canonical, "scope": "series",
                "migration_status": "needs_map_labels",
            })
            continue
        key = (canonical, index)
        if key not in features:
            features[key] = calculate_match_data_from_original(original, index)
        match = features[key]
        active.append({
            **record, "schema_version": 3, "scope": "map",
            "match_path": canonical, "map_index": index,
            "map_number": match["match_metadata"]["map_number"],
            "features": team_features(match, record["team_name"]),
        })
    return active, pending


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, records):
    temporary = path.with_name(path.name + ".migration.tmp")
    if temporary.exists():
        raise ValueError(f"移行用ファイルが既に存在します: {temporary}")
    with temporary.open("x", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def migrate(labels_path, series_data_dir, pending_path, apply=False):
    if not labels_path.is_file():
        raise ValueError(f"教師データがありません: {labels_path}")
    if labels_path.resolve() == pending_path.resolve():
        raise ValueError("教師データと保管先は別のファイルを指定してください")
    records = read_jsonl(labels_path)
    active, pending = migrate_records(records, series_data_dir)
    previous_pending = read_jsonl(pending_path)
    combined_pending = previous_pending + [record for record in pending if record not in previous_pending]
    print(f"map_records={len(active)}, series_records_needing_labels={len(combined_pending)}")
    if not apply:
        return active, combined_pending
    if records == active and combined_pending == previous_pending:
        print("移行済みです。変更はありません。")
        return active, combined_pending
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    backup = labels_path.with_name(f"{labels_path.stem}.series_backup_{stamp}{labels_path.suffix}")
    shutil.copy2(labels_path, backup)
    # The backup retains every original label before either dataset is updated.
    write_jsonl(pending_path, combined_pending)
    write_jsonl(labels_path, active)
    print(f"backup={backup}")
    print(f"pending={pending_path}")
    return active, combined_pending


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="シリーズ単位の教師データをマップ単位へ移行")
    parser.add_argument("--labels", type=Path, default=directory / "training_labels.jsonl")
    parser.add_argument("--series-data", type=Path, default=directory.parent / "series_data")
    parser.add_argument("--pending", type=Path, default=directory / "training_labels_pending_maps.jsonl")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        migrate(args.labels, args.series_data, args.pending, args.apply)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
