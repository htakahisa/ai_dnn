"""Teacher data validation, append-only collection, and grouped holdout."""

import hashlib
import json
from pathlib import Path

from tl1_dates_tactics import record_date, select_records, validate_date
from tl1_conversations_tactics import validate_history

SCHEMA_VERSION = 1


def default_group(context, question):
    text = context.strip() or question.strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def scenario_key(row):
    # Preserve scenario history without copying full teacher text into metadata.
    text = ("context:" + row["context"].strip() if row["context"].strip()
            else "question:" + row["question"].strip())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_record(row):
    if not isinstance(row, dict) or row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version=1 のオブジェクトが必要です")
    for key in ("id", "question", "answer", "group"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"{key} は空でない文字列にしてください")
    for key in ("context", "reason"):
        if not isinstance(row.get(key), str):
            raise ValueError(f"{key} は文字列にしてください（空文字も可）")
    validate_history(row.get("history", []))
    if "valid_from" in row:
        if not isinstance(row["valid_from"], str):
            raise ValueError("適用開始日は文字列にしてください（空欄なら共通）")
        if row["valid_from"]:
            validate_date(row["valid_from"])
    elif row.get("scope", "common") not in ("common", "environment"):
        raise ValueError("データの種類が不正です")
    elif row.get("scope") == "environment" and (
        not isinstance(row.get("environment_id"), str) or not row["environment_id"].strip()
    ):
        raise ValueError("環境依存のデータには対象環境を入力してください")
    return row


def load_records(path, as_of_date=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"教師データがありません: {path}。先に tl1_collect_tactics.py を実行してください")
    records, ids, prompts = [], set(), {}
    line_numbers = {}
    with path.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = validate_record(json.loads(line))
                if row["id"] in ids:
                    raise ValueError("id が重複しています")
                ids.add(row["id"])
                line_numbers[row["id"]] = line_number
                records.append(row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if as_of_date is not None:
        records = select_records(records, as_of_date)
    deduplicated = []
    for row in records:
        key = (row["context"].strip(), row["question"].strip(),
               json.dumps(row.get("history", []), ensure_ascii=False, sort_keys=True))
        if as_of_date is None:
            key = (record_date(row), row.get("environment_id", "") if record_date(row) is None else "", *key)
        if key in prompts:
            previous = prompts[key]
            if (row["answer"], row["reason"]) != (previous["answer"], previous["reason"]):
                raise ValueError(f"{path}:{line_numbers[row['id']]}: 同じ日付・状況・質問に異なる正解があります。旧回答を修正してください")
            continue
        prompts[key] = row
        deduplicated.append(row)
    if not deduplicated:
        raise ValueError("今日有効な教師データがありません。日付を確認してください" if as_of_date else "教師データが空です")
    return deduplicated


def split_records(records, fraction, seed, previous=None, previous_scenarios=None):
    """Keep scenarios/groups together, and retain assignments when resuming."""
    if not 0 <= fraction < 1:
        raise ValueError("VALIDATION_FRACTION は0以上1未満にしてください")
    groups = sorted({row["group"] for row in records})
    assignment = dict(previous or {})
    previous_scenarios = previous_scenarios or {}
    # Equal non-empty contexts represent the same scenario, even if group differs.
    parent = {group: group for group in groups}

    def root(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    contexts = {}
    for row in records:
        context = scenario_key(row)
        if context in contexts:
            parent[root(row["group"])] = root(contexts[context])
        contexts[context] = row["group"]
    components = {}
    for group in groups:
        components.setdefault(root(group), []).append(group)
    ordered = sorted(components.values(), key=lambda component:
                     hashlib.sha256(f"{seed}:{min(component)}".encode()).hexdigest())
    if not previous:
        count = (min(len(ordered) - 1, max(1, int(len(ordered) * fraction)))
                 if len(ordered) > 1 and fraction > 0 else 0)
        for index, component in enumerate(ordered):
            for group in component:
                assignment[group] = "validation" if index < count else "train"
    else:
        for component in ordered:
            old = {assignment[group] for group in component if group in assignment}
            old.update(previous_scenarios[scenario_key(row)] for row in records
                       if row["group"] in component and scenario_key(row) in previous_scenarios)
            if len(old) > 1:
                raise ValueError("新データが学習・評価の既存グループを結合しています。グループを確認して新規学習してください")
            digest = hashlib.sha256(f"{seed}:{min(component)}".encode()).hexdigest()
            choice = next(iter(old)) if old else (
                "validation" if int(digest[:8], 16) / 2**32 < fraction else "train")
            for group in component:
                assignment[group] = choice
    train = [row for row in records if assignment[row["group"]] == "train"]
    validation = [row for row in records if assignment[row["group"]] == "validation"]
    if not train:
        raise ValueError("学習側のデータがありません。新しい状況の教師データを追加してください")
    return train, validation, assignment


def scenario_assignments(records, assignment, previous=None):
    result = dict(previous or {})
    for row in records:
        key, choice = scenario_key(row), assignment[row["group"]]
        if key in result and result[key] != choice:
            raise ValueError("以前学習した状況が評価側に移っています。別の保存先で新規学習してください")
        result[key] = choice
    return result
