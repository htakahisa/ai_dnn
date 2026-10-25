"""Select common knowledge plus the newest effective date for each group."""

from datetime import date, datetime, timedelta, timezone

DATE_TIMEZONE = timezone(timedelta(hours=9), "Asia/Tokyo")
AS_OF_DATE_OVERRIDE = None  # 通常は None。過去日を検証する場合だけ YYYY-MM-DD。
KNOWLEDGE_POLICY_VERSION = 2

import hashlib
import json


def validate_date(value):
    if not isinstance(value, str):
        raise ValueError("日付は YYYY-MM-DD で入力してください")
    value = value.strip()
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError("日付は YYYY-MM-DD で入力してください（例：2026-10-12）") from None
    if parsed.isoformat() != value:
        raise ValueError("日付は YYYY-MM-DD で入力してください")
    return value


def current_date():
    return validate_date(AS_OF_DATE_OVERRIDE) if AS_OF_DATE_OVERRIDE else datetime.now(DATE_TIMEZONE).date().isoformat()


def record_date(row):
    """Empty means timeless; None means an unmigrated legacy environment label."""
    if "valid_from" in row:
        value = row["valid_from"]
        return validate_date(value) if value else ""
    if row.get("scope") == "environment":
        try:
            return validate_date(row.get("environment_id", ""))
        except ValueError:
            return None
    return ""


def select_records(rows, as_of):
    as_of = validate_date(as_of)
    latest = {}
    for row in rows:
        effective = record_date(row)
        if effective and effective <= as_of:
            latest[row["group"]] = max(latest.get(row["group"], ""), effective)
    return [row for row in rows if record_date(row) == "" or (
        record_date(row) is not None and record_date(row) == latest.get(row["group"]))]


def record_fingerprint(row):
    fields = {key: row[key] for key in ("context", "question", "answer", "reason", "group")}
    fields["valid_from"] = record_date(row)
    if row.get("history"):
        fields["history"] = row["history"]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def data_manifest(rows):
    return {row["id"]: record_fingerprint(row) for row in rows}


def knowledge_signature(rows):
    return hashlib.sha256(json.dumps(data_manifest(rows), sort_keys=True).encode()).hexdigest()


def checkpoint_issue(checkpoint, rows, *, require_current_dates=False):
    # Safely adopt a v1 checkpoint only when all its tracked data remains common
    # and unchanged. This preserves existing learning without importing old facts.
    if checkpoint.get("knowledge_policy_version") == 1 and all(record_date(row) == "" and not row.get("history") for row in rows):
        old_manifest = checkpoint.get("data_manifest", {})
        legacy = {}
        for row in rows:
            fields = {key: row[key] for key in ("context", "question", "answer", "reason", "group")}
            fields.update(scope="common", environment_id="")
            legacy[row["id"]] = hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if old_manifest and all(legacy.get(key) == value for key, value in old_manifest.items()):
            current = data_manifest(rows)
            checkpoint["knowledge_policy_version"] = KNOWLEDGE_POLICY_VERSION
            checkpoint["data_manifest"] = {key: current[key] for key in old_manifest}
    if checkpoint.get("knowledge_policy_version") != KNOWLEDGE_POLICY_VERSION:
        return "日付管理に対応していない旧モデルです。新規学習が必要です"
    previous = checkpoint.get("data_manifest")
    if not isinstance(previous, dict) or not previous:
        return "モデルの教師データ履歴がありません。新規学習が必要です"
    current = data_manifest(rows)
    if any(current.get(identifier) != fingerprint for identifier, fingerprint in previous.items()):
        return "日付による更新または教師データの修正で古い情報が含まれます。新規学習してください"
    if require_current_dates and any(record_date(row) and row["id"] not in previous for row in rows):
        return "今日有効な日付付き情報がまだ学習されていません。学習を実行してください"
    return None
