"""Deterministic hashing and JSON configuration loading."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


JsonScalar = Union[None, bool, int, float, str]
JsonValue = Union[JsonScalar, List["JsonValue"], Dict[str, "JsonValue"]]


def normalize_map_text(map_text: str) -> str:
    """Normalize only transport-level newline differences around a map.

    Row contents are not stripped, so a meaningful map edit cannot be hidden by
    normalization.
    """

    if not isinstance(map_text, str):
        raise TypeError("map_text must be str")
    normalized = map_text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not normalized:
        raise ValueError("map_text must not be empty")
    return normalized


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def map_sha256(map_text: str) -> str:
    return sha256_text(normalize_map_text(map_text))


def canonical_json(value: JsonValue) -> str:
    """Serialize JSON data in the canonical form used by checkpoint hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_sha256(value: JsonValue) -> str:
    return sha256_text(canonical_json(value))


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def load_json_config(path: Union[str, Path]) -> Dict[str, JsonValue]:
    """Load a UTF-8 JSON object and reject ambiguous duplicate keys."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_number,
        )
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a JSON object: {config_path}")
    return value


def load_hashed_json_config(
    path: Union[str, Path],
) -> Tuple[Dict[str, JsonValue], str]:
    config = load_json_config(path)
    return config, canonical_json_sha256(config)
