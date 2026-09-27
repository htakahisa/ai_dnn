"""Shared source paths and map identities for the web UI and training data."""

import json
from pathlib import Path


def original_match_path(match_path):
    path = Path(match_path)
    if path.name.endswith("_inference.json"):
        path = path.with_name(path.name[: -len("_inference.json")] + "_original.json")
    if not path.name.endswith("_original.json"):
        raise ValueError("入力は _original.json を指定してください")
    return path.as_posix()


def load_original(match_path, series_data_dir):
    root = Path(series_data_dir).resolve()
    path = (root / original_match_path(match_path)).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Invalid match path")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data.get("maps"), list) or not data["maps"]:
        raise ValueError("original にマップデータがありません")
    return path.relative_to(root).as_posix(), data


def validate_map_index(map_index, map_count):
    if type(map_index) is not int or not 0 <= map_index < map_count:
        raise ValueError("Invalid map index")
    return map_index


def record_map_index(record, map_count):
    """A series label can only be transferred unambiguously for a single map."""
    if "map_index" in record:
        return validate_map_index(record["map_index"], map_count)
    return 0 if map_count == 1 else None


def map_key(match_path, map_index):
    return f"{original_match_path(match_path)}#map={map_index}"


def team_features(match_data, team):
    if team not in match_data["map_aggregate"]:
        raise ValueError(f"original にチームがありません: {team}")
    return {
        **match_data,
        "map_aggregate": match_data["map_aggregate"][team],
        "player_stats": [p for p in match_data["player_stats"] if p.get("team") == team],
    }
