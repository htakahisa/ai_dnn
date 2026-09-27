from flask import redirect, render_template, request, url_for
import json
from pathlib import Path
from datetime import datetime, timezone
import sys

if __package__:
    from .analysis_branching import AI_ANALYSIS_RULES, AnalysisUnavailableError, analyze_match_data
    from .match_calculation import calculate_match_data_from_original
    from .map_analysis_data import (
        load_original, map_key, original_match_path, record_map_index,
        team_features, validate_map_index,
    )
else:
    from analysis_branching import AI_ANALYSIS_RULES, AnalysisUnavailableError, analyze_match_data
    from match_calculation import calculate_match_data_from_original
    from map_analysis_data import (
        load_original, map_key, original_match_path, record_map_index,
        team_features, validate_map_index,
    )

ANALYTICS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYTICS_DIR.parent
SERIES_DATA_DIR = PROJECT_ROOT / "series_data"
TRAINING_LABELS_PATH = ANALYTICS_DIR / "training_labels.jsonl"
PENDING_LABELS_PATH = ANALYTICS_DIR / "training_labels_pending_maps.jsonl"
COMPETITION_RESULTS_DIR = PROJECT_ROOT / "competition_results"
FAVORITES_PATH = ANALYTICS_DIR / "favorites.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from map_data import NEW_MAZE_STR
    MAP_GRID = [
        [int(cell) for cell in row.strip()]
        for row in NEW_MAZE_STR.strip().splitlines() if row.strip()
    ]
except (ImportError, ValueError):
    MAP_GRID = []


_original_match_path = original_match_path


def _load_replay_frames(original_data, map_index=0):
    """Recover only the selected map, matching the series and map identity."""
    maps = original_data.get("maps") or []
    validate_map_index(map_index, len(maps))
    selected = maps[map_index]
    if selected.get("replay_frames"):
        return selected["replay_frames"]
    signature = [
        (m.get("number"), m.get("seed"), m.get("score1"), m.get("score2"))
        for m in maps
    ]
    team1, team2 = original_data["team1"], original_data["team2"]
    pattern = (
        f"series_{team1.replace(' ', '_')}_vs_{team2.replace(' ', '_')}_"
        f"{original_data.get('team1_score', 0)}-{original_data.get('team2_score', 0)}_*.json"
    )
    candidates = []
    for path in COMPETITION_RESULTS_DIR.glob(pattern):
        try:
            with path.open("r", encoding="utf-8") as file:
                result = json.load(file)
            if result.get("team1") != team1 or result.get("team2") != team2:
                continue
            result_maps = result.get("maps", [])
            if [
                (m.get("number"), m.get("seed"), m.get("score1"), m.get("score2"))
                for m in result_maps
            ] != signature:
                continue
            frames = result_maps[map_index].get("replay_frames")
            if frames:
                candidates.append((path.stat().st_mtime, frames))
        except (OSError, ValueError, KeyError, IndexError):
            continue
    return max(candidates, default=(0, []), key=lambda item: item[0])[1]


def _label_records(path):
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record["match_path"] = original_match_path(record["match_path"])
                records.append(record)
            except (ValueError, KeyError, TypeError) as exc:
                print(f"Error parsing training label {path}:{line_number}: {exc}")
    return records


def _map_labels(match_path, map_index, map_count):
    labels = {}
    for record in _label_records(TRAINING_LABELS_PATH):
        if record["match_path"] != match_path:
            continue
        if record_map_index(record, map_count) != map_index:
            continue
        labels.setdefault(record["team_name"], {}).update(
            {key: value for key, value in record.get("labels", {}).items() if value is not None}
        )
    return labels


def _legacy_labels(match_path):
    labels = {}
    # Also support older, not-yet-migrated JSONL files.
    for source in (PENDING_LABELS_PATH, TRAINING_LABELS_PATH):
        for record in _label_records(source):
            if record["match_path"] == match_path and "map_index" not in record:
                labels.setdefault(record["team_name"], {}).update(record.get("labels", {}))
    return labels


def _load_match_for_path(match_path, map_index=None):
    canonical_path, original = load_original(match_path, SERIES_DATA_DIR)
    if map_index is None:
        if len(original["maps"]) != 1:
            raise ValueError("マップを選択してください")
        map_index = 0
    return calculate_match_data_from_original(original, map_index)


def _map_listing(match_path, original, timestamp):
    result = []
    for map_index, data in enumerate(original["maps"]):
        result.append({
            "folder_name": Path(match_path).parent.name,
            "path": match_path,
            "map_index": map_index,
            "key": map_key(match_path, map_index),
            "metadata": {
                "team1": original["team1"], "team2": original["team2"],
                "team1_score": data.get("score1", 0), "team2_score": data.get("score2", 0),
                "series_team1_score": original["team1_score"],
                "series_team2_score": original["team2_score"],
                "map_number": data.get("number", map_index + 1),
                "map_count": len(original["maps"]),
                "map_name": data.get("map_name", f"Map {map_index + 1}"),
                "total_rounds": len(data.get("round_records", [])),
                "timestamp": timestamp,
            },
        })
    return result


def save_training_labels(match_path, map_index=None):
    try:
        match_path, original = load_original(match_path, SERIES_DATA_DIR)
        if map_index is None:
            if len(original["maps"]) != 1:
                return "マップを選択してください", 400
            map_index = 0
        validate_map_index(map_index, len(original["maps"]))
        match_data = calculate_match_data_from_original(original, map_index)
        existing = _map_labels(match_path, map_index, len(original["maps"]))
        records = []
        for team in (original["team1"], original["team2"]):
            labels = dict(existing.get(team, {}))
            for rule in AI_ANALYSIS_RULES:
                variable = rule["variable"]
                field = f"{team}__{variable}"
                # Only fields shown/submitted by this form are updated.
                if f"{field}__present" not in request.form:
                    continue
                if rule["type"] == "bool":
                    labels[variable] = field in request.form
                else:
                    value = request.form.get(field)
                    if value not in rule["enum_values"]:
                        return f"Invalid enum value: {variable}", 400
                    labels[variable] = value
            records.append({
                "schema_version": 3, "scope": "map",
                "match_path": match_path, "map_index": map_index,
                "map_number": match_data["match_metadata"]["map_number"],
                "team_name": team, "features": team_features(match_data, team),
                "labels": labels, "labeled_at": datetime.now(timezone.utc).isoformat(),
            })
        TRAINING_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRAINING_LABELS_PATH.open("a", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return redirect(url_for("map_detail", match_path=match_path, map_index=map_index))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return f"ラベル保存に失敗しました: {exc}", 400


def match_detail(match_path, map_index=None):
    try:
        match_path, original = load_original(match_path, SERIES_DATA_DIR)
        timestamp = datetime.fromtimestamp(
            (SERIES_DATA_DIR / match_path).stat().st_mtime, timezone.utc
        ).isoformat()
        maps = _map_listing(match_path, original, timestamp)
        if map_index is None:
            if len(maps) == 1:
                return redirect(url_for("map_detail", match_path=match_path, map_index=0))
            return render_template("series.html", series=original, maps=maps)
        validate_map_index(map_index, len(maps))
        selected = original["maps"][map_index]
        match_data = calculate_match_data_from_original(original, map_index)
        match_data["match_metadata"]["timestamp"] = timestamp
        replay_frames = _load_replay_frames(original, map_index)
        replay_rounds = [
            {
                **record,
                "attacker_team": features["attacker_team"],
                "defender_team": features["defender_team"],
                "winner": features["winner_team"],
            }
            for record, features in zip(selected.get("round_records", []), match_data["round_features"])
        ]
        replay_data = {
            "team1": original["team1"], "team2": original["team2"],
            "rounds": replay_rounds,
            "players": selected.get("player_stats", []),
            "team1_score": selected.get("score1", 0),
            "team2_score": selected.get("score2", 0),
        }
        labels = _map_labels(match_path, map_index, len(maps))
        required = {rule["variable"] for rule in AI_ANALYSIS_RULES}
        missing = {
            team: sorted(required - set(labels.get(team, {})))
            for team in (original["team1"], original["team2"])
        }
        match_data["ai_analysis"] = {}
        match_data["ai_analysis_errors"] = {}
        for team in (original["team1"], original["team2"]):
            try:
                match_data["ai_analysis"][team] = analyze_match_data(team_features(match_data, team), team)
            except AnalysisUnavailableError as exc:
                match_data["ai_analysis"][team] = []
                match_data["ai_analysis_errors"][team] = str(exc)
        return render_template(
            "match.html", match=match_data, match_path=match_path, map_index=map_index,
            maps=maps, analysis_rules=AI_ANALYSIS_RULES, missing_labels=missing,
            legacy_labels=_legacy_labels(match_path) if len(maps) > 1 else {},
            original_data=replay_data, replay_frames=replay_frames, map_grid=MAP_GRID,
        )
    except FileNotFoundError:
        return "Original match file not found", 404
    except ValueError as exc:
        return str(exc), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return f"Error loading match: {exc}", 500


def get_all_matches():
    matches = []
    if SERIES_DATA_DIR.exists():
        for path in SERIES_DATA_DIR.glob("*/*_original.json"):
            try:
                match_path, original = load_original(path.relative_to(SERIES_DATA_DIR).as_posix(), SERIES_DATA_DIR)
                timestamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                matches.extend(_map_listing(match_path, original, timestamp))
            except (OSError, KeyError, ValueError) as exc:
                print(f"Error loading {path}: {exc}")
    return sorted(matches, key=lambda item: (item["metadata"]["timestamp"], -item["map_index"]), reverse=True)


def get_missing_label_stats(matches=None):
    matches = get_all_matches() if matches is None else matches
    counts = {item["path"]: item["metadata"]["map_count"] for item in matches}
    labels = {}
    for record in _label_records(TRAINING_LABELS_PATH):
        path = record["match_path"]
        if path not in counts:
            continue
        index = record_map_index(record, counts[path])
        if index is None:
            continue
        key = (path, index, record["team_name"])
        labels.setdefault(key, set()).update(
            name for name, value in record.get("labels", {}).items() if value is not None
        )
    required = {rule["variable"] for rule in AI_ANALYSIS_RULES}
    return [
        {
            "match_path": item["path"], "map_index": item["map_index"],
            "map_number": item["metadata"]["map_number"], "team_name": team,
            "missing_labels": sorted(required - labels.get((item["path"], item["map_index"], team), set())),
            "labeled_at": "",
        }
        for item in matches
        for team in (item["metadata"]["team1"], item["metadata"]["team2"])
        if required - labels.get((item["path"], item["map_index"], team), set())
    ]


def get_favorites():
    if not FAVORITES_PATH.exists():
        return []
    with FAVORITES_PATH.open("r", encoding="utf-8") as file:
        favorites = json.load(file)
    normalized = []
    for value in favorites:
        path, separator, index = value.partition("#map=")
        normalized.append(map_key(path, int(index)) if separator else original_match_path(path))
    return normalized


def save_favorite(match_path, map_index=None):
    try:
        match_path, original = load_original(match_path, SERIES_DATA_DIR)
        if map_index is None:
            if len(original["maps"]) != 1:
                return "マップを選択してください", 400
            map_index = 0
        validate_map_index(map_index, len(original["maps"]))
        favorites = get_favorites()
        # Preserve old series favorites as favorites of all its maps.
        if match_path in favorites:
            favorites.remove(match_path)
            favorites.extend(map_key(match_path, index) for index in range(len(original["maps"])))
        key = map_key(match_path, map_index)
        if key in favorites:
            favorites.remove(key)
        else:
            favorites.append(key)
        with FAVORITES_PATH.open("w", encoding="utf-8") as file:
            json.dump(list(dict.fromkeys(favorites)), file, ensure_ascii=False, indent=2)
        return redirect(url_for("index"))
    except (OSError, ValueError, KeyError) as exc:
        return f"お気に入りの保存に失敗しました: {exc}", 400


def index():
    matches = get_all_matches()
    return render_template(
        "index.html", matches=matches,
        missing_labels=get_missing_label_stats(matches), favorites=get_favorites(),
    )
