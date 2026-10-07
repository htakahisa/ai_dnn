"""Choose evaluated attack routes for the current defending AI."""

import json
import math
from pathlib import Path

from concon_v1.co1_retake_models import OPPONENT_NAMES

SELECTION_PATH = Path(__file__).resolve().parent / "data" / "attacker_selection" / "opponent_routes.json"
METRIC = "plant_or_preplant_defender_elimination"


def load_selection(path=SELECTION_PATH):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["version"] != 1 or report["metric"] != METRIC:
            raise ValueError("unsupported attacker selection report")
        if not math.isfinite(report["threshold"]) or not 0 <= report["threshold"] <= 1:
            raise ValueError("invalid selection threshold")
        if not isinstance(report["results"], list):
            raise ValueError("selection results must be a list")
        for row in report["results"]:
            if not isinstance(row["model_sha256"], str) or row["rounds"] < 1:
                raise ValueError("invalid evaluation provenance")
            if not math.isfinite(row["attack_success_rate"]) or not 0 <= row["attack_success_rate"] <= 1:
                raise ValueError("invalid attack success rate")
            if not isinstance(row["opponent"], str) or not isinstance(row["map_name"], str):
                raise ValueError("invalid evaluation target")
        keys = [(row["opponent"], row["map_name"]) for row in report["results"]]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate evaluation results")
        return report
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[ConCon attacker] cannot use evaluation {path}: {error}; using all routes", flush=True)
        return None


def route_candidates(report, opponent, map_names, model_hashes):
    """Only rank a complete set of evaluations for the loaded model bytes."""
    if report is None or opponent is None:
        return tuple(map_names)
    rows = {row["map_name"]: row for row in report["results"] if row["opponent"] == opponent}
    if any(name not in rows or rows[name]["model_sha256"] != model_hashes[name] for name in map_names):
        return tuple(map_names)
    eligible = tuple(name for name in map_names if rows[name]["attack_success_rate"] >= report["threshold"])
    if eligible:
        return eligible
    best_rate = max(rows[name]["attack_success_rate"] for name in map_names)
    return tuple(name for name in map_names if rows[name]["attack_success_rate"] == best_rate)


def defending_opponent(game):
    ai = getattr(game, "current_defender_team_ai", None)
    return OPPONENT_NAMES.get(getattr(ai, "name", None))
