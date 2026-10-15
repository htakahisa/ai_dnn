"""Compact opponent-specific route priors, cached once per controller."""

import hashlib
import json
import math
from pathlib import Path

from grid_lines import WALL_LOS_VERSION


AI_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AI_DIR.parent
SUMMARY_SCHEMA_VERSION = 1
OPPONENT_KEYS = {
    "Toru AI v4": "toru_ai_v4",
    "ConCon v1": "concon_v1",
    "Touyama Gaming v3": "touyama_gaming_v3",
    "Fnatic v3": "fnatic_v3",
    "FRC v1": "frc_v1",
    "Ghost Champions v2": "ghost_champions_v2",
}


def route_id(site, point):
    return f"{site}:{int(point[0])},{int(point[1])}"


def map_signature(maze):
    # Adding/removing waypoint markers on existing floors preserves old priors.
    lines = [line.strip().replace("6", "0").replace("7", "0")
             for line in maze.splitlines() if line.strip()]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


class RoutePriorCache:
    def __init__(self, path, maze, model_path):
        self.opponents = {}
        self.status = "missing"
        try:
            # This is the only summary read; no round/tick code accesses files.
            with Path(path).open(encoding="utf-8") as stream:
                summary = json.load(stream)
            if (summary.get("schema_version") != SUMMARY_SCHEMA_VERSION
                    or summary.get("map_signature") != map_signature(maze)
                    or summary.get("wall_los_version") != WALL_LOS_VERSION):
                raise ValueError("summary schema/map/LOS differs")
            model_hashes = summary.get("model_hashes", {})
            relative = Path(model_path).resolve().relative_to(PROJECT_DIR).as_posix()
            expected = model_hashes.get(relative)
            if not expected:
                raise ValueError("carry model was not evaluated")
            # Check our complete four-phase bundle once; guard/escort changes
            # also change the measured route win rate.
            for name, digest in model_hashes.items():
                if name.startswith("omoko_v1/"):
                    if hashlib.sha256((PROJECT_DIR / name).read_bytes()).hexdigest() != digest:
                        raise ValueError(f"model differs: {name}")
            for key, entry in summary.get("opponents", {}).items():
                if key not in OPPONENT_KEYS.values():
                    continue
                rates = entry.get("win_rates", {})
                if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
                       for v in rates.values()):
                    raise ValueError(f"invalid win rates: {key}")
                self.opponents[key] = {"players": tuple(entry["players"]), "win_rates": dict(rates)}
            self.status = "loaded"
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            self.opponents = {}
            self.status = f"unavailable: {exc}"

    def probabilities(self, game, routes, waypoints, mix):
        team_ai = getattr(game, "current_defender_team_ai", None)
        name = str(getattr(team_ai, "name", ""))
        key = OPPONENT_KEYS.get(name, name if name in OPPONENT_KEYS.values() else None)
        entry = self.opponents.get(key)
        if not entry or not routes:
            return None
        players = tuple(str(name) for name in getattr(game, "defender_roster", ()) or ())
        if players != entry["players"]:
            return None
        ids = [route_id(site, waypoints[site][index]) if waypoints.get(site) else None
               for site, index in routes]
        rates = entry["win_rates"]
        known = [rates[key] for key in ids if key in rates]
        if not known:
            return None
        # An added, unevaluated route receives the known routes' mean strength,
        # so its mixed probability remains the neutral 1/N rather than zero.
        neutral = sum(known) / len(known)
        values = [rates.get(key, neutral) for key in ids]
        total = sum(values)
        uniform = 1.0 / len(routes)
        prior = [value / total for value in values] if total else [uniform] * len(routes)
        return [(1.0 - mix) * uniform + mix * value for value in prior]
