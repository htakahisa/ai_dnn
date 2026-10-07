"""Player rankings from the active season's isolated match logs."""

import json
from pathlib import Path

from strongest_ranking import STAT_FIELDS, aggregate_summaries, summarize_record


def summarize_season_record(result, request, source):
    if result.get("status") != "completed":
        return []
    summaries = []
    for game in result.get("maps", [result]):
        if game.get("status") != "completed":
            continue
        players = []
        stats = game.get("player_stats", {})
        if not isinstance(stats, dict):
            raise ValueError("選手戦績の形式が不正です")
        for key, team_field in (("own", "own_team"), ("opponent", "opponent_team")):
            team = request[key]
            for member in team["players"]:
                name = member["name"]
                if name not in stats:
                    continue
                row = stats[name]
                players.append({"name": name, "team": game[team_field],
                                "role": row.get("role") or member.get("role", ""),
                                **{field: row.get(field) for field in STAT_FIELDS}})
        normalized = {"team1": game["own_team"], "team2": game["opponent_team"],
                      "score1": game["own_score"], "score2": game["opponent_score"],
                      "seed": game.get("seed"), "player_stats": players,
                      "round_records": game.get("round_records", [])}
        # Same MVP rule as the competition manager: most kills, then fewest
        # deaths, then roster order. Missing historical stats stay unknown.
        for number, field in ((1, "own_team"), (2, "opponent_team")):
            members = [p for p in players if p["team"] == game[field]]
            if members and all(p["kills"] is not None and p["deaths"] is not None for p in members):
                normalized[f"mvp{number}"] = {"name": max(members, key=lambda p: (p["kills"], -p["deaths"]))["name"]}
        summaries.extend(summarize_record(normalized, source))
    return summaries


def load_season_rankings(save_path, progress=None):
    directory = Path(save_path).resolve().parent
    paths = sorted((directory / "scrims").glob("*/result.json"))
    paths += sorted((directory / "tournaments").glob("*/*/result.json"))
    summaries, errors = [], []
    for index, path in enumerate(paths):
        source = path.relative_to(directory).as_posix()
        if progress:
            progress(f"戦績を確認中: {index + 1}/{len(paths)}")
        try:
            result = json.loads(path.read_text(encoding="utf-8-sig"))
            if result.get("status") != "completed":
                continue
            request = json.loads(path.with_name("request.json").read_text(encoding="utf-8-sig"))
            summaries.extend(summarize_season_record(result, request, source))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append(f"{source}: {exc}")
    return {"rows": aggregate_summaries(summaries), "maps": len(summaries), "errors": errors}
