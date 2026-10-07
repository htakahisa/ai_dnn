"""Incremental player rankings from saved competition and series records."""

from collections import Counter
import json
from pathlib import Path
import sqlite3


CACHE_VERSION = 1
STAT_FIELDS = ("kills", "deaths", "assists", "covers", "one_v_one_won", "one_v_one_lost")


def _maps(value):
    """Find series maps inside standalone, league, and tournament results."""
    if isinstance(value, dict):
        if isinstance(value.get("player_stats"), list) and "score1" in value:
            yield value
        else:
            for child in value.values():
                if isinstance(child, (dict, list)):
                    yield from _maps(child)
    elif isinstance(value, list):
        for child in value:
            yield from _maps(child)


def summarize_record(data, source):
    summaries = []
    for index, game in enumerate(_maps(data)):
        teams = sorted((str(game.get("team1", "")), str(game.get("team2", ""))))
        seed = game.get("seed")
        # Both exports of the same played map retain its seed and team names.
        identity = [teams, seed] if seed is not None else [source, index]
        players = []
        round_records = game.get("round_records", [])
        for player in game["player_stats"]:
            name, team = str(player.get("name", "")), str(player.get("team", ""))
            if not name or not team:
                continue
            roles = Counter()
            rounds = 0
            for record in round_records:
                p = record.get("players", {}).get(name, {})
                if p.get("team") == team:
                    rounds += 1
                    if p.get("role"):
                        roles[str(p["role"])] += 1
            if not rounds:
                rounds = int(game.get("total_rounds") or
                             (game.get("score1", 0) + game.get("score2", 0)))
            if not roles and player.get("role"):
                roles[str(player["role"])] = max(1, rounds)
            mvp_key = "mvp1" if team == game.get("team1") else "mvp2"
            mvp = game.get(mvp_key)
            players.append({
                "name": name, "team": team, "rounds": rounds, "roles": dict(roles),
                **{key: player.get(key) for key in STAT_FIELDS},
                "mvps": int(mvp.get("name") == name) if isinstance(mvp, dict) else None,
            })
        if players:
            summaries.append({"id": json.dumps(identity, ensure_ascii=False), "players": players})
    return summaries


def aggregate_summaries(summaries):
    totals = {}
    for game in summaries:
        for player in game["players"]:
            key = (player["team"], player["name"])
            row = totals.setdefault(key, {
                "team": key[0], "name": key[1], "maps": 0, "rounds": 0,
                "roles": Counter(), **{field: 0 for field in (*STAT_FIELDS, "mvps")},
            })
            row["maps"] += 1
            row["rounds"] += player["rounds"]
            row["roles"].update(player["roles"])
            for field in (*STAT_FIELDS, "mvps"):
                value = player.get(field)
                row[field] = row[field] + value if row[field] is not None and value is not None else None
    rows = []
    for row in totals.values():
        roles = row.pop("roles")
        row["role"] = min(roles, key=lambda role: (-roles[role], role)) if roles else "記録なし"
        kills, deaths = row["kills"], row["deaths"]
        row["kd"] = (kills / deaths if deaths else (float("inf") if kills else 0.0)) if kills is not None and deaths is not None else None
        row["kills_per_round"] = kills / row["rounds"] if kills is not None and row["rounds"] else None
        rows.append(row)
    return sorted(rows, key=lambda row: (row["team"], row["name"]))


def load_rankings(base_dir=Path("."), progress=None):
    """Read only new/changed files; retain small per-file summaries in SQLite."""
    base_dir = Path(base_dir)
    cache = base_dir / "data" / "strongest_ranking.sqlite3"
    cache.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted((base_dir / "series_data").glob("*/*_original.json"))
    sources += sorted((base_dir / "competition_results").glob("*.json"))
    errors, read_count, cached_count = [], 0, 0
    maps = {}
    with sqlite3.connect(cache) as db:
        db.execute("CREATE TABLE IF NOT EXISTS sources (path TEXT PRIMARY KEY, mtime INTEGER, size INTEGER, version INTEGER, summary TEXT)")
        existing = {row[0]: row[1:] for row in db.execute("SELECT path, mtime, size, version, summary FROM sources")}
        seen = set()
        for index, path in enumerate(sources):
            relative = path.relative_to(base_dir).as_posix()
            seen.add(relative)
            if progress:
                progress(f"記録を確認中: {index + 1}/{len(sources)}")
            try:
                stat = path.stat()
                signature = (stat.st_mtime_ns, stat.st_size, CACHE_VERSION)
                old = existing.get(relative)
                if old and old[:3] == signature:
                    summary = json.loads(old[3])
                    cached_count += 1
                else:
                    with path.open(encoding="utf-8-sig") as stream:
                        summary = summarize_record(json.load(stream), relative)
                    after = path.stat()
                    if (after.st_mtime_ns, after.st_size) != signature[:2]:
                        raise ValueError("読み込み中にファイルが更新されました。再読み込みしてください")
                    db.execute("INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?, ?)",
                               (relative, *signature, json.dumps(summary, ensure_ascii=False)))
                    read_count += 1
                for game in summary:
                    # Prefer detailed originals; competition exports supplement missing maps.
                    maps.setdefault(game["id"], game)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append(f"{relative}: {exc}")
        for removed in existing.keys() - seen:
            db.execute("DELETE FROM sources WHERE path = ?", (removed,))
    return {
        "rows": aggregate_summaries(maps.values()), "maps": len(maps),
        "read_files": read_count, "cached_files": cached_count, "errors": errors,
    }
