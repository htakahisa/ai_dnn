"""Read real competition replays and quantify Carry spawn dwell and reversals."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from map_data import NEW_MAZE_STR


def distances(grid, goal):
    result = {goal: 0}
    queue = deque([goal])
    while queue:
        pos = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = pos[0] + dr, pos[1] + dc
            if 0 <= nxt[0] < len(grid) and 0 <= nxt[1] < len(grid[0]) and grid[nxt[0]][nxt[1]] != 1 and nxt not in result:
                result[nxt] = result[pos] + 1
                queue.append(nxt)
    return result


def analyze(source):
    series = json.loads(Path(source).read_text(encoding="utf-8"))
    grid = [[int(c) for c in row.strip()] for row in NEW_MAZE_STR.strip().splitlines()]
    cache = {}
    rows = []
    for match in series["maps"]:
        frames = defaultdict(list)
        for frame in match.get("replay_frames", []):
            if not frame.get("setup"):
                frames[frame["round"]].append(frame)
        for record in match["round_records"]:
            if not any(p.get("team") == "Ghost Champions" and p.get("side") == "attacker"
                       for p in record["players"].values()):
                continue
            samples = []
            origins = {}
            for frame in frames[record["round_number"]]:
                if frame.get("planted"):
                    break
                for c in frame["chars"]:
                    if c["team"] == "A":
                        origins.setdefault(c["name"], tuple(c["pos"]))
                holder = next((c for c in frame["chars"] if c["team"] == "A" and c["alive"] and c.get("has_spike")), None)
                if holder:
                    samples.append((frame["tick"], holder["name"], tuple(holder["pos"]), frame.get("target_plant_pos")))
            spawn_ticks = reversals = 0
            for i, (_, name, pos, _) in enumerate(samples):
                origin = origins[name]
                if origin not in cache:
                    cache[origin] = distances(grid, origin)
                spawn_ticks += cache[origin].get(pos, 999) <= 6
                if i >= 2 and samples[i - 2][1] == name and pos == samples[i - 2][2] and pos != samples[i - 1][2]:
                    reversals += 1
            round_frames = frames[record["round_number"]]
            # The swap boundary can leave a stale winner label in archives.
            # Replay scores reflect the actual team A within this round.
            won = (round_frames[-1]["attacker_wins"] > round_frames[0]["attacker_wins"]
                   if round_frames else record["reason"] in ("detonated", "defender_wipe"))
            rows.append({"map": match["number"], "round": record["round_number"],
                         "gc_won": won, "reason": record["reason"],
                         "planted": record["planted"], "strategy": record["tactic"]["attacker_strategy"],
                         "carry_samples": len(samples), "spawn_samples": spawn_ticks,
                         "reversals": reversals, "first20": samples[:20]})
    plants = sum(r["planted"] for r in rows)
    return {"source": str(source), "attack_rounds": len(rows), "gc_attack_wins": sum(r["gc_won"] for r in rows),
            "plants": plants, "postplant_wins": sum(r["gc_won"] and r["planted"] for r in rows),
            "timeouts": sum(r["reason"] == "time_expired" for r in rows),
            "carry_samples": sum(r["carry_samples"] for r in rows),
            "spawn_samples": sum(r["spawn_samples"] for r in rows),
            "reversals": sum(r["reversals"] for r in rows),
            "strategies": dict(Counter(r["strategy"] for r in rows)), "rounds": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.series)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rounds"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
