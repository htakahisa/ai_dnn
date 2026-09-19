"""Read existing replays without running a match or modifying result files."""
from collections import Counter, defaultdict
from pathlib import Path
import json


def summarize(result_dir):
    groups = defaultdict(Counter)
    maps = rounds = 0
    for path in sorted(Path(result_dir).glob("series_*.json")):
        with path.open(encoding="utf-8") as stream:
            series = json.load(stream)
        for match in series.get("maps", []):
            frames = defaultdict(list)
            for frame in match.get("replay_frames", []):
                if not frame.get("setup"):
                    frames[frame["round"]].append(frame)
            used = False
            for record in match.get("round_records", []):
                if not any(p.get("team") == "Ghost Champions" and
                           p.get("side") == "attacker"
                           for p in record.get("players", {}).values()):
                    continue
                used = True
                rounds += 1
                tactic = record.get("tactic", {})
                strategy = tactic.get("attacker_strategy", "unknown")
                keys = [strategy]
                if strategy == "fake":
                    keys.append("fake_effect_positive" if
                                tactic.get("fake_effect_score", 0) > 0 else
                                "fake_effect_zero")
                samples = isolated = 0
                # Proximity is a diagnostic, not a claim of tradeable LOS.
                for frame in frames[record["round_number"]]:
                    if frame.get("planted"):
                        break
                    allies = [c for c in frame.get("chars", [])
                              if c.get("alive") and c.get("team") == "A"]
                    holder = next((c for c in allies if c.get("has_spike")), None)
                    if holder is None or len(allies) <= 1:
                        continue
                    samples += 1
                    if not any(c is not holder and
                               max(abs(c["pos"][0] - holder["pos"][0]),
                                   abs(c["pos"][1] - holder["pos"][1])) <= 3
                               for c in allies):
                        isolated += 1
                for key in keys:
                    counts = groups[key]
                    counts["rounds"] += 1
                    counts["wins"] += record.get("winner") == "attacker"
                    counts["plants"] += bool(record.get("planted"))
                    counts["carrier_samples"] += samples
                    counts["isolated_samples"] += isolated
                    counts["loss_" + record.get("reason", "unknown")] += (
                        record.get("winner") != "attacker")
            maps += used
    return {"maps": maps, "attack_rounds": rounds, "strategies": {
        name: {**dict(counts), "win_rate": counts["wins"] / counts["rounds"],
               "plant_rate": counts["plants"] / counts["rounds"],
               "isolated_sample_rate": counts["isolated_samples"] /
               max(1, counts["carrier_samples"])}
        for name, counts in sorted(groups.items())}}


if __name__ == "__main__":
    print(json.dumps(summarize(Path(__file__).resolve().parents[1] /
                               "competition_results"), ensure_ascii=False, indent=2))
