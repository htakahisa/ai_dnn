"""Summarize recorded decisions; no simulation or policy changes."""
import argparse
from collections import Counter
import json
from pathlib import Path


def analyze(data):
    rows = data["episodes"]
    trace = [t for row in rows for t in row["trace"]]
    quiet = [t for t in trace if t["before"] == t["after"] and not t["engaged"] and not t["planting"]]
    incorrect_masks = []
    blocked_by = Counter()
    for t in trace:
        for n in t["neighbors"]:
            if (not n["wall"] and not n["blockers"] and n["distance"] is not None
                    and 0 <= n["distance"] < t["distance_before"]
                    and not t.get("mask", [True]*11)[n["action"]]):
                incorrect_masks.append(dict(tick=t["tick"], pos=t["before"], neighbor=n))
        if t in quiet and not t["legal_progress"]:
            for n in t["neighbors"]:
                if n["distance"] is not None and 0 <= n["distance"] < t["distance_before"]:
                    blocked_by.update(c["name"] for c in n["blockers"])
    final_stays = [dict(tick=t["tick"], pos=t["before"], goal=t["decision_goal"], target=t["target"],
                       action=t.get("action"), final_distance=t["final_distance"])
                   for t in quiet if t.get("obs") and t["obs"][-1] and t["obs"][25] > 0]
    return dict(models=data["models"], episodes=len(rows), ticks=len(trace),
                wins=sum(r["result"]["attacker_win"] for r in rows),
                plants=sum(r["result"]["planted"] for r in rows),
                timeouts=[r["seed"] for r in rows if r["result"]["timed_out"]],
                quiet_stays=len(quiet), quiet_stays_with_legal_progress=sum(bool(t["legal_progress"]) for t in quiet),
                blocked_move_count=sum(r["blocked_moves"] for r in rows),
                incorrectly_masked_progress=incorrect_masks,
                quiet_stay_blockers=dict(blocked_by),
                quiet_stay_visible_flags=dict(Counter(str((bool(t["obs"][13]), bool(t["obs"][38])))
                                                     for t in quiet if t.get("obs"))),
                final_approach_stays=final_stays,
                quiet_stay_assignments=dict(Counter(str(t["assignment"]) for t in quiet)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(json.loads(args.trace.read_text(encoding="utf-8")))
    content = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    print(content)
