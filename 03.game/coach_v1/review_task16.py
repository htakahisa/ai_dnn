"""Summarize referee evidence around repeated kill locations for human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def review(report: dict, *, min_opponents: int = 2) -> list[dict]:
    replay_cache: dict[str, list[dict]] = {}
    round_cache: dict[str, dict] = {}
    rows = []
    for candidate in report["candidates"]:
        if candidate["kind"] != "repeated_kill_location" or len(candidate["opponents"]) < min_opponents:
            continue
        side_team = "A" if candidate["side"] == "attacker" else "D"
        seen = []
        for ref, evidence in zip(candidate["replay_refs"], candidate["evidence"]):
            path = ref["id"]
            if path not in replay_cache:
                replay_cache[path] = json.loads(Path(path).read_text(encoding="utf-8"))
                rounds_path = Path(path).with_name(Path(path).stem + "_rounds.json")
                round_cache[path] = json.loads(rounds_path.read_text(encoding="utf-8"))
            frames = replay_cache[path]
            index = int(ref["frame"])
            if not 0 < index < len(frames):
                raise ValueError(f"invalid replay frame: {ref}")
            previous, current = frames[index - 1:index + 1]
            if previous["round"] != current["round"]:
                continue
            killer_name, victim_name = evidence["nearby_enemy"], evidence["victim"]
            enemy_team = "D" if side_team == "A" else "A"
            killer = next((char for char in previous["chars"]
                           if char["team"] == enemy_team and char["name"] == killer_name), None)
            victim = next((char for char in previous["chars"]
                           if char["team"] == side_team and char["name"] == victim_name), None)
            if killer is None or victim is None:
                continue
            exact = next((event for event in round_cache[path].get("kill_events", ())
                          if event["round"] == ref["round"]
                          and event["victim"] == victim_name
                          and event["victim_team"] == side_team
                          and abs(event["tick"] - ref["tick"]) <= 1), None)
            if exact is None:
                raise ValueError(f"missing exact kill event: {ref}")
            decisions = [decision for decision in round_cache[path].get("coach_decisions", ())
                         if decision["round_number"] == ref["round"]
                         and decision["side"] == candidate["side"]
                         and decision["tick"] <= previous["tick"]]
            legal_sightings = (decisions[-1].get("sightings", ()) if decisions else ())
            killer_sighting = next((position for enemy_id, position in legal_sightings
                                    if enemy_id == killer_name), None)
            distance = abs(killer["pos"][0] - victim["pos"][0]) + abs(
                killer["pos"][1] - victim["pos"][1])
            seen.append({
                "opponent_replay": path, "round": ref["round"], "tick": ref["tick"],
                "killer": killer_name, "victim": victim_name,
                "killer_position": exact["killer_position"],
                "killer_predeath_position": killer["pos"],
                "victim_position": victim["pos"],
                "victim_facing": victim["facing"], "distance": distance,
                "killer_visible_to_coach": side_team in killer["visible_to"],
                "killer_in_actor_sightings": (killer_sighting is not None
                                              if decisions else None),
                "actor_reported_position": killer_sighting,
                "planted": bool(previous["planted"]),
            })
        matches = {entry["opponent_replay"] for entry in seen}
        rounds = {(entry["opponent_replay"], entry["round"]) for entry in seen}
        rows.append({
            "position": candidate["position"], "side": candidate["side"],
            "phase": candidate["phase"], "facing": candidate["facing"],
            "kills": candidate["count"], "opponents": candidate["opponents"],
            "matches": len(matches), "rounds": len(rounds),
            "existing_distribution": candidate["covering_watch_point_ids"],
            "uncovered_positions": candidate["uncovered_positions"],
            "predeath_killer_visible": sum(item["killer_visible_to_coach"] for item in seen),
            "predeath_killer_unseen": sum(not item["killer_visible_to_coach"] for item in seen),
            "actor_sighting_known": sum(item["killer_in_actor_sightings"] is not None
                                          for item in seen),
            "predeath_actor_sighted": sum(item["killer_in_actor_sightings"] is True
                                           for item in seen),
            "reviewed_events": len(seen), "examples": seen[:6],
        })
    rows.sort(key=lambda row: (-len(row["opponents"]), -row["rounds"], -row["kills"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        default=Path(__file__).resolve().parent / "reports/task16_candidates.json")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "reports/task16_review.json")
    args = parser.parse_args()
    rows = review(json.loads(args.report.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"locations": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
