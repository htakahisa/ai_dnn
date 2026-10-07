"""Offline postplant diagnosis; replay information never enters live policy inputs."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def alive(frame, side):
    return [c for c in frame.get("chars", ()) if c.get("team") == side and c.get("alive")]


def spike_smoked(frame):
    spike = frame.get("planted_pos")
    return bool(spike and any(list(spike) in smoke.get("cells", ())
                              for smoke in frame.get("smokes", ())))


def classify(record, frames, contact_radius=8):
    # The engine can save a terminal tick twice. Count transitions once.
    unique = {f["tick"]: f for f in frames if not f.get("setup") and f.get("planted")}
    frames = [unique[tick] for tick in sorted(unique)]
    if not frames:
        return dict(category="missing_planted_replay")
    first, last = frames[0], frames[-1]
    all_dead = next((f["tick"] for f in frames if not alive(f, "A")), None)
    contact = next((f for f in frames if any(
        "A" in enemy.get("visible_to", ()) and math.dist(friend["pos"], enemy["pos"]) <= contact_radius
        for enemy in alive(f, "D") for friend in alive(f, "A"))), None)
    flash_used = 0
    unused_flash_deaths = []
    previous = {c["name"]: c for c in first.get("chars", ()) if c.get("team") == "A"}
    for frame in frames[1:]:
        for char in frame.get("chars", ()):
            old = previous.get(char["name"])
            if char.get("team") != "A":
                continue
            if old and char.get("ability") == "FLASH":
                flash_used += max(0, old.get("ability_charges", 0) - char.get("ability_charges", 0))
                if old.get("alive") and not char.get("alive") and char.get("ability_charges", 0) > 0:
                    unused_flash_deaths.append(dict(name=char["name"], tick=frame["tick"],
                        charges=char["ability_charges"], blind=char.get("blind", 0)))
            previous[char["name"]] = char
    if record["winner"] == "attacker":
        category = "postplant_win"
    elif record["reason"] == "defused" and all_dead is not None:
        category = "defused_after_attacker_wipe"
    elif record["reason"] == "defused":
        category = "defused_with_attackers_alive"
    else:
        category = "other_postplant_loss"
    return dict(category=category, plant_tick=first["tick"], terminal_tick=last["tick"],
        alive_at_plant=len(alive(first, "A")), alive_at_terminal=len(alive(last, "A")),
        all_dead_tick=all_dead,
        ticks_dead_before_terminal=last["tick"]-all_dead if all_dead is not None else None,
        first_contact_tick=contact["tick"] if contact else None,
        alive_at_first_contact=len(alive(contact, "A")) if contact else None,
        spike_smoked_at_first_contact=spike_smoked(contact) if contact else None,
        spike_smoked_at_terminal=spike_smoked(last),
        postplant_flash_used=flash_used, unused_flash_deaths=unused_flash_deaths)


def analyze(paths, team="Ghost Champions"):
    rounds = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for map_index, mp in enumerate(data["maps"], 1):
            frames = defaultdict(list)
            for frame in mp.get("replay_frames", ()):
                frames[frame["round"]].append(frame)
            for record in mp["round_records"]:
                number = record["round_number"]
                sides = {p.get("side") for p in record.get("players", {}).values() if p.get("team") == team}
                attack = ("attacker" in sides if sides else
                          (mp.get("initial_attacker") == team) == (number <= 12))
                if number > 24 or not attack or not record.get("planted"):
                    continue
                rounds.append(dict(file=Path(path).name, map=map_index, round=number,
                    reason=record["reason"], winner=record["winner"],
                    **classify(record, frames[number])))
    losses = [r for r in rounds if r["winner"] != "attacker" and r["category"] != "missing_planted_replay"]
    return dict(planted_rounds=len(rounds), categories=dict(Counter(r["category"] for r in rounds)),
        postplant_flash_used=sum(r.get("postplant_flash_used", 0) for r in rounds),
        unused_flash_deaths=sum(len(r.get("unused_flash_deaths", ())) for r in rounds),
        losses_without_postplant_flash=sum(r["postplant_flash_used"] == 0 for r in losses),
        losses_with_spike_smoked_at_terminal=sum(r["spike_smoked_at_terminal"] for r in losses),
        rounds=rounds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--opponent", choices=("TYG", "OMG", "FRC", "FNC", "GG", "SPS"), default="OMG")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(sorted(args.folder.glob(f"series_{args.opponent}_*.json")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rounds"}, indent=2))


if __name__ == "__main__":
    main()
