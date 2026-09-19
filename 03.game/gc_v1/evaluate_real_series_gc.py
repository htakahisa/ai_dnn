"""Replay saved series seeds with actual controllers and cumulative fatigue."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluate_series(source, carry_model=None, guard_model=None, escort_model=None,
                    save_replay=False):
    import ghost_champions_v1 as gc
    previous = {attr: getattr(gc, attr) for attr in ("CARRY", "GUARD", "ESCORT")}
    try:
        return _evaluate_series(source, carry_model, guard_model, escort_model, save_replay)
    finally:
        for attr, paths in previous.items():
            setattr(gc, attr, paths)


def _evaluate_series(source, carry_model=None, guard_model=None, escort_model=None,
                     save_replay=False):
    import torch
    torch.set_num_threads(1)
    import ghost_champions_v1 as gc
    from party_presets import get_preset
    from run_competition_manager import play_map
    from gc_v1.positioning_gc import REGISTERED_PLANT_CELLS

    for attr, path in (("CARRY", carry_model), ("GUARD", guard_model),
                       ("ESCORT", escort_model)):
        if path is not None:
            setattr(gc, attr, (Path(path).resolve(),))
    models = {}
    for phase in ("carry", "guard", "escort"):
        path = getattr(gc, phase.upper())[0]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        models[phase] = {"path": str(path),
                         "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                         "positioning_version": checkpoint.get("positioning_version", 0),
                         "episode": checkpoint.get("episode")}
    original = json.loads(Path(source).read_text(encoding="utf-8"))
    team1, team2 = get_preset(original["team1"]), get_preset(original["team2"])
    controllers = original["team_controllers"]
    wins = [0, 0]
    fatigue = {}
    maps = []
    start = time.monotonic()
    for entry in original["maps"]:
        number = entry["number"]
        result = play_map(team1, team2, number, entry["seed"], False,
                          controllers[team1.name], controllers[team2.name],
                          wins[0], wins[1], original["maps_to_win"], fatigue)
        wins[0 if result.winner == team1.name else 1] += 1
        rows = [r for r in result.round_records if any(
            p.get("team") == "Ghost Champions" and p.get("side") == "attacker"
            for p in r.get("players", {}).values())]
        positions = {}
        for frame in result.replay_frames:
            if frame.get("planted") and frame.get("planted_pos") is not None:
                positions.setdefault(frame["round"], tuple(frame["planted_pos"]))
        row = {"number": number, "seed": entry["seed"],
               "score": [result.score1, result.score2], "attack_rounds": len(rows),
               "attack_wins": sum(r["winner"] == "attacker" for r in rows),
               "gc_round_wins": result.score1 if team1.name == "Ghost Champions" else result.score2,
               "total_rounds": result.score1 + result.score2,
               "plants": sum(r["planted"] for r in rows),
               "registered_plants": sum(r["planted"] and positions.get(r["round_number"])
                                        in REGISTERED_PLANT_CELLS for r in rows),
               "round_records": result.round_records}
        if save_replay:
            row["replay_frames"] = result.replay_frames
        maps.append(row)
        print(json.dumps({k: v for k, v in row.items()
                          if k not in {"round_records", "replay_frames"}}, ensure_ascii=False),
              flush=True)
    attacks = sum(m["attack_rounds"] for m in maps)
    plants = sum(m["plants"] for m in maps)
    attack_wins = sum(m["attack_wins"] for m in maps)
    round_wins = sum(m["gc_round_wins"] for m in maps)
    total_rounds = sum(m["total_rounds"] for m in maps)
    return {"source": str(source), "evaluation_context": "saved_series_seeds",
            "cumulative_fatigue": True, "models": models, "maps": maps,
            "attack_rounds": attacks, "plants": plants,
            "attack_wins": attack_wins, "attack_win_rate": attack_wins / max(1, attacks),
            "gc_round_wins": round_wins, "total_rounds": total_rounds,
            "round_win_rate": round_wins / max(1, total_rounds),
            "plant_rate": plants / max(1, attacks),
            "registered_plants": sum(m["registered_plants"] for m in maps),
            "series_score": wins, "seconds": round(time.monotonic() - start, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=Path, nargs="+", required=True)
    parser.add_argument("--carry-model", type=Path)
    parser.add_argument("--guard-model", type=Path)
    parser.add_argument("--escort-model", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--save-replay", action="store_true")
    args = parser.parse_args()
    results = [evaluate_series(source, args.carry_model, args.guard_model,
                                args.escort_model, args.save_replay)
               for source in args.series]
    if len(results) == 1:
        result = results[0]
    else:
        attacks = sum(r["attack_rounds"] for r in results)
        plants = sum(r["plants"] for r in results)
        result = {"evaluation_context": "multiple_saved_series_seeds",
                  "series": results, "attack_rounds": attacks, "plants": plants,
                  "plant_rate": plants / max(1, attacks),
                  "attack_wins": sum(r["attack_wins"] for r in results),
                  "gc_round_wins": sum(r["gc_round_wins"] for r in results),
                  "total_rounds": sum(r["total_rounds"] for r in results),
                  "registered_plants": sum(r["registered_plants"] for r in results)}
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"maps", "series"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
