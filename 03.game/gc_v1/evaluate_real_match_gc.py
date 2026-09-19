"""Compare GC checkpoints using the actual match engine and IQ perception."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--map-number", type=int, default=1)
    parser.add_argument("--carry-model", type=Path)
    parser.add_argument("--guard-model", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--save-replay", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    import ghost_champions_v1 as gc
    from party_presets import get_preset
    from run_competition_manager import play_map
    from gc_v1.positioning_gc import REGISTERED_PLANT_CELLS

    if args.carry_model:
        gc.CARRY = (args.carry_model.resolve(),)
    if args.guard_model:
        gc.GUARD = (args.guard_model.resolve(),)
    start = time.monotonic()
    result = play_map(get_preset("Ghost Champions"), get_preset("Touyama Gaming"),
                      args.map_number, args.seed, False,
                      "ghost_champions_v1", "touyama_gaming_v2")
    rows = [r for r in result.round_records if any(
        p.get("team") == "Ghost Champions" and p.get("side") == "attacker"
        for p in r.get("players", {}).values())]
    planted_positions = {}
    for frame in result.replay_frames:
        if frame.get("planted") and frame.get("planted_pos") is not None:
            planted_positions.setdefault(frame["round"], tuple(frame["planted_pos"]))
    registered = sum(r["planted"] and planted_positions.get(r["round_number"])
                     in REGISTERED_PLANT_CELLS for r in rows)
    models = {}
    for phase, path in (("carry", gc.CARRY[0]), ("guard", gc.GUARD[0])):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        models[phase] = {"path": str(path),
                         "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                         "positioning_version": checkpoint.get("positioning_version", 0),
                         "episode": checkpoint.get("episode")}
    summary = {
        "seed": args.seed, "map_number": args.map_number,
        "evaluation_context": "independent_map",
        "maps_to_win": 1, "series_score_before": [0, 0],
        "carry_model": str(args.carry_model or gc.CARRY[0]),
        "guard_model": str(args.guard_model or gc.GUARD[0]),
        "score": [result.score1, result.score2],
        "attack_rounds": len(rows), "plants": sum(r["planted"] for r in rows),
        "plant_rate": sum(r["planted"] for r in rows) / max(1, len(rows)),
        "registered_plants": registered,
        "models": models,
        "seconds": round(time.monotonic() - start, 1),
        "round_records": result.round_records,
    }
    if args.save_replay:
        summary["replay_frames"] = result.replay_frames
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in {"round_records", "replay_frames"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
