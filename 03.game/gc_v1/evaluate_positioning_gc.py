"""Evaluate GC Carry/Guard checkpoints on the same seeded positioning scenarios."""

import argparse
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("carry", "guard"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--start-mode", choices=("transition", "dispersed", "hold"))
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.phase == "guard":
        import train_attacker_guard_gc as training
        from positioning_evaluation_gc import evaluate_guard
        model = training.AttackerGuardDuelingDQN().to(training.DEVICE)
    else:
        import train_attacker_carry_gc as training
        from positioning_evaluation_gc import evaluate_carry
        model = training.AttackerCarryDuelingDQN().to(training.DEVICE)
    path = args.model or Path(training.MODEL_SAVE_PATH)
    checkpoint = torch.load(path, map_location=training.DEVICE, weights_only=False)
    if args.phase == "carry" and int(checkpoint.get("positioning_version", 0)) >= 3:
        parser.error("Carry v3 uses actual-engine observations; evaluate it with "
                     "gc_v1/evaluate_real_series_gc.py --series <saved-series.json> "
                     "--carry-model <checkpoint.pt> --json-output <report.json>")
    weights = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(weights, strict=True)
    model.eval()
    if args.phase == "guard":
        metrics = evaluate_guard(training, model, args.episodes, args.seed,
                                 int(checkpoint.get("positioning_version", 0)), args.start_mode)
    else:
        metrics = evaluate_carry(training, model, args.episodes, args.seed,
                                 None if int(checkpoint.get("positioning_version", 0)) >= 2
                                 else checkpoint.get("priority_cells", []))
    result = {"phase": args.phase, "model": str(path.resolve()), "seed": args.seed,
              "positioning_version": int(checkpoint.get("positioning_version", 0)),
              "start_mode": args.start_mode or "mixed", "metrics": metrics}
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(output + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
