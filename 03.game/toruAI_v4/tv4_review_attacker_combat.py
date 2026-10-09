"""Read-only model review: real-engine diagnostics; no training or promotion."""
from pathlib import Path
import sys
import json
from collections import Counter

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

# Normal settings. Run this directory's script without options.
OPPONENT = "frc_v1"
REVIEW_LABEL = "baseline"
COMPARE_BASELINE_AND_CANDIDATE = True
REVIEW_SEEDS = (904000042, 904000142, 904000242)
REVIEW_PRESETS = ("Eine Kleine", "SUPES", "BBL")
CHECKPOINT = HERE / "data" / "best" / OPPONENT / "attacker_analysis_best.pt"
OUTPUT_DIRECTORY = HERE / "logs" / "attacker_combat_review"
# This diagnostic may compare changed executors with the same compatible weights.
# It never treats their result as a trained best under the new conditions.
ALLOW_EXECUTOR_COMPARISON = True
TORCH_THREADS = 1
MAX_ROUND_STEPS = 400


def main():
    import torch
    from toruAI_v4.tv4_scenario import Scenario
    from toruAI_v4.tv4_learn_attacker_analysis import AttackerEncoder, AttackerAnalysisModel
    from toruAI_v4.tv4_collect_attacker_analysis import play_block
    from toruAI_v4.tv4_train_attacker_analysis import summarize
    torch.set_num_threads(TORCH_THREADS)
    scenario = Scenario()
    encoder = AttackerEncoder(scenario)
    import io
    import hashlib
    checkpoint_bytes = CHECKPOINT.read_bytes()
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    for key in ("board", "branches", "fields", "route_fields"):
        if json.dumps(saved["schema"][key], sort_keys=True) != json.dumps(encoder.schema()[key], sort_keys=True):
            raise ValueError(f"Review checkpoint input/map mismatch: {key}")
    if saved["opponent"] != OPPONENT:
        raise ValueError("Review opponent mismatch")
    if not ALLOW_EXECUTOR_COMPARISON and saved["schema"] != encoder.schema():
        raise ValueError("Review executor mismatch")
    model = AttackerAnalysisModel(len(encoder.fields), len(encoder.route_fields), len(scenario.names))
    model.load_state_dict(saved["model"])
    model.trained_rounds = saved["trained_rounds"]
    model.eval()
    config = {**saved["config"], "max_round_steps": MAX_ROUND_STEPS, "combat_diagnostics": True}
    labels = ("baseline", "candidate") if COMPARE_BASELINE_AND_CANDIDATE else (REVIEW_LABEL,)
    # Load once so concurrent training cannot change weights between the pair.
    weight_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    for label in labels:
        records = []
        review_config = {**config, "combat_policy_enabled": label != "baseline"}
        for index, seed in enumerate(REVIEW_SEEDS):
            preset = REVIEW_PRESETS[index % len(REVIEW_PRESETS)]
            print(f"Review {label} {OPPONENT} {preset} seed={seed}", flush=True)
            rounds, _ = play_block(OPPONENT, scenario, model, review_config, seed, 0., preset_name=preset)
            records.extend({"seed": seed, **r} for r in rounds)
        counts = Counter()
        for r in records:
            counts.update(r["combat_diagnostics"]["counts"])
        output = {"label": label, "checkpoint": str(CHECKPOINT), "checkpoint_set": saved["completed_sets"], "checkpoint_hash": weight_hash,
                  "checkpoint_executor": saved["schema"]["executor"], "review_executor": encoder.schema()["executor"],
                  "seeds": list(REVIEW_SEEDS), "presets": list(REVIEW_PRESETS),
                  "metrics": summarize(records), "counts": dict(counts), "rounds": records}
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIRECTORY / f"{label}.json"
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"label": label, "metrics": output["metrics"], "counts": dict(counts), "file": str(path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
