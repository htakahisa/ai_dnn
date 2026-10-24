"""Read-only BC diagnostics. No teacher collection or optimizer updates."""
from pathlib import Path
from collections import Counter, defaultdict
import json
import sys
import types
import numpy as np
import torch

AI_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = AI_ROOT/"data/attacker_preplant_bc_history_v1/latest.pt"
DATASET_PATH = AI_ROOT/"data/attacker_preplant_concon_history_v1"
OUTPUT_PATH = AI_ROOT/"logs/validation/gc3_preplant_bc_movement_audit.json"
EVALUATION_CASES = (("FNC", 20361013), ("SPS", 20365013))
BATCH_SIZE = 128
TORCH_THREADS = 1

sys.path.insert(0, str(AI_ROOT.parent))
from ghost_champions_v3.preplant.gc3_policy_attacker_history import HistoryPolicy
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import dataset_manifest, file_hash, load_shard
from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import bc_log_prob, reused_dagger_shards
from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import play_preplant
from ghost_champions_v3.gc3_rollout import module_digest
from ghost_champions_v3.gc3_engine_actions import MODES
from ghost_champions_v3.gc3_io import write_json
from grid_paths import distance_map


def label_report(model, rows):
    measurements = defaultdict(list)
    with torch.no_grad():
        for offset in range(0, len(rows), BATCH_SIZE):
            batch = rows[offset:offset+BATCH_SIZE]
            context = model.contexts(batch)
            for kind in ("mode", "position", "watch", "target", "ally"):
                selected = [i for i, row in enumerate(batch) if kind in row["selections"]]
                if not selected:
                    continue
                record = dict(phase="standard", obs=np.stack([batch[i]["obs"] for i in selected]),
                    hints=np.stack([batch[i]["hints"] for i in selected]),
                    masks={kind:np.stack([batch[i]["masks"][kind] for i in selected])})
                probabilities = model.distribution(context[selected], record, kind).probs
                for j, i in enumerate(selected):
                    row = batch[i]
                    action = row["selections"][kind]
                    probability = float(probabilities[j, action])
                    correct = int(probabilities[j].argmax()) == action
                    if kind == "watch" and row["watch_options"] is not None:
                        options = row["watch_options"]
                        probability = float(probabilities[j, options].sum())
                        correct = bool(options[int(probabilities[j].argmax())])
                    measurements[kind].append((probability, correct))
                    if kind == "mode":
                        measurements["mode_"+MODES[action]].append((probability, correct))
                    if row["obs"][11] > .5:
                        measurements["carrier_"+kind].append((probability, correct))
    return {key:dict(count=len(values), teacher_probability=float(np.mean([v[0] for v in values])),
                     top1_accuracy=float(np.mean([v[1] for v in values])))
            for key, values in measurements.items()}


def rollout_report(model, code, seed):
    trace, distances = [], None
    original = model.select
    def inspected_select(self, char, state, history):
        nonlocal distances
        command, row = original(char, state, history)
        if char.has_spike:
            grid = np.asarray(state["grid"])
            if distances is None:
                candidates = [distance_map(grid, tuple(p)) for p in np.argwhere(grid == 2)]
                distances = np.min(np.stack([np.where(d >= 0, d, 100000) for d in candidates]), axis=0)
            point = tuple(map(int, char.pos))
            moving = MODES[row["selections"]["mode"]] in ("MOVE", "MOVE_WATCH")
            destination = tuple(command[0]) if moving else point
            trace.append(dict(tick=int(state.get("battle_tick", 0)), name=str(char.name), position=list(point),
                site_distance=int(distances[point]), mode=MODES[row["selections"]["mode"]],
                destination=list(destination), stationary=destination == point,
                moves_away=int(distances[destination]) > int(distances[point])))
        return command, row
    model.select = types.MethodType(inspected_select, model)
    try:
        _, stats = play_preplant(code, seed, policy=model)
    finally:
        model.select = original
    return dict(opponent=code, seed=seed, planted=stats["planted"], reason=stats["reason"], ticks=stats["ticks"],
                closest_site_distance=min((row["site_distance"] for row in trace), default=None),
                stationary_fraction=float(np.mean([row["stationary"] for row in trace])) if trace else None,
                moves_away_fraction=float(np.mean([row["moves_away"] for row in trace])) if trace else None,
                modes=dict(Counter(row["mode"] for row in trace)), carrier_trace=trace)


def main():
    torch.set_num_threads(TORCH_THREADS)
    manifest, manifest_hash = dataset_manifest(DATASET_PATH)
    checkpoint_hash = file_hash(CHECKPOINT)
    model, metadata = HistoryPolicy.load(CHECKPOINT)
    initial_weights = module_digest((model,))
    shard = next(s for s in manifest["shards"] if s["split"] == "validation")
    rows = load_shard(DATASET_PATH/shard["file"])
    report = dict(checkpoint=str(CHECKPOINT), checkpoint_hash=checkpoint_hash,
        epoch=metadata.get("epoch"), step=metadata.get("step"), model_schema=metadata["schema_hash"],
        teacher_dataset=str(DATASET_PATH), teacher_manifest_hash=manifest_hash,
        teacher_shards_verified=len(manifest["shards"]), teacher_rounds=manifest["rounds"],
        teacher_decisions=manifest["decisions"], validation_shard=shard["file"],
        validation_labels=label_report(model, rows))
    report["rollouts"] = [rollout_report(model, code, seed) for code, seed in EVALUATION_CASES]
    # Check the corrected architecture against these exact collected observations.
    corrected = HistoryPolicy.fresh("attacker", 20261013)
    corrected_weights = module_digest((corrected,))
    loss = -bc_log_prob(corrected, rows[:BATCH_SIZE]).mean()
    loss.backward()
    report["corrected_model_gradient_check"] = dict(loss=float(loss.detach()),
        movement_gradient_finite=bool(torch.isfinite(corrected.move.weight.grad).all()),
        movement_gradient_nonzero=bool(corrected.move.weight.grad.abs().sum() > 0),
        location_gradient_nonzero=bool(corrected.features.location.weight.grad.abs().sum() > 0),
        history_gradient_nonzero=bool(corrected.action_features.weight.grad.abs().sum() > 0),
        weights_unchanged=module_digest((corrected,)) == corrected_weights)
    reused = reused_dagger_shards(dict(REUSE_DAGGER_RUN_PATH=CHECKPOINT.parent), manifest_hash)
    report["reused_dagger_shards"] = len(reused)
    report["reused_dagger_labels"] = sum(len(load_shard(Path(s["root"])/s["file"])) for s in reused)
    _, after_hash = dataset_manifest(DATASET_PATH)
    report["teacher_data_unchanged"] = after_hash == manifest_hash
    report["checkpoint_unchanged"] = file_hash(CHECKPOINT) == checkpoint_hash and module_digest((model,)) == initial_weights
    report["optimizer_updates"] = 0
    report["teacher_collection_started"] = False
    write_json(OUTPUT_PATH, report)
    print(json.dumps({key:value for key,value in report.items() if key not in ("rollouts", "validation_labels")}, indent=2))
    print("rollouts:", [{k:v for k,v in r.items() if k != "carrier_trace"} for r in report["rollouts"]])


if __name__ == "__main__":
    main()
