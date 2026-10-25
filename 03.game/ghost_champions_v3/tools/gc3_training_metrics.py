"""Read-only measurement of the exact transition values collected for PPO."""
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import numpy as np
import torch

NUMERIC_FIELDS = ("reward", "return", "advantage", "value", "discount", "age")
METRICS_FILENAME = "training_metrics.jsonl"
TRANSITIONS_FILENAME = "rollout_transitions.jsonl"


def reconcile_resume_logs(logs, step, episode):
    """Keep committed graph records; preserve the interrupted tail before trimming.

    This only reconciles observational logs against an existing checkpoint.
    It neither changes checkpoints nor restores/consumes random state.
    """
    root = Path(logs)
    plans = []
    for filename in ("episodes.jsonl", TRANSITIONS_FILENAME, METRICS_FILENAME):
        path = root/filename
        if not path.is_file():
            if filename == "episodes.jsonl" and episode:
                raise ValueError("Resume episode log is missing")
            continue
        committed_end, committed_records, tail = 0, 0, False
        with path.open("rb") as handle:
            for index, line in enumerate(handle):
                if not line.endswith(b"\n"):
                    tail = True
                    continue
                row = json.loads(line)
                keep = (index < episode if filename == "episodes.jsonl" else
                        row.get("batch_end_step", row["step"]) <= step)
                if keep:
                    if tail:
                        raise ValueError(f"Non-monotonic resume log: {filename}")
                    committed_end = handle.tell()
                    committed_records += 1
                else:
                    tail = True
        if filename == "episodes.jsonl" and committed_records != episode:
            raise ValueError("Resume episode log has fewer records than the checkpoint")
        if tail:
            plans.append((path, committed_end))
    if not plans:
        return None
    stamp = datetime.now(timezone.utc).strftime("resume_%Y%m%dT%H%M%S%fZ")
    archive = root/"recovery"/stamp
    archive.mkdir(parents=True)
    # Back up every tail before changing any active graph file.
    for path, offset in plans:
        with path.open("rb") as source, (archive/path.name).open("xb") as target:
            source.seek(offset)
            while chunk := source.read(1024*1024):
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
    (archive/"recovery.json").write_text(json.dumps(dict(
        checkpoint_step=step, checkpoint_episode=episode,
        retained_byte_offsets={path.name:offset for path, offset in plans}), indent=2), encoding="utf-8")
    for path, offset in plans:
        with path.open("r+b") as handle:
            handle.truncate(offset)
    return archive

def without_logging_blocks(source):
    # Only the explicitly marked observational hooks are excluded from the
    # training fingerprint. Policy, reward and optimizer code remain checked.
    pattern = rb"(?ms)^[ \t]*# GC3_LOGGING_ONLY_BEGIN\r?\n.*?^[ \t]*# GC3_LOGGING_ONLY_END\r?\n"
    return re.sub(pattern, b"", source)

def record_training_metrics(logs, start_step, episode, policy, rows, batch_size):
    if not rows or batch_size < 1:
        raise ValueError("Metrics need nonempty transitions and a positive batch size")
    arrays = {field: np.asarray([row[field] for row in rows], dtype=np.float64) for field in NUMERIC_FIELDS}
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Non-finite collected training value")
    entropy_sum = 0.
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            _, entropy, _ = policy.evaluate_records(rows[offset:offset+batch_size])
            if not torch.isfinite(entropy).all():
                raise ValueError("Non-finite rollout entropy")
            entropy_sum += float(entropy.sum())
    end_step = start_step+len(rows)
    metrics = dict(step=end_step, start_step=start_step, episode=episode, transitions=len(rows),
                   entropy_mean=entropy_sum/len(rows), entropy_unit="nats",
                   entropy_definition="sum of sampled action-branch head entropies; before PPO update",
                   source="collected_ppo_transitions")
    for field, array in arrays.items():
        metrics[field+"_mean"] = float(array.mean())
        metrics[field+"_min"] = float(array.min())
        metrics[field+"_max"] = float(array.max())
    metrics["reward_sum"] = float(arrays["reward"].sum())
    versions = {row["reward_version"] for row in rows if "reward_version" in row}
    if len(versions)>1:
        raise ValueError("Cannot mix reward versions in one collected batch")
    if versions:
        metrics["reward_version"] = next(iter(versions))
    component_keys = sorted({key for row in rows for key in row.get("reward_components", {})})
    if component_keys:
        component_totals = {key:sum(row.get("reward_components", {}).get(key, 0.) for row in rows)
                            for key in component_keys}
        if any(not np.isclose(sum(row.get("reward_components", {}).values()), row["reward"])
               for row in rows):
            raise ValueError("Reward breakdown does not match the actual PPO reward")
        metrics["reward_component_sums"] = component_totals
        metrics["reward_component_means"] = {key:total/len(rows) for key, total in component_totals.items()}
    root = Path(logs)
    root.mkdir(parents=True, exist_ok=True)
    with (root/TRANSITIONS_FILENAME).open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            values = {field: float(arrays[field][index]) for field in NUMERIC_FIELDS}
            if "reward_components" in row:
                values["reward_components"] = dict(row["reward_components"])
            if "reward_version" in row:
                values["reward_version"] = row["reward_version"]
            handle.write(json.dumps(dict(step=start_step+index+1, batch_end_step=end_step,
                                         phase=row["phase"], **values))+"\n")
    with (root/METRICS_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics)+"\n")
    return metrics
