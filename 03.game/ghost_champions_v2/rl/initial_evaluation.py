"""Explicit reuse of a completed historical evaluation, independent of BC data."""
import hashlib
import json
import math
from pathlib import Path
import torch
from .rollout import OPPONENTS

REFERENCE_NAME = "initial_evaluation_reference.json"


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_reference(folder):
    """Validate all six completed opponents and retain their original provenance.

    The source model/runtime may differ from the new training run. These results
    initialize PFSP and serve as a historical comparison, never as an evaluation
    of the newly collected BC model or as evidence that it passed a guardrail.
    """
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if (status.get("complete") is not True or status.get("missing_series")
            or status.get("opponents_below_minimum_attack_rounds")):
        raise ValueError("Initial evaluation must be complete")
    series = manifest.get("series_count")
    if type(series) is not int or series < 1 or set(manifest.get("opponents", ())) != set(OPPONENTS):
        raise ValueError("Initial evaluation must cover all six opponents")
    if manifest.get("controller") not in ("v2", "ghost_champions_v2"):
        raise ValueError("Initial evaluation must belong to GC v2")
    rates = {}
    for code, (name, _) in OPPONENTS.items():
        row = summary.get(name, {})
        rate, count, wins = row.get("attack_rate"), row.get("n"), row.get("wins")
        if (type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 1
                or type(count) is not int or count <= 0 or type(wins) is not int or not 0 <= wins <= count
                or not math.isclose(rate, wins / count, abs_tol=1e-12)):
            raise ValueError("Initial evaluation has invalid attack results: " + code)
        rates[code] = rate
    names = ["manifest.json", "status.json", "summary.json"]
    names += [f"series_{code}_{index:03d}.json" for code in OPPONENTS for index in range(series)]
    if any(not (folder / name).is_file() for name in names):
        raise ValueError("Initial evaluation series files are missing")
    checkpoint = dict(manifest.get("residual_checkpoint") or {})
    path = Path(checkpoint.get("path", ""))
    if not path.is_file() or _digest(path) != checkpoint.get("sha256"):
        raise ValueError("Initial evaluation source checkpoint differs or is missing")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return dict(version=1, usage="historical_baseline", folder=str(folder), rates=rates,
                hashes={name: _digest(folder / name) for name in names},
                source_checkpoint=checkpoint, source_schema_hash=payload.get("schema_hash"),
                source_runtime_snapshot=manifest.get("runtime_snapshot"),
                source_runtime_hashes=manifest.get("runtime_data_hashes", {}),
                scheduled_series=len(OPPONENTS) * series,
                evaluates_new_bc=False, initial_guardrail_passed=False)


def bind_initial_evaluation(data_dir, source):
    data_dir = Path(data_dir)
    reference = evaluation_reference(source)
    path = data_dir / REFERENCE_NAME
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != reference:
            raise ValueError("Run already references a different initial evaluation; use a new run")
    else:
        if (data_dir / "manifest.json").exists():
            raise ValueError("Cannot change the initial evaluation of an existing training run")
        data_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    return reference


def load_initial_evaluation(data_dir):
    path = Path(data_dir) / REFERENCE_NAME
    if not path.exists():
        return None
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference != evaluation_reference(reference["folder"]):
        raise ValueError("Referenced initial evaluation artifacts changed")
    return reference
