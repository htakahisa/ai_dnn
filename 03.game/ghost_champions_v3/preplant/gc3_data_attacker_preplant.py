"""Compressed, non-pickle demonstration shards, split by whole rounds."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ..gc3_observation import CELLS, OBS_DIM
from .gc3_policy_attacker_history import DATA_SCHEMA_HASH as SCHEMA_HASH
from .gc3_observation_attacker_history import MatchHistory, HISTORY_OBS_DIM
from ..gc3_io import write_json
from .gc3_rollout_attacker_preplant import choose_roster, play_preplant

KINDS = ("mode", "position", "watch", "target", "ally")
MASK_WIDTHS = dict(mode=8, position=CELLS, watch=CELLS, target=CELLS, ultimate_target=CELLS, ally=5)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_signature():
    root = Path(__file__).resolve().parents[1]
    paths = list(root.glob("gc3_*.py"))+list((root/"preplant").glob("*.py"))
    paths += list((root.parent/"concon_v1").glob("*.py"))
    paths += list((root.parent/"omoko_v1").glob("*.py"))
    paths += list(root.parent.glob("*.py"))+list((root.parent/"analytics").glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        if path.is_file() and not (path.parent == root and path.name.startswith(("gc3_train_", "gc3_collect_"))) and not (
                path.parent == root/"preplant" and path.name in (
                    "gc3_train_attacker_objective_rl.py","gc3_train_attacker_retrieval_bc.py",
                    "gc3_train_attacker_retrieval_rl.py")):
            digest.update(str(path.relative_to(root.parent)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def save_shard(path, rows):
    if not rows:
        raise ValueError("Cannot save an empty teacher shard")
    arrays = dict(obs=np.stack([r["obs"] for r in rows]).astype(np.float32),
                  selections=np.array([[r["selections"].get(kind, -1) for kind in KINDS] for r in rows], dtype=np.int32),
                  watch_options=np.packbits(np.stack([r["watch_options"] if r["watch_options"] is not None
                      else np.zeros(CELLS, dtype=np.bool_) for r in rows]), axis=1))
    if any(r.get("goal_site") is not None for r in rows):
        arrays["goal_site"] = np.array([-1 if r.get("goal_site") is None else r["goal_site"] for r in rows], dtype=np.int8)
        arrays["site_selection"] = np.array([r["selections"].get("site", -1) for r in rows], dtype=np.int8)
    # A full match is one shard, preventing context leakage across data splits.
    histories = {id(r.get("history")):r.get("history") for r in rows if r.get("history") is not None}
    if histories:
        if len(histories) != 1 or any(r.get("history") is None for r in rows):
            raise ValueError("A history shard must contain exactly one complete match")
        history = next(iter(histories.values()))
        arrays.update(history_obs=np.stack([e["obs"] for e in history.events]),
                      history_action=np.stack([e["action"] for e in history.events]),
                      history_index=np.array([r["history_index"] for r in rows], dtype=np.int32),
                      history_meta=np.array(json.dumps(dict(round_id=history.round_id, results=history.results,
                          events=[{k:v for k,v in e.items() if k not in ("obs", "action")} for e in history.events]),
                          ensure_ascii=False, default=lambda v:v.item() if isinstance(v, np.generic) else list(v))))
    for kind in MASK_WIDTHS:
        arrays["mask_"+kind] = np.packbits(np.stack([r["masks"][kind] for r in rows]), axis=1)
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def load_shard(path):
    with np.load(path, allow_pickle=False) as data:
        observations, selections = data["obs"], data["selections"]
        is_history = "history_obs" in data
        width = HISTORY_OBS_DIM if is_history else OBS_DIM
        if observations.shape != (len(selections), width) or not np.isfinite(observations).all():
            raise ValueError("Invalid teacher observations")
        history = None
        if is_history:
            history = MatchHistory()
            metadata = json.loads(str(data["history_meta"]))
            history.events = [dict(event, obs=o.copy(), action=a.copy()) for event,o,a in
                zip(metadata["events"], data["history_obs"], data["history_action"])]
            history.round_id, history.results = metadata["round_id"], metadata["results"]
            history.goal_sites = {int(e["round_id"]):int(e["goal_site"]) for e in history.events if e.get("goal_site") is not None}
            history_indices = data["history_index"].copy()
            if len(history_indices) != len(selections) or np.any(history_indices < 0) or np.any(history_indices >= len(history.events)):
                raise ValueError("Invalid chronological teacher indices")
            if np.any(np.diff(history_indices) <= 0) or any(not np.array_equal(obs, history.events[int(i)]["obs"])
                                                         for obs,i in zip(observations, history_indices)):
                raise ValueError("Teacher labels and causal observations are misaligned")
        masks = {kind:np.unpackbits(data["mask_"+kind], axis=1, count=width).astype(np.bool_)
                 for kind, width in MASK_WIDTHS.items()}
        options = np.unpackbits(data["watch_options"], axis=1, count=CELLS).astype(np.bool_)
        rows = []
        for index, observation in enumerate(observations):
            row = dict(obs=observation, phase="standard", hints=np.zeros((2, CELLS), dtype=np.float32),
                selections={kind:int(selections[index, j]) for j, kind in enumerate(KINDS) if selections[index, j] >= 0},
                masks={kind:mask[index] for kind, mask in masks.items()},
                watch_options=options[index] if options[index].any() else None)
            if history is not None:
                row.update(history=history, history_index=int(history_indices[index]))
            if "goal_site" in data:
                goal, choice = int(data["goal_site"][index]), int(data["site_selection"][index])
                if goal not in (-1, 0, 1) or choice not in (-1, 0, 1) or (choice >= 0 and choice != goal):
                    raise ValueError("Invalid own-site intention")
                row["goal_site"] = None if goal < 0 else goal
                if choice >= 0:
                    row["selections"]["site"] = choice
                    row["masks"]["site"] = np.ones(2, dtype=np.bool_)
            if any(action >= len(row["masks"][kind]) or not row["masks"][kind][action]
                   for kind, action in row["selections"].items()):
                raise ValueError("Teacher label is outside the legal mask")
            rows.append(row)
        return rows


def dataset_manifest(path):
    path = Path(path)
    manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("schema_hash") != SCHEMA_HASH or not manifest.get("history"):
        raise ValueError("Teacher dataset is incomplete or incompatible")
    for shard in manifest["shards"]:
        if file_hash(path/shard["file"]) != shard["sha256"]:
            raise ValueError("Teacher shard changed: "+shard["file"])
    if not all(any(s["split"] == split for s in manifest["shards"]) for split in ("train", "validation")):
        raise ValueError("Need both training and validation rounds")
    return manifest, file_hash(path/"manifest.json")


def collect(config):
    path, logs = config["DATASET_PATH"], config["LOG_PATH"]
    if path.exists():
        raise RuntimeError("Dataset already exists; select a new DATASET_NAME, existing data are retained")
    if logs.exists() and any(logs.iterdir()):
        raise RuntimeError("Dataset logs already exist; select a new DATASET_NAME")
    if config["ROUNDS_PER_MATCH"] < 2 or config["COLLECTION_ROUNDS"] < 2*config["ROUNDS_PER_MATCH"] or config["VALIDATION_EVERY_MATCHES"] < 2:
        raise ValueError("Need at least two multi-round matches; validation period must be >= 2")
    torch.set_num_threads(config["TORCH_THREADS"])
    path.mkdir(parents=True)
    logs.mkdir(parents=True, exist_ok=True)
    teacher_kind = config.get("TEACHER_KIND", "concon")
    manifest = dict(schema_hash=SCHEMA_HASH, source_hash=source_signature(), complete=False,
        teacher=("Omoko Gaming v1" if teacher_kind == "omoko" else "ConCon v1")+" (frozen, roster-slot adapter)",
        teacher_kind=teacher_kind, teacher_roster="Ghost Champions", student_roster="Ghost Champions",
        opponent_ai=config["FIXED_OPPONENT_AI"], opponent_codes=list(config["OPPONENT_CODES"]),
        teacher_maps=list(config["TEACHER_MAPS"]), split="whole matches including all context rounds, disjoint seeds", shards=[],
        history="all public observations/issued actions/finished outcomes; causal attention; no window truncation")
    manifest["config"] = {key:str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple)
                          else value for key, value in config.items()}
    write_json(path/"manifest.json", manifest)
    counts, attempted, skipped, carrying, retrieval, orb_collections = {}, 0, 0, 0, 0, 0
    match_rows, history, match_index = [], MatchHistory(), 0
    for episode in range(config["COLLECTION_ROUNDS"]):
        code = choose_roster(config["SEED"], episode, config["OPPONENT_CODES"])
        rows, stats = play_preplant(code, config["SEED"]+episode, teacher=True,
            teacher_maps=config["TEACHER_MAPS"], fixed_opponent_ai=config["FIXED_OPPONENT_AI"], history=history,
            teacher_kind=teacher_kind)
        if manifest.get("teacher_model_hashes") not in (None, stats["teacher_model_hashes"]):
            raise RuntimeError("ConCon checkpoints changed during collection; keep one frozen teacher dataset")
        with (logs/"episodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stats, ensure_ascii=False)+"\n")
        attempted += len(rows)+sum(stats["skipped"].values())
        skipped += sum(stats["skipped"].values())
        carrying += stats["carrying_decisions"]
        retrieval += stats["retrieval_decisions"]
        orb_collections += stats["orb_collections"]
        for mode, amount in stats["mode_counts"].items():
            counts[mode] = counts.get(mode, 0)+amount
        match_rows.extend(rows)
        if (episode+1) % config["ROUNDS_PER_MATCH"] == 0 or episode+1 == config["COLLECTION_ROUNDS"]:
            filename = f"match_{match_index:06d}.npz"
            save_shard(path/filename, match_rows)
            manifest["shards"].append(dict(file=filename, sha256=file_hash(path/filename), decisions=len(match_rows),
                seed=stats["seed"], opponent=code, teacher_map=stats["teacher_map"],
                split="validation" if match_index % config["VALIDATION_EVERY_MATCHES"] == 0 else "train"))
            match_index += 1
            match_rows, history = [], MatchHistory()
        manifest.update(rounds=episode+1, decisions=attempted-skipped, skipped=skipped,
                        mode_counts=counts, teacher_model_hashes=stats["teacher_model_hashes"],
                        carrying_decisions=carrying, retrieval_decisions=retrieval, orb_collections=orb_collections)
        write_json(path/"manifest.json", manifest)
        print(f"teacher round={episode+1}/{config['COLLECTION_ROUNDS']} roster={code} "
              f"map={stats['teacher_map']} decisions={len(rows)} planted={stats['planted']}", flush=True)
    if not counts.get("PLANT") or skipped/max(1, attempted) > config["MAX_SKIPPED_FRACTION"]:
        raise RuntimeError("Teacher coverage failed: no PLANT labels or too many unrepresentable actions; inspect episodes.jsonl")
    if config.get("REQUIRE_ORB_LABELS") and (not counts.get("COLLECT_ORB") or not orb_collections):
        raise RuntimeError("Omoko supplied no completed orb collection demonstrations; dataset kept for inspection")
    manifest.update(complete=True, skipped_fraction=skipped/max(1, attempted))
    write_json(path/"manifest.json", manifest)
    print(f"Teacher data ready: {path}; decisions={attempted-skipped}, skipped={skipped}", flush=True)
