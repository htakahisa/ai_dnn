"""Build carry route priors using consecutive twelve-round attacking halves.

Working directory: omoko_v1. Normal command:
    python ov1_evaluate_attacker_carry_routes.py
No training, checkpoint updates or production route-probability changes.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch


AI_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AI_DIR.parent
for directory in (PROJECT_DIR, AI_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

# Normal-use settings. CLI options are only temporary overrides.
ROUNDS_PER_SET = 12
SETS_PER_ROUTE = 8
ROUNDS_PER_ROUTE = ROUNDS_PER_SET * SETS_PER_ROUTE  # 96
BASE_SEED = 20261015
RESUME = True
# Keep compatible progress; replace obsolete results at the same output path.
OVERWRITE_INCOMPATIBLE = True
PARALLEL_WORKERS = 6  # One independent process per opponent; CPU inference.
OUTPUT_PATH = AI_DIR / "data" / "carry_route_evaluation" / "route_win_rates_12round_sets.json"
LOG_PATH = AI_DIR / "logs" / "ov1_evaluate_attacker_carry_routes.log"
MAX_TICKS_PER_ROUND = 500
OPPONENTS = {
    "toru_ai_v4": "Team Elites",
    "concon_v1": "Gorigons",
    "touyama_gaming_v3": "Touyama Gaming",
    "fnatic_v3": "Fnatic2023",
    "frc_v1": "Furina Classic",
    "ghost_champions_v2": "Ghost Champions",
}
SCHEMA_VERSION = 2
EVALUATION_VERSION = "12round_attack_set_full_engine_fixed_carry_route_v2"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def route_id(site, waypoint):
    return f"{site}:{int(waypoint[0])},{int(waypoint[1])}"


def read_routes():
    from ov1_learning_attacker_carry import _parse_waypoint_cells
    from ov1_map_data_carry import NEW_MAZE_STR
    return [
        {"id": route_id(site, point), "site": site, "waypoint": list(point)}
        for site, points in _parse_waypoint_cells(NEW_MAZE_STR).items()
        for point in points
    ]


def scenario_manifest(routes):
    """Invalidate resume when engine, map, roster or AI source code changes."""
    from grid_lines import WALL_LOS_VERSION
    source_files = list(PROJECT_DIR.glob("*.py"))
    # AI packages use lazy imports, so include their sources before any trials.
    for directory in PROJECT_DIR.iterdir():
        if directory.is_dir() and directory.name not in {
            ".git", ".codex", "data", "logs", "test", "analytics", "docs",
            "evaluation_debug", "competition_results", "season", "image",
        }:
            source_files.extend(
                p for p in directory.rglob("*.py")
                if not {"test", "tests", "data", "logs", "__pycache__"} & set(p.relative_to(directory).parts)
            )
    sources = {str(p.relative_to(PROJECT_DIR)).replace("\\", "/"): sha256(p)
               for p in sorted(set(source_files))}
    payload = {"evaluation_version": EVALUATION_VERSION, "wall_los_version": WALL_LOS_VERSION,
               "rounds_per_set": ROUNDS_PER_SET,
               "routes": routes, "source_hashes": sources}
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def same_scenario(saved, current):
    if saved.get("fingerprint") == current.get("fingerprint"):
        return True
    required = {"evaluation_version", "wall_los_version", "rounds_per_set", "routes", "source_hashes"}
    if not required <= saved.keys() or not required <= current.keys():
        return False
    left, right = copy.deepcopy(saved), copy.deepcopy(current)
    for manifest in (left, right):
        manifest.pop("fingerprint", None)
        # Infrastructure-only edits may resume. Any change to the actual
        # evaluation protocol must bump EVALUATION_VERSION (e.g. 1 vs 12 rounds).
        manifest["source_hashes"].pop("omoko_v1/ov1_evaluate_attacker_carry_routes.py", None)
    carry_source = "omoko_v1/ov1_learning_attacker_carry.py"
    # A single reviewed reader-only integration is compatible with already
    # running fixed-route evaluations: fixed_attacker_factory overrides the
    # changed _choose_route, and the new cache never consumes RNG state.
    if (left["source_hashes"].get(carry_source), right["source_hashes"].get(carry_source)) == (
        "9b18eb7ed98a078266fb41ad6f323eed29e09f919ba524d0e408c381647e2db9",
        "c864b41a68811b859ab2fa56cf4d3fdc8ec38d0a211df8c00e054700f871c490",
    ):
        left["source_hashes"][carry_source] = right["source_hashes"][carry_source]
        for consumer in ("omoko_v1/ov1_attacker_carry_route_priors.py",
                         "omoko_v1/ov1_summarize_attacker_carry_routes.py"):
            if consumer not in left["source_hashes"]:
                right["source_hashes"].pop(consumer, None)
    return left == right


def trial_seed(base_seed, opponent, trial):
    # Same seed across routes; independent of route insertion/order and resume.
    text = f"{base_seed}:{opponent}:{trial}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "big")


@contextmanager
def legacy_resources():
    """Keep legacy AI-relative model paths internal to the engine adapter.

    The user launches in omoko_v1; outputs always use absolute AI-local paths.
    Restore the caller's directory even when a model or simulation fails.
    """
    previous = Path.cwd()
    try:
        os.chdir(PROJECT_DIR)
        yield
    finally:
        os.chdir(previous)


class FrozenModels:
    """Cache exact checkpoint bytes and reject concurrent checkpoint changes."""

    def __init__(self, known=None):
        self.hashes = dict(known or {})
        self.bytes = {}
        self.stamps = {}
        self.original_load = None
        self.original_read_bytes = Path.read_bytes

    def validate(self):
        for relative, expected in self.hashes.items():
            path = PROJECT_DIR / relative
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Model changed since saved evaluation: {relative}; use a new output file")

    def read_model(self, source):
        path = Path(source).resolve()
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if path in self.stamps and self.stamps[path] != stamp:
            raise ValueError(f"Checkpoint changed during evaluation: {path}")
        if path not in self.bytes:
            raw = self.original_read_bytes(path)
            key = os.path.relpath(path, PROJECT_DIR).replace("\\", "/")
            digest = hashlib.sha256(raw).hexdigest()
            if key in self.hashes and self.hashes[key] != digest:
                raise ValueError(f"Checkpoint differs from previous results: {key}")
            self.hashes[key] = digest
            self.bytes[path] = raw
            self.stamps[path] = stamp
        return self.bytes[path]

    def load(self, source, *args, **kwargs):
        if not isinstance(source, (str, os.PathLike)):
            return self.original_load(source, *args, **kwargs)
        return self.original_load(io.BytesIO(self.read_model(source)), *args, **kwargs)

    @contextmanager
    def active(self):
        import torch
        self.validate()
        self.original_load = torch.load
        def read_bytes(path):
            if path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
                return self.read_model(path)
            return self.original_read_bytes(path)
        with patch.object(torch, "load", self.load), patch.object(Path, "read_bytes", read_bytes):
            yield self


def fixed_attacker_factory(factory, route):
    """Override only this evaluation instance, before the first reset_round."""
    def create():
        controller = factory()
        carry = controller.carry_controller
        point = tuple(route["waypoint"])
        index = carry.waypoint_cells_by_site[route["site"]].index(point)
        carry._choose_route = lambda current_round: (route["site"], index)
        return controller
    return create


def run_set(opponent, roster_name, route, seed):
    import numpy as np
    import torch
    from map_data import NEW_MAZE_STR
    from party_presets import get_preset
    from run_game import VisualFPSBattle, _build_team_ai
    from run_competition_manager import TeamPlayerKey, _key_for_name

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    our, other = get_preset("Omoko Gaming"), get_preset(roster_name)
    if our is None or other is None:
        raise ValueError(f"Unknown roster: {roster_name}")
    # Use the match manager's team-scoped identities. Team Elites also has
    # おもこ; plain names would merge the two players' persistent ultimate bank.
    our_keys = [TeamPlayerKey(name, f"attacker:{our.name}") for name in our.players]
    other_keys = [TeamPlayerKey(name, f"defender:{other.name}") for name in other.players]
    attacker_ai = _build_team_ai("omoko_gaming_v1", device="cpu")
    attacker_ai.attacker_factory = fixed_attacker_factory(attacker_ai.attacker_factory, route)
    game = VisualFPSBattle(
        NEW_MAZE_STR, attacker_ai, _build_team_ai(opponent, device="cpu"),
        headless=True, attacker_roster=our_keys, defender_roster=other_keys,
        spike_holder_name=_key_for_name(our_keys, our.spike_holder),
        defender_spike_holder_name=_key_for_name(other_keys, other.spike_holder),
        attacker_igl_name=_key_for_name(our_keys, our.igl),
        defender_igl_name=_key_for_name(other_keys, other.igl),
        attacker_team_name=our.name, defender_team_name=other.name, disable_side_swap=True,
    )
    game.stop_after_round = True
    game.record_replay = False
    return play_set(game, attacker_ai, route, seed)


def play_set(game, attacker_ai, route, seed):
    """Use normal round transitions on the same game and controller instances.

    stop_after_round pauses at the terminal state to record it; the usual
    current_round increment + init_round then preserves the real game's
    ultimate bank, scores and AI match memory. No reseeding between rounds.
    """
    results = []
    for round_index in range(ROUNDS_PER_SET):
        before_a, before_d = game.attacker_wins, game.defender_wins
        start_ultimate = {f"{c.team}:{c.name}": int(c.ultimate_points) for c in game.chars}
        for _ in range(MAX_TICKS_PER_ROUND):
            if game.round_over or game.match_over:
                break
            if not game.step_tick():
                break
        delta_a, delta_d = game.attacker_wins - before_a, game.defender_wins - before_d
        if (not game.round_over or (delta_a, delta_d) not in {(1, 0), (0, 1)}
                or game.current_round != round_index + 1 or game.sides_swapped):
            raise RuntimeError(f"Attacking round {round_index + 1} did not finish normally")
        carry = attacker_ai.get_attacker_controller().carry_controller
        actual = carry.waypoint_cells_by_site[carry._active_site][carry._waypoint_index]
        if carry._active_site != route["site"] or tuple(actual) != tuple(route["waypoint"]):
            raise RuntimeError("Applied carry route differs from requested route")
        results.append({
            "round_number": int(game.current_round), "won": bool(delta_a),
            "planted": bool(game.is_planted), "defused": bool(game.is_defused),
            "battle_ticks": int(game.battle_tick), "target_plant_pos": list(carry._active_target),
            "planted_pos": list(game.planted_pos) if game.planted_pos is not None else None,
            "start_ultimate_points": start_ultimate,
            "end_ultimate_points": {f"{c.team}:{c.name}": int(c.ultimate_points) for c in game.chars},
            "score_after": [int(game.attacker_wins), int(game.defender_wins)],
        })
        if round_index + 1 < ROUNDS_PER_SET:
            if game.match_over:
                raise RuntimeError("Match ended before twelve attacking rounds")
            game.current_round += 1
            game.init_round()
    return {"seed": seed, "rounds": results}


def refresh_summary(row):
    results = [result for batch in row["sets"] for result in batch["rounds"]]
    row["sets_completed"] = len(row["sets"])
    row["rounds"] = len(results)
    row["wins"] = sum(r["won"] for r in results)
    row["losses"] = row["rounds"] - row["wins"]
    row["win_rate"] = row["wins"] / row["rounds"] if row["rounds"] else None
    row["plants"] = sum(r["planted"] for r in results)


def save_report(path, report, models):
    report["model_hashes"] = dict(models.hashes)
    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            # Windows readers/antivirus may briefly deny replacement of an
            # open file; leave the previous complete JSON intact and retry.
            if attempt == 19:
                raise
            time.sleep(0.05)


def reusable_report(path, manifest, base_seed, resume):
    """Read compatible progress without writing, including for --check."""
    if not path.exists() or not resume:
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("schema_version") != SCHEMA_VERSION
                or not same_scenario(report.get("scenario", {}), manifest)
                or report.get("base_seed") != base_seed):
            raise ValueError("Saved scenario/source/seed differs")
        FrozenModels(report.get("model_hashes")).validate()
    except (ValueError, FileNotFoundError) as exc:
        if not OVERWRITE_INCOMPATIBLE:
            raise
        print(f"Starting fresh at {path}: {exc}", flush=True)
        return None
    if report["scenario"]["fingerprint"] != manifest["fingerprint"]:
        report.setdefault("infrastructure_resume_history", []).append(report["scenario"])
        report["scenario"] = manifest
    return report


def evaluate(opponents, rounds=ROUNDS_PER_ROUTE, base_seed=BASE_SEED,
             output=OUTPUT_PATH, resume=RESUME, check_only=False,
             log_path=None, stop_event=None):
    from simulation_runtime import cpu_inference
    routes = read_routes()
    if not routes or rounds < 1 or rounds % ROUNDS_PER_SET:
        raise ValueError(f"Round count must be a positive multiple of {ROUNDS_PER_SET}, with map waypoints")
    target_sets = rounds // ROUNDS_PER_SET
    if not opponents or any(key not in OPPONENTS for key in opponents):
        raise ValueError("Select at least one known opponent from OPPONENTS")
    output = Path(output).resolve()
    manifest = scenario_manifest(routes)
    report = reusable_report(output, manifest, base_seed, resume)
    if report is None:
        report = {"schema_version": SCHEMA_VERSION, "scenario": manifest,
                  "base_seed": base_seed, "evaluation": EVALUATION_VERSION,
                  "our_team": "Omoko Gaming", "our_controller": "omoko_gaming_v1",
                  "rounds_per_set": ROUNDS_PER_SET,
                  "metric": "attacker_round_win_rate_including_postplant",
                  "opponents": {}}
    models = FrozenModels(report.get("model_hashes"))
    print(f"Routes={len(routes)}, opponents={len(opponents)}, "
          f"sets/route={target_sets}, rounds/set={ROUNDS_PER_SET}, rounds/route={rounds}", flush=True)
    print(f"Output: {output}", flush=True)
    if check_only:
        models.validate()
        for route in routes:
            print(f"  {route['id']}")
        return report
    # Commit the reset before any new trials, so old results cannot survive an
    # interrupted first set. Parallel shards are replaced in the parent too.
    save_report(output, report, models)
    log_path = Path(log_path) if log_path is not None else LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log, legacy_resources(), cpu_inference(), models.active():
        try:
            for opponent in opponents:
                entry = report["opponents"].setdefault(opponent, {
                    "roster": OPPONENTS[opponent], "routes": {}, "status": "running",
                })
                if entry["roster"] != OPPONENTS[opponent]:
                    raise ValueError(f"Roster changed for {opponent}")
                entry["status"] = "running"
                entry["target_rounds_per_route"] = rounds
                entry.pop("error", None)
                try:
                    for route in routes:
                        row = entry["routes"].setdefault(route["id"], {**route, "sets": []})
                        refresh_summary(row)
                        print(f"[{opponent}] {route['id']}: {row['rounds']}/{rounds}", flush=True)
                        for trial in range(len(row["sets"]), target_sets):
                            if stop_event is not None and stop_event.is_set():
                                entry["status"] = "interrupted"
                                return report
                            with redirect_stdout(log):
                                result = run_set(opponent, entry["roster"], route,
                                                 trial_seed(base_seed, opponent, trial))
                            if len(result["rounds"]) != ROUNDS_PER_SET:
                                raise RuntimeError("Incomplete twelve-round set; result discarded")
                            row["sets"].append(result)
                            refresh_summary(row)
                            save_report(output, report, models)
                            print(f"[{opponent}] sets={row['sets_completed']}/{target_sets}, "
                                  f"rounds={row['rounds']}/{rounds}: wins={row['wins']} "
                                  f"win_rate={row['win_rate']:.1%}", flush=True)
                    entry["status"] = "complete"
                except Exception as exc:
                    entry["status"] = "error"
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"[{opponent}] ERROR: {entry['error']}", flush=True)
                save_report(output, report, models)
        finally:
            # Only complete sets are committed; restart an interrupted set
            # with its original seed so AI memory is never lost mid-set.
            save_report(output, report, models)
    errors = {
        key: report["opponents"][key].get("error", report["opponents"][key]["status"])
        for key in opponents if report["opponents"][key]["status"] != "complete"
    }
    if errors:
        details = "; ".join(f"{key}: {error}" for key, error in errors.items())
        raise RuntimeError(f"Evaluation incomplete ({details}); see output JSON")
    return report


_WORKER_STOP_EVENT = None


def _initialize_worker(stop_event):
    global _WORKER_STOP_EVENT
    _WORKER_STOP_EVENT = stop_event


def _evaluate_worker(opponent, rounds, base_seed, output, log_path):
    # Windows spawn entry point; no shared CWD, RNG, model cache or log stream.
    evaluate([opponent], rounds, base_seed, output, log_path=log_path,
             stop_event=_WORKER_STOP_EVENT)
    return opponent


def merge_worker_report(report, child, opponent, models):
    """Merge exactly one opponent, rejecting different scenarios or weights."""
    if (child.get("schema_version") != report["schema_version"]
            or not same_scenario(child.get("scenario", {}), report["scenario"])
            or child.get("base_seed") != report["base_seed"]
            or set(child.get("opponents", {})) != {opponent}):
        raise ValueError(f"Worker result differs from evaluation: {opponent}")
    incoming = child.get("model_hashes", {})
    for name, digest in incoming.items():
        if name in models.hashes and models.hashes[name] != digest:
            raise ValueError(f"Workers used different checkpoints: {name}")
    models.hashes.update(incoming)
    report["opponents"][opponent] = copy.deepcopy(child["opponents"][opponent])


def evaluate_parallel(opponents, rounds=ROUNDS_PER_ROUTE, base_seed=BASE_SEED,
                      output=OUTPUT_PATH, resume=RESUME, workers=PARALLEL_WORKERS,
                      check_only=False):
    if workers < 1:
        raise ValueError("workers must be positive")
    if len(opponents) != len(set(opponents)):
        raise ValueError("Duplicate opponents are not allowed")
    if workers == 1 or len(opponents) == 1:
        return evaluate(opponents, rounds, base_seed, output, resume, check_only)
    # Parent validates the unified output; it is the only process writing it.
    report = evaluate(opponents, rounds, base_seed, output, resume, check_only=True)
    if check_only:
        return report
    report.pop("worker_errors", None)
    output = Path(output).resolve()
    shard_dir = output.parent / f"{output.stem}_workers"
    shard_dir.mkdir(parents=True, exist_ok=True)
    models = FrozenModels(report.get("model_hashes"))
    paths = {opponent: shard_dir / f"{opponent}.json" for opponent in opponents}
    for opponent, path in paths.items():
        child = reusable_report(path, report["scenario"], base_seed, resume)
        if child is not None:
            merge_worker_report(report, child, opponent, models)
        else:
            # Import existing serial progress into the new per-opponent file.
            child = copy.deepcopy(report)
            child["opponents"] = {opponent: child["opponents"].get(opponent, {
                "roster": OPPONENTS[opponent], "routes": {}, "status": "pending",
            })}
            save_report(path, child, models)
    models.validate()
    save_report(output, report, models)
    count = min(workers, len(opponents))
    print(f"Starting {count} parallel opponent workers (12-round sets preserved)", flush=True)
    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    seen = {}
    failures = {}

    def collect():
        changed = False
        for opponent, path in paths.items():
            stamp = path.stat().st_mtime_ns
            if seen.get(opponent) != stamp:
                merge_worker_report(report, json.loads(path.read_text(encoding="utf-8")), opponent, models)
                seen[opponent] = stamp
                changed = True
        if changed:
            save_report(output, report, models)

    with ProcessPoolExecutor(max_workers=count, mp_context=context,
                             initializer=_initialize_worker, initargs=(stop_event,)) as pool:
        futures = {}
        try:
            for opponent, path in paths.items():
                log_path = LOG_PATH.with_name(f"{LOG_PATH.stem}_{opponent}{LOG_PATH.suffix}")
                future = pool.submit(_evaluate_worker, opponent, rounds, base_seed, path, log_path)
                futures[future] = opponent
            while futures:
                done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                collect()
                for future in done:
                    opponent = futures.pop(future)
                    try:
                        future.result()
                    except Exception as exc:
                        failures[opponent] = f"{type(exc).__name__}: {exc}"
                        print(f"[{opponent}] worker error: {failures[opponent]}", flush=True)
        except BaseException:
            stop_event.set()
            print("Stopping workers after their current 12-round set; saving progress...", flush=True)
            raise
        finally:
            # Wait for a current set to finish, without starting another one.
            if stop_event.is_set():
                wait(futures)
            collect()
            if failures:
                report["worker_errors"] = failures
            save_report(output, report, models)
    incomplete = [opponent for opponent in opponents
                  if report["opponents"].get(opponent, {}).get("status") != "complete"]
    if failures or incomplete:
        details = []
        for opponent in sorted(set(incomplete) | set(failures)):
            reason = failures.get(opponent) or report["opponents"].get(opponent, {}).get("error", "incomplete")
            details.append(f"{opponent}: {reason}")
        raise RuntimeError(f"Parallel evaluation incomplete ({'; '.join(details)})")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate routes/resume without playing")
    amount = parser.add_mutually_exclusive_group()
    amount.add_argument("--rounds", type=int, help="positive multiple of 12")
    amount.add_argument("--sets", type=int, help="number of twelve-round sets")
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=list(OPPONENTS))
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--workers", type=int, default=PARALLEL_WORKERS)
    args = parser.parse_args()
    started = time.monotonic()
    rounds = args.rounds if args.rounds is not None else (
        args.sets * ROUNDS_PER_SET if args.sets is not None else ROUNDS_PER_ROUTE
    )
    evaluate_parallel(args.opponents, rounds, args.seed, args.output,
                      workers=args.workers, check_only=args.check)
    print(f"Done in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
