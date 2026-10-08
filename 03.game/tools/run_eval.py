"""Reproducible BO3 evaluations using the production series engine."""
from __future__ import annotations

import argparse
import contextlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OPPONENTS = {
    "TYG": ("Touyama Gaming", "touyama_gaming_v2"),
    "OMG": ("Omoko Gaming", "omoko_gaming_v1"),
    "FRC": ("Furina Classic", "frc_v1"),
    "FNC": ("Fnatic2023", "fnatic_v3"),
    "GG": ("Gorigons", "concon_v1"),
    "SPS": ("SUPES", "toru_ai_v3.1"),
}
TEAM = "Ghost Champions"
RUNTIME_FILES=("game_core.py","character_stats.py","player_combos.py","awakening_events.py","party_presets.py","map_data.py")


def runtime_snapshot(folder):
    """Copy gameplay definitions once; game_core then loads beside its own copy."""
    folder=Path(folder).resolve()
    if not folder.exists():
        folder.mkdir(parents=True)
        for name in RUNTIME_FILES:
            shutil.copy2(ROOT/name,folder/name)
    if any(not (folder/name).is_file() for name in RUNTIME_FILES):
        raise ValueError("Runtime snapshot is incomplete; use a fresh snapshot directory")
    return folder


def wilson(wins, n):
    if not n:
        return None
    z = 1.959963984540054
    p = wins / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [center-margin, center+margin]


def matched_baseline(paths, folder):
    """Compare the same opponent/series indices, including small pilot runs."""
    matching = [Path(folder)/Path(path).name for path in paths]
    missing = [str(path) for path in matching if not path.exists()]
    if missing:
        raise ValueError(f"Matching baseline series missing: {missing}")
    return summarize(matching)


def summarize(paths):
    results = {}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        opp = data["team2"] if data["team1"] == TEAM else data["team1"]
        row = results.setdefault(opp, dict(n=0, wins=0, plants=0, postplant_wins=0,
            defuses=0, preplant_wipes=0, preplant_timeouts=0, recon_used=0, recon_available=0,
            recon_used_by_30=0, site_selected=0, site_four_plus=0, site_two=0, site_two_or_three=0, overtime_excluded=0,
            distance_10=[], distance_20=[], setups=Counter(), first_deaths=Counter()))
        for mp in data["maps"]:
            frames = defaultdict(list)
            for frame in mp.get("replay_frames", []):
                frames[frame["round"]].append(frame)
            for record in mp["round_records"]:
                rn = record["round_number"]
                if rn > 24:
                    row["overtime_excluded"] += 1
                    continue
                players = record.get("players", {})
                sides = {p.get("side") for p in players.values() if p.get("team") == TEAM}
                attacker = ("attacker" in sides if sides else
                    (mp["initial_attacker"] == TEAM) == (rn <= 12))
                if not attacker:
                    continue
                won = record["winner"] == "attacker"
                planted = bool(record["planted"])
                row["n"] += 1
                row["wins"] += won
                row["plants"] += planted
                row["postplant_wins"] += won and planted
                row["defuses"] += record["reason"] == "defused" and not won
                row["preplant_wipes"] += not planted and record["reason"] == "attacker_wipe"
                row["preplant_timeouts"] += not planted and not won and record["reason"] == "time_expired"
                tactic = record.get("tactic", {})
                setup = tactic.get("defender_initial_setup", "")
                row["setups"][setup] += 1
                counts = setup.split("-")
                if len(counts) == 3 and tactic.get("final_attack_site") in ("A", "B"):
                    row["site_selected"] += 1
                    count = int(counts[0 if tactic["final_attack_site"] == "A" else 2])
                    row["site_four_plus"] += count >= 4
                    row["site_two"] += count == 2
                    row["site_two_or_three"] += count in (2, 3)
                for name, player in players.items():
                    if player.get("team") == TEAM:
                        row["first_deaths"][name] += player.get("first_deaths", 0)
                fs = sorted(frames[rn], key=lambda f: (f["tick"], not f["setup"]))
                if fs:
                    previous = {}
                    available = {}
                    used = 0
                    used_by_30 = 0
                    for frame in fs:
                        for char in frame["chars"]:
                            if char.get("ability") == "RECON" and char["team"] == "A":
                                name, charges = char["name"], char["ability_charges"]
                                available.setdefault(name, charges)
                                spent = max(0, previous.get(name, charges) - charges)
                                used += spent
                                if not frame["setup"] and frame["tick"] <= 30:
                                    used_by_30 += spent
                                previous[name] = charges
                    row["recon_available"] += sum(available.values())
                    row["recon_used"] += used
                    row["recon_used_by_30"] += used_by_30
                active = [f for f in fs if not f["setup"]]
                plant = next((f for f in active if f["planted"]), None)
                if plant:
                    for delta in (10, 20):
                        frame = next((f for f in active if f["tick"] == plant["tick"]+delta
                            and f["planted"] and not f.get("round_over")), None)
                        if frame:
                            spike = frame["planted_pos"]
                            distances = [math.dist(c["pos"], spike) for c in frame["chars"]
                                if c["team"] == "A" and c["alive"]]
                            if distances:
                                row[f"distance_{delta}"].append(sum(distances)/len(distances))
    for row in results.values():
        row["attack_rate"] = row["wins"]/row["n"] if row["n"] else None
        row["attack_wilson95"] = wilson(row["wins"], row["n"])
        row["postplant_rate"] = row["postplant_wins"]/row["plants"] if row["plants"] else None
        row["postplant_wilson95"] = wilson(row["postplant_wins"], row["plants"])
        row["plant_rate"] = row["plants"]/row["n"] if row["n"] else None
        row["plant_wilson95"] = wilson(row["plants"], row["n"])
        row["recon_rate"] = row["recon_used"]/row["recon_available"] if row["recon_available"] else None
        row["early_recon_rate"] = row["recon_used_by_30"]/row["recon_available"] if row["recon_available"] else None
        for delta in (10, 20):
            values = row[f"distance_{delta}"]
            row[f"distance_{delta}_n"] = len(values)
            row[f"distance_{delta}"] = sum(values)/len(values) if values else None
    return results


def markdown(results, baseline=None):
    lines = ["| Opponent | Attack W/n | Wilson 95% | Plants/n (rate) | Plant Wilson 95% | Postplant W/n | Wilson 95% | Defuses | Recon used/available (rate) | Δ attack |",
        "|---|---:|---|---:|---|---:|---|---:|---:|---:|"]
    ci = lambda bounds: "—" if bounds is None else f"{bounds[0]:.1%}–{bounds[1]:.1%}"
    for opp, row in sorted(results.items()):
        old = (baseline or {}).get(opp, {}).get("attack_rate")
        delta = "—" if old is None else f"{row['attack_rate']-old:+.1%}"
        recon_rate = "—" if row['recon_rate'] is None else f"{row['recon_rate']:.1%}"
        lines.append(f"| {opp} | {row['wins']}/{row['n']} ({row['attack_rate']:.1%}) | {ci(row['attack_wilson95'])} | {row['plants']}/{row['n']} ({row['plant_rate']:.1%}) | {ci(row['plant_wilson95'])} | {row['postplant_wins']}/{row['plants']} | {ci(row['postplant_wilson95'])} | {row['defuses']} | {row['recon_used']}/{row['recon_available']} ({recon_rate}) | {delta} |")
    lines += ["", "Overtime excluded. Distances use Euclidean cells and only rounds still active at the requested tick.", "", "| Opponent | Spike +10 (n) | Spike +20 (n) | Preplant wipes/n | Preplant timeouts/n | 4+ site attacks/n | 2-player site / 2-or-3-player site |", "|---|---:|---:|---:|---:|---:|---:|"]
    for opp, r in sorted(results.items()):
        distance = lambda d: "—" if r[f"distance_{d}"] is None else f"{r[f'distance_{d}']:.2f} ({r[f'distance_{d}_n']})"
        lines.append(f"| {opp} | {distance(10)} | {distance(20)} | {r['preplant_wipes']}/{r['n']} | {r['preplant_timeouts']}/{r['n']} | {r['site_four_plus']}/{r['n']} | {r['site_two']}/{r['site_two_or_three']} |")
    return "\n".join(lines)+"\n"


def play_job(job):
    code, index, _, output, _, _ = job
    log_path = Path(output)/f"runtime_{code}_{index:03d}.log"
    with log_path.open("w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        return _play_job(job)


def _play_job(job):
    code, index, controller, output, base_seed, stage = job
    os.environ["GC_V2_STAGE"] = stage
    if os.environ.get("GC_OPPONENT_SNAPSHOT"):
        sys.path.insert(0,str(Path(os.environ["GC_OPPONENT_SNAPSHOT"]).resolve()))
    if os.environ.get("GC_RUNTIME_SNAPSHOT"):
        sys.path.insert(0,str(Path(os.environ["GC_RUNTIME_SNAPSHOT"]).resolve()))
    import torch
    torch.set_num_threads(1)
    from run_competition_manager import run_series_core, build_player_leaderboards
    name, key = OPPONENTS[code]
    first, second = (TEAM, name) if index % 2 == 0 else (name, TEAM)
    controllers = {TEAM: controller, name: key}
    started = time.monotonic()
    result = run_series_core(first, second, 2, "fixed", base_seed,
        index*100, False, lambda event: None, team_controllers=controllers)
    data = dict(mode="series", seed_mode="fixed", base_seed=base_seed,
        team_controllers=controllers, **asdict(result),
        player_leaderboards=build_player_leaderboards([result]),
        evaluation={"opponent": code, "series_index": index, "stage": stage,
            "elapsed_seconds": time.monotonic()-started})
    path = Path(output)/f"series_{code}_{index:03d}.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temp.replace(path)
    return str(path), round(time.monotonic()-started, 1)


def evaluation_status(schedule, results, output):
    missing = [f"{code}/{i}" for code in schedule["opponents"] for i in range(schedule["series_count"])
               if not (output/f"series_{code}_{i:03d}.json").exists()]
    minimum = schedule.get("minimum_attack_rounds", 100)
    below_100 = [OPPONENTS[code][0] for code in schedule["opponents"]
                 if results.get(OPPONENTS[code][0], {}).get("n", 0) < 100]
    insufficient = [OPPONENTS[code][0] for code in schedule["opponents"]
                    if results.get(OPPONENTS[code][0], {}).get("n", 0) < minimum]
    return dict(scheduled_series=len(schedule["opponents"])*schedule["series_count"],
                missing_series=missing, opponents_below_100_attack_rounds=below_100,
                minimum_attack_rounds=minimum, opponents_below_minimum_attack_rounds=insufficient,
                milestone_sample_complete=not missing and not below_100,
                complete=not missing and not insufficient)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controller", choices=("ghost_champions_v1", "ghost_champions_v2", "v1", "v2"), default="ghost_champions_v1")
    p.add_argument("--opponents", nargs="+", choices=tuple(OPPONENTS), default=list(OPPONENTS))
    p.add_argument("--series-count", type=int, default=10)
    p.add_argument("--minimum-attack-rounds", type=int, default=100,
                   help="Sample requirement; use 1 for periodic screening, 100 for formal evaluation")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-seed", type=int, default=20261007)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--stage", choices=("baseline", "hold", "recon", "sites", "profiles", "entry"), default="entry")
    p.add_argument("--baseline", type=Path)
    p.add_argument("--config", type=Path, help="Optional GC v2 config; recorded in the evaluation manifest")
    p.add_argument("--runtime-snapshot",type=Path,help="Immutable local copy of character/combo/map definitions")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    args.controller = {"v1":"ghost_champions_v1", "v2":"ghost_champions_v2"}.get(args.controller,args.controller)
    if args.config:
        os.environ["GC_V2_CONFIG"] = str(args.config.resolve())
    snapshot=args.runtime_snapshot or os.environ.get("GC_RUNTIME_SNAPSHOT")
    runtime_root=runtime_snapshot(snapshot) if snapshot else ROOT
    if snapshot:
        os.environ["GC_RUNTIME_SNAPSHOT"]=str(runtime_root)
    if os.environ.get("GC_OPPONENT_SNAPSHOT"):
        sys.path.insert(0,str(Path(os.environ["GC_OPPONENT_SNAPSHOT"]).resolve()))
    if args.minimum_attack_rounds < 1:
        p.error("minimum attack rounds must be positive")
    if args.series_count < 1 or args.workers < 1:
        p.error("series-count and workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.baseline and (args.baseline/"manifest.json").exists():
        baseline_manifest = json.loads((args.baseline/"manifest.json").read_text(encoding="utf-8"))
        if baseline_manifest["base_seed"] != args.base_seed:
            p.error("baseline seed schedule differs; evaluate a baseline with the same base seed")
    manifest = dict(controller=args.controller, opponents=args.opponents, series_count=args.series_count,
        base_seed=args.base_seed, stage=args.stage, python=sys.version, source_hashes={})
    if args.minimum_attack_rounds != 100:
        manifest["minimum_attack_rounds"] = args.minimum_attack_rounds
    manifest["runtime_data_hashes"]={name:hashlib.sha256((runtime_root/name).read_bytes()).hexdigest()
                                      for name in RUNTIME_FILES}
    if snapshot:
        manifest["runtime_snapshot"]=str(runtime_root)
    if os.environ.get("GC_OPPONENT_SNAPSHOT"):
        opponent_root=Path(os.environ["GC_OPPONENT_SNAPSHOT"]).resolve()
        manifest["opponent_snapshot"]=str(opponent_root)
        manifest["opponent_snapshot_hashes"]={str(p.relative_to(opponent_root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(opponent_root.rglob("*")) if p.is_file() and p.suffix in (".py",".pt",".pth")}
    if args.baseline and (args.baseline/"manifest.json").exists():
        old_runtime=baseline_manifest.get("runtime_data_hashes")
        if old_runtime is not None and old_runtime!=manifest["runtime_data_hashes"] and not args.summarize_only:
            p.error("baseline character/combo/map data differs; run a fresh matching baseline")
    if args.controller == "ghost_champions_v2" and os.environ.get("GC_V2_CONFIG"):
        active_config = Path(os.environ["GC_V2_CONFIG"]).resolve()
        manifest["active_config"] = dict(path=str(active_config),
            sha256=hashlib.sha256(active_config.read_bytes()).hexdigest())
    source_paths = [ROOT/"run_game.py", ROOT/"battle_logic.py", ROOT/"ghost_champions_v1.py",
                    ROOT/"ghost_champions_v1_macro.py"]
    if args.controller == "ghost_champions_v2":
        source_paths += [*sorted((ROOT/"ghost_champions_v2").glob("*.py")),
                         *sorted((ROOT/"ghost_champions_v2").glob("*.json")),
                         *sorted((ROOT/"ghost_champions_v2/rl").glob("*.py")),
                         *sorted((ROOT/"ghost_champions_v2/rl").glob("*.json"))]
        if os.environ.get("GC_V2_RL_CHECKPOINT"):
            checkpoint=Path(os.environ["GC_V2_RL_CHECKPOINT"]).resolve()
            manifest["residual_checkpoint"]=dict(path=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    for path in source_paths:
        manifest["source_hashes"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = args.output/"manifest.json"
    if manifest_path.exists() and not args.summarize_only:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            p.error("existing evaluation manifest differs; choose another output directory")
    if not args.summarize_only:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        jobs = [(code, i, args.controller, str(args.output.resolve()), args.base_seed, args.stage)
            for code in args.opponents for i in range(args.series_count)
            if not (args.output/f"series_{code}_{i:03d}.json").exists()]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(play_job, job) for job in jobs]
            for future in as_completed(futures):
                path, seconds = future.result()
                print(f"Completed {Path(path).name} ({seconds}s)", flush=True)
        if any(hashlib.sha256((runtime_root/name).read_bytes()).hexdigest()!=digest
               for name,digest in manifest["runtime_data_hashes"].items()):
            p.error("runtime character/combo/map data changed during evaluation; saved results cannot be combined")
        if os.environ.get("GC_OPPONENT_SNAPSHOT") and any(not (opponent_root/name).is_file()
                or hashlib.sha256((opponent_root/name).read_bytes()).hexdigest()!=digest
                for name,digest in manifest["opponent_snapshot_hashes"].items()):
            p.error("opponent snapshot changed during evaluation")
    paths = sorted(args.output.glob("series_*.json"))
    results = summarize(paths)
    baseline = matched_baseline(paths,args.baseline) if args.baseline else None
    if baseline is not None:
        (args.output/"baseline_summary.json").write_text(json.dumps(baseline,indent=2),encoding="utf-8")
    (args.output/"summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report = markdown(results, baseline)
    schedule = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else manifest
    status = evaluation_status(schedule, results, args.output)
    missing, below_100 = status["missing_series"], status["opponents_below_100_attack_rounds"]
    (args.output/"status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    report = (f"Evaluation schedule complete: {status['complete']}. "
              f"Formal sample complete: {status['milestone_sample_complete']}. "
              f"Missing series: {len(missing)}; opponents below 100 attacker rounds: {len(below_100)}.\n\n"
              + ("Baseline uses matching opponent/series indices and the same seed schedule.\n\n" if baseline is not None else "")
              + report)
    (args.output/"comparison.md").write_text(report, encoding="utf-8")
    from tools.gc_attack_report import analyze, report as supplied_report
    with (args.output/"supplied_report.txt").open("w", encoding="utf-8") as diagnostic, \
            contextlib.redirect_stdout(diagnostic):
        per, skipped = analyze(sorted(args.output.glob("series_*.json")), TEAM)
        supplied_report(per, TEAM)
        print(f"Overtime rounds excluded: {skipped}")
    print(report)
    if not status["complete"]:
        print("Scheduled series are missing or samples are below the required attacker rounds: evaluation is incomplete.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
