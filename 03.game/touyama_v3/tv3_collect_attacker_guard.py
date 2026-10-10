"""Collect real postplant guard start cases per opponent/site after plant training."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

# 通常はplant学習完了後、冒頭の定数を確認してオプションなしで実行。
CASES_PER_SITE = 50  # 各相手の各サイト。左右なら合計100件。
TARGET_SITES = ("L", "R")
SITE_SAMPLING = "targeted"  # 収集時だけ不足側の経路を指定。通常推論・評価は通常選択。
TARGET_OPPONENTS = ("gc_v1", "concon_v1", "omoko_v1", "fnatic_v3", "frc_v1", "toru_ai_v4")
PLANT_BEST_DIRECTORY = HERE / "data" / "best"
ANALYSIS_BEST_DIRECTORY = HERE / "data" / "best"
OUTPUT_DIRECTORY = HERE / "data" / "attacker_guard_cases"
LOG_DIRECTORY = HERE / "logs" / "attacker_guard_collection"
RESUME_COLLECTION = False
RANDOM_SEED = 42
MAX_COLLECTION_BLOCKS = 200  # 不得意な側の失敗も含む累計上限。左右の目標到達で早期終了。
MAX_ROUND_STEPS = 400
TORCH_THREADS = 1

import argparse
from collections import Counter
import hashlib
import json
import logging
import torch

from concon_v1.co1_retake_cases import save_case, case_metadata, validate_case_game, _atomic_write
from touyama_v3.tv3_collect_defender_retake import read_rows, append_row, reset_collection
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_guard_runtime import CASE_FORMAT, load_sources, play_block
from touyama_v3.tv3_learn_attacker_guard import guard_schema
from touyama_v3.tv3_collect_site_sampling import SAMPLING_VERSION, pending_site, site_sampling


def valid_guard_case(game):
    validate_case_game(game)
    if not any(c.team == "A" and c.is_alive for c in game.chars):
        raise ValueError("Guard collection requires a surviving attacker")


def collection_config(scenario, sources, seed, sites=TARGET_SITES, sampling=SITE_SAMPLING):
    return {"format": CASE_FORMAT, "schema": guard_schema(scenario), "seed": seed,
        "opponents": list(sources), "source_hashes": {k: v["hashes"] for k, v in sources.items()},
        "source_paths": {k: v["paths"] for k, v in sources.items()},
        "presets": {k: v["train_presets"] for k, v in sources.items()},
        "capture_boundary": "end_of_first_plant_tick_before_guard_actions",
        "sites": list(sites), "site_selection": sampling, "sampling_version": SAMPLING_VERSION}


def validate_output(directory, protected=()):
    directory = Path(directory).resolve()
    protected = [Path(p).resolve() for p in protected]
    protected += [HERE / "data" / name for name in ("attacker_plant", "attacker_analysis", "retake_cases", "best")]
    if directory in (HERE, HERE.parent) or any(directory == p or p in directory.parents for p in protected):
        raise ValueError("Guard collection output must be separate from plant/analysis/best/retake data")
    return directory


def collect(args):
    torch.set_num_threads(args.torch_threads)
    scenario = Scenario()
    sources = load_sources(scenario, args.opponents, args.plant_dir, args.analysis_dir)
    if any(not source["fixed_roster_training"] for source in sources.values()):
        raise ValueError("Touyama guard collection requires the matching Touyama fixed-five plant best.")
    directory = validate_output(args.output_dir, (args.plant_dir, args.analysis_dir))
    config = collection_config(scenario, sources, args.seed, args.sites, args.site_sampling)
    config_path = directory / "collection.json"
    if args.resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Collection plant/analysis/config changed; start fresh after plant training")
    else:
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")).get("format") != CASE_FORMAT:
            raise ValueError("This directory belongs to another collector")
        reset_collection(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(config_path, lambda p: p.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"))
    summary_path = directory / "collection_summary.json"
    _atomic_write(summary_path, lambda p: p.write_text(json.dumps({"complete": False}), encoding="utf-8"))
    args.log_dir.resolve().mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tv3.guard_collection")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.StreamHandler(), logging.FileHandler(args.log_dir.resolve() / "collection.log", mode="w", encoding="utf-8")]
    for h in handlers:
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(h)
    counts, blocks = Counter(), Counter()
    index, progress = directory / "cases.jsonl", directory / "blocks.jsonl"
    rows = read_rows(index)
    for row in rows:
        if Path(row["file"]).name != row["file"] or not (directory / row["file"]).is_file():
            raise ValueError("Invalid or missing guard case file")
        counts[row["opponent"], row["site"]] += 1
        blocks[row["opponent"]] = max(blocks[row["opponent"]], row["block"])
    for row in read_rows(progress):
        blocks[row["opponent"]] = max(blocks[row["opponent"]], row["block"])
    try:
        logger.info("plant学習完了後の収集。各AI・各サイト%d件、収集サイト選択=%s。", args.cases, args.site_sampling)
        for opponent in args.opponents:
            source = sources[opponent]
            logger.info("[%s] plant採用set=%d analysis採用set=%d", opponent, source["plant_set"], source["analysis_set"])
            while pending_site(counts, opponent, args.sites, args.cases) is not None and blocks[opponent] < args.max_blocks:
                requested_site = pending_site(counts, opponent, args.sites, args.cases) if args.site_sampling == "targeted" else None
                blocks[opponent] += 1
                block = blocks[opponent]
                preset = source["train_presets"][(block - 1) % len(source["train_presets"])]
                seed = args.seed + args.opponents.index(opponent) * 1_000_000 + block * 100
                def capture(game, controller, enemy):
                    side = scenario.site_of(game.planted_pos)
                    if side not in args.sites or counts[enemy, side] >= args.cases:
                        return
                    if requested_site is not None and side != requested_site:
                        raise ValueError(f"Collection site selector failed: requested={requested_site}, actual={side}")
                    try:
                        valid_guard_case(game)
                    except ValueError:
                        return
                    number = counts[enemy, side] + 1
                    filename = f"{enemy}_{side}_{number:06d}.case.gz"
                    if (directory / filename).exists():
                        raise FileExistsError(filename)
                    metadata = {**case_metadata(game, enemy), "format": CASE_FORMAT,
                        "source_hashes": source["hashes"], "preset": preset,
                        "plant_set": source["plant_set"], "analysis_set": source["analysis_set"],
                        "initial_charges": controller.guard.initial_charges,
                        "block": block, "seed": seed, "round": game.current_round,
                        "requested_site": requested_site, "site_sampling": args.site_sampling}
                    for actor, char in zip(metadata["actors"], game.chars):
                        kind = char.ability_name.lower()
                        actor["charges"][kind] = int(getattr(char, kind + "_charges", 0))
                    save_case(directory / filename, game, metadata)
                    row = {**metadata, "file": filename,
                           "sha256": hashlib.sha256((directory / filename).read_bytes()).hexdigest()}
                    append_row(index, row)
                    rows.append(row)
                    counts[enemy, side] += 1
                with site_sampling(requested_site):
                    rounds, _ = play_block(opponent, scenario, source, None, seed, capture=capture, max_steps=args.max_round_steps,
                                           preset_name=preset)
                append_row(progress, {"opponent": opponent, "block": block, "seed": seed, "requested_site": requested_site})
                logger.info("[%s] block=%d 編成=%s 指定=%s 設置=%d/12 有効ケース=%s 各目標=%d", opponent, block, preset,
                            requested_site, sum(r["planted"] for r in rounds), {s: counts[opponent, s] for s in args.sites}, args.cases)
        complete = all(counts[o, s] >= args.cases for o in args.opponents for s in args.sites)
        summary = {"complete": complete, "target_per_ai": args.cases * len(args.sites),
            "target_per_site": args.cases, "sites": args.sites,
            "counts": {o: sum(counts[o, s] for s in args.sites) for o in args.opponents},
            "by_site": {o: {s: sum(r["opponent"] == o and r["site"] == s for r in rows) for s in ("L", "R")} for o in args.opponents}}
        _atomic_write(summary_path, lambda p: p.write_text(json.dumps(summary, indent=2), encoding="utf-8"))
        logger.info("収集%s: %s", "完了" if complete else "不足あり（同条件でRESUME_COLLECTION=True）", json.dumps(summary))
        return complete
    finally:
        for h in handlers:
            h.close()
            logger.removeHandler(h)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=int, default=CASES_PER_SITE, help="cases per AI/site")
    p.add_argument("--sites", nargs="+", choices=("L", "R"), default=list(TARGET_SITES))
    p.add_argument("--site-sampling", choices=("targeted", "natural"), default=SITE_SAMPLING)
    p.add_argument("--opponents", nargs="+", choices=TARGET_OPPONENTS, default=list(TARGET_OPPONENTS))
    p.add_argument("--plant-dir", type=Path, default=PLANT_BEST_DIRECTORY)
    p.add_argument("--analysis-dir", type=Path, default=ANALYSIS_BEST_DIRECTORY)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    p.add_argument("--log-dir", type=Path, default=LOG_DIRECTORY)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=RESUME_COLLECTION)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--max-blocks", type=int, default=MAX_COLLECTION_BLOCKS)
    p.add_argument("--max-round-steps", type=int, default=MAX_ROUND_STEPS)
    p.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.cases, args.max_blocks, args.max_round_steps, args.torch_threads) < 1 or len(args.opponents) != len(set(args.opponents)) or len(args.sites) != len(set(args.sites)):
        raise ValueError("Positive counts and distinct opponents are required")
    return 0 if collect(args) else 2


if __name__ == "__main__":
    raise SystemExit(main())
