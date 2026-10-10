"""Collect real plant states using frozen opponent-specific v4 search policies."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import argparse
from collections import Counter
import hashlib
import json
import torch
from concon_v1.co1_retake_cases import save_case, case_metadata, validate_case_game, _atomic_write
from touyama_v3.tv3_defender_controller import load_policy, load_analyses, DEFENDER_BEST
from touyama_v3.tv3_scenario import Scenario, OPPONENTS
from touyama_v3.tv3_train_defender_search import rollout, eligible_presets, TRAINING_PRESETS
from touyama_v3.tv3_collect_site_sampling import SAMPLING_VERSION, pending_site, site_sampling

DEFAULT_CASES_PER_SITE = 50
DEFAULT_OUTPUT = HERE / "data" / "retake_cases"
# 通常はここを編集して、python tv3_collect_defender_retake.py をオプションなしで実行する。
DEFAULT_OPPONENTS = tuple(OPPONENTS)
DEFAULT_SITES = ("L", "R")
DEFAULT_TRAINING_PRESETS = TRAINING_PRESETS
DEFAULT_BEST_DIRECTORY = DEFENDER_BEST
DEFAULT_SEED = 42
DEFAULT_MAX_BLOCKS = 200  # 不得意な側の設置失敗を含む。左右の目標到達で早期終了。
DEFAULT_SITE_SAMPLING = "targeted"  # 不足側を指定して実際の設置まで対戦。naturalは通常選択のみ。
DEFAULT_RESUME = False  # 既存の収集データを再開・追加する場合は True


def reset_collection(directory):
    """Replace collector-owned files in the same output directory."""
    directory = Path(directory).resolve()
    targets = [directory / name for name in ("collection.json", "cases.jsonl", "blocks.jsonl")]
    targets.extend(directory.glob("*.case.gz"))
    tensor_directory = directory / "tensors"
    if tensor_directory.is_symlink() or (tensor_directory.exists() and tensor_directory.resolve().parent != directory):
        raise ValueError("Tensor directory must stay in the collection directory")
    if tensor_directory.exists():
        targets.extend(tensor_directory.glob("*.pt"))
    for path in targets:
        if path.is_symlink() or path.resolve().parent not in (directory, tensor_directory):
            raise ValueError(f"Collection file must stay in the output directory: {path}")
    for path in targets:
        path.unlink(missing_ok=True)


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def append_row(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect(args):
    torch.set_num_threads(1)
    scenario = Scenario()
    analyses, hashes = load_analyses(scenario, args.opponents, args.best_dir)
    models, signatures = {}, {}
    for opponent in args.opponents:
        path = args.best_dir.resolve() / opponent / "search_best.pt"
        models[opponent], saved = load_policy(path, "search", scenario)
        if saved.get("opponent") != opponent or saved["analysis_hashes"].get(opponent) != hashes[opponent]:
            raise ValueError(f"Search/analysis mismatch: {path}")
        models[opponent].requires_grad_(False)
        signatures[opponent] = hashlib.sha256(path.read_bytes()).hexdigest()
        eligible_presets(args.train_presets, opponent)
    directory = args.output_dir.resolve()
    config_path = directory / "collection.json"
    config = dict(format="touyama_v3_plant_cases_v1", scenario=scenario.signature,
                  seed=args.seed, presets=args.train_presets, opponents=args.opponents, sites=args.sites,
                  search_hashes=signatures, analysis_hashes=hashes,
                  site_sampling=args.site_sampling, sampling_version=SAMPLING_VERSION,
                  capture_boundary="end_of_first_plant_tick_before_retake_decisions")
    if args.resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Collection settings/search/analysis changed; use a separate output directory")
    else:
        reset_collection(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(config_path, lambda p: p.write_text(json.dumps(config, indent=2), encoding="utf-8"))
    _atomic_write(directory / "collection_summary.json", lambda p: p.write_text(json.dumps({"complete": False}), encoding="utf-8"))
    index, progress = directory / "cases.jsonl", directory / "blocks.jsonl"
    counts, blocks = Counter(), Counter()
    for row in read_rows(index):
        if not (directory / row["file"]).is_file():
            raise FileNotFoundError(row["file"])
        counts[row["opponent"], row["site"]] += 1
        blocks[row["opponent"]] = max(blocks[row["opponent"]], row["block"])
    for row in read_rows(progress):
        blocks[row["opponent"]] = max(blocks[row["opponent"]], row["block"])
    print(f"Collection: {args.cases_per_site} cases per AI/site; output={directory}", flush=True)
    for opponent in args.opponents:
        presets = eligible_presets(args.train_presets, opponent)
        while True:
            requested_site = pending_site(counts, opponent, args.sites, args.cases_per_site)
            if requested_site is None or blocks[opponent] >= args.max_blocks:
                break
            if args.site_sampling == "natural":
                requested_site = None
            blocks[opponent] += 1
            block = blocks[opponent]
            preset = presets[(block - 1) % len(presets)]
            seed = args.seed + list(OPPONENTS).index(opponent) * 1000000 + block * 100
            def capture(game, enemy, own, round_no):
                try:
                    validate_case_game(game)
                except ValueError:
                    return
                side = scenario.site_of(game.planted_pos)
                if side not in args.sites or counts[enemy, side] >= args.cases_per_site:
                    return
                if requested_site is not None and side != requested_site:
                    raise ValueError(f"Collection site selector failed: requested={requested_site}, actual={side}")
                filename = f"{enemy}_{side}_{block:05d}_{round_no:02d}.case.gz"
                if (directory / filename).exists():
                    raise FileExistsError(filename)
                metadata = case_metadata(game, enemy)
                # Extend the shared serializer's summary to every actual ability.
                for actor, char in zip(metadata["actors"], game.chars):
                    kind = getattr(char, "ability_name", "").lower()
                    actor["charges"][kind] = int(getattr(char, kind + "_charges", 0))
                metadata.update(file=filename, preset=own, block=block, round=round_no, seed=seed,
                                search_hash=signatures[enemy], analysis_hash=hashes[enemy],
                                requested_site=requested_site, site_sampling=args.site_sampling)
                save_case(directory / filename, game, metadata)
                append_row(index, metadata)
                counts[enemy, side] += 1
                print(f"  {enemy} {side}: {counts[enemy, side]}/{args.cases_per_site} ({own})", flush=True)
            with site_sampling(requested_site, attacker=opponent):
                records, _ = rollout(opponent, preset, scenario, models[opponent], None, analyses, "search", seed,
                                     plant_callback=capture)
            append_row(progress, dict(opponent=opponent, block=block, preset=preset, seed=seed,
                                     requested_site=requested_site,
                                     rounds=len(records), plants=sum(r["planted"] for r in records)))
            print(f"  {opponent} block={block} " + " ".join(f"{s}={counts[opponent, s]}/{args.cases_per_site}" for s in args.sites), flush=True)
    missing = {o: {s: args.cases_per_site - counts[o, s] for s in args.sites
                   if counts[o, s] < args.cases_per_site} for o in args.opponents}
    missing = {o: values for o, values in missing.items() if values}
    summary = dict(complete=not missing, target_per_site=args.cases_per_site, missing=missing,
                   by_site={o: {s: counts[o, s] for s in args.sites} for o in args.opponents})
    _atomic_write(directory / "collection_summary.json", lambda p: p.write_text(json.dumps(summary, indent=2), encoding="utf-8"))
    print(f"{'Incomplete' if missing else 'Complete'}: missing={missing}", flush=True)
    return not missing


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases-per-site", type=int, default=DEFAULT_CASES_PER_SITE)
    p.add_argument("--opponents", nargs="+", choices=tuple(OPPONENTS), default=list(DEFAULT_OPPONENTS))
    p.add_argument("--sites", nargs="+", choices=("L", "R"), default=list(DEFAULT_SITES))
    p.add_argument("--train-presets", nargs="+", default=list(DEFAULT_TRAINING_PRESETS))
    p.add_argument("--best-dir", type=Path, default=DEFAULT_BEST_DIRECTORY)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS, help="total 12-round blocks per AI, including resumed blocks")
    p.add_argument("--site-sampling", choices=("targeted", "natural"), default=DEFAULT_SITE_SAMPLING)
    p.add_argument("--resume", action="store_true", default=DEFAULT_RESUME)
    args = p.parse_args(argv)
    if min(args.cases_per_site, args.max_blocks) < 1 or len(set(args.opponents)) != len(args.opponents) or len(set(args.sites)) != len(args.sites):
        raise ValueError("Positive counts and distinct opponents/sites required")
    return 0 if collect(args) else 2


if __name__ == "__main__":
    raise SystemExit(main())
