"""Collect real planted-spike start cases with frozen production search."""

import argparse
from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from game_core import ROUND_DURATION_TICKS
from map_data_defender_setup import DEFENDER_SETUP_TICKS
from concon_v1.co1_attacker_common import GORIGONS
from concon_v1.co1_attacker_scenarios import GAME_MAZE_STR
from concon_v1.co1_battle_training import OPPONENTS, _run_from_project_root
from concon_v1.co1_defender_controller import ConconDefenderController
from concon_v1.co1_learn_defender_search import ConconDefenderSearchController
from concon_v1.co1_retake_cases import CASE_VERSION, case_metadata, save_case, _atomic_write

# 収集件数: 各AI・各サイトの件数。既定の5チーム × 左右2サイトで合計500件。
# 100に変更すると、既定の対象では合計1000件。コマンド引数を指定した場合は引数を優先。
DEFAULT_CASES_PER_SITE = 50
DEFAULT_MAX_ATTEMPTS_PER_TEAM = 1000
# 実際の試合上限: setup 20 tick + 設置前ラウンド 100 tick = 120 tick。
DEFAULT_MAX_TICKS = DEFENDER_SETUP_TICKS + ROUND_DURATION_TICKS
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "defender_retake_cases"


class CollectionDefenderController(ConconDefenderController):
    def decide_move(self, char, state):
        if state.get("is_planted"):
            # Planting can finish midway through character movement. Capture
            # at the end of that tick, before any learned retake decision.
            return list(char.pos), {"facing": char.facing}
        return self.search_controller.decide_move(char, state)


@_run_from_project_root
def create_game(opponent, model, seed):
    from controllers import DefaultAttackerController
    from party_presets import get_preset
    from team_ai import DualRoleTeamAI
    with contextlib.redirect_stdout(io.StringIO()):
        from run_game import VisualFPSBattle, _build_team_ai
        ai_key, roster_name = OPPONENTS[opponent]
        attackers = get_preset(roster_name)
        search = ConconDefenderSearchController(model=model, seed=seed)
        adapter = CollectionDefenderController(search_controller=search)
        defender = DualRoleTeamAI("ConCon search case collection", DefaultAttackerController,
                                  lambda: adapter, use_iq_perception=True)
        game = VisualFPSBattle(GAME_MAZE_STR, _build_team_ai(ai_key), defender, headless=True,
            attacker_roster=list(attackers.players), defender_roster=list(GORIGONS.players),
            spike_holder_name=attackers.spike_holder, defender_spike_holder_name=GORIGONS.spike_holder,
            attacker_igl_name=attackers.igl, defender_igl_name=GORIGONS.igl,
            attacker_team_name=attackers.name, defender_team_name=GORIGONS.name, disable_side_swap=True)
    game.stop_after_round, game.analytics_tracker, game.record_replay = True, None, False
    game.replay_frames = []
    return game


@_run_from_project_root
def run_to_plant(game, max_ticks):
    """Run normal setup/search, without fabrication or search exploration."""
    ticks = 0
    with contextlib.redirect_stdout(io.StringIO()):
        while not game.is_planted and not game.round_over and not game.match_over and ticks < max_ticks:
            game.step_tick()
            ticks += 1
    usable = (game.is_planted and not game.round_over and not game.match_over
              and not game.is_defused and game.detonate_timer > 0
              and any(char.team == "D" and char.is_alive for char in game.chars))
    return ticks, usable


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_json(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect(cases_per_site=DEFAULT_CASES_PER_SITE, opponents=None, sites=("L", "R"), output_dir=DEFAULT_OUTPUT,
            search_model_path=None, seed=0, max_attempts_per_team=DEFAULT_MAX_ATTEMPTS_PER_TEAM,
            max_ticks=DEFAULT_MAX_TICKS, resume=False):
    opponents = tuple(OPPONENTS if opponents is None else opponents)
    sites = tuple(sites)
    if not opponents or len(set(opponents)) != len(opponents) or any(o not in OPPONENTS for o in opponents):
        raise ValueError("choose distinct known opponents")
    if not sites or len(set(sites)) != len(sites) or any(s not in ("L", "R") for s in sites):
        raise ValueError("choose distinct L/R sites")
    if min(cases_per_site, max_attempts_per_team, max_ticks) < 1:
        raise ValueError("case and attempt/tick limits must be positive")
    directory = Path(output_dir).resolve()
    config_path = directory / "collection.json"
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise ValueError("output directory is not empty; use --resume or another --output-dir")
    if resume and not config_path.exists():
        raise ValueError("--resume requires an existing collection.json")
    search = ConconDefenderSearchController(model_path=search_model_path)
    search.model.requires_grad_(False)
    config = dict(version=CASE_VERSION, seed=seed, opponents=list(opponents), sites=list(sites),
                  search_model_sha256=hashlib.sha256(search.model_path.read_bytes()).hexdigest(),
                  map_sha256=hashlib.sha256(GAME_MAZE_STR.encode()).hexdigest())
    if resume and json.loads(config_path.read_text(encoding="utf-8"))["provenance"] != config:
        raise ValueError("resume roster/sites/seed/search weights/map differ from this dataset")
    directory.mkdir(parents=True, exist_ok=True)
    info = dict(provenance=config, search_model=str(search.model_path),
                cases_per_site=cases_per_site, capture_boundary="end_of_first_plant_tick_before_retake_decisions")
    _atomic_write(config_path, lambda path: path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"))
    index_path, rounds_path = directory / "cases.jsonl", directory / "rounds.jsonl"
    counts, attempts = Counter(), Counter()
    for row in _read_jsonl(index_path):
        if not (directory / row["file"]).is_file():
            raise FileNotFoundError(f"indexed case is missing: {row['file']}")
        counts[row["opponent"], row["site"]] += 1
        attempts[row["opponent"]] = max(attempts[row["opponent"]], row["attempt"])
    for row in _read_jsonl(rounds_path):
        attempts[row["opponent"]] = max(attempts[row["opponent"]], row["attempt"])
    print(f"Retake case collection: target={cases_per_site * len(opponents) * len(sites)} "
          f"cases_per_team_per_site={cases_per_site} search_epsilon=0.000", flush=True)
    print(f"  output={directory}", flush=True)
    while True:
        active = [o for o in opponents if attempts[o] < max_attempts_per_team
                  and any(counts[o, s] < cases_per_site for s in sites)]
        if not active:
            break
        for opponent in active:
            attempts[opponent] += 1
            attempt = attempts[opponent]
            raw = f"{seed}:{opponent}:{attempt}".encode()
            round_seed = int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")
            random.seed(round_seed)
            np.random.seed(round_seed)
            torch.manual_seed(round_seed)
            game = create_game(opponent, search.model, round_seed)
            ticks, usable = run_to_plant(game, max_ticks)
            row = dict(opponent=opponent, attempt=attempt, round_seed=round_seed,
                       ticks=ticks, planted=bool(game.is_planted), saved=False)
            if usable:
                metadata = case_metadata(game, opponent)
                site = metadata["site"]
                row["site"] = site
                if site in sites and counts[opponent, site] < cases_per_site:
                    filename = f"{opponent}_{site}_{attempt:06d}.case.gz"
                    if (directory / filename).exists():
                        raise FileExistsError(f"unindexed case already exists: {filename}")
                    metadata.update(attempt=attempt, round_seed=round_seed, file=filename)
                    save_case(directory / filename, game, metadata)
                    _append_json(index_path, metadata)
                    counts[opponent, site] += 1
                    row["saved"] = True
            _append_json(rounds_path, row)
            if row["saved"] or attempt % 10 == 0:
                progress = " ".join(f"{s}={counts[opponent, s]}/{cases_per_site}" for s in sites)
                print(f"  {opponent}: {progress} attempted_rounds={attempt}", flush=True)
    missing = {o: {s: cases_per_site - counts[o, s] for s in sites if counts[o, s] < cases_per_site}
               for o in opponents if any(counts[o, s] < cases_per_site for s in sites)}
    print(f"Collection {'incomplete' if missing else 'complete'}: saved={sum(counts.values())} "
          f"attempted_rounds={sum(attempts.values())}", flush=True)
    if missing:
        print(f"  Remaining cases: {json.dumps(missing, ensure_ascii=False)}; "
              "increase --max-attempts-per-team and use --resume", flush=True)
    return not missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-per-site", type=int, default=DEFAULT_CASES_PER_SITE,
                        help=f"cases per opponent per requested site (default {DEFAULT_CASES_PER_SITE})")
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=list(OPPONENTS))
    parser.add_argument("--sites", nargs="+", choices=("L", "R"), default=["L", "R"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--search-model", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts-per-team", type=int, default=DEFAULT_MAX_ATTEMPTS_PER_TEAM,
                        help="total attempts per team, including resumed attempts")
    parser.add_argument("--max-ticks", type=int, default=DEFAULT_MAX_TICKS,
                        help=f"maximum setup/search ticks in one attempt (default {DEFAULT_MAX_TICKS}: setup + round duration)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    return 0 if collect(args.cases_per_site, args.opponents, args.sites, args.output_dir,
                        args.search_model, args.seed, args.max_attempts_per_team, args.max_ticks, args.resume) else 2


if __name__ == "__main__":
    raise SystemExit(main())
