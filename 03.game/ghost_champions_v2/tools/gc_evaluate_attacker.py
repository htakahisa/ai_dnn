"""Run from ghost_champions_v2/: py tools/gc_evaluate_attacker.py."""
from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parents[1]
ROOT = AI_ROOT.parent
sys.path.insert(0, str(ROOT))

DEFAULT_OPPONENTS = ("OMG", "GG")
DEFAULT_SEEDS = (543720649, 3443860541)
DEFAULT_MODES = ("rule", "learned")
DEFAULT_ROUNDS = 12
DEFAULT_CHECKPOINT = AI_ROOT / "data/attacker_rl_unitstatus_20261009_recovered/latest.pt"
DEFAULT_COMBAT_GUARD = False
DEFAULT_TRACE = False
DEFAULT_COMPETITION = False

import argparse
import contextlib
from datetime import datetime
from functools import partial
import hashlib
import json
import os
import random
import numpy as np
import torch

from ghost_champions_v2.rl.rollout import OPPONENTS


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                              default=lambda value: value.item()), encoding="utf-8")


def play(opponent, seed, mode, rounds, checkpoint, directory, combat_guard=False, trace=False):
    from run_game import VisualFPSBattle, _build_team_ai
    from map_data import NEW_MAZE_STR
    from party_presets import get_preset
    from team_ai import DualRoleTeamAI
    from controllers import DefaultDefenderController
    from simulation_runtime import cpu_inference
    from ghost_champions_v2.controller import GhostChampionsV2AttackerController
    from ghost_champions_v2.rl.controller import LearnedAttackerController
    from ghost_champions_v2.geometry import los
    from gc_v1.gc_facing import facing_towards, FACING_VECTORS
    from toruAI_v4.tv4_train_defender_analysis import relocate_debug_logs

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    attacker = (GhostChampionsV2AttackerController() if mode == "rule" else
                LearnedAttackerController(checkpoint=checkpoint))
    original_decide = attacker.decide_move
    decisions, counters = [], {}

    def decide(char, state):
        result = original_decide(char, state)
        if state.get("defender_setup_active") or not char.is_alive:
            return result
        known = attacker.tactics.enemies(char, state)
        targets = [e for e in known if los(state["grid"], tuple(char.pos), tuple(e.pos),
                   state.get("smoke_cells", ()), attacker.tactics.occupied(char, state))]
        payload = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        facing = payload.get("facing", char.facing) if isinstance(payload, dict) else char.facing
        fx, fy = FACING_VECTORS.get(facing, (0., 0.))
        backwards = bool(targets) and all(fx*(e.pos[1]-char.pos[1])+fy*(e.pos[0]-char.pos[0]) < 0 for e in targets)
        counters["backwards_contact_ticks"] += backwards
        counters["contact_ticks"] += bool(targets)
        if combat_guard and targets and not getattr(char, "facing_forced_this_tick", False):
            # Experimental correction only; ability targets and objectives keep
            # their command, and audible defuse responses keep their spike aim.
            tapping = state.get("is_planted") and any(v[0] > 0 for v in (state.get("defender_defuse_info") or {}).values())
            if not tapping and (not state.get("is_planted") or backwards):
                enemy = min(targets, key=lambda e: (max(abs(e.pos[0]-char.pos[0]), abs(e.pos[1]-char.pos[1])), e.hp, str(e.name)))
                aim = facing_towards(char.pos, enemy.pos)
                if isinstance(payload, dict):
                    result = (result[0], dict(payload, facing=aim), *result[2:])
                elif payload in (None, "MOVE", "HOLD"):
                    result = (result[0] if isinstance(result, tuple) else result, {"facing": aim},
                              *(result[2:] if isinstance(result, tuple) else ()))
        if trace:
            decisions.append(dict(tick=state["battle_tick"], name=str(char.name), pos=list(char.pos),
                hp=float(char.hp), planted=bool(state.get("is_planted")), action=result,
                facing=facing, backwards=backwards, known_enemies={str(e.name):list(e.pos) for e in known},
                residual_action=getattr(attacker, "last_policy_action", None)))
        return result

    attacker.decide_move = decide
    gc, enemy = get_preset("Ghost Champions"), get_preset(OPPONENTS[opponent][0])
    with (directory / "engine.log").open("a", encoding="utf-8") as output, contextlib.redirect_stdout(output), cpu_inference():
        game = VisualFPSBattle(NEW_MAZE_STR,
            DualRoleTeamAI("GC attack diagnosis", lambda: attacker, DefaultDefenderController),
            _build_team_ai(OPPONENTS[opponent][1]), headless=True,
            attacker_roster=list(gc.players), defender_roster=list(enemy.players),
            spike_holder_name=gc.spike_holder, defender_spike_holder_name=enemy.spike_holder,
            attacker_igl_name=gc.igl, defender_igl_name=enemy.igl,
            attacker_team_name=gc.name, defender_team_name=enemy.name, disable_side_swap=True)
        game.stop_after_round = True
        game._record_replay_frame = lambda: None
        relocate_debug_logs(game.attacker_controller, directory)
        relocate_debug_logs(game.defender_controller, directory)
        rows = []
        for number in range(1, rounds + 1):
            counters.clear()
            counters.update(backwards_contact_ticks=0, contact_ticks=0)
            decisions.clear()
            plant_tick = None
            for _ in range(600):
                game.step_tick()
                if game.is_planted and plant_tick is None:
                    plant_tick = game.battle_tick
                if game.round_over or game.match_over:
                    break
            if not game.round_over:
                raise RuntimeError(f"{opponent} round {number} exceeded tick limit")
            record = game.analytics_tracker.round_records[-1]
            rows.append(dict(round=number, won=record["winner"] == "attacker", reason=record["reason"],
                planted=record["planted"], plant_tick=plant_tick,
                alive_attackers=sum(c.team=="A" and c.is_alive for c in game.chars),
                kills=5-sum(c.team=="D" and c.is_alive for c in game.chars),
                tactics=attacker.attacker_snapshot(), **counters))
            if trace:
                save(directory / f"{mode}_{opponent}_{seed}_round_{number:02d}.json", decisions)
            if game.match_over:
                break
            if number < rounds:
                game.current_round += 1
                game.init_round()
    return dict(opponent=opponent, seed=seed, mode=mode, combat_guard=combat_guard, rounds=rows,
                wins=sum(r["won"] for r in rows), plants=sum(r["planted"] for r in rows),
                postplant_wins=sum(r["won"] and r["planted"] for r in rows),
                defuses=sum(r["reason"]=="defused" for r in rows),
                backwards_contact_ticks=sum(r["backwards_contact_ticks"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=DEFAULT_OPPONENTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--modes", nargs="+", choices=("rule", "learned"), default=DEFAULT_MODES)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--combat-guard", action="store_true", default=DEFAULT_COMBAT_GUARD)
    parser.add_argument("--trace", action="store_true", default=DEFAULT_TRACE)
    parser.add_argument("--competition", action="store_true", default=DEFAULT_COMPETITION,
                        help="Use the actual competition play_map path with side swaps until match end")
    parser.add_argument("--run", default="attacker_diagnosis_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    args = parser.parse_args()
    if len(args.seeds) != len(args.opponents) or not 1 <= args.rounds <= 64:
        parser.error("One seed per opponent and 1-64 rounds are required")
    if Path(args.run).name != args.run or args.run in (".", ".."):
        parser.error("Run must be a directory name")
    if "learned" in args.modes and not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    directory = AI_ROOT / "logs" / args.run
    directory.mkdir(parents=True, exist_ok=False)
    sources = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
               for f in [*AI_ROOT.glob("*.py"), *AI_ROOT.glob("*.json"), *AI_ROOT.joinpath("rl").glob("*.py"),
                         *AI_ROOT.joinpath("defender_macro_v1").glob("*.py"), Path(__file__).resolve(),
                         ROOT/"run_game.py", ROOT/"run_competition_manager.py", ROOT/"party_presets.py",
                         ROOT/"battle_logic.py", ROOT/"game_core.py", ROOT/"character_stats.py", ROOT/"player_combos.py"]}
    metadata = dict(source_hashes=sources, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if "learned" in args.modes else None,
        python_hash_seed=os.environ.get("PYTHONHASHSEED"), combat_guard=args.combat_guard)
    save(directory / "manifest.json", metadata)
    old = Path.cwd()
    os.chdir(ROOT)
    torch.set_num_threads(1)
    try:
        results = []
        for mode in args.modes:
            for opponent, seed in zip(args.opponents, args.seeds):
                if args.competition:
                    if args.combat_guard:
                        parser.error("Experimental combat guard is only available in round blocks")
                    block = play_competition(opponent, seed, mode, args.checkpoint.resolve(), directory)
                else:
                    block = play(opponent, seed, mode, args.rounds, args.checkpoint.resolve(), directory, args.combat_guard, args.trace)
                results.append(block)
                save(directory / f"{mode}_{opponent}_{seed}.json", block)
                print(json.dumps({k:v for k,v in block.items() if k != "rounds"}), flush=True)
        save(directory / "summary.json", dict(metadata, results=results))
    finally:
        os.chdir(old)


def play_competition(opponent, seed, mode, checkpoint, directory):
    from dataclasses import asdict
    from unittest.mock import patch
    from run_competition_manager import play_map
    from party_presets import get_preset
    with patch.dict(os.environ, {"GC_V2_ATTACKER_MODE":mode,"GC_V2_RL_CHECKPOINT":str(checkpoint)}), \
         (directory/"engine.log").open("a",encoding="utf-8") as output, contextlib.redirect_stdout(output):
        result=play_map(get_preset("Ghost Champions"),get_preset(OPPONENTS[opponent][0]),
            1,seed,False,"ghost_champions_v2",OPPONENTS[opponent][1])
    save(directory/f"competition_{mode}_{opponent}_{seed}_map.json",asdict(result))
    attack=[]
    for row in result.round_records:
        sides={p.get("side") for p in row.get("players",{}).values() if p.get("team")=="Ghost Champions"}
        if "attacker" in sides:
            attack.append(row)
    return dict(opponent=opponent,seed=seed,mode=mode,competition=True,score=[result.score1,result.score2],
        attack_rounds=len(attack),wins=sum(r["winner"]=="attacker" for r in attack),
        plants=sum(r["planted"] for r in attack),defuses=sum(r["reason"]=="defused" for r in attack),
        rounds=attack)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
