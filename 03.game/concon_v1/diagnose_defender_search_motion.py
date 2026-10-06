"""Trace frozen defender models without training or changing checkpoints."""

import argparse
import contextlib
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

from concon_v1.co1_defender_scenario import get_scenario
from concon_v1.co1_defender_search_common import DefenderSearchBattleDQN, load_search_weights
from concon_v1.co1_defender_search_training import DefenderSearchEnv, OPPONENTS


def trace(model_path, opponent, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scenario = get_scenario()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = DefenderSearchBattleDQN(scenario)
    load_search_weights(model, checkpoint, scenario)
    model.eval()
    env = DefenderSearchEnv(model, seed=seed, opponents=[opponent])
    with contextlib.redirect_stdout(io.StringIO()):
        env.reset()
    rows, previous_positions = [], {}
    original = env.controller.choose_action

    def choose(char, observation, mask, context):
        action = original(char, observation, mask, context)
        memories = env.controller.sightings
        rows.append(dict(tick=context["tick"], name=char.name,
            before=list(map(int, char.pos)), action=int(action), active=bool(context["active"]),
            setup=bool(context["setup"]), fireable=bool(context["fireable"]),
            killed=bool(context["killed"]), post_kill_hold=bool(context["post_kill_hold"]),
            disclosed=[enemy.name for enemy in context["disclosed"]],
            memory={name: dict(age=context["tick"] - item["tick"], alive=item["alive"])
                    for name, item in memories.items()}, goal=list(map(int, context["goal"]))))
        return action

    env.controller.choose_action = choose
    while not env.done:
        start = len(rows)
        env.step(epsilon=0.)
        actors = {char.name: char for char in env.defenders}
        for row in rows[start:]:
            char = actors[row["name"]]
            row["after"] = list(map(int, char.pos))
            support = env.decisions[env.indices[char.name]][1]["support_goal"]
            row["support_goal"] = list(map(int, support)) if support is not None else None
            history = previous_positions.setdefault(char.name, [])
            row["reversal"] = (len(history) >= 2 and tuple(row["after"]) == history[-2]
                               and tuple(row["after"]) != history[-1])
            history.append(tuple(row["after"]))
    live = [row for row in rows if not row["setup"]]
    summary = dict(model=Path(model_path).name, episode=checkpoint.get("episode"),
        opponent=opponent, seed=seed, result=env.result(),
        reversals=sum(row["reversal"] for row in live),
        memory_only_moves=sum(row["active"] and not row["disclosed"] and row["before"] != row["after"] for row in live),
        dead_memory_decisions=sum(row["active"] and bool(row["memory"])
                                  and all(not item["alive"] for item in row["memory"].values()) for row in live))
    return dict(summary=summary, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", type=Path)
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=list(OPPONENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--output", type=Path, help="optional JSON file containing every decision")
    args = parser.parse_args()
    torch.set_num_threads(1)
    models = args.models or [get_scenario().runtime_model_path]
    traces = []
    for model in models:
        for opponent in args.opponents:
            for seed in args.seeds:
                result = trace(model, opponent, seed)
                traces.append(result)
                print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
