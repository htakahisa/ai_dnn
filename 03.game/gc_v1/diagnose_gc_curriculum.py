"""Record actual Carry decisions and executed moves without changing weights."""
from __future__ import annotations

import argparse
import copy
import hashlib
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

import train_attacker_gc_real_curriculum as trainer
from navigation_intent_gc import (navigation_intent, can_engage, CARDINAL, own_macro,
                                  carrier_screening_status)
from positioning_gc import team_plant_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--kind", default="latest", choices=("latest", "best_by_eval"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(3026091700, 5026091700, 6026091700))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    torch.set_num_threads(1)
    sources = {p: args.models_dir / f"dqn_attacker_{p}_gc_{args.kind}.pt" for p in trainer.PHASES}
    source_hashes = {p: hashlib.sha256(s.read_bytes()).hexdigest() for p, s in sources.items()}
    checkpoints = {p: torch.load(s, map_location="cpu", weights_only=False) for p, s in sources.items()}
    if len({c.get("episode") for c in checkpoints.values()}) != 1:
        raise ValueError("diagnosis requires a synchronized three-model checkpoint set")
    policies = {"carry": trainer.runtime.AttackerCarryDuelingDQN(obs_dim=checkpoints["carry"]["obs_dim"]),
                "escort": trainer.escort_runtime.DuelingQNetwork(checkpoints["escort"]["obs_dim"], 6),
                "guard": trainer.guard_runtime.AttackerGuardDuelingDQN()}
    for p in trainer.PHASES:
        policies[p].load_state_dict(checkpoints[p].get("model_state_dict", checkpoints[p]))
    session = trainer.CurriculumSession(sources, policies, 0.99)
    # Diagnose the checkpoint's actual version even when the trainer has since
    # changed; no new observation semantics or action masks are imposed.
    for p, controller in session.controllers.items():
        controller.positioning_version = checkpoints[p].get("positioning_version", 0)
    game = session.game
    macro_path = Path(own_macro(game).model_path).resolve()
    macro_hash = hashlib.sha256(macro_path.read_bytes()).hexdigest()
    for checkpoint in checkpoints.values():
        expected = checkpoint.get("macro_source", {}).get("sha256")
        if expected is not None and expected != macro_hash:
            raise ValueError("loaded Macro weights differ from the checkpoint's training Macro")
    rows, trace = [], []
    selected = {}
    choose = session.controller._select_action
    def select(obs, mask):
        padded = torch.zeros(policies["carry"].feature[0].in_features)
        padded[:len(obs)] = torch.from_numpy(obs)
        with torch.no_grad():
            q = policies["carry"](padded[None]).squeeze(0).tolist()
        action = max((i for i, valid in enumerate(mask) if valid), key=lambda i: q[i])
        # Keep the original transition collection with any newly added inputs.
        original_action = choose(padded.numpy(), mask)
        assert action == original_action
        char, state = session.decision_context
        goal, strategy, role = navigation_intent(session.controller.game, char)
        selected[session.actor] = {"action": action, "mask": mask.tolist(), "q": q,
                                 "obs": obs.tolist(), "decision_goal": goal,
                                 "decision_strategy": strategy, "decision_role": role}
        return action
    session.controller._select_action = select
    move = game.move_character
    def traced_move(char):
        if char.team != "A" or not char.has_spike or game.is_planted:
            return move(char)
        pos = tuple(char.pos)
        neighbors = []
        for action, (dr, dc) in zip((2, 4, 6, 8), CARDINAL):
            cell = (pos[0] + dr, pos[1] + dc)
            blockers = [dict(name=c.name, team=c.team, pos=tuple(c.pos)) for c in game.chars
                        if c.name != char.name and c.is_alive and tuple(c.pos) == cell]
            neighbors.append(dict(action=action, cell=cell, blockers=blockers))
        fighting = can_engage(game, char, game.chars)
        screen = carrier_screening_status(game, char, game.chars,
                                          session.__dict__.setdefault("_diagnostic_screen_maps", {}))
        designated = screen.get("designated")
        designated_pos = (tuple(map(int, designated.pos)) if designated is not None else None)
        designated_engaged = bool(designated is not None
                                  and can_engage(game, designated, game.chars))
        ally_positions = {c.name: tuple(map(int, c.pos)) for c in game.chars
                          if c.team == char.team and c.is_alive and c.name != char.name}
        selected.pop(char.name, None)
        result = move(char)
        goal, strategy, role = navigation_intent(game, char)
        target = team_plant_target(game)
        after = tuple(char.pos)
        assignment = copy.deepcopy(own_macro(game).env.assignment.get(char.name))
        decision = selected.get(char.name, {})
        decision_goal = decision.get("decision_goal", goal)
        distance = session.distances(decision_goal) if decision_goal else None
        for neighbor in neighbors:
            cell = neighbor["cell"]
            inside = 0 <= cell[0] < game.grid.shape[0] and 0 <= cell[1] < game.grid.shape[1]
            neighbor["distance"] = int(distance[cell]) if inside and distance is not None else None
            neighbor["wall"] = not inside or game.grid[cell] == 1
        improving = [n for n in neighbors if n["distance"] is not None and 0 <= n["distance"] < int(distance[pos])]
        legal_progress = [n for n in improving if decision.get("mask", [False]*11)[n["action"]]]
        trace.append(dict(tick=game.battle_tick, name=char.name, before=pos, after=after,
                          goal=goal, target=target, strategy=strategy, role=role, assignment=assignment,
                          distance_before=int(distance[pos]) if distance is not None else None,
                          distance_after=int(distance[after]) if distance is not None else None,
                          final_distance=int(session.distances(target)[after]) if target else None,
                          neighbors=neighbors, legal_progress=legal_progress,
                          engaged=fighting, planting=char.plant_timer,
                          screen_ready=bool(screen.get("screen_ready", False)),
                          formation_target=screen.get("formation_target"),
                          designated=(designated.name if designated is not None else None),
                          designated_pos=designated_pos,
                          designated_engaged=designated_engaged,
                          designated_distance=screen.get("designated_distance", -1),
                          designated_chebyshev=(max(abs(designated_pos[0] - pos[0]),
                                                    abs(designated_pos[1] - pos[1]))
                                                if designated_pos is not None else None),
                          ahead=[c.name for c in screen.get("ahead", [])],
                          near=[c.name for c in screen.get("near", [])],
                          ally_positions=ally_positions, **decision))
        return result
    game.move_character = traced_move
    for seed in args.seeds:
        for i in range(args.episodes):
            trace = []
            result = session.play(seed + i, (100, 60, 40), training=False)
            summary = dict(seed=seed + i, result=result,
                           stays=sum(t["before"] == t["after"] for t in trace),
                           at_waypoint=sum(t["distance_after"] is not None and t["distance_after"] <= 1 for t in trace),
                           quiet_stays=sum(t["before"] == t["after"] and not t["engaged"] and not t["planting"] for t in trace),
                           reversals=sum(trace[j]["after"] == trace[j - 2]["after"]
                                         and trace[j]["after"] != trace[j - 1]["after"] for j in range(2, len(trace))),
                           goal_changes=sum(t.get("decision_goal") != trace[j-1].get("decision_goal") for j,t in enumerate(trace) if j),
                           quiet_stays_with_progress=sum(t["before"] == t["after"] and not t["engaged"] and not t["planting"] and bool(t["legal_progress"]) for t in trace),
                           blocked_moves=sum(t.get("action") in (2,4,6,8) and t["before"] == t["after"] for t in trace),
                           blocked_by_ally=int(sum(any(b["team"] == "A" for b in n["blockers"])
                                                   for t in trace for n in t["neighbors"]
                                                   if n["action"] == t.get("action") and t.get("action") in (2,4,6,8))),
                           screen_ready_ticks=int(sum(t.get("screen_ready", False) for t in trace)),
                           designated_far_ticks=int(sum((t.get("designated_chebyshev") or 0) > 6 for t in trace)),
                           no_ahead_ticks=int(sum(not t.get("ahead") for t in trace)),
                           actions=dict(Counter(str(t.get("action", "scripted")) for t in trace)))
            rows.append(dict(**summary, trace=trace))
            print(json.dumps(summary), flush=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    if any(hashlib.sha256(s.read_bytes()).hexdigest() != source_hashes[p] for p,s in sources.items()):
        raise RuntimeError("source checkpoints changed during diagnosis")
    args.json_output.write_text(json.dumps(dict(macro_source=dict(path=str(macro_path), sha256=macro_hash), grid_shape=game.grid.shape,
                                          models={p: dict(path=str(sources[p]), episode=checkpoints[p].get("episode"), sha256=source_hashes[p], positioning_version=checkpoints[p].get("positioning_version")) for p in trainer.PHASES}, episodes=rows), indent=2,
                                          default=lambda value: value.item() if isinstance(value, np.generic) else list(value)), encoding="utf-8")


if __name__ == "__main__":
    main()
