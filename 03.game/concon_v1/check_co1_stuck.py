"""Check ConCon maps for prolonged blocking, waiting, and round timeouts."""

import argparse
from collections import Counter
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

from concon_v1 import co1_train_attacker as training
from concon_v1.co1_battle_training import OPPONENTS
from concon_v1.co1_attacker_common import (
    ACTION_PLANT, ACTION_WAIT, CARDINAL_MOVES, GORIGONS, SPIKE_CARRIER_INDEX,
    SharedRouteDQN, _choose_action, bfs_distance_map,
)
from concon_v1.co1_attacker_scenarios import (
    SCENARIOS, get_scenario, validate_checkpoint_scenario,
)
from game_core import ROUND_DURATION_TICKS

ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT", "PLANT")
OPPONENT_NAMES = tuple(OPPONENTS)


def waypoint_checks(scenario):
    """Show candidate distances, including candidates excluded by the limit."""
    checks = []
    for source_marker, target_marker in zip(scenario.waypoint_order, scenario.waypoint_order[1:]):
        for source in scenario.waypoint_points[source_marker]:
            distances = bfs_distance_map(scenario.grid, source)
            candidates = [
                {"pos": list(pos), "distance": int(distances[pos]),
                 "eligible": bool(0 <= distances[pos] <= scenario.max_candidate_bfs_distance)}
                for pos in scenario.waypoint_points[target_marker]
            ]
            checks.append({"from_marker": source_marker, "from": list(source),
                           "to_marker": target_marker, "candidates": candidates})
    return checks


def snapshot(env, masks, mode, actions):
    """Describe actual route constraints; parked escorts are expected to wait."""
    if mode == "battle":
        carriers = [i for i, char in enumerate(env.attackers) if char.is_alive and char.has_spike]
        grid = env.game.grid
    else:
        carriers = [SPIKE_CARRIER_INDEX]
        grid = env.scenario.grid
    actors = []
    for index, (pos, route, mask) in enumerate(zip(env.positions, env.routes, masks)):
        pos = tuple(pos)
        distance = int(route.distance_map[pos])
        occupied = {tuple(p): GORIGONS.players[j] for j, p in enumerate(env.positions)
                    if j != index and env.alive[j]}
        neighbors = []
        blockers = []
        for action, (dr, dc) in enumerate(CARDINAL_MOVES):
            dest = (pos[0] + dr, pos[1] + dc)
            inside = 0 <= dest[0] < grid.shape[0] and 0 <= dest[1] < grid.shape[1]
            terrain = int(grid[dest]) if inside else None
            towards_goal = inside and distance > 0 and int(route.distance_map[dest]) == distance - 1
            occupant = occupied.get(dest)
            if towards_goal and occupant:
                blockers.append(occupant)
            neighbors.append({"action": ACTION_NAMES[action], "pos": list(dest),
                              "terrain": terrain, "ally": occupant, "towards_goal": bool(towards_goal)})
        allowed = [ACTION_NAMES[i] for i in np.flatnonzero(mask)]
        if not env.alive[index]:
            constraint = "dead"
        elif mode == "battle" and env.retrieve_active:
            constraint = "retrieving"
        elif route.at_plant_stage and index not in carriers:
            constraint = "escort_holding"
        elif distance < 0:
            constraint = "unreachable"
        elif distance == 0 and not mask[ACTION_PLANT]:
            constraint = "sync_wait" if route.stage == 0 else "waypoint_wait"
        elif not mask[:4].any() and not mask[ACTION_PLANT]:
            constraint = "ally_blocked" if blockers else "movement_restricted"
        else:
            applied = mode != "battle" or env.policy_action_applied[index]
            constraint = ("controller_hold" if not applied else
                          "policy_wait" if actions[index] == ACTION_WAIT else "movement_not_applied")
        marker = (env.scenario.waypoint_order[route.stage]
                  if route.stage < len(env.scenario.waypoint_order) else "plant")
        actors.append({"name": GORIGONS.players[index], "alive": bool(env.alive[index]),
                       "carrier": index in carriers, "pos": list(pos), "stage": marker,
                       "goal": list(route.goal), "distance": distance, "allowed": allowed,
                       "constraint": constraint, "blockers": blockers, "neighbors": neighbors,
                       "goal_reached": distance == 0,
                       "plant_progress": int(env.attackers[index].plant_timer) if mode == "battle"
                       else env.plant_progress if index in carriers else 0})
    return actors


class StopTracker:
    """Record prolonged unchanged route states, excluding normal escort parking."""

    IGNORED = {"dead", "retrieving", "escort_holding", "waypoint_wait"}

    def __init__(self, threshold):
        self.threshold = threshold
        self.pending = {}
        self.events = []

    def update(self, tick, actors):
        for actor in actors:
            name = actor["name"]
            key = (actor["constraint"], tuple(actor["pos"]), actor["stage"],
                   tuple(actor["goal"]), actor["plant_progress"], tuple(actor["allowed"]),
                   tuple(actor["blockers"]))
            previous = self.pending.get(name)
            if actor["constraint"] in self.IGNORED or previous is None or previous["key"] != key:
                if previous and previous["event"] is not None:
                    moved = previous["key"][1] != tuple(actor["pos"])
                    previous["event"].update(
                        resolved=moved, change_tick=tick, after=actor,
                        ended_by="moved" if moved else "stage_changed" if previous["key"][2] != actor["stage"]
                        else "constraints_changed",
                    )
                if actor["constraint"] in self.IGNORED:
                    self.pending.pop(name, None)
                    continue
                previous = {"key": key, "start_tick": tick, "event": None}
                self.pending[name] = previous
            duration = tick - previous["start_tick"]
            if duration >= self.threshold:
                if previous["event"] is None:
                    previous["event"] = {"name": name, "reason": actor["constraint"],
                                         "start_tick": previous["start_tick"], "resolved": False,
                                         "ended_by": None}
                    self.events.append(previous["event"])
                previous["event"].update(end_tick=tick, stopped_ticks=duration, actors=actors)


def load_model(scenario, explicit_path=None):
    path = explicit_path or scenario.save_dir / scenario.checkpoint_filename("latest")
    if explicit_path is None and not path.is_file():
        path = scenario.model_path
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_scenario(checkpoint, scenario)
    model = SharedRouteDQN(obs_dim=scenario.obs_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, {"path": str(path.resolve()), "episode": checkpoint.get("episode")}


def run_trial(scenario, args, seed, model, opponent=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    if args.mode == "battle":
        from concon_v1.co1_battle_training import BattleRouteEnv
        with contextlib.redirect_stdout(io.StringIO()):
            env = BattleRouteEnv(seed, [opponent], model=model, map_name=scenario)
        env.game.round_timer = args.max_ticks
        observations, masks = env._collect()
    else:
        env = training.RouteEnv(seed, map_name=scenario)
        observations, masks = env.reset()
    tracker = StopTracker(args.stuck_ticks)
    first_stage_ticks = {}
    action_counts = Counter()
    within_round_limit = False
    final = []
    previous_limit = training.MAX_TICKS
    try:
        # RouteEnv's limit is module-level; restore it even if the map is invalid.
        if args.mode == "route":
            training.MAX_TICKS = args.max_ticks
        while not env.done and env.elapsed_ticks < args.max_ticks:
            if args.mode == "battle" and args.policy == "model":
                actions = None  # Production decide_move draws the actions once.
            elif args.policy == "model":
                actions = [_choose_action(model, obs, mask, args.epsilon, rng,
                                          route=route, position=pos)
                           for obs, mask, route, pos in zip(observations, masks, env.routes, env.positions)]
            elif args.policy == "move-first":
                actions = []
                for pos, route, mask in zip(env.positions, env.routes, masks):
                    moves = np.flatnonzero(mask[:4]).tolist()
                    if mask[ACTION_PLANT]:
                        action = ACTION_PLANT
                    elif mask[ACTION_WAIT] and int(route.distance_map[tuple(pos)]) == 0:
                        action = ACTION_WAIT
                    elif moves:
                        action = min(moves, key=lambda a: (
                            int(route.distance_map[pos[0] + CARDINAL_MOVES[a][0],
                                                   pos[1] + CARDINAL_MOVES[a][1]]), a))
                    else:
                        action = ACTION_WAIT
                    actions.append(action)
            else:
                actions = [int(rng.choice(np.flatnonzero(mask).tolist())) for mask in masks]
            if args.mode == "battle" and args.policy == "model":
                transition = env.step(epsilon=args.epsilon, action_rng=rng)
                actions = env.actions
            else:
                transition = env.step(actions)
                if args.mode == "battle":
                    actions = env.actions
            _, _, _, observations, masks, _ = transition
            for index, action in enumerate(actions):
                if env.alive[index] and (args.mode == "route" or env.policy_action_applied[index]):
                    action_counts[ACTION_NAMES[action]] += 1
            final = snapshot(env, masks, args.mode, actions)
            for actor in final:
                if actor["alive"]:
                    first_stage_ticks.setdefault(actor["stage"], env.elapsed_ticks)
            if not env.success:
                tracker.update(env.elapsed_ticks, final)
            within_round_limit |= bool(env.success and env.elapsed_ticks <= ROUND_DURATION_TICKS)
    finally:
        training.MAX_TICKS = previous_limit
    if env.success:
        reason = "planted"
    elif not any(env.alive):
        reason = "attacker_eliminated"
    elif args.mode == "battle" and not any(c.is_alive for c in env.game.chars if c.team == "D"):
        reason = "defender_eliminated"
    elif args.mode == "battle" and env.game.round_timer > 0 and env.game.round_over:
        reason = "round_ended"
    else:
        reason = "time_expired"
    return {"seed": seed, "opponent": opponent, "ticks": env.elapsed_ticks,
            "planted": bool(env.success), "planted_within_round_limit": within_round_limit,
            "end_reason": reason, "first_stage_ticks": first_stage_ticks,
            "action_counts": dict(action_counts), "events": tracker.events, "final": final}


def print_event(event):
    actor = next(a for a in event["actors"] if a["name"] == event["name"])
    print(f"  STOP {event['reason']} ticks={event['start_tick']}..{event['end_tick']} "
          f"duration={event['stopped_ticks']} ended_by={event['ended_by'] or 'still_stopped'}")
    print(f"    {actor['name']} carrier={actor['carrier']} pos={actor['pos']} "
          f"stage={actor['stage']} goal={actor['goal']} distance={actor['distance']} "
          f"allowed={','.join(actor['allowed'])} blockers={actor['blockers']}")
    for blocker in actor["blockers"]:
        ally = next(a for a in event["actors"] if a["name"] == blocker)
        print(f"    blocker {ally['name']} pos={ally['pos']} stage={ally['stage']} "
              f"allowed={','.join(ally['allowed'])}")


def print_timeout(result):
    """Show unfinished route state even when no prolonged stop was detected."""
    print(f"  TIMEOUT stage_first_ticks={result['first_stage_ticks']}")
    actors = [actor for actor in result["final"] if actor["alive"]]
    for stage in dict.fromkeys(actor["stage"] for actor in actors):
        stage_actors = [actor for actor in actors if actor["stage"] == stage]
        goals = {tuple(actor["goal"]) for actor in stage_actors}
        arrived = [actor["name"] for actor in stage_actors if actor["goal_reached"]]
        print(f"    stage={stage} assigned_goals={sorted(goals)} arrived={arrived}")
    for actor in actors:
        print(f"    {actor['name']} carrier={actor['carrier']} pos={actor['pos']} "
              f"stage={actor['stage']} goal={actor['goal']} distance={actor['distance']} "
              f"allowed={','.join(actor['allowed'])} blockers={actor['blockers']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-map", "--map", dest="maps", choices=SCENARIOS, nargs="+", default=["A1"])
    parser.add_argument("--mode", choices=("route", "battle"), default="route")
    parser.add_argument("--policy", choices=("random", "move-first", "model"), default="move-first",
                        help="default: prefer goal progress and follow yielding constraints")
    parser.add_argument("--rounds", type=int, default=30, help="trials per map and opponent")
    parser.add_argument("--max-ticks", type=int, default=300, help="diagnostic limit; normal rounds have 100 ticks")
    parser.add_argument("--stuck-ticks", type=int, default=10, help="unchanged ticks before reporting a stop")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path, help="checkpoint for one map; default is latest, then best")
    parser.add_argument("--epsilon", type=float, default=0.0, help="exploration probability with --policy model")
    parser.add_argument("--opponents", nargs="+", choices=OPPONENT_NAMES, default=["gc_v1"])
    parser.add_argument("--output", type=Path, help="save all events, actor states, and neighbor cells as JSON")
    args = parser.parse_args()
    if min(args.rounds, args.max_ticks, args.stuck_ticks) < 1:
        parser.error("--rounds, --max-ticks and --stuck-ticks must be positive")
    if not 0 <= args.epsilon <= 1:
        parser.error("--epsilon must be between 0 and 1")
    if args.model and (args.policy != "model" or len(args.maps) != 1):
        parser.error("--model requires --policy model and exactly one map")
    if args.epsilon and args.policy != "model":
        parser.error("--epsilon requires --policy model")
    torch.set_num_threads(1)
    reports = []
    for map_name in dict.fromkeys(args.maps):
        scenario = get_scenario(map_name)
        try:
            model, metadata = (load_model(scenario, args.model) if args.policy == "model" else (None, None))
        except (OSError, ValueError, RuntimeError) as error:
            parser.exit(2, f"map={map_name}: {error}\nUse --policy move-first to check an edited map without a checkpoint.\n")
        checks = waypoint_checks(scenario)
        print(f"map={map_name} mode={args.mode} policy={args.policy} "
              f"rounds={args.rounds} max_ticks={args.max_ticks} stuck_ticks={args.stuck_ticks}", flush=True)
        if args.policy == "random":
            print("  INFO random samples allowed moves and WAIT; ordinary moves reduce goal distance, "
                  "with separate yielding moves. Timeouts alone do not establish blocking.")
        if metadata:
            print(f"  model={metadata['path']} episode={metadata['episode']} epsilon={args.epsilon}")
        for check in checks:
            for candidate in check["candidates"]:
                if not candidate["eligible"]:
                    print(f"  EXCLUDED {check['from_marker']}{check['from']} -> "
                          f"{check['to_marker']}{candidate['pos']} distance={candidate['distance']} "
                          f"limit={scenario.max_candidate_bfs_distance}")
        trials = []
        for opponent in args.opponents if args.mode == "battle" else [None]:
            for trial in range(args.rounds):
                try:
                    result = run_trial(scenario, args, args.seed + trial, model, opponent)
                except ValueError as error:
                    parser.exit(2, f"map={map_name} seed={args.seed + trial}: {error}\n")
                trials.append(result)
                print(f"  trial={trial + 1} seed={result['seed']} opponent={opponent or '-'} "
                      f"end={result['end_reason']} ticks={result['ticks']} "
                      f"stops={len(result['events'])}", flush=True)
                for event in result["events"]:
                    print_event(event)
                if result["end_reason"] == "time_expired":
                    print_timeout(result)
        summary = {"trials": len(trials), "plants": sum(t["planted"] for t in trials),
                   "plants_within_100_ticks": sum(t["planted_within_round_limit"] for t in trials),
                   "trials_with_stops": sum(bool(t["events"]) for t in trials),
                   "stop_reasons": dict(Counter(e["reason"] for t in trials for e in t["events"]))}
        print(f"SUMMARY {map_name}: {json.dumps(summary)}", flush=True)
        reports.append({"map": map_name, "scenario_signature": scenario.signature,
                        "mode": args.mode, "policy": args.policy, "epsilon": args.epsilon,
                        "max_ticks": args.max_ticks, "stuck_ticks": args.stuck_ticks,
                        "normal_round_ticks": ROUND_DURATION_TICKS, "model": metadata,
                        "waypoint_checks": checks, "summary": summary, "trials": trials})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
