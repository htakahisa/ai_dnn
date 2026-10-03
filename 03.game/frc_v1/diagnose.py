"""Print one FRC round's action counts and movement for policy debugging."""

import argparse
from collections import Counter
import json

import torch

from frc_v1.baseline import FrcBaseline
from frc_v1.environment import FrcRoundEnvironment
from frc_v1.evaluate import OPPONENTS
from frc_v1.model import FrcPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("A", "D"), required=True)
    parser.add_argument("--mode", choices=("learned", "baseline"), default="learned")
    parser.add_argument("--seed", type=int, default=100001)
    parser.add_argument("--opponent", choices=tuple(OPPONENTS), default="Fnatic2023")
    parser.add_argument("--episode", type=int, default=1, help="reproduce this independent evaluation round")
    parser.add_argument("--checkpoint")
    parser.add_argument("--trace", action="store_true", help="print visible contacts, planting, and ability decisions")
    parser.add_argument("--trace-every", type=int, default=0, help="print positions and movement every N ticks")
    args = parser.parse_args()
    if args.episode < 1:
        parser.error("--episode must be positive")
    torch.set_num_threads(1)
    env = FrcRoundEnvironment(args.side, seed=args.seed + (args.episode - 1) * 997,
                              opponent=OPPONENTS[args.opponent], opponent_roster=args.opponent)
    observation = env.reset()
    actor = (FrcBaseline() if args.mode == "baseline" else
             FrcPolicy.load(args.checkpoint or f"frc_v1/checkpoints/{args.side}_policy.pt", side=args.side))
    if args.episode > 1 and isinstance(actor, FrcPolicy):
        actor._navigation_round = env.controller.snapshot.round_number
        actor._navigation_phase = "live"
        actor._navigation_tick = env.max_ticks
        actor._navigation_attack_episode = args.episode - 2
        actor._navigation_defense_episode = args.episode - 1
    starts = [a.position for a in env.controller.snapshot.allies]
    counts = [Counter() for _ in starts]
    moved = [0 for _ in starts]
    death_at = [None for _ in starts]
    for step in range(env.max_ticks):
        before = env.controller.snapshot.allies
        decision = actor.act(observation, env.controller.snapshot, env.controller.belief)
        if ((args.trace and (env.controller.snapshot.sightings or env.controller.snapshot.is_planted or
                            any(action.kind == "ABILITY" for action in decision.actions))) or
                args.trace_every > 0 and step % args.trace_every == 0):
            print(json.dumps({"tick": step, "phase": env.controller.snapshot.phase,
                              "planted": env.controller.snapshot.is_planted,
                              "site": decision.site, "plan": decision.phase,
                              "seen": [s.position for s in env.controller.snapshot.sightings],
                              "positions": [a.position if a.alive else None for a in before],
                              "actions": [(a.kind, a.target) for a in decision.actions]},
                             ensure_ascii=False), flush=True)
        for slot, action in enumerate(decision.actions):
            if before[slot].alive:
                counts[slot][action.kind] += 1
        result = env.step(decision)
        observation = result.observation
        for slot, after in enumerate(env.controller.snapshot.allies):
            if before[slot].alive and after.position != before[slot].position:
                moved[slot] += 1
            if before[slot].alive and not after.alive and death_at[slot] is None:
                death_at[slot] = step + 1
        if result.terminated:
            break
    print(json.dumps({"mode": args.mode, "side": args.side, "seed": args.seed,
        "opponent": args.opponent, "episode": args.episode,
        "steps": step + 1, "winner": result.metrics["winner"], "starts": starts,
        "ends": [a.position for a in env.controller.snapshot.allies],
        "alive_actions": [dict(c) for c in counts], "moved_ticks": moved,
        "death_at": death_at, "metrics": result.metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
