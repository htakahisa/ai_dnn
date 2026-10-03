"""Guard-only post-plant learning in the real engine, with five dedicated foes."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import contextlib
import copy
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import train_attacker_gc_real_curriculum as curriculum
from positioning_gc import PLANT_PATTERNS
from training_opponent_pool_gc import OPPONENT_SPECS, describe_pool

guard = curriculum.guard_runtime
PHASES = curriculum.PHASES
START_MODES = ("far", "mid", "transition", "hold", "natural")
MOVEMENT_ACTIONS = ((2, (-1, 0)), (4, (1, 0)), (6, (0, -1)), (8, (0, 1)))
REVISION = "guard_reposition_v1"
FAR_DISTANCE = 12


def route_action(grid, start, goal, chars, name, mask):
    """Occupancy-aware path label; a detour may increase static BFS distance."""
    start, goal = tuple(start), tuple(goal)
    if start == goal:
        return 0 if mask[0] else None
    blocked = {tuple(c.pos) for c in chars
               if c.name != name and getattr(c, "is_alive", True)}
    queue = deque([start])
    first = {start: None}
    while queue:
        cell = queue.popleft()
        for action, (dr, dc) in MOVEMENT_ACTIONS:
            nxt = (cell[0] + dr, cell[1] + dc)
            if (nxt in first or nxt in blocked
                    or not (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1])
                    or grid[nxt] == 1
                    or (cell == start and not mask[action])):
                continue
            first[nxt] = action if cell == start else first[cell]
            if nxt == goal:
                return first[nxt]
            queue.append(nxt)
    return None


def navigation_teacher(controller, char, state, mask):
    """Labels use only the actor's perceived chars and the public defuse alert."""
    active = controller._active_defuse_info(state)
    chars = state.get("chars", ())
    smoke = state.get("smoke_cells", ())
    if active is not None:
        defuser = next((c for c in chars if c.name == active["name"] and c.is_alive), None)
        if defuser is not None and guard._has_los(
                state["grid"], tuple(char.pos), tuple(defuser.pos), smoke):
            return 0 if mask[0] else None
        goal = state["planted_pos"]
    else:
        # A perceived enemy outside the firing cone does not justify waiting.
        # Use the same contact test as the deployed engine's automatic shooting.
        view = getattr(controller, "training_view", None)
        if view is not None and curriculum.can_engage(view, char, chars):
            return 0 if mask[0] else None
        goal = controller._assigned_guard_positions.get(char.name)
    if goal is None:
        return None
    return route_action(state["grid"], char.pos, goal, chars, char.name, mask)


class GuardProgress:
    """One-time progress and arrival rewards, with no A-B-A reward farming."""

    def __init__(self):
        self.best = {}
        self.arrived = set()

    def reward(self, name, goal, before, after, engaged, defusing, can_move):
        if before < 0 or after < 0:
            return 0.0
        key = (name, tuple(goal))
        best = self.best.get(key, before)
        improvement = max(0, best - after)
        self.best[key] = min(best, before, after)
        if engaged:
            return 0.0
        if after == 0:
            bonus = 0.02 if before == 0 and not defusing else 0.0
            if key not in self.arrived:
                self.arrived.add(key)
                bonus += 0.5
            return bonus
        # Keep successful fighting/defuse interruption higher priority than
        # positional shaping. A blocked path has no waiting penalty.
        return 0.15 * improvement - (0.08 if before == after and can_move else 0.0)


class GuardRepositionSession(curriculum.CurriculumSession):
    def __init__(self, sources, policies, gamma):
        self.start_mode_override = None
        self.preserve_opponent_stats = True
        self.custom_guard_position_rewards = True
        self.decision_snapshots = {}
        self.guard_progress = GuardProgress()
        self.guard_stats = Counter()
        self.starts = {}
        self.arrival_ticks = {}
        self.plant_started = None
        super().__init__(sources, policies, gamma)
        self.frozen_phases = {"carry", "escort"}
        self.frozen_facing_phases = {"carry", "escort"}
        self.controllers["guard"].training_navigation_teacher = self.teacher
        controller = self.controllers["guard"]
        original = controller.decide_move

        def tracked_decide(char, state):
            if char.is_alive and state.get("is_planted") and state.get("planted_pos") is not None:
                self.track_decision(char, state)
            return original(char, state)

        controller.decide_move = tracked_decide

    def teacher(self, char, state, mask):
        controller = self.controllers["guard"]
        controller.training_view = self.game.attacker_controller.inner_controller.game
        return navigation_teacher(controller, char, state, mask)

    def reset(self, seed, augment=True, randomize_target=None):
        super().reset(seed, augment, randomize_target)
        self.guard_progress = GuardProgress()
        self.guard_stats = Counter()
        self.starts = {}
        self.arrival_ticks = {}
        self.decision_snapshots = {}
        self.plant_started = None
        # Modes and opponents vary independently: a 25-episode block includes
        # every mode against every opponent, without random-policy coupling.
        self.start_mode = self.start_mode_override or START_MODES[(seed // 5) % 5]
        if self.start_mode == "natural":
            return
        game = self.game
        rng = random.Random(seed + 971)
        patterns = sorted(PLANT_PATTERNS)
        marker = patterns[(seed // 25) % len(patterns)]
        spike = tuple(rng.choice(PLANT_PATTERNS[marker]))
        game.is_planted = True
        game.planted_pos = spike
        game.target_plant_pos = spike
        game.spike_pos = None
        from game_core import SPIKE_DETONATION_TICKS
        game.detonate_timer = SPIKE_DETONATION_TICKS
        occupied = set()
        spike_dist = self.distances(spike)
        walkable = [tuple(map(int, p)) for p in zip(*np.where(game.grid != 1))]
        # Seed assignments from a near-site team, then let inference make its
        # actual nearest-unused assignment again after the distributed spawns.
        attackers = sorted((c for c in game.chars if c.team == "A"), key=lambda c: c.name)
        candidates = curriculum.guard_runtime.guard_candidates(game.grid, marker, len(attackers))
        for index, char in enumerate(attackers):
            if self.start_mode == "hold":
                cells = [candidates[index]]
            elif self.start_mode == "transition" or index < 2:
                cells = [p for p in walkable if 0 <= spike_dist[p] <= 5]
            elif self.start_mode == "mid":
                cells = [p for p in walkable if FAR_DISTANCE <= spike_dist[p] <= 35
                         and game.width // 3 <= p[1] <= 2 * game.width // 3]
            else:
                cells = [p for p in walkable if FAR_DISTANCE <= spike_dist[p] <= 40]
            cells = [p for p in cells if p not in occupied]
            if not cells:
                raise RuntimeError(f"No free start for {self.start_mode}/{char.name}")
            char.pos = list(rng.choice(cells))
            char.has_spike = False
            char.plant_timer = char.defuse_timer = 0
            occupied.add(tuple(char.pos))
        # Dedicated defenders act with their own roster stats, IQ and retake
        # policies; place them on approaches instead of replaying spawn travel.
        for char in (c for c in game.chars if c.team == "D"):
            cells = [p for p in walkable if 8 <= spike_dist[p] <= 25 and p not in occupied]
            if not cells:
                raise RuntimeError("No free defender approach")
            char.pos = list(rng.choice(cells))
            occupied.add(tuple(char.pos))
        self.controllers["guard"].reset_round()
        game.current_attacker_team_ai.perception_engine.clear_cache()
        game.current_defender_team_ai.reset_round()

    def track_decision(self, char, state):
        controller = self.controllers["guard"]
        controller._ensure_guard_assignment(char, state["grid"], state.get("chars", ()), state["planted_pos"])
        active = controller._active_defuse_info(state)
        goal = tuple(state["planted_pos"]) if active else controller._assigned_guard_positions[char.name]
        pos = tuple(char.pos)
        view = self.game.attacker_controller.inner_controller.game
        engaged = curriculum.can_engage(view, char, state.get("chars", ()))
        route = route_action(state["grid"], pos, goal, state.get("chars", ()),
                             char.name, np.ones(guard.ACTION_DIM, dtype=bool))
        self.decision_snapshots[char.name] = (
            pos, tuple(goal), int(self.distances(tuple(goal))[pos]), engaged,
            active is not None, route is not None and route != 0, self.game.battle_tick)
        if self.plant_started is None:
            self.plant_started = self.game.battle_tick
        if char.name not in self.starts:
            assigned = controller._assigned_guard_positions[char.name]
            self.starts[char.name] = int(self.distances(assigned)[pos])
            if self.starts[char.name] == 0:
                self.arrival_ticks[char.name] = 0

    def guard_tick_rewards(self):
        rewards = {}
        chars = {c.name: c for c in self.game.chars if c.team == "A"}
        for name, snapshot in self.decision_snapshots.items():
            pos, goal, before, engaged, active, can_move, tick = snapshot
            if tick != self.game.battle_tick - 1:
                continue
            char = chars[name]
            after = int(self.distances(goal)[tuple(char.pos)])
            reward = self.guard_progress.reward(
                name, goal, before, after, engaged, active, can_move) if char.is_alive else 0.0
            self.guard_stats["alive_ticks"] += int(char.is_alive)
            far = self.starts.get(name, 0) >= FAR_DISTANCE
            quiet = bool(char.is_alive and not engaged and not active and before > 0 and can_move)
            self.guard_stats["quiet_travel_ticks"] += int(quiet)
            self.guard_stats["quiet_stall_ticks"] += int(quiet and tuple(char.pos) == pos)
            self.guard_stats["far_quiet_travel_ticks"] += int(quiet and far)
            self.guard_stats["far_quiet_stall_ticks"] += int(quiet and far and tuple(char.pos) == pos)
            assigned = self.controllers["guard"]._assigned_guard_positions[name]
            if char.is_alive and tuple(char.pos) == tuple(assigned):
                self.guard_stats["position_ticks"] += 1
                self.arrival_ticks.setdefault(name, self.game.battle_tick - self.plant_started)
            # A public defuse alert has priority over returning to a guard cell.
            if active and char.is_alive:
                self.guard_stats["defuse_alert_ticks"] += 1
                if not engaged:
                    reward -= 0.15
            rewards[name] = reward
        return rewards

    def play(self, *args, **kwargs):
        row = super().play(*args, **kwargs)
        eligible = {name for name, distance in self.starts.items() if distance > 0}
        far = {name for name in eligible if self.starts[name] >= FAR_DISTANCE}
        row.update(
            start_mode=self.start_mode,
            guard_eligible=len(eligible),
            guard_arrivals=sum(name in self.arrival_ticks for name in eligible),
            far_guard_eligible=len(far),
            far_guard_arrivals=sum(name in self.arrival_ticks for name in far),
            guard_arrival_tick_sum=sum(self.arrival_ticks[n] for n in eligible if n in self.arrival_ticks),
            far_guard_arrival_tick_sum=sum(self.arrival_ticks[n] for n in far if n in self.arrival_ticks),
            guard_starts=self.starts,
            guard_arrival_ticks=self.arrival_ticks,
            defused=bool(self.game.is_defused),
            **dict(self.guard_stats),
        )
        return row


def summarize(rows, groups=True):
    def total(key):
        return sum(row.get(key, 0) for row in rows)
    planted = sum(row["planted"] for row in rows)
    metrics = {
        "episodes": len(rows),
        "postplant_episodes": planted,
        "postplant_win_rate": sum(r["planted"] and r["attacker_win"] for r in rows) / max(1, planted),
        "round_win_rate": sum(r["attacker_win"] for r in rows) / max(1, len(rows)),
        "defuse_loss_rate": total("defused") / max(1, planted),
        "arrival_rate": total("guard_arrivals") / max(1, total("guard_eligible")),
        "far_arrival_rate": total("far_guard_arrivals") / max(1, total("far_guard_eligible")),
        "far_eligible": total("far_guard_eligible"),
        "mean_arrival_ticks": total("guard_arrival_tick_sum") / max(1, total("guard_arrivals")),
        "far_mean_arrival_ticks": total("far_guard_arrival_tick_sum") / max(1, total("far_guard_arrivals")),
        "quiet_stall_rate": total("quiet_stall_ticks") / max(1, total("quiet_travel_ticks")),
        "far_quiet_stall_rate": total("far_quiet_stall_ticks") / max(1, total("far_quiet_travel_ticks")),
        "position_tick_rate": total("position_ticks") / max(1, total("alive_ticks")),
    }
    if groups:
        for key in ("opponent_name", "start_mode"):
            metrics["by_" + key] = {value: summarize([r for r in rows if r[key] == value], False)
                                    for value in sorted({r[key] for r in rows})}
    return metrics


def evaluate(session, seeds, episodes):
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        rows = [session.play(seed + index, (100, 60, 40), training=False)
                for seed in seeds for index in range(episodes)]
        return {**summarize(rows), "seeds": list(seeds), "rows": rows}
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


def selection_score(metrics, baseline, tolerance=0.05):
    safe = (metrics["postplant_win_rate"] >= baseline["postplant_win_rate"] - tolerance
            and metrics["defuse_loss_rate"] <= baseline["defuse_loss_rate"] + tolerance)
    for group in ("by_opponent_name", "by_start_mode"):
        for key, previous in baseline.get(group, {}).items():
            current = metrics.get(group, {}).get(key)
            if current is None:
                safe = False
                continue
            if current["postplant_episodes"] and previous["postplant_episodes"]:
                safe &= (current["postplant_win_rate"] >= previous["postplant_win_rate"] - tolerance
                         and current["defuse_loss_rate"] <= previous["defuse_loss_rate"] + tolerance)
    return (int(safe), metrics["far_arrival_rate"], -metrics["far_quiet_stall_rate"],
            metrics["arrival_rate"], metrics["postplant_win_rate"])


def load_policies(sources):
    checkpoints = {p: torch.load(path, map_location="cpu", weights_only=False) for p, path in sources.items()}
    policies = {
        "carry": curriculum.runtime.AttackerCarryDuelingDQN(
            obs_dim=curriculum.runtime.ORB_OBS_DIM, action_dim=curriculum.runtime.ACTION_DIM),
        "escort": curriculum.escort_runtime.DuelingQNetwork(
            curriculum.escort_runtime.ORB_OBS_DIM, curriculum.escort_runtime.N_ACTIONS),
        "guard": guard.AttackerGuardDuelingDQN(obs_dim=guard.ORB_OBS_DIM, action_dim=guard.ACTION_DIM),
    }
    for phase, policy in policies.items():
        # All three deployed checkpoints currently have these exact dimensions.
        # Refuse a fallback or an architecture migration in a Guard-only run.
        policy.load_state_dict(checkpoints[phase]["model_state_dict"], strict=True)
        policy.eval()
        if phase != "guard":
            for parameter in policy.parameters():
                parameter.requires_grad_(False)
    return checkpoints, policies


def load_guard_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (int(checkpoint.get("obs_dim", 0)) != guard.ORB_OBS_DIM
            or int(checkpoint.get("n_actions", 0)) != guard.ACTION_DIM
            or int(checkpoint.get("positioning_version", 0)) != 5):
        raise ValueError(f"Incompatible Guard continuation checkpoint: {path}")
    return checkpoint


def train(args):
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    torch.manual_seed(args.seed)
    sources = {p: getattr(args, "init_" + p).resolve() for p in PHASES}
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a fresh output directory; source and previous runs are preserved")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints, policies = load_policies(sources)
    frozen = {p: {k: v.clone() for k, v in policies[p].state_dict().items()} for p in ("carry", "escort")}
    fingerprints = curriculum.runtime_data_fingerprint()
    session = GuardRepositionSession(sources, policies, args.gamma)
    optimizer = torch.optim.Adam(policies["guard"].parameters(), lr=args.lr)
    target = copy.deepcopy(policies["guard"])
    replay, demonstrations = deque(maxlen=100_000), deque(maxlen=30_000)
    manifest = {
        "revision": REVISION,
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (Path(__file__), Path(curriculum.__file__),
                                     HERE / "training_opponent_pool_gc.py",
                                     HERE / "train_attacker_carry_gc_real.py")},
        "sources": {p: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for p, path in sources.items()},
        "opponents": list(OPPONENT_SPECS),
        "runtime_data_fingerprint": fingerprints,
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "continuation_models": {
            label: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for label, path in (("warm_start", args.warm_start_guard),
                                ("previous_winner", args.seed_guard_candidate)) if path is not None
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(describe_pool(), flush=True)
    print(f"Guard only; source episode={checkpoints['guard'].get('episode')}; "
          f"offset={args.episode_offset}; additional_episodes={args.episodes}; output={args.output_dir}", flush=True)

    def check_frozen():
        for phase, state in frozen.items():
            if any(not torch.equal(value, policies[phase].state_dict()[key]) for key, value in state.items()):
                raise RuntimeError(f"Frozen {phase} weights changed")
        if curriculum.runtime_data_fingerprint() != fingerprints:
            changed = [name for name, value in curriculum.runtime_data_fingerprint().items()
                       if fingerprints.get(name) != value]
            (args.output_dir / "failure.json").write_text(json.dumps({
                "reason": "runtime_data_changed", "changed_files": changed,
                "original_fingerprint": fingerprints,
                "current_fingerprint": curriculum.runtime_data_fingerprint(),
                "production_updated": False,
            }, indent=2), encoding="utf-8")
            raise RuntimeError("Runtime data changed during this run: " + ", ".join(changed))

    def save(kind, episode, metrics):
        check_frozen()
        payload = dict(checkpoints["guard"])
        payload.update(model_state_dict=policies["guard"].state_dict(),
                       obs_dim=guard.ORB_OBS_DIM, n_actions=guard.ACTION_DIM, positioning_version=5,
                       episode=episode, source_episode=checkpoints["guard"].get("episode"),
                       training_environment="actual_engine_guard_reposition_five_opponents",
                       training_revision=REVISION, evaluation=metrics,
                       runtime_data_fingerprint=fingerprints, sources=manifest["sources"])
        torch.save(payload, args.output_dir / f"dqn_attacker_guard_gc_{kind}.pt")

    baseline = evaluate(session, args.eval_seeds, args.eval_episodes)
    (args.output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    best_score = selection_score(baseline, baseline, args.regression_tolerance)
    best_episode = 0
    best_state = copy.deepcopy(policies["guard"].state_dict())
    evaluations = [{"episode": 0, "metrics": baseline, "score": best_score}]
    print(f"[BASELINE] episodes={baseline['episodes']} far={baseline['far_arrival_rate']:.3f}", flush=True)
    # Revalidate an earlier winner after a data revision. Its old metrics are
    # never compared with this run's baseline or used for candidate selection.
    for label, path in (("previous_winner", args.seed_guard_candidate),
                        ("warm_start", args.warm_start_guard)):
        if path is None:
            continue
        checkpoint = load_guard_checkpoint(path)
        policies["guard"].load_state_dict(checkpoint["model_state_dict"], strict=True)
        metrics = evaluate(session, args.eval_seeds, args.eval_episodes)
        check_frozen()
        score = selection_score(metrics, baseline, args.regression_tolerance)
        candidate_episode = int(checkpoint.get("episode", args.episode_offset))
        evaluations.append({"episode": candidate_episode, "phase": "revalidation",
                            "candidate": label, "metrics": metrics, "score": score})
        if score[0] and score > best_score:
            best_score, best_episode = score, candidate_episode
            best_state = copy.deepcopy(policies["guard"].state_dict())
            save("best_by_eval", best_episode, metrics)
        (args.output_dir / "evaluations.json").write_text(json.dumps(evaluations, indent=2), encoding="utf-8")
        print(f"[REVALIDATE {label}] episode={candidate_episode} "
              f"far={metrics['far_arrival_rate']:.3f} safe={bool(score[0])}", flush=True)
    if args.warm_start_guard is not None:
        policies["guard"].load_state_dict(load_guard_checkpoint(args.warm_start_guard)["model_state_dict"], strict=True)
    else:
        policies["guard"].load_state_dict(checkpoints["guard"]["model_state_dict"], strict=True)
    target.load_state_dict(policies["guard"].state_dict())
    history = []
    started = time.monotonic()
    interrupted = False
    episode = args.episode_offset
    try:
        for local_episode in range(1, args.episodes + 1):
            episode = args.episode_offset + local_episode
            probability = (0.8 * max(0.0, 1.0 - local_episode / args.bootstrap_episodes)
                           if args.bootstrap_episodes else 0.0)
            row = session.play(args.seed + episode - 1, (100, 60, 40),
                               epsilon=max(0.03, 0.15 * (1 - episode / (args.episode_offset + args.episodes))),
                               teacher_probability=probability, orb_teacher_probability=0.0)
            row["episode"] = episode
            row["guard_demo_count"] = len(session.demonstrations["guard"])
            history.append(row)
            with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            replay.extend(curriculum.n_step_transitions(session.transitions["guard"], args.gamma, 3))
            demonstrations.extend(session.demonstrations["guard"])
            td_loss = demo_loss = None
            for _ in range(args.updates):
                td_loss = curriculum.optimize(policies["guard"], target, optimizer, replay, args.gamma)
                demo_loss = curriculum.optimize_demonstrations(
                    policies["guard"], optimizer, demonstrations, args.demo_weight)
            if episode % 10 == 0:
                target.load_state_dict(policies["guard"].state_dict())
            if episode % 5 == 0:
                print(f"[EP {episode}] opponent={row['opponent_name']} mode={row['start_mode']} "
                      f"far={row['far_guard_arrivals']}/{row['far_guard_eligible']} "
                      f"demo={row['guard_demo_count']} td={td_loss} demo_loss={demo_loss} "
                      f"seconds={time.monotonic()-started:.0f}", flush=True)
            if episode % 25 == 0:
                save("latest", episode, {"status": "evaluation_pending"})
            if episode % args.eval_interval == 0 or local_episode == args.episodes:
                metrics = evaluate(session, args.eval_seeds, args.eval_episodes)
                score = selection_score(metrics, baseline, args.regression_tolerance)
                evaluations.append({"episode": episode, "metrics": metrics, "score": score})
                save("latest", episode, metrics)
                if score[0] and score > best_score:
                    best_score, best_episode = score, episode
                    best_state = copy.deepcopy(policies["guard"].state_dict())
                    save("best_by_eval", episode, metrics)
                (args.output_dir / "evaluations.json").write_text(json.dumps(evaluations, indent=2), encoding="utf-8")
                print(f"[EVAL {episode}] far={metrics['far_arrival_rate']:.3f} "
                      f"stall={metrics['far_quiet_stall_rate']:.3f} "
                      f"postplant_win={metrics['postplant_win_rate']:.3f} safe={bool(score[0])} "
                      f"best={best_episode}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        save("latest", episode, {"status": "interrupted"})
    check_frozen()
    # Holdout seeds were not used in selection. Compare the unchanged source
    # against the evaluation winner, with identical scenarios and opponents.
    policies["guard"].load_state_dict(checkpoints["guard"]["model_state_dict"])
    holdout_baseline = evaluate(session, args.holdout_seeds, args.eval_episodes)
    policies["guard"].load_state_dict(best_state)
    holdout_candidate = evaluate(session, args.holdout_seeds, args.eval_episodes)
    safe = selection_score(holdout_candidate, holdout_baseline, args.regression_tolerance)[0]
    approved = bool(best_episode and safe and holdout_candidate["far_arrival_rate"] > holdout_baseline["far_arrival_rate"])
    report = {"best_episode": best_episode, "completed_episodes": episode,
              "interrupted": interrupted, "frozen_weights_unchanged": True,
              "candidate_qualified": approved,
              "baseline": holdout_baseline, "candidate": holdout_candidate}
    (args.output_dir / "holdout.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if approved:
        save("selected", best_episode, holdout_candidate)
    print(json.dumps({k: v for k, v in report.items() if k not in ("baseline", "candidate")}), flush=True)


def main():
    import ghost_champions_v1 as gc
    parser = argparse.ArgumentParser(description=__doc__)
    for phase in PHASES:
        parser.add_argument("--init-" + phase, type=Path,
                            default=gc._first_existing(getattr(gc, phase.upper())))
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "data" / ("guard_reposition_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--warm-start-guard", type=Path,
                        help="Start further optimization from a saved Guard; baseline stays init-guard")
    parser.add_argument("--seed-guard-candidate", type=Path,
                        help="Revalidate a previous evaluation winner using the current runtime data")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[3126100200, 5126100200, 7126100200])
    parser.add_argument("--holdout-seeds", type=int, nargs="+", default=[9226100200, 11226100200])
    parser.add_argument("--seed", type=int, default=2026100200)
    parser.add_argument("--bootstrap-episodes", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--demo-weight", type=float, default=1.0)
    parser.add_argument("--regression-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    if any(getattr(args, "init_" + p) is None or not getattr(args, "init_" + p).is_file() for p in PHASES):
        parser.error("All deployed Carry/Escort/Guard source checkpoints are required")
    if min(args.episodes, args.eval_interval, args.eval_episodes, args.updates) <= 0:
        parser.error("Episode and update counts must be positive")
    if args.episode_offset < 0:
        parser.error("episode-offset must be nonnegative")
    for path in (args.warm_start_guard, args.seed_guard_candidate):
        if path is not None and not path.is_file():
            parser.error(f"Continuation checkpoint does not exist: {path}")
    if not (0 < args.gamma <= 1 and args.lr > 0 and args.demo_weight >= 0
            and 0 <= args.regression_tolerance <= 1 and args.bootstrap_episodes >= 0):
        parser.error("Invalid learning or regression parameters")
    if args.eval_episodes % 25:
        parser.error("eval-episodes must be a multiple of 25 for balanced opponents and start modes")
    evaluation_cases = {s + i for s in args.eval_seeds for i in range(args.eval_episodes)}
    holdout_cases = {s + i for s in args.holdout_seeds for i in range(args.eval_episodes)}
    if evaluation_cases & holdout_cases:
        parser.error("Evaluation and holdout seeds must be independent")
    train(args)


if __name__ == "__main__":
    main()
