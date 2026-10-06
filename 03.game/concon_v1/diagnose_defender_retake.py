"""Trace failed retakes with frozen latest/foundation weights; never optimize."""

import argparse
from collections import Counter, defaultdict
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

from concon_v1.co1_attacker_common import bfs_distance_map
from concon_v1.co1_battle_training import OPPONENTS
from concon_v1.co1_defender_retake_training import DefenderRetakeEnv, TrainingRetakeController
from concon_v1.co1_learn_defender_search import ConconDefenderSearchController
from concon_v1.co1_retake_common import RetakeDQN, DEFUSE_ACTION, FEATURE_DIM, MOVES, MAP_CHANNELS
from concon_v1.co1_retake_scenarios import get_scenario

BASE = Path(__file__).resolve().parent
DEFAULT_CASES_DIR = BASE / "data" / "defender_retake_cases"
DEFAULT_LOG = BASE / "data" / "defender_retake_L_data" / "retake_training_log.jsonl"
DEFAULT_OUTPUT = BASE / "data" / "diagnostics" / "defender_retake_motion.json"


def foundation_context(model, observation):
    maps = observation[:model.map_size].reshape(MAP_CHANNELS, model.height, model.width)
    features = observation[model.map_size:]
    position = (round(float(features[0]) * model.height), round(float(features[1]) * model.width))
    spike = (round(float(features[2]) * model.height), round(float(features[3]) * model.width))
    row, col = position
    sr, sc = spike
    plant = int(model.foundation_plant_lookup[sr * model.width + sc]) if model.foundation else -1
    index = max(plant, 0) * model.height * model.width + row * model.width + col
    reasons = []
    if not model.foundation:
        reasons.append("no_foundation")
    if features[16] != 0:
        reasons.append("fireable")
    if plant < 0:
        reasons.append("unknown_plant")
    elif not bool(model.foundation_trained[index]):
        reasons.append("untrained_position")
    basic = model.foundation_values.weight[index].detach().cpu().tolist() if model.foundation and plant >= 0 else [0.] * 6
    return dict(active=not reasons, reasons=reasons, basic=basic)


def action_label(action):
    if action == DEFUSE_ACTION:
        return "defuse"
    if action < 40:
        return ("north", "south", "west", "east", "wait")[action // 8]
    if action < 112:
        return ("smoke", "flash", "recon")[(action // 8 - 5) // 3]
    return "ultimate"


class DiagnosticController(TrainingRetakeController):
    def choose_action(self, char, observation, mask, context):
        action = super().choose_action(char, observation, mask, context)
        with torch.no_grad():
            values = self.model(torch.as_tensor(observation).unsqueeze(0))[0].cpu().numpy()
        legal = np.where(mask, values, -np.inf)
        greedy = int(legal.argmax())
        basis = foundation_context(self.model, observation)
        real = self.env.defenders[self.env.indices[char.name]]
        true_spike = tuple(self.env.game.planted_pos)
        true_distance = int(self.env.true_distances[tuple(real.pos)])
        real_near = max(abs(real.pos[0] - true_spike[0]), abs(real.pos[1] - true_spike[1])) <= 1
        legal_defuse = bool(mask[DEFUSE_ACTION])
        counter = self.env.counters
        counter["decisions"] += 1
        counter["foundation_active"] += int(basis["active"])
        for reason in basis["reasons"]:
            counter["foundation_off_" + reason] += 1
        counter["real_near_decisions"] += int(real_near)
        counter["legal_defuse_decisions"] += int(legal_defuse)
        counter["legal_defuse_not_selected"] += int(legal_defuse and action != DEFUSE_ACTION)
        counter["legal_defuse_greedy_rejected"] += int(legal_defuse and greedy != DEFUSE_ACTION)
        counter["waiting_decisions"] += int(context["waiting"])
        counter["waiting_at_goal"] += int(context["waiting"] and context["reward_distances"][context["position"]] == 0)
        counter["selected_" + action_label(action)] += 1
        counter["greedy_" + action_label(greedy)] += 1
        goal_distance = int(context["reward_distances"][context["position"]])
        shrinking = []
        for op, (dr, dc) in enumerate(MOVES[:4]):
            point = context["position"][0] + dr, context["position"][1] + dc
            if mask[op * 8:(op + 1) * 8].any() and context["reward_distances"][point] < goal_distance:
                shrinking.extend(range(op * 8, (op + 1) * 8))
        if shrinking:
            counter["goal_progress_available"] += 1
            counter["greedy_progress_rejected"] += int(greedy not in shrinking)
            counter["active_foundation_progress_rejected"] += int(basis["active"] and greedy not in shrinking)
        event = dict(tick=self.env.elapsed_ticks + 1, actor=char.name, pos=list(real.pos),
                     true_distance=true_distance, real_near=real_near, legal_defuse=legal_defuse,
                     remaining=float(self.env.game.detonate_timer), waiting=bool(context["waiting"]),
                     goal=list(context["goal"]), goal_distance=goal_distance,
                     fireable=bool(context["fireable"]), action=action_label(action), greedy=action_label(greedy),
                     q_selected=float(values[action]), q_greedy=float(values[greedy]),
                     q_defuse=float(values[DEFUSE_ACTION]) if legal_defuse else None,
                     q_best_progress=max(float(legal[a]) for a in shrinking) if shrinking else None,
                     foundation=basis)
        self.env.trace.append(event)
        return action


class DiagnosticEnv(DefenderRetakeEnv):
    def reset_case(self, path, tensor_cache=None):
        result = super().reset_case(path, tensor_cache)
        self.counters, self.trace = Counter(), []
        self.true_distances = bfs_distance_map(self.game.grid, tuple(self.game.planted_pos))
        self.actor_stats = {char.name: dict(start=list(char.pos), min_distance=int(self.true_distances[tuple(char.pos)]),
                            first_arrival_tick=None, max_defuse_ticks=0) for char in self.defenders if char.is_alive}
        for site in self.retakes:
            self.retakes[site] = DiagnosticController(self, site)
        self.observe_positions()
        return result

    def observe_positions(self):
        spike = self.game.planted_pos
        for char in self.defenders:
            if char.name not in self.actor_stats:
                continue
            stats = self.actor_stats[char.name]
            stats["max_defuse_ticks"] = max(stats["max_defuse_ticks"], int(char.defuse_timer))
            if not char.is_alive:
                continue
            distance = int(self.true_distances[tuple(char.pos)])
            if distance >= 0:
                stats["min_distance"] = min(stats["min_distance"], distance)
            if max(abs(char.pos[0] - spike[0]), abs(char.pos[1] - spike[1])) <= 1 and stats["first_arrival_tick"] is None:
                stats["first_arrival_tick"] = self.elapsed_ticks
                stats["remaining_at_arrival"] = float(self.game.detonate_timer)

    def step(self, epsilon=0.):
        result = super().step(epsilon)
        self.observe_positions()
        return result


def select_failed_cases(log, per_group, first_episode, last_episode):
    rows = []
    for line in Path(log).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    starts = [index for index, row in enumerate(rows) if row.get("round") == 1]
    if starts:
        rows = rows[starts[-1]:]
    selected, counts, seen = [], Counter(), set()
    for row in rows:
        if not first_episode <= row.get("episode", 0) <= last_episode or row.get("defused") or not row.get("case_file"):
            continue
        key = row["opponent"], row["site"]
        filename = Path(row["case_file"]).name
        if counts[key] < per_group and filename not in seen:
            counts[key] += 1
            seen.add(filename)
            selected.append(row)
    if not selected:
        raise ValueError("no failed saved cases in the requested training window")
    return selected


def diagnose(cases_dir=DEFAULT_CASES_DIR, log=DEFAULT_LOG, output=DEFAULT_OUTPUT,
             per_group=2, first_episode=501, last_episode=600, variants=("latest", "foundation")):
    torch.set_num_threads(1)
    if per_group < 1:
        raise ValueError("cases per opponent/site must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Check report permissions before performing the rollouts.
    probe = output.with_suffix(".probe")
    probe.write_text("", encoding="utf-8")
    probe.unlink()
    selected = select_failed_cases(log, per_group, first_episode, last_episode)
    search = ConconDefenderSearchController()
    report = dict(source_window=[first_episode, last_episode], cases=len(selected), checkpoints={}, variants={})
    tensor_cache = {}
    checkpoint_cache = {}
    for variant in variants:
        models, distances = {}, {}
        report["checkpoints"][variant] = {}
        for site in ("L", "R"):
            scenario = get_scenario(site)
            kind = "foundation" if variant == "foundation" else "latest"
            path = scenario.model_path(kind)
            if (kind, site) not in checkpoint_cache:
                checkpoint_cache[kind, site] = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint = checkpoint_cache[kind, site]
            model = RetakeDQN(scenario, foundation=checkpoint.get("foundation_version") == 1)
            model.load_state_dict(checkpoint["model_state_dict"])
            if variant == "latest_without_foundation":
                model.foundation = False
            model.eval().requires_grad_(False)
            models[site] = model
            distances[site] = checkpoint.get("ability_distances", checkpoint.get("ability_distance", 6))
            report["checkpoints"][variant][site] = dict(path=str(path), episode=checkpoint["episode"])
        env = DiagnosticEnv(models, search.model, opponents=OPPONENTS, ability_distance=distances)
        results, totals = [], Counter()
        for row in selected:
            seed = int.from_bytes(hashlib.sha256(row["case_file"].encode()).digest()[:4], "big")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env.action_rng.seed(seed)
            env.reset_case(Path(cases_dir).resolve() / Path(row["case_file"]).name, tensor_cache)
            epsilon = float(row["epsilon"])
            while not env.done:
                env.step(epsilon)
            arrived = any(stats["first_arrival_tick"] is not None for stats in env.actor_stats.values())
            totals.update(env.counters)
            totals["rounds"] += 1
            totals["arrival_rounds"] += int(arrived)
            totals["defuse_attempt_rounds"] += int(env.metrics["defuse_decisions"] > 0)
            totals["defused_rounds"] += int(env.game.is_defused)
            item = dict(case_file=Path(row["case_file"]).name, source_episode=row["episode"], epsilon=epsilon,
                        result=env.result(), counters=dict(env.counters), actors=env.actor_stats, trace=env.trace)
            results.append(item)
            print(f"{variant}: {row['opponent']} {row['site']} arrived={arrived} "
                  f"defuse_actions={env.metrics['defuse_decisions']} end={env.end_reason}", flush=True)
        report["variants"][variant] = dict(totals=dict(totals), rounds=results)
        print(f"{variant} totals: {json.dumps(dict(totals))}", flush=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved diagnostic: {output.resolve()}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-group", type=int, default=2)
    args = parser.parse_args()
    diagnose(per_group=args.per_group)
