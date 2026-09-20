"""Train Carry in the actual 5v5 engine, including Macro escorts and IQ.

The physics and opponent actions come from the real match. Only the Carry
network's action selection is replaced for exploration. Checkpoints stay in
a separate directory until full-series validation passes.
"""
from __future__ import annotations

import argparse
from collections import deque
import contextlib
from dataclasses import dataclass
import io
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import learning_attacker_carry_gc as runtime
from positioning_gc import (REGISTERED_PLANT_CELLS, PLANT_PATTERNS,
                            team_plant_target, validate_tactical_maps)

VERSION = 3


def expanded_state(checkpoint):
    state = dict(checkpoint["model_state_dict"])
    weights = state["feature.0.weight"]
    if weights.shape[1] == runtime.OBS_DIM:
        expanded = weights.new_zeros((weights.shape[0], runtime.REAL_OBS_DIM))
        expanded[:, :runtime.OBS_DIM] = weights
        # Legacy runtime always supplied zero here (the engine clears the
        # flag before deciding). Preserve its deployed policy initially and
        # learn the corrected previous-move feature from actual transitions.
        expanded[:, 3] = 0
        state["feature.0.weight"] = expanded
    return state


@dataclass
class Pending:
    obs: np.ndarray
    action: int
    reward: float = 0.0
    ticks: int = 0


class RealCarrySession:
    def __init__(self, init_model, policy, consume, gamma=0.99):
        from run_game import VisualFPSBattle, _build_team_ai
        from party_presets import get_preset
        from map_data import NEW_MAZE_STR
        from run_competition_manager import TeamPlayerKey
        gc = get_preset("Ghost Champions")
        opponent = get_preset("Touyama Gaming")
        with contextlib.redirect_stdout(io.StringIO()):
            self.game = VisualFPSBattle(
                NEW_MAZE_STR, _build_team_ai("ghost_champions_v1"),
                _build_team_ai("touyama_gaming_v2"), headless=True,
                attacker_roster=[TeamPlayerKey(n, "team1:" + gc.name) for n in gc.players],
                defender_roster=[TeamPlayerKey(n, "team2:" + opponent.name) for n in opponent.players],
                spike_holder_name=gc.spike_holder, defender_spike_holder_name=opponent.spike_holder,
                attacker_igl_name=gc.igl, defender_igl_name=opponent.igl,
                attacker_team_name=gc.name, defender_team_name=opponent.name,
                disable_side_swap=True, series_context={"maps_to_win": 2})
        inner = self.game.attacker_controller.inner_controller
        validate_tactical_maps(self.game.grid)
        self.controller = runtime.LearningAttackerCarryGCController(
            model_path=str(init_model), device="cpu", greedy=True)
        inner.carry = self.controller
        self.controller.set_game(self.game)
        if inner.macro_controller is not None:
            inner.macro_controller.learned_positioning = True
        self.controller.policy_net = policy
        self.controller.positioning_version = VERSION
        self.controller._select_action = self.choose_action
        self.controller.greedy = True
        self.controller.debug = False
        self.policy = policy
        self.consume = consume
        self.gamma = gamma
        self.pending = {}
        self.epsilon = 0.0
        self.maps = {}
        # Retain the engine's round termination but leave its terminal actors
        # available to reward the final action before resetting the episode.
        self.game.check_match_winner = lambda: None
        self.game._record_replay_frame = lambda: None
        self.game.analytics_tracker = None

    def choose_action(self, obs, mask):
        holder = next(c for c in self.game.chars
                      if c.team == "A" and c.is_alive and c.has_spike)
        name = holder.name
        previous = self.pending.pop(name, None)
        if previous is not None:
            self.consume(previous, obs.copy(), mask.copy(), False)
        if random.random() < self.epsilon:
            valid = np.flatnonzero(mask)
            action = int(random.choice(valid)) if len(valid) else 0
        else:
            with torch.no_grad():
                values = self.policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
            action = int(np.argmax(np.where(mask, values, -np.inf)))
        self.pending[name] = Pending(obs.copy(), action)
        return action

    def distances(self, target):
        if target not in self.maps:
            self.maps[target] = runtime._bfs_distance_map(self.game.grid, target)
        return self.maps[target]

    def reset(self, seed, augment=True, randomize_target=None):
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        game = self.game
        game.current_round = random.randint(1, 24) if augment else 1
        game.attacker_wins = random.randint(0, 5) if augment else 0
        game.defender_wins = random.randint(0, 12) if augment else 0
        game.match_over = False
        game.overtime = False
        game.match_stats.clear()
        game.team_round_loss_streak.clear()
        game.player_mental_fatigue = {
            str(n): random.uniform(-0.05, 0.35) if augment else 0.0
            for n in game.attacker_roster + game.defender_roster}
        self.pending.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            game.init_round()
            while game.defender_setup_phase.active:
                game._run_defender_setup_tick()
        # Cover all registered plans during training. Evaluation uses the
        # engine's normal initial target and the production Macro decisions.
        use_random_target = augment if randomize_target is None else randomize_target
        if use_random_target:
            marker = random.choice(list(PLANT_PATTERNS))
            game.target_plant_pos = random.choice(PLANT_PATTERNS[marker])

    def tick(self):
        game = self.game
        before = {c.name: (tuple(c.pos), c.hp, c.plant_timer, c.kills, c.is_alive)
                  for c in game.chars}
        game._build_occupancy_counts()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for char in game._move_order():
                    if char.is_alive:
                        game.move_character(char)
                game.process_battle()
                game._advance_combo_announcement()
        finally:
            game._clear_occupancy_counts()
        terminal = game.is_planted or game.round_over
        target = team_plant_target(game)
        distance = self.distances(target) if target is not None else None
        smoke = game._smoke_cells()
        for char in game.chars:
            pending = self.pending.get(char.name)
            if pending is None:
                continue
            pos, hp, progress, kills, alive = before[char.name]
            reward = -0.03 - 0.002 * game.battle_tick
            if distance is not None and distance[pos] >= 0 and distance[tuple(char.pos)] >= 0:
                reward += 0.15 * (distance[pos] - distance[tuple(char.pos)])
            reward -= 0.025 * max(0, hp - char.hp)
            reward += 0.4 * max(0, char.kills - kills)
            reward += 0.8 * max(0, char.plant_timer - progress)
            if progress > 0 and char.plant_timer < progress and not game.is_planted:
                reward -= 0.6 * progress
            support = sum(c is not char and c.team == "A" and c.is_alive
                          and max(abs(c.pos[0] - char.pos[0]), abs(c.pos[1] - char.pos[1])) <= 3
                          for c in game.chars)
            if support == 0 and any(c.team == "D" and c.is_alive
                                    and runtime._has_los(game.grid, char.pos, c.pos, smoke)
                                    for c in game.chars):
                reward -= 0.12
            dead = alive and not char.is_alive
            if dead:
                reward -= 8.0
            if terminal:
                if game.is_planted:
                    reward += 20.0
                    if tuple(game.planted_pos) in REGISTERED_PLANT_CELLS:
                        reward += 3.0
                elif game.round_timer <= 0:
                    reward -= 12.0
                elif not any(c.team == "A" and c.is_alive for c in game.chars):
                    reward -= 12.0
                else:
                    reward += 8.0  # Defender wipe is also a valid round win.
            pending.reward += (self.gamma ** pending.ticks) * reward
            pending.ticks += 1
            if dead or terminal:
                self.pending.pop(char.name)
                self.consume(pending, np.zeros(runtime.REAL_OBS_DIM, dtype=np.float32),
                             np.ones(runtime.ACTION_DIM, dtype=bool), True)
        return terminal

    def run_episode(self, seed, epsilon=0.0, augment=True, after_tick=None):
        self.epsilon = epsilon
        self.reset(seed, augment)
        while not self.tick():
            if after_tick:
                after_tick()
        if after_tick:
            after_tick()
        game = self.game
        return {"planted": bool(game.is_planted),
                "registered": bool(game.is_planted and tuple(game.planted_pos) in REGISTERED_PLANT_CELLS),
                "ticks": game.battle_tick,
                "attacker_win": bool(not game.is_planted and any(
                    c.team == "A" and c.is_alive for c in game.chars)
                    and not any(c.team == "D" and c.is_alive for c in game.chars))}


def optimize(policy, target, optimizer, replay, gamma, batch_size=128):
    if len(replay) < batch_size:
        return None
    rows = random.sample(list(replay), batch_size)
    obs, action, reward, next_obs, mask, terminal, steps = zip(*rows)
    obs, next_obs = torch.tensor(np.array(obs)), torch.tensor(np.array(next_obs))
    with torch.no_grad():
        next_actions = policy(next_obs).masked_fill(~torch.tensor(np.array(mask)), -torch.inf).argmax(1)
        future = target(next_obs).gather(1, next_actions[:, None]).squeeze(1)
        expected = torch.tensor(reward) + torch.tensor([gamma ** n for n in steps]) * future * (1 - torch.tensor(terminal).float())
    predicted = policy(obs).gather(1, torch.tensor(action)[:, None]).squeeze(1)
    loss = torch.nn.functional.smooth_l1_loss(predicted, expected)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach())


def train(args):
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.init_model, map_location="cpu", weights_only=False)
    source_hash = hashlib.sha256(args.init_model.read_bytes()).hexdigest()
    policy = runtime.AttackerCarryDuelingDQN(obs_dim=runtime.REAL_OBS_DIM)
    incompatible = policy.load_state_dict(expanded_state(checkpoint), strict=False)
    missing = [key for key in incompatible.missing_keys
               if not key.startswith("facing_head.")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Carry checkpoint keys mismatch: missing={missing}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    target = runtime.AttackerCarryDuelingDQN(obs_dim=runtime.REAL_OBS_DIM)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    replay = deque(maxlen=100_000)
    def consume(pending, obs, mask, terminal):
        replay.append((pending.obs, pending.action, pending.reward, obs, mask, terminal, pending.ticks))
    session = RealCarrySession(args.init_model, policy, consume, gamma=args.gamma)
    steps = 0
    def after_tick():
        nonlocal steps
        steps += 1
        if steps % 2 == 0:
            optimize(policy, target, optimizer, replay, args.gamma)
        if steps % 500 == 0:
            target.load_state_dict(policy.state_dict())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = (-1, -1, -1)
    def save(name, episode, evaluation):
        payload = dict(checkpoint)
        payload.update(model_state_dict=policy.state_dict(), obs_dim=runtime.REAL_OBS_DIM,
                       n_actions=runtime.ACTION_DIM, positioning_version=VERSION,
                       episode=episode, training_environment="actual_engine_5v5_iq_macro_escort",
                       evaluation=evaluation, source_model=str(args.init_model),
                       source_model_sha256=source_hash,
                       success_rate=evaluation["plant_rate"],
                       training_parameters={"seed": args.seed, "gamma": args.gamma,
                                            "lr": args.lr, "episodes": args.episodes})
        torch.save(payload, args.output_dir / name)
    start = time.monotonic()
    for episode in range(1, args.episodes + 1):
        epsilon = max(0.03, 0.20 * (1 - episode / args.episodes))
        row = session.run_episode(args.seed + episode, epsilon, after_tick=after_tick)
        history.append(row)
        if episode % 10 == 0:
            recent = history[-20:]
            print(f"[REAL EP {episode}] plant={sum(r['planted'] for r in recent)/len(recent):.3f} "
                  f"epsilon={epsilon:.3f} steps={steps} seconds={time.monotonic()-start:.1f}", flush=True)
        if episode % args.eval_interval == 0 or episode == args.episodes:
            previous_consume = session.consume
            session.consume = lambda *a: None
            previous_py, previous_np = random.getstate(), np.random.get_state()
            evaluation = [session.run_episode(args.eval_seed + i, augment=False)
                          for i in range(args.eval_episodes)]
            session.consume = previous_consume
            random.setstate(previous_py)
            np.random.set_state(previous_np)
            metrics = {"episodes": len(evaluation), "seed": args.eval_seed,
                       "plant_rate": sum(r['planted'] for r in evaluation)/len(evaluation),
                       "registered_plant_rate": sum(r['registered'] for r in evaluation)/len(evaluation),
                       "entry_ticks": sum(r['ticks'] for r in evaluation)/len(evaluation),
                       "environment": "actual_engine_5v5_iq_macro_escort"}
            score = (metrics["plant_rate"], metrics["registered_plant_rate"], -metrics["entry_ticks"])
            save("dqn_attacker_carry_gc_latest.pt", episode, metrics)
            if score > best:
                best = score
                save("dqn_attacker_carry_gc_best_by_eval.pt", episode, metrics)
            print("[REAL EVAL] " + json.dumps(metrics), flush=True)
    (args.output_dir / "training_history.json").write_text(json.dumps(history), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026091700)
    parser.add_argument("--eval-seed", type=int, default=2026091900)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()
    if min(args.episodes, args.eval_episodes, args.eval_interval) < 1:
        parser.error("episode counts and intervals must be positive")
    if not 0 < args.gamma <= 1 or args.lr <= 0:
        parser.error("gamma must be in (0, 1] and learning rate must be positive")
    train(args)


if __name__ == "__main__":
    main()
