from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import torch

from gc_v1.train_defender_setup_gc import (
    SetupQNet,
    TrainingGCDefender,
    _build_training_attacker,
    _ghost_champions_roster,
    _resolve_opponent,
    resolve_device,
    OBS_DIM,
    ACTION_DIM,
    CANDIDATES,
    GC_ROSTER_ORDER,
)
from controllers import DefaultAttackerController, DefaultDefenderController
from map_data import NEW_MAZE_STR
from roster_utils import build_two_balanced_rosters
from run_game import VisualFPSBattle
from team_ai import DualRoleTeamAI

import inspect
import random
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "defender_setup_gc_data"
DEFAULT_MODELS = (
    DATA_DIR / "dqn_defender_setup_gc_interrupt.pt",
    DATA_DIR / "dqn_defender_setup_gc_latest.pt",
    DATA_DIR / "dqn_defender_setup_gc_best.pt",
    DATA_DIR / "dqn_defender_setup_gc_final.pt",
)

def resolve_model_path(value):
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    for path in DEFAULT_MODELS:
        if path.exists():
            return path
    raise FileNotFoundError("No Setup checkpoint found.")

def load_model(path, device):
    data = torch.load(path, map_location=device, weights_only=False)
    if int(data.get("obs_dim", -1)) != OBS_DIM:
        raise RuntimeError(f"OBS_DIM mismatch: checkpoint={data.get('obs_dim')} current={OBS_DIM}")
    if int(data.get("action_dim", -1)) != ACTION_DIM:
        raise RuntimeError(f"ACTION_DIM mismatch: checkpoint={data.get('action_dim')} current={ACTION_DIM}")
    checkpoint_candidates = data.get("candidate_positions")
    if checkpoint_candidates is not None:
        normalized = [tuple(map(int, p)) for p in checkpoint_candidates]
        if normalized != list(CANDIDATES):
            raise RuntimeError("candidate_positions differ from current map")
    model = SetupQNet().to(device)
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    return model, {
        "episode": int(data.get("episode", 0)),
        "global_step": int(data.get("global_step", 0)),
        "best_win_rate": float(data.get("best_win_rate", -1.0)),
    }


class CachedEvaluationRuntime:
    """Heavy controllers are constructed once and reused across matches."""

    def __init__(self, model, device, seed, opponent_mode):
        self.model = model
        self.device = device
        self.rng = random.Random(seed)
        self.opponent_mode = opponent_mode

        # GC Defender is the expensive side: its constructor loads
        # Search / Retake / Opening checkpoints. Construct exactly once.
        self.defender = TrainingGCDefender(
            model=model,
            device=device,
            epsilon=0.0,
            rng=self.rng,
            transition_sink=lambda _t: None,
            greedy=True,
        )

        # Cache one attacker controller per opponent type used by this eval.
        self.attackers = {}

        self.attacker_roster, _ = build_two_balanced_rosters()
        self.defender_roster = _ghost_champions_roster()

    def _attacker(self, opponent_key):
        if opponent_key not in self.attackers:
            self.attackers[opponent_key] = _build_training_attacker(opponent_key)
        return self.attackers[opponent_key]

    def _prepare_defender_for_match(self):
        # These fields are match-local in TrainingGCDefender.
        self.defender._round_reward_total = 0.0
        self.defender._completed_rounds = 0
        self.defender._prev_attacker_wins = 0
        self.defender._prev_defender_wins = 0

        # Clear any leftover setup decision from the previous match.
        planner = self.defender.trainable_setup_planner
        planner.reset_round()

    def make_game(self, opponent_key):
        self._prepare_defender_for_match()

        attacker_controller = self._attacker(opponent_key)

        attacker_team = DualRoleTeamAI(
            name=f"SetupEval-A[{opponent_key}]",
            # Return the cached controller instead of reconstructing it.
            attacker_factory=lambda ctrl=attacker_controller: ctrl,
            defender_factory=lambda: DefaultDefenderController(),
            use_iq_perception=False,
        )
        defender_team = DualRoleTeamAI(
            name="SetupEval-GC-D",
            attacker_factory=lambda: DefaultAttackerController(),
            # Return the same GC Defender with already-loaded models.
            defender_factory=lambda: self.defender,
            use_iq_perception=False,
        )

        kwargs = {
            "headless": True,
            "attacker_roster": self.attacker_roster,
            "defender_roster": self.defender_roster,
        }
        if "disable_side_swap" in inspect.signature(
            VisualFPSBattle.__init__
        ).parameters:
            kwargs["disable_side_swap"] = True

        game = VisualFPSBattle(
            NEW_MAZE_STR,
            attacker_team,
            defender_team,
            **kwargs,
        )

        # Ensure both cached controllers now point at this new match.
        self.defender.set_game(game)
        if hasattr(attacker_controller, "set_game"):
            attacker_controller.set_game(game)

        return game

    def run_one(self, opponent_key):
        game = self.make_game(opponent_key)
        game.run_headless_loop()
        self.defender.finalize_match()

        a = int(getattr(game, "attacker_wins", 0))
        d = int(getattr(game, "defender_wins", 0))
        return {
            "attacker_rounds": a,
            "defender_rounds": d,
            "defender_win": int(d > a),
            "round_reward": float(self.defender.average_round_reward),
        }

    def evaluate(self, matches):
        self.model.eval()

        wins = 0
        rounds_a = []
        rounds_d = []
        rewards = []
        opponents = {}

        for _ in range(matches):
            opponent = _resolve_opponent(self.opponent_mode, self.rng)
            opponents[opponent] = opponents.get(opponent, 0) + 1

            result = self.run_one(opponent)
            wins += result["defender_win"]
            rounds_a.append(result["attacker_rounds"])
            rounds_d.append(result["defender_rounds"])
            rewards.append(result["round_reward"])

        return {
            "win_rate": wins / max(1, matches),
            "avg_rounds_a": float(np.mean(rounds_a)) if rounds_a else 0.0,
            "avg_rounds_d": float(np.mean(rounds_d)) if rounds_d else 0.0,
            "avg_round_reward": float(np.mean(rewards)) if rewards else 0.0,
            "opponents": opponents,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--matches", type=int, default=20)
    p.add_argument("--opponent", choices=("default", "toru_ai_v3", "mixed"), default="toru_ai_v3")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    model_path = resolve_model_path(args.model)
    model, meta = load_model(model_path, device)

    print("=" * 72)
    print("GC DEFENDER SETUP EVALUATION")
    print("=" * 72)
    print(f"model       : {model_path}")
    print(f"episode     : {meta['episode']}")
    print(f"global_step : {meta['global_step']}")
    print(f"bestWR(meta): {meta['best_win_rate']:.3f}")
    print(f"device      : {device}")
    print(f"matches     : {args.matches}")
    print(f"opponent    : {args.opponent}")
    print()

    # 評価中のGC内部ログは大量かつ低速なので破棄する。
    # 最終的な評価結果だけ表示する。
    # Construct expensive GC/Toru controllers once, then reuse them.
    # Their internal per-round state is still reset by the game normally.
    with open(os.devnull, "w", encoding="utf-8") as _devnull:
        with contextlib.redirect_stdout(_devnull):
            runtime = CachedEvaluationRuntime(
                model=model,
                device=device,
                seed=args.seed,
                opponent_mode=args.opponent,
            )
            metrics = runtime.evaluate(args.matches)

    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"win_rate        : {metrics['win_rate']:.3f}")
    print(f"avg_rounds_a    : {metrics['avg_rounds_a']:.3f}")
    print(f"avg_rounds_d    : {metrics['avg_rounds_d']:.3f}")
    print(f"avg_round_reward: {metrics['avg_round_reward']:.4f}")
    print(f"opponents       : {metrics['opponents']}")

if __name__ == "__main__":
    main()
