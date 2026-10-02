"""Evaluate ConCon attacker routes in fresh, single-round games against real defenders."""

import argparse
import contextlib
import hashlib
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from party_presets import get_preset
from run_game import VisualFPSBattle, _build_team_ai
from concon_v1.co1_attacker_controller import ConconAttackerController
from concon_v1.co1_battle_training import _run_from_project_root
from concon_v1.co1_learn_attacker import DEFAULT_MODEL_PATH

from concon_v1.co1_attacker_scenarios import SCENARIOS, get_scenario, validate_checkpoint_scenario


DEFAULT_ROUNDS = 36

OPPONENTS = {
    "omoko_v1": ("omoko_gaming_v1", "Omoko Gaming"),
    "touyama_v2": ("touyama_gaming_v2", "Touyama Gaming"),
    "fnatic_v3": ("fnatic_v3", "Fnatic2023"),
    "gc_v1": ("gc_v1", "Ghost Champions"),
    # Toru AI is a controller, with no dedicated roster preset.
    "toru_ai_v3.1": ("toru_ai_v3.1", "Team Elites"),
}


class LimitedRoundBattle(VisualFPSBattle):
    def move_character(self, char):
        was_planted = self.is_planted
        had_spike = char.team == "A" and char.has_spike and not was_planted
        result = super().move_character(char)
        if had_spike and char.is_alive and not self.is_planted:
            row, col = map(int, char.pos)
            if self.grid[row, col] == 2 and getattr(self, "first_site_tick", None) is None:
                self.first_site_tick = self.battle_tick
            self.max_plant_progress = max(
                getattr(self, "max_plant_progress", 0), int(char.plant_timer)
            )
        if not was_planted and self.is_planted:
            self.plant_alive_counts = {
                "A": sum(c.is_alive for c in self.chars if c.team == "A"),
                "D": sum(c.is_alive for c in self.chars if c.team == "D"),
            }
            self.plant_tick = self.battle_tick
        return result

    def _record_spike_drop(self, carrier):
        self.spike_drop_count = getattr(self, "spike_drop_count", 0) + 1
        if getattr(self, "spike_drop_tick", None) is None:
            self.spike_drop_tick = self.battle_tick
            self.spike_drop_pos = [int(value) for value in carrier.pos]
            route = self.attacker_controller.route_controller._routes.get(carrier.name)
            self.spike_drop_goal = (
                [int(value) for value in route.goal] if route is not None else None
            )

    def _kill_character(self, shooter, target, *, credit_kill=True, during_battle=True):
        if target.team == "A" and target.has_spike and not self.is_planted:
            self._record_spike_drop(target)
        return super()._kill_character(
            shooter, target, credit_kill=credit_kill, during_battle=during_battle
        )

    def process_battle(self):
        # Some status effects mark the carrier dead without calling _kill_character.
        # The base battle process drops that carrier's spike later in this tick.
        if not self.is_planted:
            for carrier in self.chars:
                if carrier.team == "A" and not carrier.is_alive and carrier.has_spike:
                    self._record_spike_drop(carrier)
        return super().process_battle()

    def check_match_winner(self):
        previous_attacker_wins = self.previous_attacker_wins
        # Each evaluation game contains one round. The normal match handler
        # would otherwise continue until a team reaches its win threshold.
        attacker_alive = sum(c.is_alive for c in self.chars if c.team == "A")
        defender_alive = sum(c.is_alive for c in self.chars if c.team == "D")
        if self.is_defused:
            end_reason = "defused"
        elif self.is_planted:
            end_reason = "detonated" if self.detonate_timer <= 0 else "defender_eliminated"
        elif self.round_timer <= 0:
            end_reason = "time_expired"
        else:
            end_reason = "attacker_eliminated" if not attacker_alive else "defender_eliminated"
        self.round_results.append({
            "winner": "A" if self.attacker_wins > previous_attacker_wins else "D",
            "planted": bool(self.is_planted),
            "end_reason": end_reason,
            "end_tick": self.battle_tick,
            "plant_tick": getattr(self, "plant_tick", None),
            "route_pattern": self.attacker_controller.route_controller._pattern_index,
            "first_site_tick": getattr(self, "first_site_tick", None),
            "max_plant_progress": getattr(self, "max_plant_progress", 0),
            "spike_drop_tick": getattr(self, "spike_drop_tick", None),
            "spike_drop_count": getattr(self, "spike_drop_count", 0),
            "spike_drop_pos": getattr(self, "spike_drop_pos", None),
            "spike_drop_goal": getattr(self, "spike_drop_goal", None),
            "planted_pos": (
                [int(value) for value in self.planted_pos]
                if self.planted_pos is not None else None
            ),
            "attacker_alive_at_end": attacker_alive,
            "defender_alive_at_end": defender_alive,
            "attacker_kills": sum(int(c.round_kills) for c in self.chars if c.team == "A"),
            "defender_kills": sum(int(c.round_kills) for c in self.chars if c.team == "D"),
            "attacker_alive_at_plant": (
                self.plant_alive_counts["A"] if getattr(self, "plant_alive_counts", None) else None
            ),
            "defender_alive_at_plant": (
                self.plant_alive_counts["D"] if getattr(self, "plant_alive_counts", None) else None
            ),
        })
        # Preserve normal round-end bookkeeping, but keep this first-round game
        # from transitioning to a second round before its result is recorded.
        self.stop_after_round = True
        super().check_match_winner()
        self.plant_alive_counts = None
        self.plant_tick = None
        self.first_site_tick = None
        self.max_plant_progress = 0
        self.spike_drop_tick = None
        self.spike_drop_count = 0
        self.spike_drop_pos = None
        self.spike_drop_goal = None
        self.previous_attacker_wins = self.attacker_wins
        if len(self.round_results) >= self.round_limit:
            self.match_over = True
        else:
            self.current_round += 1
            self.init_round()


def summarize_spike_drops(round_results):
    """Count drop events across all rounds, independently of plant outcomes."""
    drop_counts = [int(record.get("spike_drop_count", 0)) for record in round_results]
    return {
        "spike_drop_events": sum(drop_counts),
        "spike_drop_rounds": sum(count > 0 for count in drop_counts),
        "plants_after_drop": sum(
            count > 0 and bool(record["planted"])
            for count, record in zip(drop_counts, round_results)
        ),
        "no_plant_after_drop": sum(
            count > 0 and not record["planted"]
            for count, record in zip(drop_counts, round_results)
        ),
    }


@_run_from_project_root
def evaluate(opponent, rounds=3, seed=0, model_path=None,
             frozen_checkpoint=None, map_name="A1"):
    scenario = get_scenario(map_name)
    model_path = scenario.model_path if model_path is None else Path(model_path)
    if opponent not in OPPONENTS:
        raise ValueError(f"unknown opponent: {opponent}")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    # Every trial creates a controller. Keep its weights fixed even if a
    # concurrent training process replaces the checkpoint between trials.
    checkpoint_bytes = (
        bytes(frozen_checkpoint) if frozen_checkpoint is not None
        else Path(model_path).read_bytes()
    )
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu",
                            weights_only=False)
    validate_checkpoint_scenario(checkpoint, scenario)
    model_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    ai_key, preset_name = OPPONENTS[opponent]
    attackers = get_preset("Gorigons")
    defenders = get_preset(preset_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    route_rng = random.Random(seed)
    results = []
    for trial in range(1, rounds + 1):
        # Match training's one-new-game-per-episode structure. Keep the RNG
        # streams advancing so the trials do not repeat the same setup.
        with contextlib.redirect_stdout(io.StringIO()):
            attacker_ai = _build_team_ai("concon_v1")
            attacker_ai.attacker_factory = lambda: ConconAttackerController(
                checkpoint_bytes=checkpoint_bytes, map_name=scenario
            )
            game = LimitedRoundBattle(
                scenario.game_map,
                attacker_ai,
                _build_team_ai(ai_key),
                headless=True,
                attacker_roster=list(attackers.players),
                defender_roster=list(defenders.players),
                spike_holder_name=attackers.spike_holder,
                defender_spike_holder_name=defenders.spike_holder,
                attacker_igl_name=attackers.igl,
                defender_igl_name=defenders.igl,
                attacker_team_name=attackers.name,
                defender_team_name=defenders.name,
                disable_side_swap=True,
            )
            game.round_limit = 1
            game.round_results = []
            game.previous_attacker_wins = 0
            game.analytics_tracker = None
            controller_rng = game.attacker_controller.route_controller.rng
            controller_rng.setstate(route_rng.getstate())
            game.run()
            route_rng.setstate(controller_rng.getstate())
        if len(game.round_results) != 1:
            raise RuntimeError("evaluation game did not finish exactly one round")
        results.append({**game.round_results[0], "trial": trial})
    plant_count = sum(r["planted"] for r in results)
    plant_results = [r for r in results if r["attacker_alive_at_plant"] is not None]
    no_plant_results = [r for r in results if not r["planted"]]
    drop_summary = summarize_spike_drops(results)
    end_reasons = dict(Counter(r["end_reason"] for r in results))
    no_plant_reasons = dict(Counter(r["end_reason"] for r in no_plant_results))
    return {
        "map_name": scenario.map_name,
        "opponent": opponent,
        "roster": preset_name,
        "model_episode": checkpoint.get("episode"),
        "model_success100_at_save": checkpoint.get("success_rate"),
        "model_attacker_perception": checkpoint.get("attacker_perception", "legacy_unknown"),
        "model_sha256": model_sha256,
        "rounds": len(results),
        "attacker_wins": sum(r["winner"] == "A" for r in results),
        "plants": plant_count,
        "plant_success_rate": plant_count / len(results) if results else 0.0,
        "avg_attacker_alive_at_plant": (
            sum(r["attacker_alive_at_plant"] for r in plant_results) / len(plant_results)
            if plant_results else None
        ),
        "avg_defender_alive_at_plant": (
            sum(r["defender_alive_at_plant"] for r in plant_results) / len(plant_results)
            if plant_results else None
        ),
        "attacker_kills": sum(r["attacker_kills"] for r in results),
        "defender_kills": sum(r["defender_kills"] for r in results),
        "end_reasons": end_reasons,
        "no_plant_reasons": no_plant_reasons,
        **drop_summary,
        "no_plant_site_reached": sum(r["first_site_tick"] is not None for r in no_plant_results),
        "no_plant_started": sum(r["max_plant_progress"] > 0 for r in no_plant_results),
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-map", "--map", dest="map_name", choices=SCENARIOS, default="A1")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help="number of independent single-round games per opponent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path,
                        help="checkpoint to freeze for every opponent and trial")
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS,
                        default=list(OPPONENTS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    scenario = get_scenario(args.map_name)
    model_path = args.model if args.model is not None else scenario.model_path
    frozen_checkpoint = model_path.read_bytes()
    results = []
    for opponent in args.opponents:
        result = evaluate(opponent, args.rounds, args.seed,
                          frozen_checkpoint=frozen_checkpoint, map_name=scenario)
        results.append(result)
        print(f"map={result['map_name']} model episode={result['model_episode']} "
              f"success100_at_save={result['model_success100_at_save']} "
              f"attacker_perception={result['model_attacker_perception']} "
              f"sha256={result['model_sha256'][:12]}", flush=True)
        plant_alive = (
            f"{result['avg_attacker_alive_at_plant']:.2f}:"
            f"{result['avg_defender_alive_at_plant']:.2f}"
            if result["avg_attacker_alive_at_plant"] is not None else "-:-"
        )
        print(f"{opponent}: plant success {result['plants']}/{result['rounds']} "
              f"({result['plant_success_rate']:.1%}), "
              f"avg alive at plant A:D {plant_alive}",
              flush=True)
        print(f"  spike drop events: {result['spike_drop_events']}, "
              f"rounds with drop: {result['spike_drop_rounds']}; "
              f"planted after drop: {result['plants_after_drop']}, "
              f"no plant after drop: {result['no_plant_after_drop']}",
              flush=True)
        print(f"  no plant: {len(result['details']) - result['plants']} rounds, "
              f"site reached {result['no_plant_site_reached']}, "
              f"plant started {result['no_plant_started']}; "
              f"end reasons {result['no_plant_reasons']}",
              flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                               encoding="utf-8")


if __name__ == "__main__":
    main()
