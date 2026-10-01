"""Evaluate the ConCon A1 attacker against real defender controllers."""

import argparse
import contextlib
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from map_data import NEW_MAZE_STR
from party_presets import get_preset
from run_game import VisualFPSBattle, _build_team_ai
from concon_v1.co1_battle_training import _run_from_project_root


DEFAULT_ROUNDS = 12

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

    def _kill_character(self, shooter, target, *, credit_kill=True, during_battle=True):
        if (target.team == "A" and target.has_spike and not self.is_planted
                and getattr(self, "spike_drop_tick", None) is None):
            self.spike_drop_tick = self.battle_tick
            self.spike_drop_pos = [int(value) for value in target.pos]
            route = self.attacker_controller.route_controller._routes.get(target.name)
            self.spike_drop_goal = (
                [int(value) for value in route.goal] if route is not None else None
            )
        return super()._kill_character(
            shooter, target, credit_kill=credit_kill, during_battle=during_battle
        )

    def check_match_winner(self):
        previous_attacker_wins = self.previous_attacker_wins
        self.stop_after_round = True
        super().check_match_winner()
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
        self.plant_alive_counts = None
        self.plant_tick = None
        self.first_site_tick = None
        self.max_plant_progress = 0
        self.spike_drop_tick = None
        self.spike_drop_pos = None
        self.spike_drop_goal = None
        self.previous_attacker_wins = self.attacker_wins
        if len(self.round_results) >= self.round_limit:
            self.match_over = True
        else:
            self.current_round += 1
            self.init_round()


@_run_from_project_root
def evaluate(opponent, rounds=3, seed=0):
    if opponent not in OPPONENTS:
        raise ValueError(f"unknown opponent: {opponent}")
    ai_key, preset_name = OPPONENTS[opponent]
    attackers = get_preset("Gorigons")
    defenders = get_preset(preset_name)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    with contextlib.redirect_stdout(io.StringIO()):
        game = LimitedRoundBattle(
            NEW_MAZE_STR,
            _build_team_ai("concon_v1"),
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
        game.round_limit = rounds
        game.round_results = []
        game.previous_attacker_wins = 0
        game.attacker_controller.route_controller.rng.seed(seed)
        game.run()
    results = game.round_results
    plant_count = sum(r["planted"] for r in results)
    plant_results = [r for r in results if r["attacker_alive_at_plant"] is not None]
    no_plant_results = [r for r in results if not r["planted"]]
    end_reasons = dict(Counter(r["end_reason"] for r in results))
    no_plant_reasons = dict(Counter(r["end_reason"] for r in no_plant_results))
    return {
        "opponent": opponent,
        "roster": preset_name,
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
        "no_plant_carrier_drops": sum(r["spike_drop_tick"] is not None for r in no_plant_results),
        "no_plant_site_reached": sum(r["first_site_tick"] is not None for r in no_plant_results),
        "no_plant_started": sum(r["max_plant_progress"] > 0 for r in no_plant_results),
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS,
                        default=list(OPPONENTS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    results = []
    for opponent in args.opponents:
        result = evaluate(opponent, args.rounds, args.seed)
        results.append(result)
        plant_alive = (
            f"{result['avg_attacker_alive_at_plant']:.2f}:"
            f"{result['avg_defender_alive_at_plant']:.2f}"
            if result["avg_attacker_alive_at_plant"] is not None else "-:-"
        )
        print(f"{opponent}: plant success {result['plants']}/{result['rounds']} "
              f"({result['plant_success_rate']:.1%}), "
              f"avg alive at plant A:D {plant_alive}",
              flush=True)
        print(f"  no plant: {len(result['details']) - result['plants']} rounds, "
              f"carrier dropped {result['no_plant_carrier_drops']}, "
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
