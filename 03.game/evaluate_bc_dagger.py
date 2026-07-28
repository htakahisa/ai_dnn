"""
BC / DAgger モデル実戦評価スクリプト

実行:
    python evaluate_bc_dagger.py
    python evaluate_bc_dagger.py --matches 30
    python evaluate_bc_dagger.py --models policy_dagger_final.pt
"""

from __future__ import annotations

import argparse
import inspect
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from controllers import DefaultAttackerController, DefaultDefenderController
from dagger_train import (
    DIRECTION_BY_ACTION,
    choose_ability_target,
    get_game_observation,
    load_policy,
    observation_to_vector,
)
from map_data import NEW_MAZE_STR
from roster_utils import build_two_balanced_rosters
from run_game import VisualFPSBattle


ACTION_NAMES = {
    0: "MOVE_UP",
    1: "MOVE_DOWN",
    2: "MOVE_LEFT",
    3: "MOVE_RIGHT",
    4: "MOVE_UP_LEFT",
    5: "MOVE_UP_RIGHT",
    6: "MOVE_DOWN_LEFT",
    7: "MOVE_DOWN_RIGHT",
    8: "SMOKE",
    9: "FLASH",
    10: "RECON",
    11: "STOP",
    12: "PLANT",
}

ABILITY_BY_ACTION = {8: "SMOKE", 9: "FLASH", 10: "RECON"}
GROUP_ORDER = ["MOVE", "STOP", "PLANT", "SMOKE", "FLASH", "RECON", "UNKNOWN"]


def group_action_index(action_idx: int) -> str:
    if action_idx in DIRECTION_BY_ACTION:
        return "MOVE"
    return {8: "SMOKE", 9: "FLASH", 10: "RECON", 11: "STOP", 12: "PLANT"}.get(
        action_idx, "UNKNOWN"
    )


def group_controller_result(char: Any, result: Any) -> str:
    if result is None:
        return "UNKNOWN"
    if isinstance(result, tuple) and len(result) == 2:
        second = result[1]
        if isinstance(second, dict):
            ability = str(second.get("ability", "")).upper()
            return ability if ability in {"SMOKE", "FLASH", "RECON"} else "UNKNOWN"
        if isinstance(second, str):
            return "PLANT" if second.upper() == "PLANT" else "UNKNOWN"
    try:
        move_pos = result[0] if isinstance(result, tuple) else result
        return "STOP" if list(map(int, move_pos)) == list(map(int, char.pos)) else "MOVE"
    except (TypeError, ValueError, IndexError):
        return "UNKNOWN"


def character_is_alive(char: Any) -> bool:
    return bool(getattr(char, "is_alive", True))


def decode_evaluation_action(
    action_idx: int,
    char: Any,
    game_state: dict,
    expert_result: Any,
) -> Any:
    grid = game_state["grid"]

    if action_idx in DIRECTION_BY_ACTION:
        dr, dc = DIRECTION_BY_ACTION[action_idx]
        nr = int(char.pos[0] + dr)
        nc = int(char.pos[1] + dc)

        valid = (
            0 <= nr < grid.shape[0]
            and 0 <= nc < grid.shape[1]
            and grid[nr, nc] != 1
        )
        if valid:
            for other in game_state.get("chars", []):
                if other is char or not character_is_alive(other):
                    continue
                if int(other.pos[0]) == nr and int(other.pos[1]) == nc:
                    valid = False
                    break
        return [nr, nc] if valid else list(char.pos)

    if action_idx in ABILITY_BY_ACTION:
        ability_name = ABILITY_BY_ACTION[action_idx]
        charge_name = {
            "SMOKE": "smoke_charges",
            "FLASH": "flash_charges",
            "RECON": "recon_charges",
        }[ability_name]
        if getattr(char, charge_name, 0) <= 0:
            return list(char.pos)
        target = choose_ability_target(ability_name, char, game_state, expert_result)
        return list(char.pos), {"ability": ability_name, "target": target}

    if action_idx == 12:
        return list(char.pos), "PLANT"

    return list(char.pos)


@dataclass
class ControllerStatistics:
    predicted: Counter = field(default_factory=Counter)
    executed: Counter = field(default_factory=Counter)
    detailed: Counter = field(default_factory=Counter)
    total_predictions: int = 0
    predicted_moves: int = 0
    invalid_moves: int = 0
    confidence_sum: float = 0.0
    probability_sums: np.ndarray = field(
        default_factory=lambda: np.zeros(13, dtype=np.float64)
    )
    plant_predictions: int = 0
    plant_executions: int = 0
    observed_plant_transitions: int = 0

    def merge(self, other: "ControllerStatistics") -> None:
        self.predicted.update(other.predicted)
        self.executed.update(other.executed)
        self.detailed.update(other.detailed)
        self.total_predictions += other.total_predictions
        self.predicted_moves += other.predicted_moves
        self.invalid_moves += other.invalid_moves
        self.confidence_sum += other.confidence_sum
        self.probability_sums += other.probability_sums
        self.plant_predictions += other.plant_predictions
        self.plant_executions += other.plant_executions
        self.observed_plant_transitions += other.observed_plant_transitions


class EvaluationAttackerController:
    """学習済みモデルだけでアタッカーを操作し、行動統計を収集する。"""

    def __init__(self, model: torch.nn.Module, obs_size: int, device: str) -> None:
        self.model = model
        self.obs_size = int(obs_size)
        self.device = device
        self.target_helper = DefaultAttackerController()
        self.game = None
        self.stats = ControllerStatistics()
        self._plant_seen_this_round = False
        self._last_round_marker = None

    def set_game(self, game: VisualFPSBattle) -> None:
        self.game = game

    def reset_round(self) -> None:
        if hasattr(self.target_helper, "reset_round"):
            self.target_helper.reset_round()
        self._plant_seen_this_round = False
        self._last_round_marker = getattr(self.game, "current_round", None)

    def _update_real_plant_transition(self) -> None:
        if self.game is None:
            return
        current_round = getattr(self.game, "current_round", None)
        if current_round != self._last_round_marker:
            self._last_round_marker = current_round
            self._plant_seen_this_round = False
        if bool(getattr(self.game, "is_planted", False)) and not self._plant_seen_this_round:
            self.stats.observed_plant_transitions += 1
            self._plant_seen_this_round = True

    def _is_invalid_move(self, action_idx: int, char: Any, game_state: dict) -> bool:
        if action_idx not in DIRECTION_BY_ACTION:
            return False
        grid = game_state["grid"]
        dr, dc = DIRECTION_BY_ACTION[action_idx]
        nr = int(char.pos[0] + dr)
        nc = int(char.pos[1] + dc)
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            return True
        if grid[nr, nc] == 1:
            return True
        for other in game_state.get("chars", []):
            if other is char or not character_is_alive(other):
                continue
            if int(other.pos[0]) == nr and int(other.pos[1]) == nc:
                return True
        return False

    def decide_move(self, char: Any, game_state: dict) -> Any:
        if self.game is None:
            raise RuntimeError("EvaluationAttackerController.set_game(game) が未実行です。")

        self._update_real_plant_transition()
        observation = get_game_observation(self.game, char)
        obs_vec = observation_to_vector(observation)
        if obs_vec.shape[0] != self.obs_size:
            raise ValueError(
                f"観測次元がモデルと一致しません: observation={obs_vec.shape[0]}, "
                f"model={self.obs_size}"
            )

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_vec).unsqueeze(0).to(self.device)
            logits = self.model(obs_tensor)
            probabilities = torch.softmax(logits, dim=1)
            action_idx = int(probabilities.argmax(dim=1).item())
            confidence = float(probabilities[0, action_idx].item())
            probability_array = probabilities[0].detach().cpu().numpy()

        action_group = group_action_index(action_idx)
        self.stats.total_predictions += 1
        self.stats.predicted[action_group] += 1
        self.stats.detailed[ACTION_NAMES.get(action_idx, f"UNKNOWN_{action_idx}")] += 1
        self.stats.confidence_sum += confidence
        if len(probability_array) == 13:
            self.stats.probability_sums += probability_array

        if action_idx in DIRECTION_BY_ACTION:
            self.stats.predicted_moves += 1
            if self._is_invalid_move(action_idx, char, game_state):
                self.stats.invalid_moves += 1
        if action_idx == 12:
            self.stats.plant_predictions += 1

        # 現モデルは標的座標を出力しないため、アビリティ標的だけ既存AIで補う。
        expert_result = self.target_helper.decide_move(char, game_state)
        result = decode_evaluation_action(action_idx, char, game_state, expert_result)
        executed_group = group_controller_result(char, result)
        self.stats.executed[executed_group] += 1
        if executed_group == "PLANT":
            self.stats.plant_executions += 1
        return result


@dataclass
class ModelEvaluationResult:
    model_path: str
    matches: int = 0
    attacker_match_wins: int = 0
    defender_match_wins: int = 0
    draws_or_unknown: int = 0
    attacker_rounds: int = 0
    defender_rounds: int = 0
    attacker_kills: int = 0
    attacker_deaths: int = 0
    attacker_13_0: int = 0
    defender_13_0: int = 0
    matches_with_plant: int = 0
    controller_stats: ControllerStatistics = field(default_factory=ControllerStatistics)

    @property
    def win_rate(self) -> float:
        return self.attacker_match_wins / self.matches if self.matches else 0.0

    @property
    def average_attacker_rounds(self) -> float:
        return self.attacker_rounds / self.matches if self.matches else 0.0

    @property
    def average_defender_rounds(self) -> float:
        return self.defender_rounds / self.matches if self.matches else 0.0

    @property
    def average_round_difference(self) -> float:
        return self.average_attacker_rounds - self.average_defender_rounds

    @property
    def average_attacker_kills(self) -> float:
        return self.attacker_kills / self.matches if self.matches else 0.0

    @property
    def average_attacker_deaths(self) -> float:
        return self.attacker_deaths / self.matches if self.matches else 0.0


class ModelEvaluator:
    """複数のBC / DAggerモデルをHeadless試合で比較する。"""

    def __init__(
        self,
        model_paths: list[Path],
        matches_per_model: int = 20,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        if matches_per_model <= 0:
            raise ValueError("matches_per_modelは1以上にしてください。")
        self.model_paths = [Path(path) for path in model_paths]
        self.matches_per_model = int(matches_per_model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(seed)

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _make_game(self, controller: EvaluationAttackerController) -> VisualFPSBattle:
        attacker_roster, defender_roster = build_two_balanced_rosters()
        kwargs = {
            "headless": True,
            "attacker_roster": attacker_roster,
            "defender_roster": defender_roster,
        }
        signature = inspect.signature(VisualFPSBattle.__init__)
        if "disable_side_swap" in signature.parameters:
            kwargs["disable_side_swap"] = True
        game = VisualFPSBattle(
            NEW_MAZE_STR,
            controller,
            DefaultDefenderController(),
            **kwargs,
        )
        controller.set_game(game)
        return game

    def _extract_team_kd(self, game: VisualFPSBattle, team: str) -> tuple[int, int]:
        names = {
            str(getattr(char, "name", ""))
            for char in getattr(game, "chars", [])
            if getattr(char, "team", None) == team
        }
        kills = deaths = 0
        match_stats = getattr(game, "match_stats", {})
        if isinstance(match_stats, dict):
            for name in names:
                data = match_stats.get(name, {})
                if isinstance(data, dict):
                    kills += int(data.get("kills", 0))
                    deaths += int(data.get("deaths", 0))
        return kills, deaths

    def evaluate_model(self, model_path: Path) -> ModelEvaluationResult:
        if not model_path.exists():
            raise FileNotFoundError(f"モデルがありません: {model_path}")
        model, obs_size = load_policy(model_path, self.device)
        result = ModelEvaluationResult(model_path=str(model_path))

        print("\n" + "=" * 76)
        print(f"評価開始: {model_path}")
        print(f"device={self.device} matches={self.matches_per_model}")
        print("=" * 76)

        for match_index in range(self.matches_per_model):
            self._seed_everything(self.seed + match_index)
            controller = EvaluationAttackerController(model, obs_size, self.device)
            game = self._make_game(controller)
            game.run_headless_loop()
            controller._update_real_plant_transition()

            attacker_score = int(getattr(game, "attacker_wins", 0))
            defender_score = int(getattr(game, "defender_wins", 0))
            result.matches += 1
            result.attacker_rounds += attacker_score
            result.defender_rounds += defender_score

            if attacker_score > defender_score:
                result.attacker_match_wins += 1
            elif defender_score > attacker_score:
                result.defender_match_wins += 1
            else:
                result.draws_or_unknown += 1

            result.attacker_13_0 += int(attacker_score == 13 and defender_score == 0)
            result.defender_13_0 += int(defender_score == 13 and attacker_score == 0)
            result.matches_with_plant += int(controller.stats.observed_plant_transitions > 0)

            kills, deaths = self._extract_team_kd(game, "A")
            result.attacker_kills += kills
            result.attacker_deaths += deaths
            result.controller_stats.merge(controller.stats)

            print(
                f"[{match_index + 1:>3}/{self.matches_per_model}] "
                f"A {attacker_score:>2} - {defender_score:<2} D | "
                f"plant={controller.stats.observed_plant_transitions} | "
                f"invalid={controller.stats.invalid_moves}/"
                f"{controller.stats.predicted_moves}"
            )
        return result

    def evaluate_all(self) -> list[ModelEvaluationResult]:
        return [self.evaluate_model(path) for path in self.model_paths]

    def _print_counter(self, title: str, counter: Counter, total: int) -> None:
        print(f"\n{title}\n" + "-" * 60)
        for name in GROUP_ORDER:
            count = int(counter.get(name, 0))
            percentage = count / total * 100.0 if total else 0.0
            print(f"{name:<18}: {count:>9} ({percentage:>6.2f}%)")

    def print_result(self, result: ModelEvaluationResult) -> None:
        stats = result.controller_stats
        total = stats.total_predictions
        print("\n" + "=" * 76)
        print(f"MODEL: {result.model_path}")
        print("=" * 76)
        print(f"試合数                  : {result.matches}")
        print(f"アタッカー勝利          : {result.attacker_match_wins}")
        print(f"ディフェンダー勝利      : {result.defender_match_wins}")
        print(f"勝率                    : {result.win_rate * 100.0:.2f}%")
        print(f"平均取得ラウンド        : {result.average_attacker_rounds:.2f}")
        print(f"平均失点ラウンド        : {result.average_defender_rounds:.2f}")
        print(f"平均ラウンド差          : {result.average_round_difference:+.2f}")
        print(f"アタッカー13-0          : {result.attacker_13_0}")
        print(f"ディフェンダー13-0      : {result.defender_13_0}")
        print(f"平均アタッカーキル      : {result.average_attacker_kills:.2f}")
        print(f"平均アタッカーデス      : {result.average_attacker_deaths:.2f}")
        print(f"設置が確認された試合    : {result.matches_with_plant}/{result.matches}")

        self._print_counter("Model Predicted Actions", stats.predicted, total)
        executed_total = sum(stats.executed.values())
        self._print_counter("Executed Actions", stats.executed, executed_total)

        invalid_rate = stats.invalid_moves / stats.predicted_moves if stats.predicted_moves else 0.0
        average_confidence = stats.confidence_sum / total if total else 0.0
        print("\nInvalid Move Statistics\n" + "-" * 60)
        print(f"Predicted MOVE          : {stats.predicted_moves}")
        print(f"Invalid MOVE            : {stats.invalid_moves}")
        print(f"Invalid / MOVE          : {invalid_rate * 100.0:.2f}%")
        print("\nPlant / Confidence\n" + "-" * 60)
        print(f"PLANT predictions       : {stats.plant_predictions}")
        print(f"PLANT executions        : {stats.plant_executions}")
        print(f"Observed plant rounds   : {stats.observed_plant_transitions}")
        print(f"Average max probability : {average_confidence:.4f}")

        if total:
            average_probabilities = stats.probability_sums / total
            print("\nAverage Probability by Action\n" + "-" * 60)
            for action_idx in range(13):
                print(f"{ACTION_NAMES[action_idx]:<18}: {average_probabilities[action_idx]:.4f}")

    def print_comparison(self, results: list[ModelEvaluationResult]) -> None:
        print("\n" + "=" * 100)
        print("MODEL COMPARISON")
        print("=" * 100)
        print(
            f"{'Model':<32}{'Win%':>8}{'Avg A':>8}{'Avg D':>8}{'Diff':>8}"
            f"{'Invalid%':>10}{'PlantMatch%':>13}{'Confidence':>12}"
        )
        print("-" * 100)
        for result in results:
            stats = result.controller_stats
            invalid_rate = stats.invalid_moves / stats.predicted_moves if stats.predicted_moves else 0.0
            plant_match_rate = result.matches_with_plant / result.matches if result.matches else 0.0
            confidence = stats.confidence_sum / stats.total_predictions if stats.total_predictions else 0.0
            print(
                f"{Path(result.model_path).name:<32}"
                f"{result.win_rate * 100.0:>7.2f}%"
                f"{result.average_attacker_rounds:>8.2f}"
                f"{result.average_defender_rounds:>8.2f}"
                f"{result.average_round_difference:>+8.2f}"
                f"{invalid_rate * 100.0:>9.2f}%"
                f"{plant_match_rate * 100.0:>12.2f}%"
                f"{confidence:>12.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        type=Path,
        default=[
            Path("policy_bc_final.pt"),
            Path("policy_dagger_iter_1.pt"),
            Path("policy_dagger_iter_2.pt"),
            Path("policy_dagger_iter_3.pt"),
            Path("policy_dagger_final.pt"),
        ],
    )
    parser.add_argument("--matches", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing_models = [path for path in args.models if path.exists()]
    for path in args.models:
        if not path.exists():
            print(f"スキップ: モデルがありません: {path}")
    if not existing_models:
        raise FileNotFoundError("評価可能なモデルが1つもありません。")

    evaluator = ModelEvaluator(
        model_paths=existing_models,
        matches_per_model=args.matches,
        device=args.device,
        seed=args.seed,
    )
    results = evaluator.evaluate_all()
    for result in results:
        evaluator.print_result(result)
    evaluator.print_comparison(results)


if __name__ == "__main__":
    main()
