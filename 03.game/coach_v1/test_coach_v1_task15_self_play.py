"""Task 15 opponent-pool, self-play, privacy, and promotion tests."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from defender_setup_phase import DefenderSetupPhase

from coach_v1.common.constants import FIXED_ROSTER
from coach_v1.common.types import Side
from coach_v1.full_match import FullMatchResult, FullMatchSummary, run_headless_full_match
from coach_v1.opponent_pool import (
    OpponentKind, OpponentPool, OpponentSpec, load_opponent_pool,
)
from coach_v1.task15_self_play import (
    PoolEvaluation, PoolMatchRecord, PromotionCriteria, build_opponent_team,
    decide_promotion, evaluate_pool,
)
from coach_v1.team_ai import build_coach_v1_team
from coach_v1.training.coach_trainer import CoachTrainer
from coach_v1.training.opponent_pool_imitation import (
    RecordingTeacherPolicy, balanced_pool_samples, optimize_pool_samples,
    prepare_candidate_directory,
)
from coach_v1.training.opponent_pool_policy_gradient import (
    RoundRewardPolicy, _coach_round_results, optimize_round_rewards,
)
from coach_v1.test_coach_v1_task03_team_perception import FakeCharacter, FakeGame
from coach_v1.test_coach_v1_task14_full_match import FacingActor, StayCoach, stay_team
from coach_v1.perception.team_perception import TeamPerceptionBuilder
from map_data import NEW_MAZE_STR


def summary(seed: int, side: str, coach_score: int, opponent_score: int):
    value = FullMatchSummary(
        seed=seed, coach_started_as=side,
        coach_score=coach_score, opponent_score=opponent_score,
        winner="coach_v1" if coach_score > opponent_score else "opponent",
        rounds=coach_score + opponent_score, overtime=False, replay_frames=1,
        round_reasons={}, planted_rounds=0, defused_rounds=0,
        detonation_rounds=0, elimination_rounds=0, timeout_rounds=0,
        setup_rounds=1, live_rounds=1,
        attacker_decisions=1, defender_decisions=1,
        attacker_actions=5, defender_actions=5,
        memory_reset_ok=True, side_checkpoint_switch_ok=True, replay_ok=True,
    )
    return FullMatchResult(value, tuple(), tuple())


def evaluation(scores: tuple[tuple[str, int, str, int, int], ...]) -> PoolEvaluation:
    records = tuple(PoolMatchRecord(
        opponent_id=opponent, opponent_kind=(
            OpponentKind.HISTORICAL_COACH.value
            if opponent.startswith("history") else OpponentKind.EXISTING_AI.value
        ),
        difficulty=2, seed=seed, coach_started_as=side,
        coach_score=ours, opponent_score=theirs, rounds=ours + theirs,
        won=ours > theirs, memory_reset_ok=True,
        side_checkpoint_switch_ok=True, replay_ok=True,
    ) for opponent, seed, side, ours, theirs in scores)
    return PoolEvaluation("a.pt", "a" * 64, "d.pt", "d" * 64, records, {})


class OpponentPoolTest(unittest.TestCase):
    def test_default_pool_is_pinned_and_covers_required_opponents(self):
        pool = load_opponent_pool()
        self.assertTrue({"omoko_v1", "touyama_v2", "gc_v1"}.issubset(
            {item.opponent_id for item in pool.opponents}
        ))
        historical = [item for item in pool.opponents
                      if item.kind is OpponentKind.HISTORICAL_COACH]
        self.assertEqual(1, len(historical))
        historical[0].verify_checkpoints()
        self.assertNotIn("task12", str(historical[0].attacker_checkpoint))
        self.assertNotIn("task13", str(historical[0].defender_checkpoint))

    def test_sampling_is_seeded_and_weakness_weighted(self):
        specs = tuple(OpponentSpec(
            opponent_id=name, kind=OpponentKind.EXISTING_AI,
            preset="Ghost Champions", difficulty=1, weight=1.0, ai_key="default",
        ) for name in ("weak", "strong"))
        pool = OpponentPool(specs, require_coverage=False)
        first = pool.sample(seed=15, count=200,
                            point_rates={"weak": 0.0, "strong": 1.0})
        second = pool.sample(seed=15, count=200,
                             point_rates={"weak": 0.0, "strong": 1.0})
        self.assertEqual(first, second)
        self.assertGreater(sum(item.opponent_id == "weak" for item in first),
                           sum(item.opponent_id == "strong" for item in first))

    def test_pool_evaluation_alternates_sides_and_enables_only_historical_mirror(self):
        existing = OpponentSpec(
            "existing", OpponentKind.EXISTING_AI, "Ghost Champions", 1, 1.0,
            ai_key="default",
        )
        # Hash verification is handled separately by the default-pool test.
        historical = next(item for item in load_opponent_pool().opponents
                          if item.kind is OpponentKind.HISTORICAL_COACH)
        pool = OpponentPool((existing, historical), require_coverage=False)
        calls = []

        def runner(**kwargs):
            calls.append(kwargs)
            return summary(kwargs["seed"], kwargs["coach_starts_as"].value, 13, 8)

        attacker = Path("coach_v1/checkpoints/coach/attacker/task12/latest.pt")
        defender = Path("coach_v1/checkpoints/coach/defender/task13/latest.pt")
        with (
            patch("coach_v1.task15_self_play.build_coach_v1_team",
                  return_value=object()),
            patch("coach_v1.task15_self_play.build_opponent_team",
                  return_value=object()),
        ):
            result = evaluate_pool(
                attacker_checkpoint=attacker, defender_checkpoint=defender,
                pool=pool, seeds=(1, 2), match_runner=runner,
            )
        self.assertEqual(4, len(result.matches))
        self.assertEqual(
            [Side.ATTACKER, Side.DEFENDER, Side.ATTACKER, Side.DEFENDER],
            [item["coach_starts_as"] for item in calls],
        )
        self.assertEqual([False, False, True, True],
                         [item["allow_mirrored_roster"] for item in calls])


class SelfPlayAndPromotionTest(unittest.TestCase):
    def test_historical_opponent_actor_cannot_observe_hidden_enemy_position(self):
        spec = next(item for item in load_opponent_pool().opponents
                    if item.kind is OpponentKind.HISTORICAL_COACH)
        observations = []
        grid = np.array([[int(cell) for cell in line]
                         for line in NEW_MAZE_STR.strip().splitlines()], dtype=np.int8)
        for enemy_position in ((23, 6), (23, 7)):
            allies = [FakeCharacter(slot.character_name, "D", (1, 18 + slot.slot),
                                    facing="N") for slot in FIXED_ROSTER]
            enemy = FakeCharacter("hidden", "A", enemy_position, facing="S")
            game = FakeGame(grid, allies + [enemy])
            with (
                patch("coach_v1.team_ai.load_defender_coach",
                      return_value=StayCoach("historical")),
                patch("coach_v1.team_ai._CHARACTER_POLICY_TYPES", (FacingActor,) * 5),
            ):
                team = build_opponent_team(spec)
                team.bind_game(game)
                controller = team.get_defender_controller()
                controller.decide_move(allies[0], {})
            self.assertFalse(controller._snapshot.sightings)
            observations.append(controller.coach.calls[0])
        np.testing.assert_array_equal(observations[0][0], observations[1][0])
        np.testing.assert_array_equal(observations[0][1], observations[1][1])

    def test_real_short_mirrored_roster_match_uses_team_identity(self):
        roster = tuple(slot.character_name for slot in FIXED_ROSTER)

        def early_halftime(game):
            if not game.sides_swapped and game.current_round >= 2:
                game._swap_sides()
                game.sides_swapped = True

        with (
            patch("run_game.ROUND_DURATION_TICKS", 1),
            patch("battle_logic.WINNING_ROUNDS", 3),
            patch("run_game.VisualFPSBattle._swap_sides_if_needed", early_halftime),
            patch("run_game.DefenderSetupPhase",
                  side_effect=lambda: DefenderSetupPhase(setup_ticks=2)),
            patch.object(TeamPerceptionBuilder, "_visible_cells",
                         return_value=frozenset()),
            patch.object(TeamPerceptionBuilder, "_sighting_for", return_value=None),
        ):
            result = run_headless_full_match(
                coach_team_ai=stay_team(), opponent_team_ai=stay_team(),
                opponent_roster=roster, opponent_spike_holder="ごんた",
                opponent_igl="ごりまる", seed=1500,
                opponent_team_name="historical", allow_mirrored_roster=True,
            )
        self.assertEqual(4, result.summary.rounds)
        self.assertTrue(result.summary.side_checkpoint_switch_ok)
        self.assertTrue(result.summary.memory_reset_ok)

    def test_promotion_rejects_regression_and_accepts_non_regression(self):
        schedule = (
            ("omoko", 1, "attacker", 13, 8),
            ("omoko", 2, "defender", 13, 10),
            ("history_old", 1, "attacker", 13, 9),
            ("history_old", 2, "defender", 9, 13),
        )
        incumbent = evaluation(schedule)
        self.assertEqual(incumbent, PoolEvaluation.from_dict(incumbent.to_dict()))
        accepted = decide_promotion(
            incumbent, incumbent, historical_opponents=("history_old",),
        )
        self.assertTrue(accepted.promoted)

        regressed = evaluation((
            ("omoko", 1, "attacker", 13, 8),
            ("omoko", 2, "defender", 8, 13),
            ("history_old", 1, "attacker", 7, 13),
            ("history_old", 2, "defender", 8, 13),
        ))
        rejected = decide_promotion(
            regressed, incumbent, historical_opponents=("history_old",),
            criteria=PromotionCriteria(maximum_opponent_point_rate_drop=0.0),
        )
        self.assertFalse(rejected.promoted)
        self.assertTrue(any("regressed" in reason for reason in rejected.reasons))
        self.assertTrue(any("historical" in reason for reason in rejected.reasons))


class PoolDaggerTest(unittest.TestCase):
    def test_recorded_teacher_samples_do_not_change_with_hidden_enemy_truth(self):
        observations = []
        grid = np.array([[int(cell) for cell in line]
                         for line in NEW_MAZE_STR.strip().splitlines()], dtype=np.int8)
        trainer = CoachTrainer(Side.ATTACKER, seed=15)
        for enemy_position in ((4, 6), (4, 7)):
            allies = [FakeCharacter(slot.character_name, "A", (23, 18 + slot.slot),
                                    facing="E") for slot in FIXED_ROSTER]
            game = FakeGame(
                grid, allies + [FakeCharacter("hidden", "D", enemy_position, facing="N")]
            )
            recorder = RecordingTeacherPolicy(trainer)
            with patch("coach_v1.team_ai._CHARACTER_POLICY_TYPES", (FacingActor,) * 5):
                candidate = build_coach_v1_team(attacker_coach=recorder)
                candidate.bind_game(game)
                controller = candidate.get_attacker_controller()
                controller.decide_move(allies[0], {})
            self.assertFalse(controller._snapshot.sightings)
            observations.append(recorder.samples[0])
        for index in range(8):
            np.testing.assert_array_equal(observations[0][index], observations[1][index])

    def test_candidate_copy_balancing_and_optimization_are_isolated(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source", root / "target"
            initial = CoachTrainer(Side.ATTACKER, seed=15, directory=source)
            initial.save()
            prepare_candidate_directory(source=source, target=target)
            resumed = CoachTrainer(Side.ATTACKER, seed=15, directory=target)
            resumed.resume()
            self.assertEqual(initial.training_step, resumed.training_step)
            self.assertTrue((source / "latest.pt").is_file())

            grid = np.array([[int(cell) for cell in line]
                             for line in NEW_MAZE_STR.strip().splitlines()], dtype=np.int8)
            allies = [FakeCharacter(slot.character_name, "A", (23, 18 + slot.slot),
                                    facing="E") for slot in FIXED_ROSTER]
            game = FakeGame(
                grid, allies + [FakeCharacter("hidden", "D", (4, 6), facing="N")]
            )
            recorder = RecordingTeacherPolicy(resumed)
            with patch("coach_v1.team_ai._CHARACTER_POLICY_TYPES", (FacingActor,) * 5):
                candidate = build_coach_v1_team(attacker_coach=recorder)
                candidate.bind_game(game)
                candidate.get_attacker_controller().decide_move(allies[0], {})
            selected, buckets = balanced_pool_samples(
                recorder.samples, side=Side.ATTACKER, samples_per_bucket=2,
            )
            before = {name: value.detach().clone()
                      for name, value in resumed.actor.state_dict().items()}
            loss = optimize_pool_samples(resumed, selected, epochs=1)
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(2, len(selected))
            self.assertTrue(buckets)
            self.assertTrue(any(
                not torch.equal(before[name], value)
                for name, value in resumed.actor.state_dict().items()
            ))
            self.assertEqual(0, initial.training_step)


class PoolRoundRewardTest(unittest.TestCase):
    def test_round_reward_samples_are_private_and_update_actor_only(self):
        samples = []
        grid = np.array([[int(cell) for cell in line]
                         for line in NEW_MAZE_STR.strip().splitlines()], dtype=np.int8)
        trainer = CoachTrainer(Side.ATTACKER, seed=15)
        for enemy_position in ((4, 6), (4, 7)):
            allies = [FakeCharacter(slot.character_name, "A", (23, 18 + slot.slot),
                                    facing="E") for slot in FIXED_ROSTER]
            game = FakeGame(
                grid, allies + [FakeCharacter("hidden", "D", enemy_position, facing="N")]
            )
            recorder = RoundRewardPolicy(trainer)
            with patch("coach_v1.team_ai._CHARACTER_POLICY_TYPES", (FacingActor,) * 5):
                candidate = build_coach_v1_team(attacker_coach=recorder)
                candidate.bind_game(game)
                controller = candidate.get_attacker_controller()
                torch.manual_seed(99)
                controller.decide_move(allies[0], {})
            self.assertFalse(controller._snapshot.sightings)
            recorder.finish()
            samples.append(recorder.segments[0][0])
        for index in range(5):
            np.testing.assert_array_equal(samples[0][index], samples[1][index])

        before = {name: value.detach().clone()
                  for name, value in trainer.actor.state_dict().items()}
        loss = optimize_round_rewards(trainer, ((samples[0],),), (True,))
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(
            not torch.equal(before[name], value)
            for name, value in trainer.actor.state_dict().items()
        ))

    def test_round_records_map_results_to_the_coach_side(self):
        result = FullMatchResult(
            summary(1, "attacker", 1, 1).summary,
            tuple(),
            (
                {"winner": "attacker", "players": {
                    "ours": {"team": "candidate", "side": "attacker"},
                    "theirs": {"team": "opponent", "side": "defender"},
                }},
                {"winner": "attacker", "players": {
                    "ours": {"team": "candidate", "side": "defender"},
                    "theirs": {"team": "opponent", "side": "attacker"},
                }},
            ),
        )
        outcomes = _coach_round_results(result, coach_team_name="candidate")
        self.assertEqual([True], outcomes[Side.ATTACKER])
        self.assertEqual([False], outcomes[Side.DEFENDER])


if __name__ == "__main__":
    unittest.main()
