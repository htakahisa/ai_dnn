"""V3-specific regressions. No collection or optimizer updates are executed."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import torch

from frc_v1 import FACING
from frc_v1.perception import FrcPerceptionBuilder, Sighting
from touyama_v3.tv3_scenario import Scenario, OPPONENTS
from touyama_v3.tv3_character_stats_touyama import TOUYAMA_ROSTER_ORDER
from touyama_v3.tv3_model import SiteModel
from touyama_v3.tv3_observer import FeatureHistory
from touyama_v3.tv3_defender_policy import DefenderDQN, OBS_DIM, assign_goals, staging_positions
from touyama_v3.tv3_defender_controller import TouyamaV3DefenderController, OPPONENT_NAMES
from touyama_v3.tv3_attacker_guard_controller import TouyamaV3AttackerGuardController
from touyama_v3.tv3_learn_attacker_guard import ACTION_DIM, guard_schema
from touyama_v3.tv3_attacker_route_planner import AdaptiveAttackPlanner
from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder
from touyama_v3.test.tv3_fixtures import defender_world
from touyama_v3.test.tv3_fixtures import attacker_world
from touyama_v3.test.test_tv3_attacker_route_planner import PublicContactModel


class StayPolicy(torch.nn.Module):
    def forward(self, x):
        q = torch.zeros((len(x), ACTION_DIM))
        q[:, 4] = 100  # STAY facing S, even under a public smoke-defuse notification.
        return q


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def defender(self, game):
        analysis = SiteModel(len(FeatureHistory(game).fields))
        policy = DefenderDQN(OBS_DIM)
        with torch.no_grad():
            for p in policy.parameters():
                p.zero_()
        controller = TouyamaV3DefenderController('gc_v1', scenario=game, search=policy,
            retake={}, analyses={'gc_v1': analysis})
        controller.set_game(game)
        return controller

    def test_all_opponents_and_fixed_roster_include_toru_v4_instead_of_v3(self):
        from touyama_v3.tv3_attacker_rosters import eligible_attacker_presets
        self.assertEqual(len(OPPONENTS), 6)
        self.assertEqual(OPPONENTS['toru_ai_v4'][0], 'toru_ai_v4')
        self.assertNotIn('toru_ai_v3', OPPONENTS)
        self.assertNotIn('touyama_v2', OPPONENTS)
        self.assertEqual(OPPONENTS['concon_v1'], ('concon_v1', 'Gorigons'))
        self.assertEqual(OPPONENT_NAMES['ConCon v1'], 'concon_v1')
        self.assertNotIn('Touyama Gaming v2', OPPONENT_NAMES)
        self.assertEqual(OPPONENT_NAMES['Toru AI v4'], 'toru_ai_v4')
        for opponent in OPPONENTS:
            self.assertEqual(eligible_attacker_presets(('Touyama Gaming',), opponent), ['Touyama Gaming'])
        with self.assertRaises(ValueError):
            eligible_attacker_presets(('Gorigons',), 'gc_v1')

    def test_initial_markers_are_targets_and_capitals_are_look_points(self):
        scenario = Scenario()
        self.assertEqual([p.watch for p in scenario.posts], [(7,3),(11,16),(11,21),(8,31),(2,40)])
        self.assertEqual([p.look for p in scenario.posts], [(15,3),(14,19),(13,23),(14,31),(14,40)])
        game = defender_world(scenario)
        snapshot = FrcPerceptionBuilder('D').build(game)
        goals = assign_goals(snapshot, scenario, [.5,.5], staging_positions(scenario))
        for ally in snapshot.allies:
            self.assertEqual(goals[ally.slot], scenario.post_for(ally).watch)
        self.assertEqual(scenario.grid[16,23], 5)

    def test_initial_facing_then_learned_choice_and_round_reset(self):
        game = defender_world(Scenario())
        controller = self.defender(game)
        controller.prepare_team_tick()
        expected = dict(zip(TOUYAMA_ROSTER_ORDER, ('S','SE','SE','S','S')))
        for name, (_, action) in controller.actions.items():
            self.assertEqual(action['facing'], expected[name])
        game.battle_tick += 1
        controller.prepare_team_tick()
        self.assertTrue(all(action['facing'] == 'N' for _, action in controller.actions.values()))
        controller.reset_round()
        self.assertEqual(controller.initial_arrived, set())
        self.assertEqual(controller.initial_faced, set())

    def test_hidden_enemy_state_does_not_change_initial_inputs(self):
        game = defender_world(Scenario())
        first = self.defender(game)
        first.prepare_team_tick()
        for c in game.chars[5:]:
            c.pos = [22,30]
            c.has_spike = False
        second = self.defender(game)
        second.prepare_team_tick()
        self.assertEqual(first.actions, second.actions)
        for name in first.inputs:
            np.testing.assert_array_equal(first.inputs[name].observation, second.inputs[name].observation)

    def test_grouped_approach_has_no_scout_or_flank_split(self):
        game = attacker_world()
        snapshot = replace(FrcPerceptionBuilder('A').build(game), sightings=())
        encoder = AttackerEncoder(game)
        planner = AdaptiveAttackPlanner()
        plan = planner.update(snapshot, encoder.observe(snapshot, []), encoder, PublicContactModel())
        self.assertNotEqual(plan.phase, 'entry')
        self.assertEqual(plan.scout_goals, {})
        self.assertEqual(planner.flank_entries, {})

    def test_formation_cost_handles_missing_plan_and_uses_executed_tick_context(self):
        from touyama_v3.tv3_train_attacker_plant import group_separation_cost
        scenario = SimpleNamespace(grid=np.zeros((3,12), dtype=np.int32))
        state = SimpleNamespace(allies=(SimpleNamespace(alive=True, has_spike=True, position=(1,1)),))
        before, combat = SimpleNamespace(blind=0), SimpleNamespace(contacts=0)
        controller = SimpleNamespace(attack_plan=SimpleNamespace(phase='approach'))
        executed_plan = controller.attack_plan
        controller.attack_plan = None  # Next-tick replanning has no feasible route.
        self.assertEqual(group_separation_cost(scenario, state, (1,10), before, combat,
                                              controller.attack_plan, False), 0.)
        self.assertAlmostEqual(group_separation_cost(scenario, state, (1,10), before, combat,
                                                    executed_plan, False), .05)
        self.assertEqual(group_separation_cost(scenario, state, (1,10), before, combat,
                                              SimpleNamespace(phase='entry'), False), 0.)
        self.assertEqual(group_separation_cost(scenario, state, (1,10), before, combat,
                                              executed_plan, True), 0.)

    def test_source_defaults_use_fixed_roster_and_separate_evaluation_seeds(self):
        from touyama_v3 import tv3_train_defender_search as search
        from touyama_v3 import tv3_train_defender_analysis as site
        from touyama_v3 import tv3_train_attacker_analysis as analysis
        from touyama_v3 import tv3_train_attacker_plant as plant
        for module in (search, site, analysis, plant):
            self.assertEqual(module.TRAINING_PRESETS, ('Touyama Gaming',))
            self.assertEqual(module.EVALUATION_PRESETS, ('Touyama Gaming',))
        args = search.parse_arguments([])
        self.assertFalse(args.resume)
        self.assertEqual(set(args.opponents), set(OPPONENTS))
        plan = search.evaluation_plan(('gc_v1',), args.eval_presets, 3, args.seed)
        self.assertEqual(len({row[2] for row in plan}), 3)
        self.assertTrue(all(row[2] > args.seed for row in plan))

    def test_short_crossfire_reposition_faces_the_public_enemy(self):
        from touyama_v3.tv3_attacker_combat import AttackerCombatCoach
        from grid_lines import line_cells
        game = attacker_world()
        snapshot = FrcPerceptionBuilder('A').build(game)
        allies = tuple(replace(a, position=(4,1) if i == 0 else (3,1) if i == 1 else (0,i),
                               alive=i < 2, blind=0, hp=100, max_hp=100, facing='E')
                       for i, a in enumerate(snapshot.allies))
        state = replace(snapshot, allies=allies, sightings=(Sighting(0,(4,5),'normal'),), smoke_cells=frozenset())
        scenario = SimpleNamespace(clear=lambda p,q: True, _line_cells=line_cells,
            neighbors=lambda p: [(p[0]+dr,p[1]+dc) for dr,dc in ((1,0),(-1,0),(0,1),(0,-1))
                                 if 0 <= p[0]+dr < 9 and 0 <= p[1]+dc < 9])
        advice = AttackerCombatCoach(scenario).advise(state, allies[0], allies[0].position, (4,5))
        self.assertEqual(advice.reason, 'crossfire_reposition')
        self.assertEqual(sum(abs(a-b) for a,b in zip(advice.position, allies[0].position)), 1)
        self.assertIn(advice.facing, ('E','NE','SE'))

    def test_smoke_defuse_preserves_learned_alternatives_and_teaches_approach(self):
        game = attacker_world()
        cells = game.sites['L']
        for c, pos in zip(game.chars[:5], ((11,1),(11,2),(11,3),(11,4),(11,5))):
            c.pos = list(pos)
            c.has_spike = False
        game.is_planted, game.planted_pos = True, cells[0]
        game.battle_tick, game.detonate_timer = 20, 54
        game.active_defuser_name = 'hidden_defuser'
        game._smoke_cells = lambda: {cells[0]}
        controller = TouyamaV3AttackerGuardController(game, StayPolicy())
        controller.set_game(game)
        controller.prepare_team_tick()
        pressure = [x for x in controller.inputs.values() if x.defuse_pressure]
        self.assertTrue(pressure)
        for name, (selected, inputs, ally) in controller.plans.items():
            self.assertTrue(inputs.mask[selected])
            if inputs.defuse_pressure and inputs.distances[ally.position] > 0:
                self.assertTrue(inputs.mask[4])
                self.assertEqual(selected, 4)
        self.assertTrue(any(x.actions[x.teacher].kind != 'STAY' for x in pressure))
        self.assertIn('learned_smoke_defuse', guard_schema(game)['tactics'])

    def test_toru_baseline_updates_are_rejected_without_running_games(self):
        from touyama_v3 import tv3_opponents as module
        with patch.object(module, '_frozen_hashes', {'model': 'old'}), \
             patch.object(module, 'toru_baseline_hashes', return_value={'model': 'new'}):
            with self.assertRaisesRegex(ValueError, 'baseline changed'):
                module.build_opponent_ai('toru_ai_v4')

    def test_guard_handoff_waits_for_the_completed_plant_tick(self):
        from touyama_v3.tv3_attacker_guard_controller import TouyamaV3AttackerPlantGuardController
        game = attacker_world()
        game.battle_tick = 7
        game.is_planted, game.planted_pos = True, game.sites['L'][0]
        combined = TouyamaV3AttackerPlantGuardController(game, None, None, StayPolicy())
        combined.set_game(game)
        combined.plant.cache = (1, 'live', 7, False)
        combined.prepare_team_tick()
        self.assertIsNone(combined.guard)
        destination, _ = combined.decide_move(game.chars[0], {'is_planted': True})
        self.assertEqual(destination, list(game.chars[0].pos))
        game.battle_tick = 8
        combined.prepare_team_tick()
        self.assertIsNotNone(combined.guard)
        self.assertEqual(combined.guard.start_tick, 8)
        self.assertIs(combined.guard.sensor, combined.plant.sensor)

    def test_game_entrypoints_and_same_names_on_both_sides(self):
        from run_game import VisualFPSBattle, _build_team_ai
        from party_presets import get_preset
        from touyama_v3.tv3_opponents import scoped_presets
        preset, enemy = scoped_presets(get_preset('Touyama Gaming'), get_preset('Touyama Gaming'))
        from controllers import DefaultAttackerController, DefaultDefenderController
        from team_ai import DualRoleTeamAI
        team = _build_team_ai('touyama_gaming_v3', device='cpu')
        other = DualRoleTeamAI('Unknown opponent', DefaultAttackerController, DefaultDefenderController)
        game = VisualFPSBattle(Scenario().maze, team, other, headless=True,
            attacker_roster=list(preset.players), defender_roster=list(enemy.players),
            spike_holder_name=preset.spike_holder, defender_spike_holder_name=preset.spike_holder,
            attacker_igl_name=preset.igl, defender_igl_name=preset.igl,
            attacker_team_name=preset.name, defender_team_name=preset.name, disable_side_swap=True)
        game.analytics_tracker = None
        game._record_replay_frame = lambda: None
        self.assertEqual(len(game.chars), 10)
        self.assertEqual(len({c.name for c in game.chars}), 10)
        for _ in range(2):
            game.step_tick()


if __name__ == '__main__':
    unittest.main()
