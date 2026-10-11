"""Deadline examples, timeout-first ranking and weights-only initialization."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import io
import tempfile
import unittest
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from frc_v1.actions import FrcAction
from touyama_v3.tv3_model_extensions import ExtensionContext, build_extensions
from touyama_v3 import tv3_custom_train_attacker_plant_frc_v1_R_v3 as custom
from touyama_v3.tv3_train_attacker_plant import best_rank, initialize_custom_weights


class DeadlineTests(unittest.TestCase):
    def setUp(self):
        grid = np.zeros((7, 10), dtype=int)
        grid[2, 7] = 2
        self.snapshot = NS(grid=tuple(map(tuple, grid)), round_timer=18, sightings=())
        self.ally = NS(has_spike=True, position=(4, 7), plant_progress=0, hp=60,
                       max_hp=100, blind=0, ability_name='SMOKE')
        self.context = ExtensionContext('attacker_plant', 'frc_v1', 'R', self.snapshot, self.ally)
        actions = tuple(FrcAction(kind, 'N') for kind in ('STAY','N','E','S','W') for _ in range(8))
        actions += tuple(FrcAction('ABILITY', 'N') for _ in range(16)) + (FrcAction('PLANT','N'), FrcAction('COLLECT_ORB','N'))
        mask = np.ones(58, bool); mask[56] = False
        self.inputs = NS(mask=mask, actions=actions, teacher=24, combat=NS(reason='cover_retreat', facing='N'))

    def test_deadline_teaches_progress_instead_of_repeated_retreat(self):
        custom.adjust_inputs(self.context, self.inputs)
        self.assertEqual(self.inputs.teacher, 8)
        self.assertTrue(self.inputs.mask[self.inputs.teacher])

    def test_legal_plant_takes_priority_over_combat_teacher(self):
        self.ally.position = (2, 7)
        self.snapshot.round_timer = 16
        self.inputs.mask[56] = True
        custom.adjust_inputs(self.context, self.inputs)
        self.assertEqual(self.inputs.teacher, 56)

    def test_masked_forward_is_not_enabled(self):
        self.inputs.mask[8:16] = False
        before = self.inputs.mask.copy()
        custom.adjust_inputs(self.context, self.inputs)
        self.assertEqual(self.inputs.teacher, 24)
        np.testing.assert_array_equal(before, self.inputs.mask)

    def test_early_round_impossible_completion_and_other_scope_keep_teacher(self):
        for timer in (60, 5):
            self.snapshot.round_timer = timer
            custom.adjust_inputs(self.context, self.inputs)
            self.assertEqual(self.inputs.teacher, 24)
        self.snapshot.round_timer = 18
        for changes in (dict(site='L'), dict(opponent='omoko_v1'), dict(role='attacker_guard')):
            custom.adjust_inputs(replace(self.context, **changes), self.inputs)
            self.assertEqual(self.inputs.teacher, 24)

    def test_wait_and_reverse_cost_but_completion_has_no_extra_cost(self):
        for index, penalty in ((0, custom.DEADLINE_IDLE_PENALTY), (24, custom.DEADLINE_REVERSE_PENALTY), (8, 0.), (56, 0.)):
            context = replace(self.context, action=self.inputs.actions[index])
            self.assertAlmostEqual(custom.adjust_reward(context, 1.), 1.-penalty)
        self.ally.has_spike = False
        self.assertEqual(custom.adjust_reward(replace(self.context, action=self.inputs.actions[0]), 1.), 1.)

    def test_partial_plant_uses_remaining_work(self):
        self.ally.position = (2, 7)
        self.ally.plant_progress = 3
        self.snapshot.round_timer = 1
        self.inputs.mask[56] = True
        custom.adjust_inputs(self.context, self.inputs)
        self.assertEqual(self.inputs.teacher, 56)

    def test_safer_plant_goal_and_side_step_do_not_teach_exposed_plant(self):
        from frc_v1.perception import Sighting
        from touyama_v3.tv3_scenario import Scenario
        scenario = Scenario()
        snapshot = NS(grid=tuple(map(tuple, scenario.grid)), round_timer=16,
                      sightings=(), allies=(), smoke_cells=())
        self.ally.position = (11,40)
        self.ally.slot, self.ally.reveal = 1, 0
        context = replace(self.context, snapshot=snapshot, goal=(10,40),
                          threats=(Sighting(2,(7,38),'history'),))
        goal = custom.adjust_goal(context, (10,40))
        self.assertEqual(goal,(10,41))
        self.ally.position = (10,40)
        context = replace(context, goal=goal)
        self.inputs.mask[56] = True
        custom.adjust_inputs(context,self.inputs)
        self.assertEqual(self.inputs.actions[self.inputs.teacher].kind,'E')
        self.assertEqual(custom.adjust_reward(replace(context,action=self.inputs.actions[16]),1.),1.)

    def test_deadline_helper_yields_even_when_already_at_its_own_goal(self):
        from frc_v1.perception import Sighting
        from touyama_v3.tv3_scenario import Scenario
        scenario = Scenario()
        self.ally.position = (12,40)
        self.ally.slot, self.ally.alive, self.ally.reveal = 1, True, 0
        helper = NS(position=(10,41), slot=2, alive=True, has_spike=False, reveal=0)
        snapshot = NS(grid=tuple(map(tuple,scenario.grid)), round_timer=18,
                      sightings=(), smoke_cells=(), allies=(self.ally,helper))
        context = replace(self.context,snapshot=snapshot,ally=helper,carrier_goal=(10,41),
                          threats=(Sighting(2,(7,38),'history'),))
        self.assertEqual(custom.adjust_goal(context,(10,41)),(10,42))
        helper.position = (10,42)
        self.assertEqual(custom.adjust_goal(context,(10,41)),(10,42))
        snapshot.round_timer = 80
        self.assertEqual(custom.adjust_goal(context,(10,41)),(10,41))

    def test_timeout_first_beats_more_advantaged_plants(self):
        metrics = dict(rounds=72, timeouts=12, plant_rate=60/72, competitive_plant_rate=54/72,
                       advantaged_plant_rate=43/72, ability_advantage_plant_rate=13/72,
                       mean_damage=138., mean_plant_ticks=62.)
        better = {**metrics, 'timeouts': 11, 'advantaged_plant_rate': 30/72}
        self.assertGreater(best_rank(better, timeout_first=True), best_rank(metrics, timeout_first=True))
        self.assertLess(best_rank(better), best_rank(metrics))
        equal = {**metrics, 'competitive_plant_rate': 55/72, 'advantaged_plant_rate': 40/72}
        self.assertGreater(best_rank(equal, timeout_first=True), best_rank(metrics, timeout_first=True))

    def test_new_profile_copies_only_custom_weights_and_preserves_new_contract(self):
        from touyama_v3.tv3_scenario import Scenario
        from touyama_v3.tv3_learn_attacker_plant import PlantDQN, policy_schema, attach_common_policy
        scenario = Scenario()
        common = PlantDQN(16)
        buffer = io.BytesIO()
        torch.save(dict(schema=policy_schema(scenario,16), model=common.state_dict(),
                        opponent='frc_v1', analysis_hash='fixed'), buffer)
        source, target = PlantDQN(19), PlantDQN(19)
        for model in (source, target):
            attach_common_policy(model, buffer.getvalue(), scenario, 'frc_v1', 'fixed', 'R')
        old = build_extensions(('tv3_custom_train_attacker_plant_frc_v1_R',), role='attacker_plant', opponent='frc_v1').contract
        new = build_extensions(('tv3_custom_train_attacker_plant_frc_v1_R_v3',), role='attacker_plant', opponent='frc_v1').contract
        target.custom_extension_contract = new
        schema = policy_schema(scenario,19,custom_extensions=new,weight_routing=target.weight_routing)
        state = dict(schema=policy_schema(scenario,19,custom_extensions=old,weight_routing=source.weight_routing),
                     opponent='frc_v1',analysis_hash='fixed',model=source.state_dict(),completed_sets=5,
                     common_policy_payload=buffer.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.pt'
            torch.save(state, path)
            initialize_custom_weights(target,path,scenario,'frc_v1','fixed',schema)
            self.assertEqual(target.custom_extension_contract,new)
            self.assertTrue(all(torch.equal(v,target.state_dict()[k]) for k,v in source.state_dict().items()))
            self.assertTrue(all(not p.requires_grad for p in target.common_policy.parameters()))
            with self.assertRaises(ValueError):
                initialize_custom_weights(target,path,scenario,'frc_v1','other',schema)


if __name__ == '__main__':
    unittest.main()
