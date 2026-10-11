"""Scoped custom hooks, checkpoint contracts and unchanged common behavior."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import copy
import io
import tempfile
import unittest
from unittest.mock import patch
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from touyama_v3 import tv3_model_extensions as ext
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_learn_attacker_plant import PlantDQN,policy_schema,load_plant,attach_common_policy
from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController
from touyama_v3.tv3_train_attacker_plant import plant_output_directories,warm_start_schema_compatible

TEMPLATE='tv3_custom_train_attacker_plant_frc_v1_R'


class ModelExtensionTests(unittest.TestCase):
    def setUp(self):
        self.context=ext.ExtensionContext('attacker_plant','frc_v1','R',NS(grid=((0,0),(0,0))))

    def test_default_and_template_preserve_common_values(self):
        pipeline=ext.build_extensions((),role='attacker_plant',opponent='frc_v1')
        self.assertIsNone(pipeline.contract)
        template=ext.build_extensions((TEMPLATE,),role='attacker_plant',opponent='frc_v1')
        for p in (pipeline,template):
            self.assertEqual(p.adjust_goal(self.context,(0,1)),(0,1))
            self.assertEqual(p.adjust_reward(self.context,3.5),3.5)

    def test_scope_gates_opponent_site_and_role_and_common_hook_order(self):
        calls=[]
        def hook(label):
            def adjust(context,reward):
                calls.append(label)
                return reward+1
            return NS(adjust_reward=adjust)
        pipeline=ext.ExtensionPipeline('attacker_plant','frc_v1',(
            (hook('common'),dict(id='common',site='*')),
            (hook('right'),dict(id='right',site='R'))))
        self.assertEqual(pipeline.adjust_reward(self.context,0),2)
        self.assertEqual(calls,['common','right'])
        calls.clear()
        self.assertEqual(pipeline.adjust_reward(replace(self.context,site='L'),0),1)
        self.assertEqual(calls,['common'])
        for context in (replace(self.context,opponent='omoko_v1'),replace(self.context,role='defender_retake')):
            calls.clear()
            self.assertEqual(pipeline.adjust_reward(context,0),0)
            self.assertEqual(calls,[])
        self.assertIsNone(ext.build_extensions((TEMPLATE,),role='attacker_plant',opponent='omoko_v1').contract)

    def test_saved_contract_is_exact_and_changed_scope_is_rejected(self):
        pipeline=ext.build_extensions((TEMPLATE,),role='attacker_plant',opponent='frc_v1')
        contract=pipeline.contract
        self.assertEqual(ext.load_extensions(contract,role='attacker_plant',opponent='frc_v1').contract,contract)
        for key,value in [('revision',contract['extensions'][0]['revision']+1),('site','L')]:
            wrong=copy.deepcopy(contract)
            wrong['extensions'][0][key]=value
            with self.assertRaises(ValueError):
                ext.load_extensions(wrong,role='attacker_plant',opponent='frc_v1')
        with self.assertRaises(ValueError):
            ext.load_extensions(contract,role='attacker_plant',opponent='omoko_v1')

    def test_changed_source_is_rejected_even_if_revision_was_not_bumped(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/f'{TEMPLATE}.py'
            path.write_bytes((ext.EXTENSION_DIRECTORY/f'{TEMPLATE}.py').read_bytes())
            with patch.object(ext,'EXTENSION_DIRECTORY',Path(directory)):
                contract=ext.build_extensions((TEMPLATE,),role='attacker_plant',opponent='frc_v1').contract
                path.write_bytes(path.read_bytes()+b'\n# changed source\n')
                with self.assertRaisesRegex(ValueError,'source changed'):
                    ext.load_extensions(contract,role='attacker_plant',opponent='frc_v1')

    def test_inputs_cannot_enable_blocked_actions_or_change_dimensions(self):
        def inputs():
            return NS(observation=np.zeros(3),mask=np.array([True,False]),actions=('stay','move'),teacher=0)
        def enable(context,value):
            value.mask[1]=True
        pipeline=ext.ExtensionPipeline('attacker_plant','frc_v1',((NS(adjust_inputs=enable),dict(site='R')),))
        with self.assertRaisesRegex(ValueError,'cannot enable'):
            pipeline.adjust_inputs(self.context,inputs())
        def reshape(context,value):
            value.observation=np.zeros(4)
        pipeline=ext.ExtensionPipeline('attacker_plant','frc_v1',((NS(adjust_inputs=reshape),dict(site='R')),))
        with self.assertRaises(ValueError):
            pipeline.adjust_inputs(self.context,inputs())

    def test_invalid_goal_reward_and_duplicate_registration_are_rejected(self):
        pipeline=ext.ExtensionPipeline('attacker_plant','frc_v1',((NS(adjust_goal=lambda c,g:(3,3)),dict(site='R')),))
        with self.assertRaises(ValueError):
            pipeline.adjust_goal(self.context,(0,0))
        pipeline=ext.ExtensionPipeline('attacker_plant','frc_v1',((NS(adjust_reward=lambda c,r:float('nan')),dict(site='R')),))
        with self.assertRaises(ValueError):
            pipeline.adjust_reward(self.context,1)
        with self.assertRaises(ValueError):
            ext.build_extensions((TEMPLATE,TEMPLATE),role='attacker_plant',opponent='frc_v1')

    def test_common_and_custom_checkpoints_have_separate_schema_and_directories(self):
        scenario=Scenario()
        common=policy_schema(scenario,19)
        contract=ext.build_extensions((TEMPLATE,),role='attacker_plant',opponent='frc_v1').contract
        custom=policy_schema(scenario,19,custom_extensions=contract)
        self.assertNotIn('custom_extensions',common)
        self.assertFalse(warm_start_schema_compatible(common,custom))
        self.assertFalse(warm_start_schema_compatible(policy_schema(scenario,16),custom))
        normal=plant_output_directories(Path('data'),Path('best'),'frc_v1')
        separate=plant_output_directories(Path('data'),Path('best'),'frc_v1',contract)
        self.assertEqual(normal,(Path('data/frc_v1'),Path('best/frc_v1')))
        self.assertNotEqual(normal,separate)
        self.assertIn('custom_profiles',separate[0].parts)
        model=PlantDQN(19)
        common=PlantDQN(19)
        buffer=io.BytesIO()
        torch.save(dict(schema=policy_schema(scenario,19),opponent='frc_v1',
                        analysis_hash='analysis',model=common.state_dict()),buffer)
        payload=buffer.getvalue()
        routing=attach_common_policy(model,payload,scenario,'frc_v1','analysis','R')
        custom=policy_schema(scenario,19,custom_extensions=contract,weight_routing=routing)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'custom.pt'
            torch.save(dict(schema=custom,opponent='frc_v1',analysis_hash='analysis',model=model.state_dict(),common_policy_payload=payload),path)
            loaded,_=load_plant(path,scenario,'frc_v1','analysis')
            self.assertEqual(loaded.custom_extension_contract,contract)
            self.assertEqual(loaded.opponent,'frc_v1')
            self.assertEqual(loaded.weight_scope,'R')
            self.assertTrue(all(not p.requires_grad for p in loaded.common_policy.parameters()))
            self.assertFalse({id(p) for p in loaded.parameters()} & {id(p) for p in loaded.common_policy.parameters()})
            before={k:v.clone() for k,v in loaded.common_policy.state_dict().items()}
            with torch.no_grad():
                loaded.net[0].weight.add_(1.)
            self.assertTrue(all(torch.equal(v,loaded.common_policy.state_dict()[k]) for k,v in before.items()))
            controller=TouyamaV3AttackerPlantController.__new__(TouyamaV3AttackerPlantController)
            controller.root_policy=loaded
            controller.training=True
            controller.common_encoder=NS(execution_version=16)
            controller.custom_encoder=NS(execution_version=19)
            controller.select_weight_policy('L')
            self.assertIs(controller.policy,loaded.common_policy)
            self.assertEqual(controller.encoder.execution_version,16)
            self.assertFalse(controller.training_current_policy)
            controller.select_weight_policy('R')
            self.assertIs(controller.policy,loaded)
            self.assertEqual(controller.encoder.execution_version,19)
            self.assertTrue(controller.training_current_policy)
            tampered=dict(routing,common_sha256='0'*64)
            torch.save(dict(schema=policy_schema(scenario,19,custom_extensions=contract,weight_routing=tampered),
                opponent='frc_v1',analysis_hash='analysis',model=model.state_dict(),common_policy_payload=payload),path)
            with self.assertRaisesRegex(ValueError,'hash/routing'):
                load_plant(path,scenario,'frc_v1','analysis')

    def test_replacement_ability_target_must_be_legal_in_the_real_public_mask(self):
        from frc_v1.actions import FrcAction,build_masks
        from frc_v1.perception import FrcPerceptionBuilder
        from touyama_v3.test.tv3_fixtures import attacker_world
        snapshot=FrcPerceptionBuilder('A').build(attacker_world())
        ally=next(a for a in snapshot.allies if a.ability_name=='FLASH')
        masks=build_masks(snapshot)
        index=int(np.flatnonzero(masks.target[ally.slot,0])[0])
        target=divmod(index,len(snapshot.grid[0]))
        context=replace(self.context,snapshot=snapshot,ally=ally)
        def make_inputs():
            actions=[FrcAction('STAY')]*58
            actions[40]=FrcAction('ABILITY','N',None)
            mask=np.zeros(58,dtype=bool);mask[0]=True
            return NS(actions=tuple(actions),mask=mask,observation=np.zeros(3),teacher=0)
        def hook(context,inputs):
            actions=list(inputs.actions);actions[40]=FrcAction('ABILITY','N',target)
            inputs.actions=tuple(actions);inputs.mask[40]=True
        pipeline=ext.ExtensionPipeline('attacker_plant','frc_v1',((NS(adjust_inputs=hook),dict(site='R')),))
        self.assertTrue(pipeline.adjust_inputs(context,make_inputs()).mask[40])
        target=(-1,-1)
        with self.assertRaises(ValueError):
            pipeline.adjust_inputs(context,make_inputs())


if __name__=='__main__':
    unittest.main()
