"""Major AI/roster pairing, lazy freezing and safe macro-only continuation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
from ghost_champions_v3.macro_v1.gc3_policy_attacker_macro import MacroPolicy
from ghost_champions_v3.macro_v1.gc3_opponents_attacker_macro import MAIN_OPPONENT_AIS, opponent_settings, FrozenOpponent
from ghost_champions_v3.macro_v1.gc3_training_attacker_macro import initialize, rng_state
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash
from ghost_champions_v3.gc3_rollout import module_digest


class MajorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_seven_major_pairs_and_original_fixed_mode(self):
        self.assertEqual(set(MAIN_OPPONENT_AIS),set(("FNC","FRC","TYG","OMG","SPS","GRG","GC")))
        for code,key in MAIN_OPPONENT_AIS.items():
            self.assertEqual(opponent_settings(dict(OPPONENT_MODE="major_main"),code),dict(opponent_ai=key,expected_opponent_files=None))
            self.assertEqual(opponent_settings({},code),dict(opponent_ai="default"))
        with self.assertRaisesRegex(ValueError,'corresponding'):
            opponent_settings(dict(OPPONENT_MODE="major_main",OPPONENT_AIS={'FNC':'default'}),'FNC')

    def test_lazily_loaded_model_is_frozen_before_first_forward(self):
        class Lazy:
            def decide_move(self,char,state):
                self.model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.9))
                def check(model,inputs):
                    if model.training or any(p.requires_grad for p in model.parameters()):
                        raise AssertionError('Lazy model not frozen before forward')
                handle=self.model.register_forward_pre_hook(check)
                try:return self.model(torch.ones(1,2))
                finally:handle.remove()
        opponent=FrozenOpponent(Lazy())
        result=opponent.decide_move(None,{})
        self.assertFalse(result.requires_grad)
        self.assertFalse(opponent.inner.model.training)
        opponent.verify()
        with torch.no_grad():opponent.inner.model[0].weight.add_(1)
        with self.assertRaisesRegex(RuntimeError,'updated'):
            opponent.verify()

    def test_major_continuation_keeps_best_weights_and_inherited_step(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);pre=root/'pre.pt';guard=root/'guard.pt';source=root/'source.pt';validation=root/'opponents.json'
            pre.write_bytes(b'pre');guard.write_bytes(b'guard')
            model=MacroPolicy.fresh('attacker',88)
            hashes=dict(PREPLANT_CHECKPOINT=file_hash(pre),POSTPLANT_CHECKPOINT=file_hash(guard))
            model.save(source,trained=True,stage='macro_rl',step=11519,lower_checkpoint_hashes=hashes)
            validation.write_text(json.dumps(dict(complete=True,source_hash='fixture',opponent_ais=MAIN_OPPONENT_AIS,opponent_checkpoint_hashes={})))
            config=dict(RESUME=False,SEED=89,PREPLANT_CHECKPOINT=pre,POSTPLANT_CHECKPOINT=guard,
                INITIAL_CHECKPOINT=source,OPPONENT_MODE='major_main',OPPONENT_AIS=MAIN_OPPONENT_AIS,
                OPPONENT_VALIDATION=validation,LEARNING_RATE=.0001)
            data=root/'new';data.mkdir()
            initial_hash=file_hash(source)
            with patch('ghost_champions_v3.macro_v1.gc3_training_attacker_macro.source_signature',return_value='fixture'):
                policy,optimizer,saved=initialize(config,data)
                self.assertIsNone(saved)
                self.assertEqual(len(optimizer.state),0)
                self.assertEqual(module_digest([model]),module_digest([policy]))
                self.assertEqual(torch.load(data/'initial.pt',weights_only=True)['step'],11519)
                policy.save(data/'latest.pt',trained=True,step=11519,optimizer=optimizer.state_dict(),**rng_state())
                resumed,_,saved=initialize(dict(config,RESUME=True),data)
                self.assertEqual(module_digest([policy]),module_digest([resumed]))
                self.assertEqual(saved['step'],11519)
            self.assertEqual(file_hash(source),initial_hash)


if __name__=='__main__':unittest.main()
