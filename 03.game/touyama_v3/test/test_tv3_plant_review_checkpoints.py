"""Review snapshots preserve stages and runs without pretending to be evaluated."""
from pathlib import Path
import sys
import tempfile
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from touyama_v3.tv3_train_attacker_plant import review_run_directory,save_review_weights


class ReviewCheckpointTests(unittest.TestCase):
    def test_runs_and_stages_keep_independent_weights_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            first=review_run_directory(directory,19)
            second=review_run_directory(directory,19)
            self.assertNotEqual(first,second)
            model=torch.nn.Linear(1,1,bias=False)
            model.warm_start_hash='original-best-hash'
            with torch.no_grad():
                model.weight.fill_(2.)
            zero=save_review_weights(first,model,0,'frc_v1',{'version':19},{},'analysis-hash',0)
            with torch.no_grad():
                model.weight.fill_(3.)
            one=save_review_weights(first,model,1,'frc_v1',{'version':19},{},'analysis-hash',0)
            other=save_review_weights(second,model,0,'frc_v1',{'version':19},{},'analysis-hash',0)
            a=torch.load(zero,weights_only=True)
            b=torch.load(one,weights_only=True)
            self.assertEqual(a['model']['weight'].item(),2.)
            self.assertEqual(b['model']['weight'].item(),3.)
            self.assertTrue(other.exists())
            self.assertEqual(a['analysis_hash'],'analysis-hash')
            self.assertEqual(a['warm_start_hash'],'original-best-hash')
            self.assertEqual(b['completed_sets'],1)
            self.assertEqual(b['evaluation_status'],'not_evaluated')
            self.assertFalse({'optimizer','replay','evaluation'}.intersection(a))


if __name__=='__main__':
    unittest.main()
