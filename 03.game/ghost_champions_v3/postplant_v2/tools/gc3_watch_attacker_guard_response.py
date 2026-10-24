"""View the best guard from the corrected response run; run from postplant_v2."""
from pathlib import Path
import sys

GUARD_ROOT=Path(__file__).resolve().parents[1]
RUN_NAME='attacker_postplant_response_v3'
CHECKPOINT_NAME='best.pt'
sys.path.insert(0,str(GUARD_ROOT.parents[1]))

if __name__=='__main__':
    from ghost_champions_v3.postplant_v2.tools import gc3_watch_attacker_guard as viewer
    from ghost_champions_v3.postplant_v2 import gc3_train_attacker_guard_response as settings
    viewer.RUN_NAME=RUN_NAME;viewer.CHECKPOINT_NAME=CHECKPOINT_NAME;viewer.SETTINGS_MODULE=settings
    viewer.main()
