"""Learn public defuse responses in a new run. Run from postplant_v2 with no options."""
from pathlib import Path
import sys

GUARD_ROOT=Path(__file__).resolve().parent
AI_ROOT=GUARD_ROOT.parent
sys.path.insert(0,str(AI_ROOT.parent))
from ghost_champions_v3.postplant_v2.gc3_train_attacker_guard import *

RUN_NAME='attacker_postplant_response_v2'
START_CHECKPOINT=GUARD_ROOT/'data/attacker_postplant_antidefuse_v1/best.pt'
RUN_PATH=GUARD_ROOT/'data'/RUN_NAME
LOG_PATH=GUARD_ROOT/'logs'/RUN_NAME
RESUME=False
TRAINING_STEPS=20000
RESPONSE_LOSS_WEIGHT=1.
RESPONSE_SAMPLE_FRACTION=.5
ANCHOR_KL_WEIGHT=2.
EVALUATION_PATIENCE=2
KL_BACKTRACK_STEPS=6

if __name__=='__main__':
    from ghost_champions_v3.postplant_v2.gc3_training_attacker_guard import train
    train({name:value for name,value in globals().copy().items() if name.isupper()})
