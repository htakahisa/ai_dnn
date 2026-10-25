"""Fresh micro BC from existing teacher data; never load an older micro model."""
from pathlib import Path
import sys

AI_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(AI_ROOT.parent))
RUN_NAME="attacker_preplant_retrieval_bc_v2"
DATASET_PATH=AI_ROOT/"data/attacker_preplant_omoko_site_v1"
INITIAL_CHECKPOINT=None
REQUIRE_FRESH_INITIALIZATION=True
REUSE_DAGGER_RUN_PATH=None
SITE_GOAL_POLICY=True
NAVIGATION_FEATURES=True
RETRIEVAL_FEATURES=True
EPOCHS=12
RESUME=False
SEED=20261025
LEARNING_RATE=.0001
BATCH_SIZE=128
BC_ORB_LABEL_WEIGHT=5.0
TORCH_THREADS=1
FIXED_OPPONENT_AI="default"
OPPONENT_CODES=("FNC","FRC","TYG","OMG","SPS","GRG","GC")
EVALUATION_ROUNDS_PER_OPPONENT=4
MIN_BC_PLANT_RATE=.5
MIN_BC_ORB_COLLECTIONS=1
POSTPLANT_CHECKPOINT=AI_ROOT/"data/attacker_postplant_fixed_v1/best.pt"
ENABLE_DAGGER=False
RUN_PATH=AI_ROOT/"data"/RUN_NAME
LOG_PATH=AI_ROOT/"logs"/RUN_NAME

if __name__=="__main__":
    from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import train_bc
    train_bc({key:value for key,value in globals().copy().items() if key.isupper()})
