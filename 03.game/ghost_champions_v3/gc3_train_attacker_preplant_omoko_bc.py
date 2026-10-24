"""Fine-tune GC v3 best with Omoko site/utility/orb demonstrations."""
from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_ROOT.parent))

RUN_NAME = "attacker_preplant_omoko_bc_site_v1"
DATASET_NAME = "attacker_preplant_omoko_site_v1"
INITIAL_CHECKPOINT = AI_ROOT/"data/attacker_preplant_rl_history_v2/best.pt"
SITE_GOAL_POLICY = True
REUSE_DAGGER_RUN_PATH = None  # Existing ConCon data/files remain available.
EPOCHS = 6
RESUME = False
SEED = 20261015
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
BC_ORB_LABEL_WEIGHT = 5.0  # BC loss weighting only; does not choose an action.
TORCH_THREADS = 1
FIXED_OPPONENT_AI = "default"
OPPONENT_CODES = ("FNC", "FRC", "TYG", "OMG", "SPS", "GRG", "GC")
EVALUATION_ROUNDS_PER_OPPONENT = 4
MIN_BC_PLANT_RATE = 0.01
MIN_BC_ORB_COLLECTIONS = 1  # Actual complete collections; sampled action alone is insufficient.
ENABLE_DAGGER = False
TEACHER_KIND = "omoko"
DAGGER_ROUNDS_PER_EPOCH = 7
DAGGER_TEACHER_MAPS = ()
DATASET_PATH = AI_ROOT/"data"/DATASET_NAME
RUN_PATH = AI_ROOT/"data"/RUN_NAME
LOG_PATH = AI_ROOT/"logs"/RUN_NAME

if __name__ == "__main__":
    from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import train_bc
    train_bc({k:v for k,v in globals().copy().items() if k.isupper()})
