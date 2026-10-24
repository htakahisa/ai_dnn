"""Live charts for the restarted micro run; run from ghost_champions_v3."""
from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "attacker_preplant_retrieval_v2"
BC_RUN = "attacker_preplant_retrieval_bc_v2"
REFRESH_SECONDS = 5
ROLLING_EPISODES = 20
REPORT_NAME = "gc3_attacker_micro_retrieval_training_progress"

sys.path.insert(0, str(AI_ROOT.parent))

if __name__ == "__main__":
    from ghost_champions_v3.tools import gc3_watch_attacker_preplant_metrics as charts
    charts.RL_RUN = RUN_NAME
    charts.BC_RUN = BC_RUN
    charts.REFRESH_SECONDS = REFRESH_SECONDS
    charts.ROLLING_EPISODES = ROLLING_EPISODES
    charts.REPORT_NAME = REPORT_NAME
    charts.main()
