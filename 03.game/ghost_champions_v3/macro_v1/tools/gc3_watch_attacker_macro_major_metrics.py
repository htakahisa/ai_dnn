"""Live charts for continuation against main AIs; never start learning."""
from pathlib import Path
import sys

MACRO_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "attacker_macro_major_v1"
REFRESH_SECONDS = 5
ROLLING_EPISODES = 20
REPORT_NAME = "gc3_attacker_macro_major_training_progress"

sys.path.insert(0, str(MACRO_ROOT.parents[1]))

if __name__ == "__main__":
    from ghost_champions_v3.macro_v1.tools import gc3_watch_attacker_macro_metrics as charts
    charts.RUN_NAME = RUN_NAME
    charts.REFRESH_SECONDS = REFRESH_SECONDS
    charts.ROLLING_EPISODES = ROLLING_EPISODES
    charts.REPORT_NAME = REPORT_NAME
    charts.main()
