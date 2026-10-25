"""Monitor the corrected response run. Run from postplant_v2 without options."""
from pathlib import Path
import sys

GUARD_ROOT=Path(__file__).resolve().parents[1]
RUN_NAME='attacker_postplant_response_v3'
REFRESH_SECONDS=5
sys.path.insert(0,str(GUARD_ROOT.parents[1]))

if __name__=='__main__':
    from ghost_champions_v3.postplant_v2.tools import gc3_watch_attacker_guard_metrics as monitor
    monitor.RUN_NAME=RUN_NAME;monitor.REFRESH_SECONDS=REFRESH_SECONDS
    monitor.main()
