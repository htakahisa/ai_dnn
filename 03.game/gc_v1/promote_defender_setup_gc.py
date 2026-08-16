"""Promote a selected GC Defender Setup checkpoint to runtime best."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import torch


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "defender_setup_gc_data"

DEFAULT_SOURCE = DATA_DIR / "dqn_defender_setup_gc_interrupt.pt"
BEST_PATH = DATA_DIR / "dqn_defender_setup_gc_best.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="checkpoint to promote",
    )
    args = p.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()

    if not source.exists():
        raise FileNotFoundError(source)

    data = torch.load(
        source,
        map_location="cpu",
        weights_only=False,
    )

    required = (
        "model_state_dict",
        "obs_dim",
        "action_dim",
        "candidate_positions",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(
            f"Not a valid GC Setup checkpoint; missing={missing}"
        )

    BEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    backup = None
    if BEST_PATH.exists():
        backup = BEST_PATH.with_suffix(".pt.bak")
        shutil.copy2(BEST_PATH, backup)

    shutil.copy2(source, BEST_PATH)

    print(f"[PROMOTE] source : {source}")
    print(f"[PROMOTE] best   : {BEST_PATH}")
    if backup is not None:
        print(f"[PROMOTE] backup : {backup}")
    print(
        f"[PROMOTE] episode={int(data.get('episode', 0))} "
        f"global_step={int(data.get('global_step', 0))} "
        f"bestWR(meta)={float(data.get('best_win_rate', -1.0)):.3f}"
    )


if __name__ == "__main__":
    main()
