"""Validate and display the fixed-map coach_v1 watch points."""

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

from map_data import NEW_MAZE_STR

from coach_v1.common.constants import WATCH_POINTS_CONFIG_PATH
from coach_v1.common.watch_points import (
    ALLOWED_SITUATIONS,
    load_watch_points,
    render_watch_points,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=WATCH_POINTS_CONFIG_PATH, help="JSON config path"
    )
    parser.add_argument("--side", choices=("attacker", "defender"))
    parser.add_argument("--situation", choices=sorted(ALLOWED_SITUATIONS))
    parser.add_argument(
        "--no-map", action="store_true", help="validate and print hashes only"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_watch_points(args.config, NEW_MAZE_STR)
    print(f"valid: {len(config.points)} watch points")
    print(f"map_sha256: {config.map_hash}")
    print(f"watch_points_sha256: {config.config_hash}")
    if not args.no_map:
        print(
            render_watch_points(
                config,
                NEW_MAZE_STR,
                side=args.side,
                situation=args.situation,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
