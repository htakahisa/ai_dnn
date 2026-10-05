"""Train the defender search basic positioning model without a mode option."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concon_v1.co1_train_defender_search import main as train_main


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    train_main([*args, "--mode", "positioning"])


if __name__ == "__main__":
    main()
