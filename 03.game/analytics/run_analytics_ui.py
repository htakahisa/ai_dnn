"""Launch the competition-results analytics viewer."""

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analytics_ui import main  # noqa: E402


if __name__ == "__main__":
    main()
