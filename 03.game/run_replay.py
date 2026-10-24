"""Launch the existing match results and replay viewer."""

from pathlib import Path
import subprocess
import sys


def main() -> int:
    analytics_dir = Path(__file__).resolve().parent / "analytics"
    launcher = analytics_dir / "run_analytics_ui.py"
    if not launcher.is_file():
        raise FileNotFoundError(f"Analytics launcher was not found: {launcher}")
    return subprocess.call(
        [sys.executable, ".\\run_analytics_ui.py"],
        cwd=analytics_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
