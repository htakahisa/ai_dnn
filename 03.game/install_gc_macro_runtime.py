"""
Install GC Macro runtime into the existing game without overwriting
ghost_champions_v1.py.

Usage:
    Put this file in 03.game root together with:
      - run_game.py
      - ghost_champions_v1.py
      - gc_v1/

    Put these two generated runtime files in gc_v1/:
      - learning_attacker_macro_gc_runtime.py
      - train_attacker_macro_gc_v28.py

    Put the final model at:
      gc_v1/data/attacker_macro_gc_data/dqn_attacker_macro_gc_final.pt

    Put ghost_champions_v1_macro.py in 03.game root.

Then run this installer once.
"""

from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
RUN_GAME = ROOT / "run_game.py"
WRAPPER = ROOT / "ghost_champions_v1_macro.py"
GC_DIR = ROOT / "gc_v1"
RUNTIME = GC_DIR / "learning_attacker_macro_gc_runtime.py"
TRAIN_V28 = GC_DIR / "train_attacker_macro_gc_v28.py"
MODEL = (
    GC_DIR
    / "data"
    / "attacker_macro_gc_data"
    / "dqn_attacker_macro_gc_final.pt"
)

required = [RUN_GAME, WRAPPER, RUNTIME, TRAIN_V28, MODEL]
missing = [p for p in required if not p.is_file()]
if missing:
    raise FileNotFoundError(
        "Missing required files:\n"
        + "\n".join(f"  - {p}" for p in missing)
    )

backup = RUN_GAME.with_suffix(".py.bak_gc_macro")
if not backup.exists():
    shutil.copy2(RUN_GAME, backup)
    print(f"[BACKUP] {backup.name}")

text = RUN_GAME.read_text(encoding="utf-8")

# Replace the canonical GC import while preserving all existing menu keys.
patterns = [
    (
        r"from\s+ghost_champions_v1\s+import\s+"
        r"GhostChampionsV1AttackerController\s*,\s*"
        r"GhostChampionsV1DefenderController",
        "from ghost_champions_v1_macro import "
        "GhostChampionsV1AttackerController, "
        "GhostChampionsV1DefenderController",
    ),
    (
        r"from\s+ghost_champions_v1\s+import\s*\(\s*"
        r"GhostChampionsV1AttackerController\s*,\s*"
        r"GhostChampionsV1DefenderController\s*,?\s*\)",
        "from ghost_champions_v1_macro import ("
        "GhostChampionsV1AttackerController, "
        "GhostChampionsV1DefenderController)",
    ),
]

changed = False
for pattern, repl in patterns:
    new_text, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if n:
        text = new_text
        changed = True
        break

if not changed:
    # Some versions import the module/builder rather than both classes.
    if "ghost_champions_v1_macro" in text:
        print("[INFO] run_game.py already uses Macro wrapper.")
        changed = True
    else:
        raise RuntimeError(
            "Could not find the Ghost Champions v1 import in run_game.py. "
            "No file was modified. Send the current run_game.py if this occurs."
        )

RUN_GAME.write_text(text, encoding="utf-8")
compile(RUN_GAME.read_text(encoding="utf-8"), str(RUN_GAME), "exec")
compile(WRAPPER.read_text(encoding="utf-8"), str(WRAPPER), "exec")
compile(RUNTIME.read_text(encoding="utf-8"), str(RUNTIME), "exec")

print("[OK] GC Macro practical integration installed.")
print("[OK] Existing Ghost Champions menu key is unchanged.")
print("[OK] Guard/Retrieve remain existing phase models.")
print("[OK] Pre-plant Carry/Escort MOVE is now guided by Macro DQN.")
print(f"[MODEL] {MODEL}")
