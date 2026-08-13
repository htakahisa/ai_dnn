from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "gc_v1" / "learning_attacker_macro_gc_runtime.py"

if not TARGET.is_file():
    raise FileNotFoundError(f"Not found: {TARGET}")

text = TARGET.read_text(encoding="utf-8")
old = '    def reset_round(self):\n        self.env.reset(\n            forced_strategy="DEFAULT",\n            forced_curriculum_mode="FREE",\n        )\n        self._last_real_tick = None\n        self._last_macro_decision_tick = None\n        self._last_strategy = self.env.current_strategy\n        self._last_q_values = None\n'
new = '    def reset_round(self):\n        # Opening strategy is NOT forced to DEFAULT.\n        # The first real-game tick immediately runs the DQN decision.\n        self.env.reset(\n            forced_curriculum_mode="FREE",\n        )\n        self._last_real_tick = None\n        self._last_macro_decision_tick = None\n        self._last_strategy = self.env.current_strategy\n        self._last_q_values = None\n'

if old not in text:
    if new in text:
        print("[INFO] Opening-DQN patch is already installed.")
    else:
        raise RuntimeError(
            "Expected reset_round() block was not found. No file was modified."
        )
else:
    backup = TARGET.with_suffix(".py.bak_opening_default")
    if not backup.exists():
        shutil.copy2(TARGET, backup)
        print(f"[BACKUP] {backup}")

    text = text.replace(old, new, 1)
    compile(text, str(TARGET), "exec")
    TARGET.write_text(text, encoding="utf-8")
    print("[OK] Opening DEFAULT force removed.")
    print("[OK] First real tick can immediately choose Rush/Split/DEFAULT/etc.")
    print("[OK] No retraining required.")
