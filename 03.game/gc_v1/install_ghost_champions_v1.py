
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

FILES = {
    "run_game.py": ROOT / "run_game.py",
    "roster_select.py": ROOT / "roster_select.py",
    "run_competition_manager.py": ROOT / "run_competition_manager.py",
}

for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"{name} が見つかりません: {path}")

def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak_gc_v1")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[BACKUP] {bak.name}")

# run_game.py
p = FILES["run_game.py"]
backup(p)
t = p.read_text(encoding="utf-8")

import_line = (
    "from ghost_champions_v1 import "
    "GhostChampionsV1AttackerController, GhostChampionsV1DefenderController\n"
)
if import_line not in t:
    marker = "from team_ai import DualRoleTeamAI\n"
    if marker not in t:
        raise RuntimeError("run_game.py: team_ai import が見つかりません")
    t = t.replace(marker, marker + import_line, 1)

team_block = '''    if normalized in {
        "ghost_champions_v1",
        "ghost_champions",
        "gc_v1",
        "gc1",
        "ghost champions v1",
    }:
        return DualRoleTeamAI(
            name="Ghost Champions v1",
            attacker_factory=lambda: GhostChampionsV1AttackerController(greedy=True),
            defender_factory=lambda: GhostChampionsV1DefenderController(greedy=True),
        )

'''
if team_block.strip() not in t:
    marker = '    if normalized == "toru_ai_v3":\n'
    if marker not in t:
        raise RuntimeError("run_game.py: _build_team_ai の Toru AI v3 分岐が見つかりません")
    t = t.replace(marker, team_block + marker, 1)

attacker_block = '''    if normalized in {
        "ghost_champions_v1",
        "ghost_champions",
        "gc_v1",
        "gc1",
        "ghost champions v1",
    }:
        return GhostChampionsV1AttackerController(greedy=True)

'''
attacker_fn = t.find("def _build_attacker_controller")
if attacker_fn != -1:
    next_def = t.find("\ndef ", attacker_fn + 1)
    next_class = t.find("\nclass ", attacker_fn + 1)
    ends = [x for x in (next_def, next_class) if x != -1]
    segment_end = min(ends) if ends else len(t)
    segment = t[attacker_fn:segment_end]
    if "GhostChampionsV1AttackerController" not in segment:
        norm_pos = t.find("normalized =", attacker_fn, segment_end)
        if norm_pos == -1:
            raise RuntimeError("run_game.py: attacker normalized が見つかりません")
        insert_at = t.find("\n", norm_pos) + 1
        t = t[:insert_at] + "\n" + attacker_block + t[insert_at:]

defender_block = '''    if normalized in {
        "ghost_champions_v1",
        "ghost_champions",
        "gc_v1",
        "gc1",
        "ghost champions v1",
    }:
        return GhostChampionsV1DefenderController(greedy=True)

'''
defender_fn = t.find("def _build_defender_controller")
if defender_fn != -1:
    next_def = t.find("\ndef ", defender_fn + 1)
    next_class = t.find("\nclass ", defender_fn + 1)
    ends = [x for x in (next_def, next_class) if x != -1]
    segment_end = min(ends) if ends else len(t)
    segment = t[defender_fn:segment_end]
    if "GhostChampionsV1DefenderController" not in segment:
        body_start = t.find("\n", defender_fn) + 1
        if "normalized =" not in segment:
            norm_stmt = '    normalized = str(key or "default").strip().lower()\n'
            t = t[:body_start] + norm_stmt + t[body_start:]
            segment_end += len(norm_stmt)
        norm_pos = t.find("normalized =", defender_fn, segment_end)
        insert_at = t.find("\n", norm_pos) + 1
        t = t[:insert_at] + "\n" + defender_block + t[insert_at:]

p.write_text(t, encoding="utf-8")
print("[OK] run_game.py")

# roster_select.py
p = FILES["roster_select.py"]
backup(p)
t = p.read_text(encoding="utf-8")

def add_dict_option(text, dict_name, option_line):
    start = text.find(dict_name + " = {")
    if start == -1:
        raise RuntimeError(f"roster_select.py: {dict_name} が見つかりません")
    end = text.find("}", start)
    segment = text[start:end]
    if '"Ghost Champions v1"' in segment:
        return text
    line_end = text.find("\n", start) + 1
    return text[:line_end] + option_line + text[line_end:]

t = add_dict_option(
    t, "ATTACKER_CONTROLLER_OPTIONS",
    '    "Ghost Champions v1": "ghost_champions_v1",\n'
)
t = add_dict_option(
    t, "DEFENDER_CONTROLLER_OPTIONS",
    '    "Ghost Champions v1": "ghost_champions_v1",\n'
)
p.write_text(t, encoding="utf-8")
print("[OK] roster_select.py")

# run_competition_manager.py
p = FILES["run_competition_manager.py"]
backup(p)
t = p.read_text(encoding="utf-8")

start = t.find("CONTROLLER_OPTIONS = {")
if start == -1:
    raise RuntimeError("run_competition_manager.py: CONTROLLER_OPTIONS が見つかりません")
end = t.find("}", start)
segment = t[start:end]
if '"Ghost Champions v1"' not in segment:
    line_end = t.find("\n", start) + 1
    t = (
        t[:line_end]
        + '    "Ghost Champions v1": "ghost_champions_v1",\n'
        + t[line_end:]
    )
p.write_text(t, encoding="utf-8")
print("[OK] run_competition_manager.py")

for name, path in FILES.items():
    compile(path.read_text(encoding="utf-8"), str(path), "exec")

print("")
print("Ghost Champions v1 registration complete.")
print("Backups: *.bak_gc_v1")
