"""Back up and install GC attacker phase checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

GC_DIR = Path(__file__).resolve().parent
PHASES = ("carry", "escort", "retrieve", "guard")
TARGETS = {
    phase: GC_DIR / "data" / f"attacker_{phase}_gc_data" / f"dqn_attacker_{phase}_gc_best_by_eval.pt"
    for phase in PHASES
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temp = Path(name)
    try:
        shutil.copy2(source, temp)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for phase in PHASES:
        p.add_argument(f"--{phase}-source", type=Path, help=f"trained {phase} checkpoint")
    p.add_argument("--backup-root", type=Path, default=GC_DIR / "data" / "attacker_gc_model_backups")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    sources = {phase: getattr(args, f"{phase}_source") for phase in PHASES}
    # No arguments means promote the four canonical files (useful for a
    # previously completed run); specifying sources limits promotion to those phases.
    explicit = any(source is not None for source in sources.values())
    for phase, source in sources.items():
        if source is None and explicit:
            continue
        if source is None:
            source = TARGETS[phase]
        source = source.expanduser().resolve()
        if not source.is_file():
            if any(getattr(args, f"{other}_source") is not None for other in PHASES):
                sources[phase] = None
                continue
            raise FileNotFoundError(f"{phase} source model not found: {source}")
        sources[phase] = source

    active = {phase: source for phase, source in sources.items() if source is not None}
    if not active:
        raise ValueError("no attacker phase source models were provided")
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = args.backup_root.expanduser().resolve() / stamp
    manifest = {"operation": "gc_attacker_model_promotion", "timestamp": datetime.now(timezone.utc).isoformat(), "backup_dir": str(backup_dir), "models": {}}
    for phase, source in active.items():
        target = TARGETS[phase]
        item = {"source": str(source), "source_sha256": digest(source), "target": str(target), "target_existed": target.is_file()}
        item["target_sha256_before"] = digest(target) if target.is_file() else None
        item["backup"] = str(backup_dir / target.name) if target.is_file() else None
        manifest["models"][phase] = item
        print(f"  {phase}: {source} -> {target}")
    if args.dry_run:
        print("[DRY-RUN] no files changed")
        return 0

    backup_dir.mkdir(parents=True, exist_ok=False)
    for phase, source in active.items():
        target = TARGETS[phase]
        if target.is_file():
            shutil.copy2(target, backup_dir / target.name)
        atomic_copy(source, target)
        after = digest(target)
        manifest["models"][phase]["target_sha256_after"] = after
        if after != manifest["models"][phase]["source_sha256"]:
            raise IOError(f"installed hash mismatch for {phase}")
    manifest_path = backup_dir / "promotion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[BACKUP] {backup_dir}")
    print(f"[MANIFEST] {manifest_path}")
    print("[DONE] attacker models are active at the runtime paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
