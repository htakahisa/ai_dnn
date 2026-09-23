"""Promote the trained GC defender models into the runtime paths.

The runtime controller reads the two ``best_by_eval.pt`` files below
``gc_v1/data``.  This command backs up the files currently at those paths,
records hashes and source metadata, then atomically installs the requested
Search and Retake checkpoints.

Examples (from ``03.game``)::

    python -m gc_v1.promote_defender_gc
    python -m gc_v1.promote_defender_gc \
        --search-source path/to/search.pt \
        --retake-source path/to/retake.pt

Use ``--dry-run`` to inspect the operation without changing any files.
"""

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
DEFAULT_TARGETS = {
    "search": GC_DIR / "data" / "defender_search_gc_data" / "dqn_defender_search_gc_best_by_eval.pt",
    "retake": GC_DIR / "data" / "defender_retake_gc_data" / "dqn_defender_retake_gc_best_by_eval.pt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_existing(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        shutil.copy2(source, temp_path)
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-source", type=Path, default=DEFAULT_TARGETS["search"])
    parser.add_argument("--retake-source", type=Path, default=DEFAULT_TARGETS["retake"])
    parser.add_argument("--search-target", type=Path, default=DEFAULT_TARGETS["search"])
    parser.add_argument("--retake-target", type=Path, default=DEFAULT_TARGETS["retake"])
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=GC_DIR / "data" / "defender_gc_model_backups",
        help="Directory under which timestamped backups are created.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan only.")
    parser.add_argument(
        "--backup-only",
        action="store_true",
        help="Back up the current runtime files without installing a source checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = {
        "search": (args.search_source, args.search_target),
        "retake": (args.retake_source, args.retake_target),
    }

    resolved = {}
    for phase, (source, target) in pairs.items():
        source = resolve_existing(source, f"{phase} source model")
        target = target.expanduser().resolve()
        resolved[phase] = (source, target)

    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = args.backup_root.expanduser().resolve() / timestamp
    manifest = {
        "operation": "gc_defender_model_promotion",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backup_dir": str(backup_dir),
        "dry_run": args.dry_run,
        "models": {},
    }

    for phase, (source, target) in resolved.items():
        entry = {
            "source": str(source),
            "source_sha256": sha256(source),
            "target": str(target),
            "target_existed": target.is_file(),
        }
        if target.is_file():
            entry["target_sha256_before"] = sha256(target)
            entry["backup"] = str(backup_dir / target.name)
        else:
            entry["target_sha256_before"] = None
            entry["backup"] = None
        manifest["models"][phase] = entry

    print("[PROMOTE] GC defender Search/Retake")
    for phase, entry in manifest["models"].items():
        print(f"  {phase}: {entry['source']} -> {entry['target']}")
        print(f"    source_sha256={entry['source_sha256']}")
        if entry["target_existed"]:
            print(f"    backup={entry['backup']}")

    if args.dry_run:
        print("[DRY-RUN] no files changed")
        return 0

    backup_dir.mkdir(parents=True, exist_ok=False)
    for phase, (source, target) in resolved.items():
        if target.is_file():
            shutil.copy2(target, backup_dir / target.name)
        if not args.backup_only:
            atomic_copy(source, target)
            installed_hash = sha256(target)
            manifest["models"][phase]["target_sha256_after"] = installed_hash
            if installed_hash != manifest["models"][phase]["source_sha256"]:
                raise IOError(f"installed hash mismatch for {phase}: {target}")

    manifest_path = backup_dir / "promotion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[BACKUP] {backup_dir}")
    print(f"[MANIFEST] {manifest_path}")
    if args.backup_only:
        print("[DONE] current defender runtime models backed up; no model was installed")
    else:
        print("[DONE] learned defender models are now active at the runtime paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
