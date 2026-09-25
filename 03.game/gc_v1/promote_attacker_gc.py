"""Back up and install GC attacker phase checkpoints.

Without source arguments, deploy the latest training best-by-eval bundle for
Carry/Escort/Guard, even when it did not pass safety selection.  Use
--selected to require an approved trained bundle instead.  Explicit phase
sources remain supported.
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
PHASES = ("carry", "escort", "retrieve", "guard")
TARGETS = {
    phase: GC_DIR / "data" / f"attacker_{phase}_gc_data" / f"dqn_attacker_{phase}_gc_best_by_eval.pt"
    for phase in PHASES
}
RUN_ROOT = GC_DIR / "data" / "attacker_gc_escort_support_current"
SELECTED_BUNDLE = RUN_ROOT / "selected" / "best_escort_support_ab_bundle.json"
CANDIDATE_BUNDLE = RUN_ROOT / "training" / "best_by_eval_bundle.json"
AUTO_PHASES = ("carry", "escort", "guard")


def bundle_sources(bundle_path: Path, require_trained: bool) -> tuple[dict[str, Path], dict]:
    """Resolve a coherent bundle without accepting paths outside its folder."""
    bundle_path = bundle_path.expanduser().resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"attacker bundle not found: {bundle_path}")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if require_trained and bundle.get("selected_branch") != "trained":
        raise ValueError(
            f"no trained attacker bundle passed selection: {bundle_path} "
            f"(selected_branch={bundle.get('selected_branch')!r}). "
            "Use --candidate only if you intentionally accept an unselected model."
        )
    names = bundle.get("models")
    if not isinstance(names, dict):
        raise ValueError(f"bundle has no model mapping: {bundle_path}")
    directory = bundle_path.parent
    sources = {}
    for phase in AUTO_PHASES:
        name = names.get(phase)
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"invalid {phase} model filename in {bundle_path}: {name!r}")
        source = (directory / name).resolve()
        if source.parent != directory or not source.is_file():
            raise FileNotFoundError(f"{phase} model from bundle not found: {source}")
        sources[phase] = source
    return sources, bundle


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
    p.add_argument(
        "--selected", action="store_true",
        help="require and deploy the trained bundle that passed safety selection",
    )
    p.add_argument("--backup-root", type=Path, default=GC_DIR / "data" / "attacker_gc_model_backups")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    sources = {phase: getattr(args, f"{phase}_source") for phase in PHASES}
    explicit = any(source is not None for source in sources.values())
    if explicit and args.selected:
        p.error("--selected cannot be combined with explicit phase sources")
    bundle = None
    if not explicit:
        bundle_path = SELECTED_BUNDLE if args.selected else CANDIDATE_BUNDLE
        discovered, bundle = bundle_sources(
            bundle_path, require_trained=args.selected,
        )
        sources.update(discovered)
        print(
            f"[AUTO SOURCE] {bundle_path} episode={bundle.get('episode')} "
            f"mode={'selected' if args.selected else 'BEST BY EVAL (may be unsafe)'}"
        )
        if not args.selected:
            evaluation = bundle.get("evaluation", {})
            print(
                "[WARN] automatic best-by-eval deployment bypasses safety selection; "
                f"worst_timeout_rate={evaluation.get('worst_timeout_rate')} "
                f"round_win_rate={evaluation.get('round_win_rate')}"
            )
    for phase, source in sources.items():
        if source is None:
            continue
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"{phase} source model not found: {source}")
        sources[phase] = source

    active = {phase: source for phase, source in sources.items() if source is not None}
    if not active:
        raise ValueError("no attacker phase source models were provided")
    if not explicit and all(
        target.is_file() and digest(active[phase]) == digest(target)
        for phase, target in TARGETS.items() if phase in active
    ):
        raise ValueError("auto-selected attacker models are already active; nothing to promote")
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
