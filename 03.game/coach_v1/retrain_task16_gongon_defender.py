"""Fine-tune Gongon's defender-only actor for the adopted watch points."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from coach_v1.common.constants import CHARACTER_CHECKPOINT_PATHS, REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.models import CharacterModelConfig
from coach_v1.retrain_task16_watch_points import OLD_WATCH_HASH, NEW_WATCH_HASH
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.character_trainer import CharacterTrainer, load_character_model


def main() -> None:
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    source = CHARACTER_CHECKPOINT_PATHS["gongon"] / "defender_best.pt"
    target = root / "checkpoints" / "experiments" / "task16_watch_added" / "characters" / "gongon_defender"
    if (target / "latest.pt").exists():
        raise FileExistsError(f"candidate already exists: {target}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload["metadata"]["watch_points_hash"] != OLD_WATCH_HASH:
        raise ValueError("unexpected Gongon defender source hash")
    config = CharacterModelConfig(**payload["metadata"]["model_config"])
    train = [item.example for item in build_character_curriculum(1, seed=2601, count=720)
             if item.side is Side.DEFENDER]
    validation = [item.example for item in build_character_curriculum(1, seed=3601, count=160)
                  if item.side is Side.DEFENDER]
    holdout = [item.example for item in build_character_curriculum(1, seed=4601, count=160)
               if item.side is Side.DEFENDER]
    trainer = CharacterTrainer(1, seed=5601, config=config, learning_rate=0.0001,
                               directory=target)
    if trainer.encoder.watch_points_hash != NEW_WATCH_HASH:
        raise ValueError("watch-point configuration changed during retraining")
    trainer.model.load_state_dict(payload["model_state_dict"], strict=True)
    baseline = trainer.evaluate(holdout).to_dict()
    trainer.fit(train, validation, epochs=8)
    trainer.model = load_character_model(target / "best.pt", slot=1)
    candidate = trainer.evaluate(holdout).to_dict()
    report = {"source": str(source), "candidate": str(target / "best.pt"),
              "old_watch_hash": OLD_WATCH_HASH, "new_watch_hash": NEW_WATCH_HASH,
              "train_count": len(train), "validation_count": len(validation),
              "holdout_count": len(holdout), "baseline": baseline,
              "candidate_holdout": candidate}
    output = REPORTS_DIR / "task16_watch_gongon_defender_retrain.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
