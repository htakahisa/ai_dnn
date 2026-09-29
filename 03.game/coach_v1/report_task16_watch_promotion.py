"""Apply the existing Task 15 promotion gate to the new watch-point team."""

from __future__ import annotations

import json

from coach_v1.common.constants import REPORTS_DIR
from coach_v1.task15_self_play import PoolEvaluation, decide_promotion


def _load(name: str) -> PoolEvaluation:
    return PoolEvaluation.from_dict(json.loads(
        (REPORTS_DIR / name).read_text(encoding="utf-8")))


def main() -> None:
    existing = _load("task16_watch_added_eval.json")
    historical = _load("task16_watch_added_historical_eval.json")
    incumbent = _load("task15_holdout_promoted_1510.json")
    if (existing.attacker_sha256 != historical.attacker_sha256
            or existing.defender_sha256 != historical.defender_sha256):
        raise ValueError("candidate checkpoint hashes differ between evaluations")
    candidate = PoolEvaluation(
        existing.attacker_checkpoint, existing.attacker_sha256,
        existing.defender_checkpoint, existing.defender_sha256,
        existing.matches + historical.matches,
        {**existing.recommended_weights, **historical.recommended_weights},
    )
    decision = decide_promotion(
        candidate, incumbent,
        historical_opponents=("coach_v1_task11_6",),
    )
    payload = {"incumbent": incumbent.to_dict(), "candidate": candidate.to_dict(),
               "promotion": decision.to_dict()}
    output = REPORTS_DIR / "task16_watch_added_promotion.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"incumbent": incumbent.overall(),
                      "candidate": candidate.overall(),
                      "promotion": decision.to_dict()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
