"""Compare frozen BC checkpoints on new seeds; this never starts training."""
from pathlib import Path
import sys
import json
import time
import torch

AI_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "attacker_preplant_bc_history_v2"
CHECKPOINTS = (("best", AI_ROOT/"data"/RUN_NAME/"best.pt"),
               ("latest", AI_ROOT/"data"/RUN_NAME/"latest.pt"))
OPPONENT_CODES = ("FNC", "FRC", "TYG", "OMG", "SPS", "GRG", "GC")
FIXED_OPPONENT_AI = "default"
SEED = 2030101100  # Disjoint from collection, DAgger and epoch-selection seeds.
GROUPS_PER_OPPONENT = 2
ROUNDS_PER_GROUP = 4
TORCH_THREADS = 1
OUTPUT_PATH = AI_ROOT/"logs/validation/gc3_preplant_bc_history_v2_holdout_comparison.json"

sys.path.insert(0, str(AI_ROOT.parent))
from ghost_champions_v3.preplant.gc3_policy_attacker_history import HistoryPolicy
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash
from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import play_preplant
from ghost_champions_v3.preplant.gc3_observation_attacker_history import MatchHistory
from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import rng_state, restore_rng
from ghost_champions_v3.gc3_io import write_json


def summarize(rounds):
    count = len(rounds)
    plants = sum(row["planted"] for row in rounds)
    successes = sum(row["success"] for row in rounds)
    return dict(rounds=count, plants=plants, plant_rate=plants/count,
        successes=successes, success_rate=successes/count,
        carrier_closest_site_distance=sum(r["carrier_closest_site_distance"] for r in rounds)/count,
        stationary_fraction=sum(r["carrier_stationary_fraction"] for r in rounds)/count,
        moves_away_fraction=sum(r["carrier_moves_away_fraction"] for r in rounds)/count,
        reasons={reason:sum(r["reason"] == reason for r in rounds) for reason in sorted({r["reason"] for r in rounds})})


def main():
    torch.set_num_threads(TORCH_THREADS)
    state = rng_state()
    report = dict(complete=False, stage="evaluation_only", seed=SEED,
        groups_per_opponent=GROUPS_PER_OPPONENT, rounds_per_group=ROUNDS_PER_GROUP,
        opponent_codes=list(OPPONENT_CODES), fixed_opponent_ai=FIXED_OPPONENT_AI,
        sampled=True, full_match_evaluated=False, optimizer_updates=0, models={})
    started = time.monotonic()
    try:
        for name, checkpoint in CHECKPOINTS:
            before = file_hash(checkpoint)
            policy, metadata = HistoryPolicy.load(checkpoint)
            policy.requires_grad_(False)
            rounds = []
            for index, code in enumerate(OPPONENT_CODES):
                for group in range(GROUPS_PER_OPPONENT):
                    history = MatchHistory()
                    for repeat in range(ROUNDS_PER_GROUP):
                        seed = SEED+index*10000+group*100+repeat
                        _, stats = play_preplant(code, seed, policy=policy,
                            fixed_opponent_ai=FIXED_OPPONENT_AI, history=history)
                        rounds.append(stats)
                per_opponent = summarize([r for r in rounds if r["opponent"] == code])
                report["models"][name] = dict(checkpoint=str(checkpoint), epoch=metadata.get("epoch"),
                    step=metadata.get("step"), summary=summarize(rounds), rounds=rounds,
                    checkpoint_hash=before, checkpoint_unchanged=file_hash(checkpoint) == before,
                    per_opponent={c:summarize([r for r in rounds if r["opponent"] == c])
                                  for c in OPPONENT_CODES if any(r["opponent"] == c for r in rounds)})
                report["elapsed_seconds"] = time.monotonic()-started
                write_json(OUTPUT_PATH, report)
                print(f"{name} epoch={metadata.get('epoch')} {code} plant={per_opponent['plants']}/{per_opponent['rounds']} "
                      f"total={sum(r['planted'] for r in rounds)}/{len(rounds)} "
                      f"elapsed={report['elapsed_seconds']:.0f}s", flush=True)
        report["complete"] = True
        write_json(OUTPUT_PATH, report)
        print(json.dumps({name:data["summary"] for name,data in report["models"].items()}, indent=2), flush=True)
    finally:
        restore_rng(state)


if __name__ == "__main__":
    main()
