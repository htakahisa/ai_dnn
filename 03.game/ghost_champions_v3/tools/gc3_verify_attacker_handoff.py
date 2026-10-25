"""Verify frozen transfer and actual round credit; never train or collect labels."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import sys
import torch

AI_ROOT = Path(__file__).resolve().parents[1]
INITIAL_CHECKPOINT = AI_ROOT/"data/attacker_preplant_omoko_rl_resources_v2/latest.pt"
POSTPLANT_CHECKPOINT = AI_ROOT/"data/attacker_postplant_fixed_v1/best.pt"
BASELINE_ROOT = AI_ROOT/"logs/validation/gc3_preplant_handoff_audit/latest"
CASES = (("FNC", 20361017, "fixed_FNC_0.json"),
         ("SPS", 2030142100, "holdout_SPS_0.json"),
         ("GC", 20367017, "fixed_GC_0.json"))
TORCH_THREADS = 1
GAMMA = .99
OUTPUT_PATH = AI_ROOT/"logs/validation/gc3_preplant_handoff_fix.json"

sys.path.insert(0, str(AI_ROOT.parent))
from ghost_champions_v3.preplant.gc3_policy_attacker_history import load_preplant_policy
from ghost_champions_v3.preplant.gc3_policy_attacker_site import SiteHistoryPolicy
from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import play_preplant, preplant_reward_settings
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash, source_signature
from ghost_champions_v3.preplant.gc3_state_attacker_resources import frozen_guard
from ghost_champions_v3.gc3_rollout import module_digest
from ghost_champions_v3.gc3_io import write_json
from ghost_champions_v3.gc3_reward_phase import WIN_REWARD, LOSS_REWARD


def main():
    torch.set_num_threads(TORCH_THREADS)
    files = (INITIAL_CHECKPOINT, POSTPLANT_CHECKPOINT)
    hashes = {str(path):file_hash(path) for path in files}
    old, metadata = load_preplant_policy(INITIAL_CHECKPOINT)
    old_weights = module_digest([old])
    policy = SiteHistoryPolicy.with_navigation(old).requires_grad_(False).eval()
    policy_weights = module_digest([policy])
    guard = frozen_guard(str(POSTPLANT_CHECKPOINT), file_hash(POSTPLANT_CHECKPOINT))
    guard_weights = module_digest([guard])
    rounds = []
    for code, seed, baseline_file in CASES:
        with torch.no_grad(), redirect_stdout(io.StringIO()):
            rows, stats = play_preplant(code, seed, policy=policy, gamma=GAMMA, postplant_checkpoint=POSTPLANT_CHECKPOINT)
        baseline = json.loads((BASELINE_ROOT/baseline_file).read_text(encoding="utf-8"))["stats"]
        fields = ("planted", "plant", "reason", "ticks", "orb_collections", "actual_round_success", "actual_round_reason",
                  "ultimate_points_start", "ultimate_points_round_end")
        if not all(stats[key] == baseline[key] for key in fields):
            raise RuntimeError("Zero-initialized navigation changed frozen scenario outcome")
        if stats["success"] != stats["actual_round_success"]:
            raise RuntimeError("Plant was incorrectly counted as success")
        last = {name:trajectory[-1] for name, trajectory in _by_agent(rows).items()}
        expected = WIN_REWARD if stats["actual_round_success"] else LOSS_REWARD
        for row in rows:
            if "outcome" in row["reward_components"]:
                raise RuntimeError("Legacy plant-success reward leaked into PPO")
        for row in last.values():
            actual = row["reward_components"].get("round_outcome", 0.)
            if abs(actual-expected*GAMMA**(row["age"]-1)) > 1e-5 or row["discount"] != 0.:
                raise RuntimeError("Actual outcome/time discount was not credited exactly once")
        selected = rows[::max(1, len(rows)//8)]
        with torch.no_grad():
            logp, _, values = policy.evaluate_records(selected)
        torch.testing.assert_close(logp, torch.tensor([r["logp"] for r in selected]), rtol=4e-6, atol=4e-6)
        torch.testing.assert_close(values, torch.tensor([r["value"] for r in selected]), rtol=4e-6, atol=4e-6)
        rounds.append(dict(stats=stats, old_success=baseline["success"], sampled_replay_matches=True,
                    last_preplant_outcome_by_actor={name:row["reward_components"]["round_outcome"] for name,row in last.items()},
                    last_preplant_age_by_actor={name:row["age"] for name,row in last.items()}, baseline_outcome_unchanged=True))
        print(code, seed, "planted", stats["planted"], "won", stats["success"], stats["actual_round_reason"], flush=True)
    if hashes != {str(path):file_hash(path) for path in files} or old_weights != module_digest([old]) or policy_weights != module_digest([policy]) or guard_weights != module_digest([guard]):
        raise RuntimeError("Frozen verification changed checkpoint or weights")
    write_json(OUTPUT_PATH, dict(complete=True, stage="frozen_verification", optimizer_updates=0,
        initial_step=metadata["step"], checkpoint_files_unchanged=True, old_and_new_student_weights_unchanged=True,
        guard_weights_unchanged=True, source_hash=source_signature(), reward_settings=preplant_reward_settings(), rounds=rounds))
    print(OUTPUT_PATH)


def _by_agent(rows):
    from ghost_champions_v3.preplant.gc3_policy_attacker_history import actor_key
    grouped = {}
    for row in rows:
        grouped.setdefault(str(actor_key(row["obs"])), []).append(row)
    return grouped


if __name__ == "__main__":
    main()
