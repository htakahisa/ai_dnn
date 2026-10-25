"""Read-only validation of all main opponents before explicit continuation."""
from pathlib import Path
import json
import sys
import torch

MACRO_ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = MACRO_ROOT.parent
INITIAL_CHECKPOINT = MACRO_ROOT/"data/attacker_macro_v1/best.pt"
PREPLANT_CHECKPOINT = AI_ROOT/"data/attacker_preplant_omoko_rl_handoff_v3/best.pt"
POSTPLANT_CHECKPOINT = AI_ROOT/"data/attacker_postplant_fixed_v1/best.pt"
SEED = 2031102200
TORCH_THREADS = 1
OUTPUT_PATH = MACRO_ROOT/"logs/validation/gc3_macro_major_opponents.json"

sys.path.insert(0, str(AI_ROOT.parent))
from ghost_champions_v3.macro_v1.gc3_opponents_attacker_macro import MAIN_OPPONENT_AIS, capture_checkpoint_loads
from ghost_champions_v3.macro_v1.gc3_policy_attacker_macro import MacroPolicy
from ghost_champions_v3.macro_v1.gc3_rollout_attacker_macro import play_macro, load_lower_policies
from ghost_champions_v3.macro_v1.gc3_training_attacker_macro import source_signature
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash
from ghost_champions_v3.gc3_rollout import module_digest
from ghost_champions_v3.gc3_io import write_json


def main():
    torch.set_num_threads(TORCH_THREADS)
    macro, metadata = MacroPolicy.load(INITIAL_CHECKPOINT)
    macro.requires_grad_(False).eval()
    preplant, guard = load_lower_policies(PREPLANT_CHECKPOINT, POSTPLANT_CHECKPOINT)
    hashes = {str(p):file_hash(p) for p in (INITIAL_CHECKPOINT, PREPLANT_CHECKPOINT, POSTPLANT_CHECKPOINT)}
    before = module_digest([macro, preplant, guard])
    rounds = []; assets = {}
    # ConCon creates retake controllers lazily; validate both actual production
    # checkpoints even if this particular smoke round fails before planting.
    with capture_checkpoint_loads() as concon_files:
        from concon_v1.co1_learn_defender_retake import ConconDefenderRetakeController
        from concon_v1.co1_retake_models import runtime_model_path
        for site in ("L", "R"):
            ConconDefenderRetakeController(site, model_path=runtime_model_path(site))
    assets.update(concon_files)
    for i,(code,key) in enumerate(MAIN_OPPONENT_AIS.items()):
        with torch.no_grad():
            rows, stats, _ = play_macro(code, SEED+i, policy=macro, preplant=preplant, guard=guard, opponent_ai=key)
            chosen = rows[::max(1,len(rows)//8)]
            logp, _, values = macro.evaluate_records(chosen)
        torch.testing.assert_close(logp, torch.tensor([r["logp"] for r in chosen]), atol=5e-6, rtol=5e-6)
        torch.testing.assert_close(values, torch.tensor([r["value"] for r in chosen]), atol=5e-6, rtol=5e-6)
        assets.update(stats["opponent_checkpoint_hashes"])
        rounds.append(stats)
        write_json(OUTPUT_PATH, dict(complete=False, rounds=rounds, source_hash=source_signature(),
            opponent_ais=MAIN_OPPONENT_AIS, opponent_checkpoint_hashes=assets, optimizer_updates=0))
        print(code,key,stats["reason"],"planted",stats["planted"],"reward",stats["environment_reward"],flush=True)
    if before != module_digest([macro,preplant,guard]) or hashes != {p:file_hash(p) for p in hashes}:
        raise RuntimeError("Frozen main-AI verification changed learner/lower weights")
    if any(file_hash(p)!=value for p,value in assets.items()):
        raise RuntimeError("Main opponent files changed during verification")
    write_json(OUTPUT_PATH, dict(complete=True, source_hash=source_signature(), opponent_ais=MAIN_OPPONENT_AIS,
        opponent_checkpoint_hashes=assets, optimizer_updates=0, macro_step=metadata["step"],
        all_weights_unchanged=True, existing_files_unchanged=True, rounds=rounds))


if __name__ == "__main__":
    main()
