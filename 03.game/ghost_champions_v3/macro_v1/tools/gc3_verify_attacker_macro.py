"""Frozen macro/lower integration check; no training and no demonstration data."""
from pathlib import Path
import json
import sys
import torch

MACRO_ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = MACRO_ROOT.parent
PREPLANT_CHECKPOINT = AI_ROOT/"data/attacker_preplant_omoko_rl_handoff_v3/best.pt"
POSTPLANT_CHECKPOINT = AI_ROOT/"data/attacker_postplant_fixed_v1/best.pt"
MACRO_SEED = 20261022
CASES = (("FNC", 20361017), ("SPS", 20361018))
TORCH_THREADS = 1
OUTPUT_PATH = MACRO_ROOT/"logs/validation/gc3_macro_frozen_probe.json"

sys.path.insert(0, str(AI_ROOT.parent))
from ghost_champions_v3.macro_v1.gc3_policy_attacker_macro import MacroPolicy
from ghost_champions_v3.macro_v1.gc3_rollout_attacker_macro import play_macro, load_lower_policies
from ghost_champions_v3.macro_v1.gc3_training_attacker_macro import source_signature
from ghost_champions_v3.macro_v1.gc3_reward_attacker_macro import reward_settings
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash
from ghost_champions_v3.preplant.gc3_observation_attacker_history import MatchHistory
from ghost_champions_v3.gc3_rollout import module_digest
from ghost_champions_v3.gc3_io import write_json


def main():
    torch.set_num_threads(TORCH_THREADS)
    preplant, guard = load_lower_policies(PREPLANT_CHECKPOINT, POSTPLANT_CHECKPOINT)
    macro = MacroPolicy.fresh("attacker", MACRO_SEED).requires_grad_(False).eval()
    digest = module_digest([macro, preplant, guard])
    hashes = {str(p):file_hash(p) for p in (PREPLANT_CHECKPOINT, POSTPLANT_CHECKPOINT)}
    history = MatchHistory(); memory = None; rounds = []
    for code, seed in CASES:
        with torch.no_grad():
            rows, stats, memory = play_macro(code, seed, policy=macro, preplant=preplant, guard=guard,
                                             history=history, macro_history=memory)
            selected = rows[::max(1, len(rows)//12)]
            logp, _, values = macro.evaluate_records(selected)
        torch.testing.assert_close(logp, torch.tensor([r["logp"] for r in selected]), atol=4e-6, rtol=4e-6)
        torch.testing.assert_close(values, torch.tensor([r["value"] for r in selected]), atol=4e-6, rtol=4e-6)
        if stats["decisions"] != stats["ticks"] or any(abs(sum(r["reward_components"].values())-r["reward"])>1e-6 for r in rows):
            raise RuntimeError("Macro did not run once per tick or reward breakdown was invalid")
        parts = stats["reward_component_sums"]
        expected = parts.get("kill", 0)+parts.get("death", 0)+(10 if stats["planted"] else 0)+(30 if stats["success"] else -30)
        bonus = 10 if stats["success"] and stats["planted"] and not stats["defuse_ever_started"] else 0
        if stats["environment_reward"] != expected+bonus or parts.get("no_defuse_start", 0) != bonus:
            raise RuntimeError("Macro event rewards differ from user settings")
        rounds.append(stats)
        print(json.dumps(stats), flush=True)
    if digest != module_digest([macro, preplant, guard]) or hashes != {str(p):file_hash(p) for p in (PREPLANT_CHECKPOINT, POSTPLANT_CHECKPOINT)}:
        raise RuntimeError("Frozen verification changed models")
    write_json(OUTPUT_PATH, dict(complete=True, source_hash=source_signature(), optimizer_updates=0,
        model_files_unchanged=True, all_weights_unchanged=True, sampled_replay_matches=True,
        full_history_events=len(memory.events), history_rounds=history.round_id+1,
        reward_settings=reward_settings(), rounds=rounds))


if __name__ == "__main__":
    main()
