"""Verify normal run_game factories and saved best models without training."""
from pathlib import Path
import sys
import contextlib
import io
import json
import torch

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

CHECK_OPPONENTS = ('gc_v1', 'touyama_v2', 'omoko_v1', 'fnatic_v3', 'frc_v1', 'toru_ai_v3')
LIVE_ROUND_OPPONENTS = ('omoko_v1', 'frc_v1')
OWN_PRESETS = ('Eine Kleine', 'SUPES', 'BBL')
SEED = 804000042  # Saved frc_v1 independent evaluation seed.
MAX_ROUND_STEPS = 400
OUTPUT_PATH = HERE/'logs'/'best_runtime_check.json'
TORCH_THREADS = 1


def main():
    from toruAI_v4.tv4_scenario import Scenario, OPPONENTS
    from toruAI_v4.tv4_train_defender_analysis import legacy_root, seed_all, relocate_debug_logs
    from toruAI_v4.tv4_train_defender_search import eligible_presets
    from simulation_runtime import cpu_inference
    from party_presets import get_preset
    torch.set_num_threads(TORCH_THREADS)
    scenario = Scenario()
    results = []
    for opponent in CHECK_OPPONENTS:
        for side in ('A', 'D'):
            with legacy_root(), cpu_inference(enabled=True), contextlib.redirect_stdout(io.StringIO()):
                from run_game import VisualFPSBattle, _build_team_ai
                seed_all(SEED)
                own = get_preset(eligible_presets(OWN_PRESETS, opponent)[0])
                enemy = get_preset(OPPONENTS[opponent][1])
                attack_roster, defense_roster = (own, enemy) if side=='A' else (enemy, own)
                game = VisualFPSBattle(scenario.maze,
                    _build_team_ai('toru_ai_v4' if side=='A' else OPPONENTS[opponent][0], device='cpu'),
                    _build_team_ai('toru_ai_v4' if side=='D' else OPPONENTS[opponent][0], device='cpu'),
                    headless=True, attacker_roster=list(attack_roster.players), defender_roster=list(defense_roster.players),
                    spike_holder_name=attack_roster.spike_holder, defender_spike_holder_name=defense_roster.spike_holder,
                    attacker_igl_name=attack_roster.igl, defender_igl_name=defense_roster.igl,
                    attacker_team_name=attack_roster.name, defender_team_name=defense_roster.name, disable_side_swap=True)
                game.stop_after_round, game.analytics_tracker = True, None
                game._record_replay_frame = lambda: None
                controller = game.attacker_controller if side=='A' else game.defender_controller
                relocate_debug_logs(game.defender_controller if side=='A' else game.attacker_controller, None)
                if controller.learned is None:
                    raise ValueError(f'{opponent} {side}: generic fallback: {controller.status}')
                if side=='A' and controller.learned.guard_policy is None:
                    raise ValueError(f'{opponent}: guard fallback: {controller.status}')
                if side=='D' and set(controller.learned.retake) != {'L','R'}:
                    raise ValueError(f'{opponent}: retake fallback: {controller.status}')
                steps = 0
                guard_seen = retake_seen = False
                if opponent in LIVE_ROUND_OPPONENTS:
                    while not game.round_over and not game.match_over:
                        game.step_tick(); steps += 1
                        guard_seen |= side=='A' and controller.learned.guard is not None
                        retake_seen |= side=='D' and bool(game.is_planted)
                        if steps > MAX_ROUND_STEPS:
                            raise RuntimeError('Runtime check exceeded round step budget')
                result = dict(opponent=opponent, role=side, status=controller.status,
                    completed_live_round=opponent in LIVE_ROUND_OPPONENTS, steps=steps,
                    guard_activated=guard_seen, retake_activated=retake_seen)
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='status'}, ensure_ascii=False), flush=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf8')


if __name__=='__main__':
    main()
