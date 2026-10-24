"""Full attacking rounds: macro learns, all lower/opponent weights stay frozen."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import torch
from ..gc3_policy import FreshPolicy
from ..gc3_rollout import Recorder, engine_directory, freeze_modules, module_digest
from ..preplant.gc3_policy_attacker_history import load_preplant_policy
from ..preplant.gc3_data_attacker_preplant import file_hash
from ..preplant.gc3_rollout_attacker_preplant import MAJOR_TEAMS, distinguish_same_roster
from ..preplant.gc3_observation_attacker_history import MatchHistory
from ..preplant.gc3_state_attacker_resources import MatchResources
from .gc3_learn_attacker_macro import MacroAttackerController
from .gc3_reward_attacker_macro import MacroRewards, REWARD_VERSION
from .gc3_opponents_attacker_macro import MAIN_OPPONENT_AIS, build_opponent, capture_checkpoint_loads

MAX_ROUND_TICKS = 400


def load_lower_policies(preplant_checkpoint, guard_checkpoint):
    preplant, before = load_preplant_policy(preplant_checkpoint)
    guard, after = FreshPolicy.load(guard_checkpoint, role="attacker")
    if not before.get("trained") or not after.get("trained") or not hasattr(preplant, "site"):
        raise ValueError("Macro requires trained site-conditioned preplant and guard checkpoints")
    return preplant.requires_grad_(False).eval(), guard.requires_grad_(False).eval()


def play_macro(code, seed, *, policy, preplant, guard, history=None, gamma=.99, gae_lambda=.95,
               macro_history=None, opponent_ai="default", expected_opponent_files=None):
    if code not in MAJOR_TEAMS:
        raise ValueError("Unknown major roster")
    if opponent_ai not in ("default", MAIN_OPPONENT_AIS[code]):
        raise ValueError("Opponent AI does not match the major team roster")
    history = history if history is not None else MatchHistory()
    if not hasattr(history, "match_resources"):
        history.match_resources = MatchResources()
    recorder = Recorder()
    raw = MacroAttackerController(macro_policy=policy, preplant_policy=preplant, guard_policy=guard,
                                 history=history, recorder=recorder)
    if macro_history is not None:
        raw.macro_history = macro_history
    with engine_directory(), redirect_stdout(io.StringIO()), capture_checkpoint_loads() as opponent_files:
        from run_game import VisualFPSBattle, _build_team_ai
        from run_competition_manager import seed_all
        from simulation_runtime import cpu_inference
        from team_ai import DualRoleTeamAI
        from controllers import DefaultDefenderController
        from party_presets import get_preset
        from map_data import NEW_MAZE_STR
        seed_all(seed)
        own, opposing = get_preset("Ghost Champions"), get_preset(MAJOR_TEAMS[code])
        with cpu_inference():
            team = DualRoleTeamAI("GC v3 macro", lambda:raw, DefaultDefenderController)
            game = VisualFPSBattle(NEW_MAZE_STR, team, build_opponent(opponent_ai), headless=True,
                disable_side_swap=True, attacker_roster=list(own.players), defender_roster=list(opposing.players),
                spike_holder_name=own.spike_holder, defender_spike_holder_name=opposing.spike_holder,
                attacker_igl_name=own.igl, defender_igl_name=opposing.igl)
            game.stop_after_round = True
            distinguish_same_roster(game)
            history.match_resources.restore(game, code)
            frozen = [preplant, guard]+freeze_modules(game.defender_controller, excluded=(game, policy))
            digest = module_digest(frozen)
            with MacroRewards(game) as rewards:
                for tick in range(MAX_ROUND_TICKS):
                    game._simulate_tick()
                    terminal = bool(game.round_over or game.match_over)
                    parts = rewards.finish_tick(terminal, bool(game.attacker_wins))
                    recorder.reward(sum(parts.values()), gamma, terminal, components=parts)
                    if terminal:
                        break
                else:
                    raise RuntimeError("Macro round exceeded engine limit")
            if module_digest(frozen) != digest:
                raise RuntimeError("Macro rollout changed frozen lower/opponent weights")
            game.defender_controller.verify()
            if any(file_hash(path) != digest for path,digest in opponent_files.items()):
                raise RuntimeError("Opponent checkpoint files changed during rollout")
            if expected_opponent_files is not None and any(expected_opponent_files.get(path)!=digest for path,digest in opponent_files.items()):
                raise RuntimeError("Main opponent weights differ from the frozen run manifest")
            history.match_resources.capture(game, code)
            rows = recorder.advantages(gae_lambda)
            for row in rows:
                row["reward_version"] = REWARD_VERSION
            result = game.analytics_tracker.round_records[-1]
            stats = dict(phase="macro", opponent=code, opponent_ai=opponent_ai, seed=seed,
                opponent_checkpoint_hashes=opponent_files,
                decisions=len(rows), ticks=tick+1, success=bool(game.attacker_wins), actual_round_success=bool(game.attacker_wins),
                planted=bool(game.is_planted), plant=list(game.planted_pos) if game.planted_pos else None,
                reason=result["reason"], actual_round_reason=result["reason"],
                defuse_ever_started=rewards.defuse_ever_started, reward_component_sums=dict(rewards.totals),
                environment_reward=sum(rewards.totals.values()), reward=REWARD_VERSION,
                site_switches=raw.switches, final_goal_site=("A", "B")[raw.current_goal],
                decisions_preplant=sum(not r["obs"][2] for r in rows), decisions_postplant=sum(bool(r["obs"][2]) for r in rows),
                lower_weights_unchanged=True, opponent_weights_unchanged=True, macro_only_learning=True)
            history.finish_round(stats)
            raw.macro_history.sync_public_events()
            return rows, stats, raw.macro_history
