"""Real-engine setup and preplant defense episodes; retake is excluded."""
from functools import partial
import gc
import math
from gc_v1.gc_facing import facing_towards
from ghost_champions_v2.geometry import los
from ghost_champions_v2.rl.rollout import Recorder, OPPONENTS, engine_directory, freeze_modules, module_digest
from .gc3_learn_defender_search import GhostChampionsV3SearchController
from .gc3_spatial_policy import DEFENDER_ARRIVAL_RADIUS


def play_search_round(code, seed, *, site="normal", policy=None, mode="teacher", baseline=False,
                      gamma=.99, baseline_checkpoint=None, gae_lambda=.95):
    recorder = Recorder()
    with engine_directory():
        from simulation_runtime import cpu_inference
        from run_competition_manager import seed_all
        from run_game import VisualFPSBattle, _build_team_ai
        from map_data import NEW_MAZE_STR
        from party_presets import get_preset
        from team_ai import DualRoleTeamAI
        from ghost_champions_v2.rl.controller import LearnedAttackerController
        from ghost_champions_v2.defender_macro_v1.controller import GhostChampionsV2DefenderController
        seed_all(seed)
        own, opponent = get_preset("Ghost Champions"), get_preset(OPPONENTS[code][0])
        factory = GhostChampionsV2DefenderController if baseline else partial(GhostChampionsV3SearchController,
                      search_policy=policy, search_mode=mode, search_recorder=recorder)
        with cpu_inference():
            game = VisualFPSBattle(NEW_MAZE_STR, _build_team_ai(OPPONENTS[code][1]),
                    DualRoleTeamAI("GC search", partial(LearnedAttackerController, checkpoint=baseline_checkpoint), factory),
                    headless=True, disable_side_swap=True, attacker_roster=list(opponent.players), defender_roster=list(own.players),
                    spike_holder_name=opponent.spike_holder, defender_spike_holder_name=own.spike_holder,
                    attacker_igl_name=opponent.igl, defender_igl_name=own.igl)
            game.stop_after_round = True
            controller = getattr(game.defender_controller, "inner", game.defender_controller)
            attacker = getattr(game.attacker_controller, "inner", game.attacker_controller)
            actor = getattr(attacker, "actor", None)
            if actor is not None and hasattr(actor, "collect_statistics"):
                actor.collect_statistics = False
            frozen = freeze_modules(game.attacker_controller, excluded=(game, policy))
            # All inherited defender models remain inference-only.
            inherited = freeze_modules(controller, excluded=(game, policy, getattr(controller, "initial_policy", None)))
            digest = module_digest(frozen + inherited)
            allies = [c for c in game.chars if c.team == "D"]
            enemies = [c for c in game.chars if c.team == "A"]
            def charges():
                return sum(sum(getattr(c, a+"_charges", 0) for a in ("smoke", "flash", "recon", "dance", "ash")) for c in allies)
            def potential():
                result = .1*(sum(c.is_alive for c in allies)-sum(c.is_alive for c in enemies))
                if not baseline:
                    for c in allies:
                        if c.is_alive and str(c.name) in controller.initial_goals:
                            distance = max(0., math.dist(c.pos, controller.initial_goals[str(c.name)])-DEFENDER_ARRIVAL_RADIUS)
                            result += .02*max(0., 1-distance/10)
                return result
            initial_charges = charges()
            revealed, blinded = set(), set()
            aim_ticks = coverage_ticks = alive_ticks = 0
            terminal = False
            for tick in range(400):
                setup = game.defender_setup_phase.active
                before = potential()
                old_hp = sum(c.hp for c in enemies if c.is_alive)
                game._simulate_tick()
                terminal = game.is_planted or game.round_over or game.match_over
                if not setup:
                    own_recon = any(b.get("team") == "D" for b in game.recon_bursts)
                    own_flash = any(b.get("team") == "D" for b in game.flash_bursts)
                    new_reveals = {str(c.name) for c in enemies if c.is_alive and c.reveal_remaining > 0}-revealed if own_recon else set()
                    new_blinds = {str(c.name) for c in enemies if c.is_alive and c.blind_remaining > 0}-blinded if own_flash else set()
                    revealed.update(new_reveals)
                    blinded.update(new_blinds)
                    reward = gamma*(0. if terminal else potential())-before + .02*(len(new_reveals)+len(new_blinds))
                    reward += .0002*max(0, old_hp-sum(c.hp for c in enemies if c.is_alive))
                    if terminal:
                        reward += 1. if game.defender_wins and not game.is_planted else -1.
                    recorder.reward(reward, gamma, terminal)
                    for c in allies:
                        name = str(c.name)
                        if not c.is_alive:
                            if name in recorder.latest:
                                recorder.latest.pop(name)["discount"] = 0.
                            continue
                        alive_ticks += 1
                        if not baseline and name in controller.initial_aims:
                            aim = controller.initial_aims[name]
                            aim_ticks += c.facing == facing_towards(c.pos, aim)
                            coverage_ticks += los(game.grid, tuple(c.pos), aim, game._smoke_cells())
                if terminal:
                    break
            if not terminal:
                raise RuntimeError("Defender search episode exceeded tick limit")
            success = bool(game.defender_wins and not game.is_planted)
            record = game.analytics_tracker.round_records[-1] if game.analytics_tracker.round_records else {}
            stats = dict(opponent=code, seed=seed, site=site, winner="defender" if success else "attacker",
                         reason="planted" if game.is_planted else record.get("reason", "unknown"),
                         planted=bool(game.is_planted), search_success=success, decisions=len(recorder.rows),
                         ability_uses=initial_charges-charges(), revealed=len(revealed), blinded=len(blinded),
                         aim_ticks=aim_ticks, coverage_ticks=coverage_ticks, alive_ticks=alive_ticks, ticks=tick+1,
                         search_only=True, retake_trained=False, opponent_weights_unchanged=True)
            if module_digest(frozen + inherited) != digest:
                raise RuntimeError("Frozen opponent or inherited defender weights changed")
            if policy is not None and any(not p.requires_grad for p in policy.parameters()):
                raise RuntimeError("Search policy was unexpectedly frozen")
            rows = recorder.advantages(gae_lambda)
            del game
    gc.collect()
    return rows, stats
