"""Postplant-only real-engine episodes and outcome/effect-based rewards."""
from functools import partial
import math
import random
import numpy as np
from gc_v1.gc_facing import facing_towards
from ghost_champions_v2.geometry import valid, los, watch_cells
from ghost_champions_v2.rl.rollout import Recorder, OPPONENTS, engine_directory, freeze_modules, module_digest
from . import BASELINE_CHECKPOINT
from .gc3_map_attacker_guard import GUARD_MAP, validate_map, site_plan
from .gc3_learn_attacker_guard import GhostChampionsV3AttackerController
from .gc3_spatial_policy import POSTPLANT_ARRIVAL_RADIUS


def setup_case(game, site, seed):
    """Varied plants, retake approaches, survivors, spent skills and time left."""
    rng = random.Random(seed)
    plan = GUARD_MAP[site]
    validate_map(game.grid)
    game.defender_setup_phase.finish()
    game.stop_after_round = True
    game.is_planted = True
    game.is_defused = False
    game.planted_pos = rng.choice(plan["plants"])
    plan = site_plan(game.planted_pos)
    game.spike_pos = None
    game.analytics_tracker._planted_pos = game.planted_pos
    game.battle_tick = rng.randint(15, 45)
    from game_core import SPIKE_DETONATION_TICKS
    game.detonate_timer = rng.choice((SPIKE_DETONATION_TICKS, SPIKE_DETONATION_TICKS, 20, 12))
    attackers = sorted((c for c in game.chars if c.team == "A"), key=lambda c: str(c.name))
    defenders = sorted((c for c in game.chars if c.team == "D"), key=lambda c: str(c.name))
    starting = [p.position for p in plan["posts"]]
    # A third of cases start from scattered, suboptimal plant-time positions.
    if seed % 3 == 0:
        starting = [tuple(map(int, p)) for p in np.argwhere(game.grid != 1)
                    if math.dist(p, game.planted_pos) <= 7 and tuple(p) not in plan["retake_spawns"]]
    rng.shuffle(starting)
    retake = list(plan["retake_spawns"])
    rng.shuffle(retake)
    for team, points in ((attackers, starting), (defenders, retake)):
        for char, point in zip(team, points):
            char.pos = list(point)
            char.has_spike = False
            char.plant_timer = char.defuse_timer = 0
            char.hp = max(35, int(char.max_hp * rng.uniform(.55, 1.)))
            char.facing = rng.choice(("N", "S", "E", "W"))
            char.ultimate_points = char.ultimate_cost if seed % 4 == 0 else rng.randrange(max(1, char.ultimate_cost))
            for ability in ("smoke", "flash", "recon", "dance", "ash"):
                attr = ability + "_charges"
                if hasattr(char, attr) and rng.random() < .3:
                    setattr(char, attr, max(0, getattr(char, attr)-1))
        survivors = rng.randint(2, 5)
        for char in rng.sample(team, len(team)-survivors):
            char.hp = 0
            char.is_alive = False
    return dict(site=site, plant=game.planted_pos, time_left=game.detonate_timer,
                attackers=sum(c.is_alive for c in attackers), defenders=sum(c.is_alive for c in defenders))


def guard_potential(game, controller):
    """Reward-only geometry; actor still receives IQ-filtered public state."""
    if game.round_over:
        return 0.
    total = 0.
    allies = [c for c in game.chars if c.team == "A" and c.is_alive]
    for char in allies:
        post = controller.guard_plan.posts.get(str(char.name))
        if post:
            distance = max(0., math.dist(char.pos, post.position)-POSTPLANT_ARRIVAL_RADIUS)
            total += .03 * max(0., 1-distance/10)
            total += .015 * (char.facing == facing_towards(char.pos, post.aim))
        total += .025 * any(los(game.grid, tuple(char.pos), p, game._smoke_cells()) for p in watch_cells(game.grid, game.planted_pos))
    total -= .03 * sum(math.dist(a.pos, b.pos) < 2 for i, a in enumerate(allies) for b in allies[i+1:])
    return total


def play_guard_round(code, seed, *, site="A", policy=None, mode="teacher", baseline=False, gamma=.99,
                     baseline_checkpoint=BASELINE_CHECKPOINT, gae_lambda=.95):
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
        factory = (partial(LearnedAttackerController, checkpoint=baseline_checkpoint) if baseline else
                   partial(GhostChampionsV3AttackerController, checkpoint=baseline_checkpoint,
                           guard_policy=policy, guard_mode=mode, guard_recorder=recorder))
        with cpu_inference():
            game = VisualFPSBattle(NEW_MAZE_STR, DualRoleTeamAI("GC guard", factory, GhostChampionsV2DefenderController),
                    _build_team_ai(OPPONENTS[code][1]), headless=True, disable_side_swap=True,
                    attacker_roster=list(own.players), defender_roster=list(opponent.players),
                    spike_holder_name=own.spike_holder, defender_spike_holder_name=opponent.spike_holder,
                    attacker_igl_name=own.igl, defender_igl_name=opponent.igl)
            case = setup_case(game, site, seed)
            controller = getattr(game.attacker_controller, "inner", game.attacker_controller)
            frozen = freeze_modules(game.defender_controller, excluded=(game, policy))
            digest = module_digest(frozen)
            preplant_digest = module_digest([controller.policy])
            own_chars = [c for c in game.chars if c.team == "A"]
            enemies = [c for c in game.chars if c.team == "D"]
            used = reveals = blinds = 0
            aim_ticks = coverage_ticks = ticks = alive_ticks = 0
            seen_revealed, seen_blinded = set(), set()
            initial_charges = sum(sum(getattr(c, a+"_charges", 0) for a in ("smoke", "flash", "recon", "dance", "ash")) for c in own_chars)
            for _ in range(160):
                before = guard_potential(game, controller) if not baseline else 0.
                old_hp = sum(c.hp for c in enemies if c.is_alive)
                game._simulate_tick()
                ticks += 1
                terminal = game.round_over or game.match_over
                after = guard_potential(game, controller) if not baseline else 0.
                # Effect credit is training-only, attached to an A-owned cast.
                owned_recon = any(b.get("team") == "A" for b in game.recon_bursts)
                owned_flash = any(b.get("team") == "A" for b in game.flash_bursts)
                new_reveals = {str(c.name) for c in enemies if c.is_alive and c.reveal_remaining > 0} - seen_revealed if owned_recon else set()
                new_blinds = {str(c.name) for c in enemies if c.is_alive and c.blind_remaining > 0} - seen_blinded if owned_flash else set()
                seen_revealed.update(new_reveals)
                seen_blinded.update(new_blinds)
                reveals += len(new_reveals)
                blinds += len(new_blinds)
                # No reward for merely issuing/consuming a skill.
                damage = max(0, old_hp-sum(c.hp for c in enemies if c.is_alive))
                reward = gamma*after-before + .02*len(new_reveals) + .02*len(new_blinds) + .0002*damage
                if terminal:
                    reward += 1. if game.attacker_wins else -1.
                recorder.reward(reward, gamma, terminal)
                for c in own_chars:
                    if not c.is_alive:
                        if str(c.name) in recorder.latest:
                            recorder.latest[str(c.name)]["discount"] = 0.
                            recorder.latest.pop(str(c.name), None)
                        continue
                    alive_ticks += 1
                    # The same static nearest-post reference measures v2 and v3.
                    post = min(site_plan(game.planted_pos)["posts"], key=lambda p: math.dist(c.pos, p.position))
                    aim_ticks += bool(post and c.facing == facing_towards(c.pos, post.aim))
                    coverage_ticks += any(los(game.grid, tuple(c.pos), p, game._smoke_cells()) for p in watch_cells(game.grid, game.planted_pos))
                if terminal:
                    break
            if not terminal:
                raise RuntimeError("Guard round did not terminate")
            final_charges = sum(sum(getattr(c, a+"_charges", 0) for a in ("smoke", "flash", "recon", "dance", "ash")) for c in own_chars)
            used = initial_charges-final_charges
            if module_digest(frozen) != digest or module_digest([controller.policy]) != preplant_digest:
                raise RuntimeError("Frozen opponent or v2 preplant weights changed")
            record = game.analytics_tracker.round_records[-1]
            stats = dict(opponent=code, seed=seed, **case, winner=record["winner"], reason=record["reason"],
                         decisions=len(recorder.rows), ability_uses=used, revealed=reveals, blinded=blinds,
                         aim_ticks=aim_ticks, coverage_ticks=coverage_ticks, alive_ticks=alive_ticks, ticks=ticks,
                         guard_only=True, preplant_weights_unchanged=True, opponent_weights_unchanged=True)
            return recorder.advantages(gae_lambda), stats
