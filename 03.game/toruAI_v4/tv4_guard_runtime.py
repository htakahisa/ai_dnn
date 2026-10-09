"""Real plant-boundary collection, guard case replay, and independent full matches."""
import contextlib
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import torch

from concon_v1.co1_retake_cases import load_case
from toruAI_v4.tv4_scenario import OPPONENTS
from toruAI_v4.tv4_learn_attacker_analysis import AttackerEncoder, AttackerAnalysisModel
from toruAI_v4.tv4_learn_attacker_plant import PlantDQN, policy_schema
from toruAI_v4.tv4_learn_attacker_guard import OBS_DIM, ACTION_DIM
from toruAI_v4.tv4_learn_attacker_guard import crossfire_score
from toruAI_v4.tv4_attacker_guard_controller import ToruV4AttackerPlantGuardController
from toruAI_v4.tv4_train_defender_analysis import legacy_root, seed_all, relocate_debug_logs
from toruAI_v4.tv4_attacker_rosters import source_presets

CASE_FORMAT = "toru_v4_attacker_guard_cases_v1"
DEFAULT_REWARDS = {"win": 8., "loss": -8., "survivor": .1, "reserve": .05, "tick": .003,
                   "progress": .02, "damage": .006, "death": 1., "kill": .1,
                   "team_damage": .001, "ability_use": .01, "delay": .02,
                   "blind_cover": .05, "crossfire": .02,
                   "defuse_approach": .1, "defuse_stall": .05}


def read_checkpoint(path):
    # One file read gives matching weights and hash even at an atomic replacement boundary.
    payload = Path(path).read_bytes()
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True), hashlib.sha256(payload).hexdigest()


def load_sources(scenario, opponents, best_dir, analysis_dir):
    """Read best directly. No archived/copied model files are created."""
    result = {}
    for opponent in opponents:
        plant_path = Path(best_dir).resolve() / opponent / "attacker_plant_best.pt"
        analysis_path = Path(analysis_dir).resolve() / opponent / "attacker_analysis_best.pt"
        plant_state, plant_hash = read_checkpoint(plant_path)
        analysis_state, analysis_hash = read_checkpoint(analysis_path)
        encoder = AttackerEncoder(scenario)
        if plant_state["opponent"] != opponent or analysis_state["opponent"] != opponent:
            raise ValueError(f"Source opponent mismatch: {opponent}")
        if plant_state["schema"] != policy_schema(scenario) or json.dumps(analysis_state["schema"], sort_keys=True) != json.dumps(encoder.schema(), sort_keys=True):
            raise ValueError(f"Source map/schema mismatch: {opponent}")
        if plant_state["analysis_hash"] != analysis_hash:
            raise ValueError(f"Plant and analysis best do not match: {opponent}")
        plant = PlantDQN()
        plant.load_state_dict(plant_state["model"])
        analysis = AttackerAnalysisModel(len(encoder.fields), len(encoder.route_fields), len(scenario.names))
        analysis.load_state_dict(analysis_state["model"])
        analysis.trained_rounds = analysis_state["trained_rounds"]
        for model in (plant, analysis):
            model.eval()
            model.requires_grad_(False)
        train_presets, eval_presets = source_presets(plant_state["config"], opponent)
        result[opponent] = {"plant": plant, "analysis": analysis,
            "hashes": {"plant": plant_hash, "analysis": analysis_hash},
            "plant_set": plant_state["completed_sets"], "analysis_set": analysis_state["completed_sets"],
            "preset": train_presets[0], "train_presets": train_presets, "eval_presets": eval_presets,
            "generic_roster_training": "train_presets" in plant_state["config"],
            "paths": {"plant": str(plant_path), "analysis": str(analysis_path)}}
    return result


def unwrap_combined(controller):
    while not isinstance(controller, ToruV4AttackerPlantGuardController):
        controller = vars(controller).get("inner")
        if controller is None:
            raise ValueError("Case does not contain the attacker plant/guard controller")
    return controller


def build_game(opponent, scenario, source, guard_policy, seed, preset_name=None):
    from run_game import VisualFPSBattle, _build_team_ai
    from party_presets import get_preset
    from controllers import DefaultDefenderController
    from team_ai import DualRoleTeamAI
    seed_all(seed)
    own, enemy = get_preset(preset_name or source["preset"]), get_preset(OPPONENTS[opponent][1])
    if set(own.players) & set(enemy.players):
        raise ValueError("Own and enemy rosters must be disjoint")
    controller = ToruV4AttackerPlantGuardController(scenario, source["analysis"], source["plant"], guard_policy, seed=seed + 19)
    team = DualRoleTeamAI("Toru v4 plant and guard", lambda: controller, DefaultDefenderController)
    game = VisualFPSBattle(scenario.maze, team, _build_team_ai(OPPONENTS[opponent][0], device="cpu"), headless=True,
        attacker_roster=list(own.players), defender_roster=list(enemy.players),
        spike_holder_name=own.spike_holder, defender_spike_holder_name=enemy.spike_holder,
        attacker_igl_name=own.igl, defender_igl_name=enemy.igl,
        attacker_team_name=own.name, defender_team_name=enemy.name, disable_side_swap=True)
    game.stop_after_round, game.analytics_tracker = True, None
    game._record_replay_frame = lambda: None
    relocate_debug_logs(game.defender_controller, None)
    return game, controller


def play_round(game, controller, opponent, rewards, max_steps, capture=None):
    """Only decisions actually executed after the plant boundary enter guard replay."""
    own = [c for c in game.chars if c.team == "A"]
    enemy = [c for c in game.chars if c.team == "D"]
    score_before = game.attacker_wins
    transitions, last_indices = [], {}
    damage = uses = total_reward = 0.
    uses_by_kind = Counter()
    steps = 0
    while not game.round_over and not game.match_over:
        guard_before = controller.guard
        if guard_before is not None:
            guard_before.decisions = {}
            guard_before.prepare_team_tick()
        else:
            controller.prepare_team_tick()
        hp = {str(getattr(c, "base_name", c.name)): max(0., c.hp) if c.is_alive else 0. for c in own}
        kills = {str(getattr(c, "base_name", c.name)): c.round_kills for c in own}
        charges = {str(getattr(c, "base_name", c.name)): int(getattr(c, c.ability_name.lower() + "_charges", 0)) for c in own}
        enemy_hp = sum(max(0., c.hp) for c in enemy if c.is_alive)
        bomb_before = float(game.detonate_timer)
        defuse_notified = guard_before is not None and guard_before.snapshot.defuse_notified
        game.step_tick()
        steps += 1
        if steps > max_steps:
            raise RuntimeError(f"Guard watchdog exceeded: {opponent} round {game.current_round}")
        if guard_before is None:
            if game.is_planted and not game.round_over:
                controller.mark_plant_boundary()
                if capture is not None:
                    capture(game, controller, opponent)
            continue
        current = {str(getattr(c, "base_name", c.name)): c for c in own}
        tick_damage = {n: max(0., hp[n] - (max(0., c.hp) if c.is_alive else 0.)) for n, c in current.items()}
        damage += sum(tick_damage.values())
        for n, c in current.items():
            spent = max(0, charges[n] - int(getattr(c, c.ability_name.lower() + "_charges", 0)))
            uses += spent
            uses_by_kind[c.ability_name] += spent
        enemy_damage = max(0., enemy_hp - sum(max(0., c.hp) for c in enemy if c.is_alive))
        decisions = dict(guard_before.decisions)
        terminal = bool(game.round_over or game.match_over)
        if not terminal:
            guard_before.prepare_team_tick()
        for name, (chosen, inputs, before) in decisions.items():
            char = current[name]
            old, new = inputs.distances[before.position], inputs.distances[tuple(char.pos)]
            progress = float(np.clip(old - new, -1, 1)) if old >= 0 and new >= 0 else 0.
            reward = -rewards["tick"] + rewards["progress"] * progress - rewards["damage"] * tick_damage[name]
            reward += rewards["kill"] * (char.round_kills - kills[name]) + rewards["team_damage"] * min(200., enemy_damage)
            reward -= rewards["death"] * int(not char.is_alive)
            reward -= rewards["ability_use"] * int(inputs.actions[chosen].kind in ("ABILITY", "ULTIMATE"))
            if not game.is_defused and not defuse_notified:
                reward += rewards["delay"] * max(0., min(1., bomb_before - game.detonate_timer))
            if inputs.defuse_pressure and old > 0:
                reward += rewards.get("defuse_approach", .1) * progress
                if progress <= 0 and before.movement_disabled == 0:
                    reward -= rewards.get("defuse_stall", .05)
            if inputs.blind and inputs.threats and not inputs.defuse_pressure:
                exposed_before = sum(controller.scenario.clear(before.position, t) for t in inputs.threats)
                exposed_after = sum(controller.scenario.clear(tuple(char.pos), t) for t in inputs.threats)
                reward += rewards["blind_cover"] * float(np.clip(exposed_before - exposed_after, -1, 1))
            # Potential differences do not pay repeatedly for camping at a crossfire position.
            reward += rewards["crossfire"] * (crossfire_score(controller.scenario, tuple(char.pos), inputs.teammates, inputs.threats)
                                             - crossfire_score(controller.scenario, before.position, inputs.teammates, inputs.threats))
            done = terminal or not char.is_alive
            next_input = guard_before.inputs.get(name)
            if next_input is None:
                done = True
            next_obs = next_input.observation if not done else np.zeros(OBS_DIM, np.float32)
            next_mask = next_input.mask.copy() if not done else np.ones(ACTION_DIM, bool)
            transitions.append([inputs.observation.astype(np.float16), chosen, reward,
                next_obs.astype(np.float16), next_mask, float(done), inputs.teacher, inputs.mask.copy()])
            last_indices[name] = len(transitions) - 1
            total_reward += reward
    if not game.round_over:
        raise RuntimeError("Guard rollout ended without a real round result")
    won = game.attacker_wins > score_before
    guard = controller.guard
    played = guard is not None
    alive = sum(c.is_alive for c in own)
    remaining = sum(int(getattr(c, c.ability_name.lower() + "_charges", 0)) for c in own if c.is_alive)
    initial_abilities = sum(guard.initial_charges.values()) if played else 0
    reserve = min(1., remaining / initial_abilities) if initial_abilities else None
    if played:
        terminal_reward = (rewards["win"] + rewards["survivor"] * alive + rewards["reserve"] * (reserve or 0.)
                           if won else rewards["loss"])
        for index in last_indices.values():
            transitions[index][2] += terminal_reward
            transitions[index][5] = 1.
            total_reward += terminal_reward
    reason = ("defused" if game.is_defused else "exploded" if game.is_planted and game.detonate_timer <= 0
              else "defenders_eliminated" if not any(c.is_alive for c in enemy)
              else "preplant_timeout" if not game.is_planted and game.round_timer <= 0 else "attackers_eliminated")
    record = {"opponent": opponent, "round": game.current_round, "planted": bool(game.is_planted),
        "guard_played": played, "guard_won": bool(won) if played else None, "full_round_won": bool(won),
        "site": controller.scenario.site_of(game.planted_pos) if game.is_planted else None,
        "reason": reason, "alive": alive, "enemy_alive": sum(c.is_alive for c in enemy),
        "initial_alive": guard.initial_alive if played else 0,
        "initial_enemy_alive": guard.initial_enemy_alive if played else 0,
        "initial_abilities": initial_abilities, "remaining_abilities_alive": remaining if played else 0,
        "uses": uses, "uses_by_kind": {k: v for k, v in uses_by_kind.items() if v}, "damage": damage, "reward": total_reward,
        "guard_ticks": int(game.battle_tick - guard.start_tick) if played else 0}
    return record, transitions


def play_block(opponent, scenario, source, guard_policy, seed, *, capture=None, rewards=None, max_steps=400, preset_name=None):
    from simulation_runtime import cpu_inference
    rounds, transitions, history = [], [], []
    with legacy_root(), open(os.devnull, "w", encoding="utf-8") as quiet, \
            contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet), cpu_inference(enabled=True):
        game, controller = build_game(opponent, scenario, source, guard_policy, seed, preset_name)
        for round_number in range(1, 13):
            controller.plant.previous_rounds = list(history)
            record, rows = play_round(game, controller, opponent, rewards or DEFAULT_REWARDS, max_steps, capture)
            record["preset"] = preset_name or source["preset"]
            rounds.append(record)
            transitions.extend(rows)
            history.append({"site": record["site"], "winner": "A" if record["full_round_won"] else "D"})
            if round_number < 12:
                game.current_round += 1
                game.init_round()
    return rounds, transitions


def replay_case(path, scenario, policy, seed, *, rewards=None, epsilon=0., teacher_probability=0., tensor_cache=None, max_steps=400):
    from simulation_runtime import cpu_inference
    with legacy_root(), open(os.devnull, "w", encoding="utf-8") as quiet, \
            contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet), cpu_inference(enabled=True):
        seed_all(seed)
        game, metadata = load_case(path, restore_rng=False, device="cpu", tensor_cache=tensor_cache)
        if metadata.get("format") != CASE_FORMAT:
            raise ValueError("This is not an attacker guard case")
        controller = unwrap_combined(game.attacker_controller)
        if controller.guard is None or controller.guard.start_tick != game.battle_tick:
            raise ValueError("Guard case is not at the end-of-plant-tick boundary")
        controller.guard_policy = controller.guard.policy = policy
        controller.guard.training = True
        controller.guard.epsilon, controller.guard.teacher_probability = epsilon, teacher_probability
        controller.guard.rng = np.random.default_rng(seed + 31)
        controller.guard.cache = None
        controller.guard.inputs, controller.guard.actions, controller.guard.plans, controller.guard.decisions = {}, {}, {}, {}
        # Do not reset game, opponent controller, sensor, history, characters or effects.
        relocate_debug_logs(game.defender_controller, None)
        record, transitions = play_round(game, controller, metadata["opponent"], rewards or DEFAULT_REWARDS, max_steps)
    return [record], transitions


def summarize_guard(rounds):
    guarded = [r for r in rounds if r["guard_played"]]
    def rate(key, denominator):
        return sum(r[key] for r in guarded) / denominator if denominator else None
    initial_alive = sum(r["initial_alive"] for r in guarded)
    initial_enemy_alive = sum(r["initial_enemy_alive"] for r in guarded)
    initial_abilities = sum(r["initial_abilities"] for r in guarded)
    average = lambda key: float(np.mean([r[key] for r in guarded])) if guarded else None
    metrics = {"rounds": len(rounds), "guard_rounds": len(guarded),
        "guard_wins": sum(r["guard_won"] for r in guarded),
        "guard_win_rate": rate("guard_won", len(guarded)),
        "full_win_rate": sum(r["full_round_won"] for r in rounds) / len(rounds) if rounds else None,
        "plant_rate": sum(r["planted"] for r in rounds) / len(rounds) if rounds else None,
        "defuses": sum(r["reason"] == "defused" for r in guarded),
        "explosions": sum(r["reason"] == "exploded" for r in guarded),
        "mean_alive": average("alive"), "mean_enemy_alive": average("enemy_alive"),
        "ally_survival_rate": rate("alive", initial_alive), "enemy_survival_rate": rate("enemy_alive", initial_enemy_alive),
        "ability_use_rate": rate("uses", initial_abilities), "ability_reserve_rate": rate("remaining_abilities_alive", initial_abilities),
        "mean_uses": average("uses"), "mean_damage": average("damage"), "mean_reward": average("reward")}
    metrics["by_site"] = {side: {"rounds": len(rows), "win_rate": sum(r["guard_won"] for r in rows) / len(rows) if rows else None}
        for side in ("L", "R") for rows in ([r for r in guarded if r["site"] == side],)}
    totals = Counter()
    for row in guarded:
        totals.update(row.get("uses_by_kind", {}))
    metrics["uses_by_kind"] = dict(totals)
    return metrics


def format_guard(metrics):
    percent = lambda x: "対象なし" if x is None else f"{x:.1%}"
    people = lambda x: "対象なし" if x is None else f"{x:.2f}人"
    return (f"guard勝率={percent(metrics['guard_win_rate'])} ({metrics['guard_wins']}/{metrics['guard_rounds']}) "
            f"全体勝率={percent(metrics['full_win_rate'])} プラント率={percent(metrics['plant_rate'])} "
            f"味方生存={people(metrics['mean_alive'])} ({percent(metrics['ally_survival_rate'])}) "
            f"敵生存={people(metrics['mean_enemy_alive'])} ({percent(metrics['enemy_survival_rate'])}) "
            f"ability温存率={percent(metrics['ability_reserve_rate'])} 使用率={percent(metrics['ability_use_rate'])} "
            f"解除された回数={metrics['defuses']} 爆発={metrics['explosions']} "
            f"能力別使用={metrics.get('uses_by_kind', {})}")


def guard_best_rank(metrics):
    if not metrics["guard_rounds"]:
        raise ValueError("No guard episodes: cannot select a best guard")
    return (metrics["guard_win_rate"], metrics["full_win_rate"], metrics["mean_alive"],
            -metrics["mean_damage"], metrics["ability_reserve_rate"] or 0., -metrics["mean_uses"])
