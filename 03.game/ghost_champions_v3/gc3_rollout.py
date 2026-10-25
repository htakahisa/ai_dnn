"""Real-game episodes with training-only contribution and potential rewards."""
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import gc
import hashlib
import os
import random
import numpy as np
import torch
from .gc3_learn_attacker_guard import GhostChampionsV3AttackerController
from .gc3_learn_defender_search import GhostChampionsV3SearchController
from .gc3_reward_phase import PhaseRewards, REWARD_VERSION

ROOT = Path(__file__).resolve().parent.parent
OPPONENTS = {"TYG": ("Touyama Gaming", "touyama_gaming_v2"), "OMG": ("Omoko Gaming", "omoko_gaming_v1"),
             "FRC": ("Furina Classic", "frc_v1"), "FNC": ("Fnatic2023", "fnatic_v3"),
             "GG": ("Gorigons", "concon_v1"), "SPS": ("SUPES", "toru_ai_v3.1")}
# ConCon's attacker requires its own roster and cannot drive GC characters.
POSTPLANT_ENTRY_ATTACKER_CODES = ("TYG", "OMG", "FRC", "FNC", "SPS")
POSTPLANT_ENTRY_MAX_ATTEMPTS = 24
POSTPLANT_ENTRY_MAX_TICKS = 400
POSTPLANT_ENTRY_SEED_STRIDE = 1000000


class NoPostplantEntry(RuntimeError):
    pass


@contextmanager
def engine_directory():
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def freeze_modules(controller, excluded=()):
    queue, seen, modules = [controller], {id(x) for x in excluded}, []
    while queue:
        item = queue.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, torch.nn.Module):
            item.eval()
            for parameter in item.parameters():
                parameter.requires_grad_(False)
            modules.append(item)
        elif isinstance(item, dict):
            queue.extend(item.values())
        elif isinstance(item, (list, tuple)):
            queue.extend(item)
        elif hasattr(item, "__dict__") and not isinstance(item, type):
            queue.extend(value for key, value in vars(item).items() if key not in
                         ("game", "_game", "_real", "real_game", "_real_game", "encoder", "sensor", "memory"))
    return modules


def module_digest(modules):
    digest = hashlib.sha256()
    for module in modules:
        for key, value in sorted(module.state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class Recorder:
    def __init__(self):
        self.rows, self.latest = [], {}
        self.by_agent = defaultdict(list)

    def record(self, name, record):
        row = dict(record, reward=0., age=0, discount=1., reward_components={})
        self.rows.append(row)
        self.latest[name] = row
        self.by_agent[name].append(row)

    def reward(self, value, gamma, terminal, individual=None, components=None, individual_components=None):
        shared = dict(components) if components is not None else {"environment": value}
        if not np.isclose(sum(shared.values()), value):
            raise ValueError("Shared reward components do not sum to reward")
        for name, row in self.latest.items():
            personal = dict((individual_components or {}).get(name, {}))
            if individual is not None:
                amount = individual.get(name, 0.)
                if individual_components is None:
                    personal["individual"] = amount
                elif not np.isclose(sum(personal.values()), amount):
                    raise ValueError("Individual reward components do not sum to reward")
            parts = dict(shared)
            for key, amount in personal.items():
                parts[key] = parts.get(key, 0.)+amount
            discount = gamma**row["age"]
            row["reward"] += discount*sum(parts.values())
            for key, amount in parts.items():
                row["reward_components"][key] = row["reward_components"].get(key, 0.)+discount*amount
            row["age"] += 1
            row["discount"] = 0. if terminal else gamma**row["age"]

    def advantages(self, lam):
        for trajectory in self.by_agent.values():
            next_value, next_advantage = 0., 0.
            for row in reversed(trajectory):
                delta = row["reward"]+row["discount"]*next_value-row["value"]
                advantage = delta+row["discount"]*lam**row["age"]*next_advantage
                row["advantage"], row["return"] = advantage, advantage+row["value"]
                next_value, next_advantage = row["value"], advantage
        return self.rows


def setup_postplant(game, seed):
    """Stop at an actual plant; preserve every character/effect/timer in place.

    The caller supplies a frozen pre-plant attacker. No GC learning decisions
    are recorded until it is replaced at this phase boundary.
    """
    for _ in range(POSTPLANT_ENTRY_MAX_TICKS):
        if game.round_over or game.match_over:
            break
        if game.is_planted:
            return tuple(game.planted_pos)
        game._simulate_tick()
    if game.is_planted and not game.round_over and not game.match_over:
        return tuple(game.planted_pos)
    raise NoPostplantEntry(f"Seed {seed}: game ended without a live post-plant phase")


def phase_start_snapshot(game):
    fields = ("hp", "max_hp", "shield_hp", "is_alive", "facing", "has_spike", "plant_timer",
              "defuse_timer", "ultimate_points", "blind_remaining", "reveal_remaining",
              "electric_remaining", "ability_seal_remaining", "fate_loom_remaining",
              *(a+"_charges" for a in ("recon", "flash", "smoke", "ramp", "dance", "ash")))
    return dict(battle_tick=game.battle_tick, detonate_timer=game.detonate_timer,
                chars=[dict(name=str(c.name), team=c.team, pos=list(c.pos),
                            **{key:getattr(c, key, 0) for key in fields}) for c in game.chars])


def play_phase(code, seed, *, phase, policy=None, gamma=.99, gae_lambda=.95,
               entry_attacker_codes=POSTPLANT_ENTRY_ATTACKER_CODES,
               entry_max_attempts=POSTPLANT_ENTRY_MAX_ATTEMPTS):
    if phase not in ("search", "postplant") or code not in OPPONENTS:
        raise ValueError("Invalid training phase or opponent")
    if phase == "postplant" and (not entry_attacker_codes or entry_max_attempts < 1 or
                                 any(source not in POSTPLANT_ENTRY_ATTACKER_CODES for source in entry_attacker_codes)):
        raise ValueError("Invalid post-plant entry configuration")
    recorder = Recorder()
    with engine_directory():
        from simulation_runtime import cpu_inference
        from run_competition_manager import seed_all
        from run_game import VisualFPSBattle, _build_team_ai
        from map_data import NEW_MAZE_STR
        from party_presets import get_preset
        from team_ai import DualRoleTeamAI
        seed_all(seed)
        own, opponent = get_preset("Ghost Champions"), get_preset(OPPONENTS[code][0])
        attacker = partial(GhostChampionsV3AttackerController, policy=policy if phase == "postplant" else None,
                           recorder=recorder if phase == "postplant" else None, record_phase="postplant")
        defender = partial(GhostChampionsV3SearchController, policy=policy if phase == "search" else None,
                           recorder=recorder if phase == "search" else None, record_phase="search")
        own_ai = DualRoleTeamAI("GC v3 scratch", attacker, defender)
        attack_roster, defend_roster = (own, opponent) if phase == "postplant" else (opponent, own)
        with cpu_inference():
            def make_game(attack_ai, defend_ai):
                game = VisualFPSBattle(NEW_MAZE_STR, attack_ai, defend_ai, headless=True, disable_side_swap=True,
                    attacker_roster=list(attack_roster.players), defender_roster=list(defend_roster.players),
                    spike_holder_name=attack_roster.spike_holder, defender_spike_holder_name=defend_roster.spike_holder,
                    attacker_igl_name=attack_roster.igl, defender_igl_name=defend_roster.igl)
                game.stop_after_round = True
                return game

            entry = None
            warmup_frozen, warmup_digest = [], None
            if phase == "postplant":
                offset = random.Random(seed).randrange(len(entry_attacker_codes))
                for attempt in range(entry_max_attempts):
                    entry_seed = seed+attempt*POSTPLANT_ENTRY_SEED_STRIDE
                    seed_all(entry_seed)
                    source = entry_attacker_codes[(offset+attempt) % len(entry_attacker_codes)]
                    warmup_ai = _build_team_ai(OPPONENTS[source][1])
                    opponent_ai = _build_team_ai(OPPONENTS[code][1])
                    game = make_game(warmup_ai, opponent_ai)
                    warmup_frozen = freeze_modules(game.attacker_controller, excluded=(game, policy))
                    frozen = freeze_modules(game.defender_controller, excluded=(game, policy))
                    warmup_digest, digest = module_digest(warmup_frozen), module_digest(frozen)
                    try:
                        plant = setup_postplant(game, entry_seed)
                    except NoPostplantEntry:
                        if module_digest(warmup_frozen) != warmup_digest or module_digest(frozen) != digest:
                            raise RuntimeError("Frozen entry/opponent weights changed")
                        del game, warmup_ai, opponent_ai
                        gc.collect()
                        continue
                    entry = dict(method="actual_plant", attacker=source, seed=entry_seed,
                                 attempts=attempt+1, state=phase_start_snapshot(game))
                    # Preserve team perception accumulated before the plant as well.
                    own_ai.perception_engine = warmup_ai.perception_engine
                    game.current_attacker_team_ai = game.initial_attacker_team_ai = own_ai
                    own_ai.bind_game(game)
                    game.attacker_controller = own_ai.get_attacker_controller()
                    if phase_start_snapshot(game) != entry["state"]:
                        raise RuntimeError("Post-plant controller handoff changed character state")
                    break
                else:
                    raise NoPostplantEntry(f"{code}, seed {seed}: no plant in {entry_max_attempts} actual rounds; "
                                          "no random-position fallback was used")
            else:
                opponent_ai = _build_team_ai(OPPONENTS[code][1])
                game = make_game(opponent_ai, own_ai)
                plant = None
            opposite = game.defender_controller if phase == "postplant" else game.attacker_controller
            inner = getattr(opposite, "inner", opposite)
            actor = getattr(inner, "actor", None)
            if actor is not None and hasattr(actor, "collect_statistics"):
                actor.collect_statistics = False
            if phase == "search":
                frozen = freeze_modules(opposite, excluded=(game, policy))
                digest = module_digest(frozen)
            own_team = "A" if phase == "postplant" else "D"
            own_chars = [c for c in game.chars if c.team == own_team]
            def charges():
                return sum(sum(getattr(c, a+"_charges", 0) for a in ("recon", "flash", "smoke", "ramp", "dance", "ash")) for c in own_chars)
            before_charges = charges()
            terminal = False
            with PhaseRewards(game, phase, gamma) as rewards:
                for tick in range(400):
                    rewards.begin_tick()
                    game._simulate_tick()
                    terminal = game.round_over or game.match_over or (phase == "search" and game.is_planted)
                    success = bool(game.attacker_wins) if phase == "postplant" else bool(game.defender_wins and not game.is_planted)
                    shared, personal = rewards.finish_tick(terminal, success)
                    recorder.reward(sum(shared.values()), gamma, terminal,
                                    components=shared, individual_components=personal)
                    if terminal:
                        break
            if not terminal:
                raise RuntimeError("Training episode exceeded engine tick limit")
            if module_digest(frozen) != digest:
                raise RuntimeError("Frozen opponent weights changed")
            if warmup_digest is not None and module_digest(warmup_frozen) != warmup_digest:
                raise RuntimeError("Frozen entry attacker weights changed")
            if policy is not None and any(not p.requires_grad for p in policy.parameters()):
                raise RuntimeError("GC v3 policy was unexpectedly frozen")
            result = game.analytics_tracker.round_records[-1] if game.analytics_tracker.round_records else {}
            stats = dict(phase=phase, opponent=code, seed=seed, success=success,
                         winner=own_team if success else ("D" if own_team == "A" else "A"),
                         reason="planted" if phase == "search" and game.is_planted else result.get("reason", "unknown"),
                         plant=plant, planted=bool(game.is_planted), phase_start=entry,
                         decisions=len(recorder.rows), ticks=tick+1,
                         setup_decisions=sum(r["phase"] == "defender" for r in recorder.rows),
                         ability_uses=before_charges-charges(), opponent_weights_unchanged=True,
                         opponent_model_digest=digest, reward=REWARD_VERSION,
                         reward_component_sums={key:sum(row["reward_components"].get(key, 0.) for row in recorder.rows)
                                                for key in sorted({key for row in recorder.rows for key in row["reward_components"]})})
            rows = recorder.advantages(gae_lambda)
            for row in rows:
                row["reward_version"] = REWARD_VERSION
            del game
    gc.collect()
    return rows, stats
