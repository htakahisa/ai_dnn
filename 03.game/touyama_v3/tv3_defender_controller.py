"""New generic Touyama v3 defender: frozen site analysis, search and retake policies."""
from pathlib import Path
import hashlib
import numpy as np
import torch

from frc_v1.perception import FrcPerceptionBuilder
from frc_v1.actions import build_masks, to_game_action, MOVE_STEPS
from touyama_v3.tv3_observer import FeatureHistory
from touyama_v3.tv3_model import SiteModel, VERSION as ANALYSIS_VERSION
from touyama_v3.tv3_scenario import Scenario, OPPONENTS
from touyama_v3.tv3_learn_public_hazards import hazard_schema
from touyama_v3.tv3_defender_policy import (
    DefenderDQN, PolicyEncoder, POLICY_VERSION, OBS_DIM, ACTION_DIM, MOVEMENTS, staging_positions, legacy_staging_positions, assign_goals,
)

HERE = Path(__file__).resolve().parent
DEFENDER_BEST = HERE / "data" / "best"
OPPONENT_NAMES = {"Ghost Champions v1": "gc_v1", "ConCon v1": "concon_v1",
                  "Omoko Gaming v1": "omoko_v1", "Fnatic v3": "fnatic_v3", "FRC v1": "frc_v1",
                  "Toru AI v4": "toru_ai_v4"}


def policy_metadata(scenario, phase="search"):
    metadata = {"version": POLICY_VERSION, "board": scenario.grid.tolist(), "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM, "staging": staging_positions(scenario),
            "search_objective": "survival_resources_readiness_plant_allowed_v1",
            "roster_scope": "touyama_five_shared_weights", "sensor": "frc_public_team_v1", "scenario": scenario.signature,
            "public_hazards": hazard_schema()}
    if phase == "retake":
        from touyama_v3.tv3_retake_combat import retake_combat_schema, RETAKE_OBS_DIM
        metadata = {**metadata, "obs_dim": RETAKE_OBS_DIM, "retake_combat": retake_combat_schema()}
    return metadata


def normalize_policy_schema(saved, scenario):
    """Accept the previous rally geometry without discarding learned weights."""
    expected = policy_metadata(scenario)
    legacy = {**expected, "staging": legacy_staging_positions(scenario)}
    old = saved.get("schema", {})
    geometry_only = {**old, "staging": expected["staging"]} == expected
    valid_cells = isinstance(old.get("staging"), dict) and set(old["staging"]) == {"L", "R"} and all(
        len(points) >= 5 and all(isinstance(p, (list, tuple)) and len(p) == 2
                               and all(isinstance(x, int) for x in p)
                               and 0 <= p[0] < scenario.grid.shape[0] and 0 <= p[1] < scenario.grid.shape[1]
                               and scenario.grid[tuple(p)] != 1 for p in points)
        for points in old["staging"].values())
    if saved.get("schema") == legacy or (geometry_only and valid_cells):
        saved = {**saved, "schema": expected}
    return saved


def load_policy(path, phase, scenario):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    saved = normalize_policy_schema(saved, scenario)
    if saved.get("phase") != phase or saved.get("schema") != policy_metadata(scenario, phase):
        raise ValueError(f"Generic defender checkpoint/schema mismatch: {path}")
    if phase == "retake":
        from touyama_v3.tv3_retake_coordination import RETAKE_VERSION, retake_layout
        if saved.get("retake_version") != RETAKE_VERSION or saved.get("retake_layout") != retake_layout(scenario):
            raise ValueError(f"Retake coordination/utility map changed: {path}")
    with torch.random.fork_rng(devices=[]):
        model = DefenderDQN(policy_metadata(scenario, phase)["obs_dim"])
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved


def load_analyses(scenario, opponents, directory=None):
    directory = Path(directory) if directory else HERE / "data" / "best"
    fields = FeatureHistory(scenario).fields
    models, signatures = {}, {}
    for opponent in opponents:
        path = directory / opponent / "defender_analysis_best.pt"
        if not path.is_file():
            # Older exports remain readable; new training uses the role name.
            for name in ("analysis_best.pt", "best.pt"):
                legacy = directory / opponent / name
                if legacy.is_file():
                    path = legacy
                    break
        if not path.is_file():
            raise FileNotFoundError(f"Site analysis best is required: {path}")
        # Freeze an exact file snapshot, including its public feature schema.
        content = path.read_bytes()
        import io
        saved = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
        if (saved.get("version") != ANALYSIS_VERSION or saved.get("opponent") != opponent or
                saved.get("fields") != fields or saved.get("scenario") != scenario.signature):
            raise ValueError(f"Site analysis checkpoint mismatch: {path}")
        with torch.random.fork_rng(devices=[]):
            model = SiteModel(len(fields))
        model.load_state_dict(saved["model"])
        model.eval()
        models[opponent] = model
        signatures[opponent] = hashlib.sha256(content).hexdigest()
    return models, signatures


class TouyamaV3DefenderController:
    handles_team_perception = True

    def __init__(self, opponent=None, *, scenario=None, search=None, retake=None, analyses=None,
                 search_path=None, retake_path=None, training=False, seed=0):
        self.scenario = scenario or Scenario()
        self.opponent, self.training = opponent, training
        self.retake_combat_enabled = True  # False is reserved for diagnostic comparisons.
        self.retake_deadline_enabled = True
        self.legacy_dead_defuser_mask = False  # Diagnostic reproduction only.
        self.allow_bootstrap_retake = training
        self.staging = staging_positions(self.scenario)
        self.search = search
        self.retake = retake
        self.retake_directory = Path(retake_path) if retake_path else DEFENDER_BEST
        self.search_state = None
        self.search_hash = None
        self.search_path = Path(search_path) if search_path else None
        if search is None and opponent is not None:
            self.load_opponent_search()
        self.analyses = analyses
        self.encoder = PolicyEncoder(self.scenario)
        self.sensor = FrcPerceptionBuilder("D")
        self.rng = np.random.default_rng(seed)
        self.epsilon = self.teacher_probability = 0.
        self.learning_phase = None
        self.stop_at_plant = False
        self.previous_rounds = []
        self.game = None
        self.reset_round()

    def load_opponent_search(self):
        source = self.search_path or DEFENDER_BEST / self.opponent / "search_best.pt"
        model, saved = load_policy(source, "search", self.scenario)
        if saved.get("opponent") != self.opponent:
            raise ValueError(f"Search model belongs to a different opponent: {source}")
        self.search, self.search_state = model, saved
        self.search_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    def set_game(self, game):
        self.game = getattr(game, "_real", game)
        if self.opponent is None:
            public_name = self.game.current_attacker_team_ai.name
            self.opponent = OPPONENT_NAMES.get(public_name)
            if self.opponent is None:
                raise ValueError(f"Touyama v3 needs an analysis model for opponent AI: {public_name}")
        if self.search is None:
            self.load_opponent_search()
        if self.analyses is None:
            self.analyses, signatures = load_analyses(self.scenario, (self.opponent,))
            if self.search_state is not None and self.search_state["analysis_hashes"].get(self.opponent) != signatures[self.opponent]:
                raise ValueError("The site-analysis best changed since search training; retrain/update the defender explicitly")
        if self.retake is None and not self.allow_bootstrap_retake:
            self.retake = {}
            for side in ("L", "R"):
                path = self.retake_directory / self.opponent / f"retake_{side}_best.pt"
                model, saved = load_policy(path, "retake", self.scenario)
                if saved.get("opponent") != self.opponent or saved.get("site") != side:
                    raise ValueError(f"Retake opponent/site mismatch: {path}")
                if self.search_hash is not None and saved.get("frozen_search_hash") != self.search_hash:
                    raise ValueError("Retake was trained with a different search model")
                self.retake[side] = model

    def reset_round(self):
        from touyama_v3.tv3_retake_coordination import RetakeAssembly
        from touyama_v3.tv3_retake_combat import RetakeEncoder, LegacyRetakeEncoder
        self.assembly = RetakeAssembly(self.scenario)
        self.retake_encoder = RetakeEncoder(self.scenario)
        self.legacy_retake_encoder = LegacyRetakeEncoder(self.scenario)
        self.initial_arrived = set()
        self.initial_faced = set()
        self.sensor.reset()
        self.history = FeatureHistory(self.scenario)
        self.probabilities = np.asarray([.5, .5], np.float32)
        self.cache = None
        self.inputs, self.actions, self.plans, self.decisions = {}, {}, {}, {}
        self.snapshot = None
        self.goal_key = None
        self.goal_cache = {}

    def prepare_team_tick(self):
        if self.stop_at_plant and self.game.is_planted:
            return
        setup = self.game.defender_setup_phase.active
        key = (self.game.current_round, "setup" if setup else "live",
               self.game.defender_setup_phase.ticks_remaining if setup else self.game.battle_tick,
               bool(self.game.is_planted))
        if key == self.cache:
            return
        snapshot = self.sensor.build(self.game)
        if not setup and not snapshot.is_planted:
            observation = self.history.encode(snapshot, self.previous_rounds)
            self.probabilities = self.analyses[self.opponent].probabilities(observation)
        elif snapshot.is_planted:
            self.history.encode(snapshot, self.previous_rounds)
        self.snapshot, self.cache = snapshot, key
        goal_site = (self.scenario.site_of(snapshot.spike_planted) if snapshot.is_planted else
                     ("L", "R")[int(np.argmax(self.probabilities))] if not setup and max(self.probabilities) >= .65 else "watch")
        goal_key = ("retake" if snapshot.is_planted else "search", goal_site,
                    tuple(a.slot for a in snapshot.allies if a.alive))
        if goal_key != self.goal_key:
            self.goal_cache = assign_goals(snapshot, self.scenario, self.probabilities, self.staging)
            self.goal_key = goal_key
        goals = dict(self.goal_cache)
        if not snapshot.is_planted and goal_site == "watch":
            for ally in snapshot.allies:
                if not ally.alive:
                    continue
                post = self.scenario.post_for(ally)
                if ally.position == post.watch:
                    self.initial_arrived.add(ally.slot)
                if not setup and ally.slot in self.initial_arrived:
                    hiding = (snapshot.tick + ally.slot * 2) % 8 >= 4
                    goals[ally.slot] = post.retreat if hiding else post.watch
        if snapshot.is_planted:
            goals = self.assembly.goals(snapshot)
            if self.retake_deadline_enabled:
                goals = self.retake_encoder.roles.goals(snapshot, goals, self.assembly.launched)
        masks = build_masks(snapshot)
        if snapshot.is_planted and self.legacy_dead_defuser_mask:
            from dataclasses import replace
            masks = replace(masks, kind=masks.kind.copy())
            active = next((a.slot for a in snapshot.allies if a.defuse_progress > 0), None)
            for a in snapshot.allies:
                if active not in (None, a.slot):
                    masks.kind[a.slot, 6] = False
        self.inputs, self.actions, self.plans = {}, {}, {}
        phase = "retake" if snapshot.is_planted else "search"
        model = self.retake if snapshot.is_planted else self.search
        if snapshot.is_planted and isinstance(model, dict):
            model = model[self.scenario.site_of(snapshot.spike_planted)]
        reserved = set()
        names = {a.slot: a.name for a in snapshot.allies}
        for ally in sorted(snapshot.allies, key=lambda a: (a.position, a.ability_name)):
            if not ally.alive:
                continue
            encoder = self.retake_encoder if snapshot.is_planted and self.retake_combat_enabled else self.encoder
            if snapshot.is_planted and self.retake_combat_enabled and not self.retake_deadline_enabled:
                encoder = self.legacy_retake_encoder
            inputs = encoder.encode(snapshot, ally, goals[ally.slot], self.probabilities, self.history.tracks, masks)
            if snapshot.is_planted and not self.assembly.launched and ally.slot in self.assembly.ready:
                # Wait in the assigned rally area while retaining combat/utility.
                for index in np.flatnonzero(inputs.mask[:40]):
                    kind = MOVEMENTS[index // 8]
                    dr, dc = MOVE_STEPS.get(kind, (0, 0))
                    p = ally.position[0] + dr, ally.position[1] + dc
                    if kind != "STAY" and (inputs.distances[p] < 0 or inputs.distances[p] > 2):
                        inputs.mask[index] = False
                if inputs.teacher < 40 and not inputs.combat_contact:
                    stays = [i for i in np.flatnonzero(inputs.mask[:40]) if MOVEMENTS[i // 8] == "STAY"]
                    if stays:
                        inputs.teacher = int(min(stays, key=lambda i: inputs.actions[i].facing != inputs.actions[inputs.teacher].facing))
            for action in np.flatnonzero(inputs.mask[:40]):
                kind = MOVEMENTS[action // 8]
                dr, dc = MOVE_STEPS.get(kind, (0, 0))
                if kind != "STAY" and (ally.position[0] + dr, ally.position[1] + dc) in reserved:
                    inputs.mask[action] = False
            if not inputs.mask[inputs.teacher]:
                inputs.teacher = int(np.flatnonzero(inputs.mask)[0])
            post = self.scenario.post_for(ally)
            if not snapshot.is_planted and not inputs.combat_contact and ally.position == post.watch and ally.slot not in self.initial_faced:
                from touyama_v3.tv3_observer import facing
                from frc_v1 import FACING
                self.initial_faced.add(ally.slot)
                aim = ally.facing if ally.forced_facing else facing(ally.position, post.look)
                candidates = [i for i in np.flatnonzero(inputs.mask[:8]) if FACING[i] == aim]
                chosen = int(candidates[0]) if candidates else inputs.teacher
            elif setup or model is None:
                chosen = inputs.teacher
            elif self.training and phase == self.learning_phase and self.rng.random() < self.teacher_probability:
                chosen = inputs.teacher
            elif self.training and phase == self.learning_phase and self.rng.random() < self.epsilon:
                chosen = int(self.rng.choice(np.flatnonzero(inputs.mask)))
            else:
                with torch.no_grad():
                    values = model(torch.tensor(inputs.observation).unsqueeze(0))[0].numpy()
                chosen = int(np.where(inputs.mask, values, -np.inf).argmax())
            action = inputs.actions[chosen]
            self.inputs[ally.name] = inputs
            self.plans[ally.name] = (phase, chosen, inputs, ally)
            self.actions[ally.name] = to_game_action(snapshot, ally.slot, action, names)
            if action.kind in MOVE_STEPS:
                dr, dc = MOVE_STEPS[action.kind]
                reserved.add((ally.position[0] + dr, ally.position[1] + dc))
            else:
                reserved.add(ally.position)

    def decide_move(self, char, state):
        # Planting can finish during attacker movement, before the defenders.
        # Refresh then; a cached pre-plant movement must not execute as retake.
        if self.stop_at_plant and state.get("is_planted"):
            return list(char.pos), {"facing": char.facing}
        self.prepare_team_tick()
        name = str(getattr(char, "base_name", char.name))
        if name in self.plans and self.snapshot.phase != "setup":
            self.decisions[name] = self.plans[name]
        return self.actions.get(name, (list(char.pos), {"facing": char.facing}))

    def record_opponent_round_end(self):
        site = self.scenario.site_of(self.game.planted_pos) if self.game.is_planted else None
        # Public result notification; winner is the last scored round.
        winner = "D" if self.game.is_defused or (not self.game.is_planted and
                 (self.game.round_timer <= 0 or not any(c.is_alive for c in self.game.chars if c.team == "A"))) else "A"
        self.previous_rounds.append({"site": site, "winner": winner})
