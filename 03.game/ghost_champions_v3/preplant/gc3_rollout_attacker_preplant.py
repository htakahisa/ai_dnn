"""Actual pre-plant rounds against one fixed rule AI and varied roster stats."""
from collections import Counter
import random
import numpy as np
from grid_paths import distance_map
from ..gc3_rollout import engine_directory, Recorder, freeze_modules, module_digest
from ..gc3_reward_phase import PhaseRewards, reward_settings, WIN_REWARD, LOSS_REWARD
from ..gc3_engine_actions import MODES
from ..gc3_observation import ARENA_SHAPE
from .gc3_learn_attacker_preplant import PreplantController
from .gc3_teacher_attacker_actions import RecordingTeacher
from .gc3_observation_attacker_history import MatchHistory
from .gc3_teacher_attacker_roster import RosterTeacher
from .gc3_state_attacker_resources import MatchResources, complete_round, point_snapshot, POSTPLANT_CHECKPOINT
from .gc3_reward_attacker_objectives import MicroObjectives, objective_reward_settings

MAJOR_TEAMS = {"FNC": "Fnatic2023", "FRC": "Furina Classic", "TYG": "Touyama Gaming",
               "OMG": "Omoko Gaming", "SPS": "SUPES", "GRG": "Gorigons", "GC": "Ghost Champions"}
PREPLANT_REWARD_VERSION = "gc3-preplant-personal-retrieval-v6"
SITE_PROGRESS_WEIGHT = 1.0
RETRIEVAL_PROGRESS_WEIGHT = 6.0
DISTANCE_SCALE = 12.0
MAX_EPISODE_TICKS = 400
ORB_ULTIMATE_REWARD_PER_POINT = 0.20
BATTLE_TIME_COST_PER_TICK = 0.01


def choose_roster(seed, episode, codes):
    """Shuffle each complete roster cycle instead of accidentally omitting a team."""
    if not codes or len(set(codes)) != len(codes) or any(code not in MAJOR_TEAMS for code in codes):
        raise ValueError("Invalid major-team roster list")
    cycle, index = divmod(episode, len(codes))
    order = list(codes)
    random.Random(seed+cycle).shuffle(order)
    return order[index]


def preplant_reward_settings():
    return dict(reward_settings(), **objective_reward_settings(), REWARD_VERSION=PREPLANT_REWARD_VERSION,
                SITE_PROGRESS_WEIGHT=SITE_PROGRESS_WEIGHT, RETRIEVAL_PROGRESS_WEIGHT=RETRIEVAL_PROGRESS_WEIGHT,
                DISTANCE_SCALE=DISTANCE_SCALE,
                success="actual attacker round win after frozen postplant, not merely planting",
                site_distance="nearest legal plant cell within the actor's selected target site",
                placement_hints=False, ORB_ULTIMATE_REWARD_PER_POINT=ORB_ULTIMATE_REWARD_PER_POINT,
                orb_reward="personal reward only for actual ULT points gained at completed collection",
                resource_carry="actual remaining ULT points after frozen postplant; reset for each rollout match",
                BATTLE_TIME_COST_PER_TICK=BATTLE_TIME_COST_PER_TICK,
                time_cost="elapsed live preplant clock ticks, including moves, waits and channels; no setup cost",
                postplant_credit="one actual round outcome, discounted over actual frozen postplant ticks")


def credit_round_outcome(recorder, postplant_ticks, gamma, success):
    """Advance elapsed frozen ticks without recording or learning guard actions."""
    for _ in range(max(0, postplant_ticks-1)):
        recorder.reward(0., gamma, False, components={})
    if postplant_ticks:
        outcome = WIN_REWARD if success else LOSS_REWARD
        recorder.reward(outcome, gamma, True, components={"round_outcome":outcome})


class PreplantRewards(PhaseRewards):
    def __init__(self, game, gamma, goal_provider=None):
        super().__init__(game, "preplant", gamma)
        self.goal_provider = goal_provider
        self.goal_distances = {}
        for site in (0, 1):
            points = [p for p in np.argwhere(game.grid == 2) if int(p[1] >= game.grid.shape[1]/2) == site]
            distances = [distance_map(game.grid, tuple(point)) for point in points]
            if not distances:
                raise ValueError("Both target sites need legal plant cells")
            self.goal_distances[site] = np.min(np.stack([np.where(d >= 0, d, 100000) for d in distances]), axis=0)
        # Kept for legacy descriptive metrics only. Rewards never read this union.
        self.site_distances = np.minimum(self.goal_distances[0], self.goal_distances[1])
        self.drop = None
        self.drop_distances = None
        self.completed_orbs = 0
        self.orb_points_gained = 0
        self.objectives = MicroObjectives(game, gamma, self.current_goal, self.site_potential,
            retrieval_weight=RETRIEVAL_PROGRESS_WEIGHT, distance_scale=DISTANCE_SCALE)

    def __enter__(self):
        super().__enter__()
        self.original_move = self.game.move_character
        self.game.move_character = self.move_character
        self.original_shots = self.game._resolve_all_shots
        self.game._resolve_all_shots = self.resolve_all_shots
        return self

    def __exit__(self, *args):
        self.game.move_character = self.original_move
        self.game._resolve_all_shots = self.original_shots
        return super().__exit__(*args)

    def resolve_all_shots(self, *args, **kwargs):
        self.objectives.observe_before_shots()
        return self.original_shots(*args, **kwargs)

    def move_character(self, char):
        point = tuple(char.pos)
        had_orb = point in self.game.available_orbs
        before_points = int(char.ultimate_points)
        before = dict(timer=int(char.plant_timer), has_spike=bool(char.has_spike),
                      planted=bool(self.game.is_planted), pos=point)
        result = self.original_move(char)
        if hasattr(self, "objectives"):
            self.objectives.observe_move(char, before)
        if (char.team == "A" and had_orb and point not in self.game.available_orbs
                and getattr(char, "collecting_orb_this_tick", False)):
            # Observe immediately after movement, before kills or passive ULT
            # consumption later in this tick can change the point balance.
            gained = max(0, int(char.ultimate_points)-before_points)
            self.completed_orbs += 1
            self.orb_points_gained += gained
            self.pending[str(char.name)]["orb_ultimate_gain"] += ORB_ULTIMATE_REWARD_PER_POINT*gained
        return result

    def current_goal(self):
        if self.goal_provider is not None:
            return self.goal_provider()
        target = getattr(self.game, "target_plant_pos", None)
        return None if target is None else int(target[1] >= self.game.grid.shape[1]/2)

    def site_potential(self, position, goal):
        if position is None or goal is None:
            return 0.
        distance = int(self.goal_distances[int(goal)][tuple(position)])
        return SITE_PROGRESS_WEIGHT/(1+distance/DISTANCE_SCALE) if distance < 100000 else 0.

    def begin_tick(self):
        self.tick_round_timer = self.game.round_timer
        self.objectives.begin_tick()
        super().begin_tick()

    def finish_tick(self, terminal, success):
        shared, personal = super().finish_tick(terminal, success)
        # Objective progress uses bounded new-best improvements, not a
        # potential reset at a pickup/plant. Combat shaping remains unchanged.
        shared.pop("site_progress", None)
        shared.pop("retrieval_progress", None)
        objective_shared, objective_personal = self.objectives.finish_tick(terminal)
        shared.update(objective_shared)
        for name, parts in objective_personal.items():
            row = personal.setdefault(name, {})
            for key, amount in parts.items():
                row[key] = row.get(key, 0.)+amount
        # A completed plant earns its own bonus, separately from the actual
        # win/loss delivered after frozen guard play.
        if terminal and self.game.is_planted and not self.game.round_over:
            shared.pop("outcome", None)
        elif "outcome" in shared:
            shared["round_outcome"] = shared.pop("outcome")
        elapsed = max(0, self.tick_round_timer-self.game.round_timer)
        shared["battle_time"] = -BATTLE_TIME_COST_PER_TICK*elapsed
        return shared, personal

    def objective_potential(self):
        carrier = next((c for c in self.own.values() if c.is_alive and c.has_spike), None)
        site = 0.
        retrieval = 0.
        if carrier is not None:
            site = self.site_potential(carrier.pos, self.current_goal())
        elif self.game.spike_pos is not None:
            drop = tuple(self.game.spike_pos)
            if drop != self.drop:
                self.drop, self.drop_distances = drop, distance_map(self.game.grid, drop)
            distances = [int(self.drop_distances[tuple(c.pos)]) for c in self.own.values() if c.is_alive]
            distances = [d for d in distances if d >= 0]
            if distances:
                retrieval = RETRIEVAL_PROGRESS_WEIGHT/(1+min(distances)/DISTANCE_SCALE)
        return dict(site_progress=site, retrieval_progress=retrieval)


def distinguish_same_roster(game):
    """The engine keys effects and analytics by name; give mirror opponents unique IDs.

    Keep base_name, display_name, stats, combo bonuses and IGL flags unchanged.
    This is applied once after the engine has created both real rosters.
    """
    names = {str(c.name) for c in game.chars if c.team == "A"}
    aliases = {}
    for char in game.chars:
        if char.team == "D" and str(char.name) in names:
            original = str(char.name)
            char.name = "D:"+original
            aliases[char.name] = original
    if aliases:
        if "D:"+str(game.defender_igl_name) in aliases:
            game.defender_igl_name = "D:"+str(game.defender_igl_name)
        if "D:"+str(game.defender_spike_holder_name) in aliases:
            game.defender_spike_holder_name = "D:"+str(game.defender_spike_holder_name)
        game.match_stats = {str(c.name): dict(kills=0, deaths=0,
                                             ultimate_points=c.ultimate_points) for c in game.chars}
        game.analytics_tracker.register_players(game.chars,
            {"A":game.attacker_team_name, "D":game.defender_team_name})
        game.analytics_tracker.begin_round_tactics(game.width,
            [tuple(map(int, p)) for p in np.argwhere(game.grid == 2)], game.chars)
    return aliases


def carrier_movement_metrics(rows, site_distances):
    """Measured public decisions for evaluation logs, never a policy preference."""
    distances, stationary, away = [], [], []
    for row in rows:
        if row["obs"][11] <= .5:
            continue
        point = tuple(np.rint(row["obs"][:2]*ARENA_SHAPE).astype(int))
        mode = MODES[row["selections"]["mode"]]
        destination = np.unravel_index(row["selections"]["position"], ARENA_SHAPE) if mode in ("MOVE", "MOVE_WATCH") else point
        distance = int(site_distances[point])
        distances.append(distance)
        stationary.append(destination == point)
        away.append(int(site_distances[destination]) > distance)
    return dict(carrier_decisions=len(distances),
        carrier_start_site_distance=distances[0] if distances else None,
        carrier_closest_site_distance=min(distances) if distances else None,
        carrier_stationary_fraction=float(np.mean(stationary)) if stationary else None,
        carrier_moves_away_fraction=float(np.mean(away)) if away else None)


def play_preplant(code, seed, *, policy=None, teacher=False, teacher_maps=("A1", "A2", "A3", "A4"),
                  fixed_opponent_ai="default", gamma=.99, gae_lambda=.95, history=None, label_policy=None,
                  teacher_kind="concon", match_resources=None, postplant_checkpoint=POSTPLANT_CHECKPOINT):
    if code not in MAJOR_TEAMS or fixed_opponent_ai != "default":
        raise ValueError("Pre-plant opponents must use the fixed default rule AI")
    if teacher == (policy is not None):
        raise ValueError("Supply either a student policy or a ConCon teacher")
    if label_policy is not None and not teacher:
        raise ValueError("A label_policy requires the frozen ConCon teacher")
    recorder = Recorder()
    history = history if history is not None else MatchHistory()
    if match_resources is None:
        # A repeated caller sharing history also shares the physical resource
        # state. Separate matches/checkpoints cannot leak points to each other.
        if not hasattr(history, "match_resources"):
            history.match_resources = MatchResources()
        match_resources = history.match_resources
    with engine_directory():
        from run_game import VisualFPSBattle, _build_team_ai
        from run_competition_manager import seed_all
        from simulation_runtime import cpu_inference
        from team_ai import DualRoleTeamAI
        from controllers import DefaultDefenderController
        from party_presets import get_preset
        from map_data import NEW_MAZE_STR
        seed_all(seed)
        # Demonstrations and deployment now use the very same real GC roster.
        own = get_preset("Ghost Champions")
        opposing = get_preset(MAJOR_TEAMS[code])
        with cpu_inference():
            if teacher:
                if teacher_kind == "omoko":
                    from .gc3_teacher_attacker_omoko import OmokoTeacher
                    coach = OmokoTeacher()
                elif teacher_kind == "concon":
                    from concon_v1.co1_attacker_controller import ConconRoundAttackerController
                    coach = RosterTeacher(ConconRoundAttackerController(map_names=teacher_maps, seed=seed, selection_path=None))
                else:
                    raise ValueError("Unknown preplant teacher")
                raw = RecordingTeacher(coach, history=history, executing_policy=label_policy)
                factory = lambda: raw
            else:
                raw = PreplantController(policy=policy, recorder=recorder, history=history)
                factory = lambda: raw
            own_ai = DualRoleTeamAI("ConCon teacher" if teacher else "GC v3 pre-plant", factory, DefaultDefenderController)
            opponent_ai = _build_team_ai(fixed_opponent_ai)
            game = VisualFPSBattle(NEW_MAZE_STR, own_ai, opponent_ai, headless=True, disable_side_swap=True,
                attacker_roster=list(own.players), defender_roster=list(opposing.players),
                spike_holder_name=own.spike_holder, defender_spike_holder_name=opposing.spike_holder,
                attacker_igl_name=own.igl, defender_igl_name=opposing.igl)
            game.stop_after_round = True
            aliases = distinguish_same_roster(game)
            match_resources.restore(game, code)
            start_points = point_snapshot(game)
            frozen = freeze_modules(game.defender_controller, excluded=(game, policy, label_policy))
            if teacher:
                frozen += freeze_modules(raw.teacher, excluded=(game, policy, label_policy))
            before = module_digest(frozen)
            success = False
            def selected_goal():
                goal = history.goal_sites.get(history.round_id)
                if goal is not None:
                    return goal
                if teacher:
                    return getattr(raw.teacher, "target_site", None)
                # Existing checkpoints remain usable for explicit comparison.
                if not hasattr(policy, "site"):
                    return int(game.target_plant_pos[1] >= game.width/2) if game.target_plant_pos is not None else None
                return None
            with PreplantRewards(game, gamma, selected_goal) as rewards:
                for tick in range(MAX_EPISODE_TICKS):
                    rewards.begin_tick()
                    game._simulate_tick()
                    terminal = bool(game.is_planted or game.round_over or game.match_over)
                    success = bool(game.attacker_wins)
                    shared, personal = rewards.finish_tick(terminal, success)
                    if not teacher:
                        deferred = terminal and game.is_planted and not game.round_over
                        recorder.reward(sum(shared.values()), gamma, terminal and not deferred,
                                        components=shared, individual_components=personal)
                    if terminal:
                        break
                else:
                    raise RuntimeError("Pre-plant round exceeded the engine limit")
            phase_points = point_snapshot(game)
            postplant_ticks = complete_round(game, history, postplant_checkpoint)
            success = bool(game.attacker_wins)
            if not teacher:
                credit_round_outcome(recorder, postplant_ticks, gamma, success)
            match_resources.capture(game, code)
            if module_digest(frozen) != before:
                raise RuntimeError("Frozen teacher/opponent changed")
            result = game.analytics_tracker.round_records[-1] if game.analytics_tracker.round_records else {}
            rows = raw.rows if teacher else recorder.advantages(gae_lambda)
            stats = dict(phase="preplant", opponent=code, opponent_ai=fixed_opponent_ai, seed=seed,
                success=success, planted=bool(game.is_planted),
                plant=list(game.planted_pos) if game.planted_pos is not None else None,
                reason="planted" if game.is_planted else result.get("reason", "unknown"),
                decisions=len(rows), ticks=tick+1, attacker_roster=own.name, defender_roster=opposing.name,
                duplicate_name_aliases=aliases, opponent_weights_unchanged=True,
                target_site=("A", "B")[selected_goal()] if selected_goal() is not None else None,
                planted_site=("A" if game.planted_pos[1] < game.width/2 else "B") if game.is_planted else None,
                orb_collections=rewards.completed_orbs, orb_ultimate_points_gained=rewards.orb_points_gained,
                spike_pickups=rewards.objectives.pickups, plant_cancellations=rewards.objectives.cancellations,
                plant_completions=rewards.objectives.plants,
                ultimate_points_start=start_points, ultimate_points_preplant_end=phase_points,
                ultimate_points_round_end=point_snapshot(game), postplant_resource_ticks=postplant_ticks,
                actual_round_success=bool(game.attacker_wins), actual_round_reason=result.get("reason", "unknown"),
                postplant_policy_learning=False, setup_learning_included=True,
                reward=PREPLANT_REWARD_VERSION, reward_component_sums={}, full_match_evaluated=False,
                actual_round_evaluated=True, learning_objective="actual_round_win")
            if teacher:
                stats.update(teacher="Omoko Gaming v1" if teacher_kind == "omoko" else "ConCon v1", teacher_map=raw.teacher.current_map_name,
                             teacher_model_hashes=raw.teacher.model_hashes, skipped=dict(raw.skipped),
                             mode_counts=dict(Counter(MODES[r["selections"]["mode"]] for r in rows)),
                             carrying_decisions=sum(r["carrying"] for r in rows),
                             retrieval_decisions=sum(r["retrieving"] for r in rows))
                stats["executed_by"] = "sampled student (DAgger labels from teacher)" if label_policy is not None else "teacher"
            else:
                goal = selected_goal()
                stats.update(carrier_movement_metrics(rows, rewards.goal_distances[goal] if goal is not None else rewards.site_distances))
                for row in rows:
                    row["reward_version"] = PREPLANT_REWARD_VERSION
                stats["reward_component_sums"] = {key:sum(r["reward_components"].get(key, 0.) for r in rows)
                    for key in sorted({key for r in rows for key in r["reward_components"]})}
            history.finish_round(stats)
            stats.update(history_events=len(history.events), history_round=history.round_id,
                         history_scope="all public events in this match; no truncation")
            return rows, stats
