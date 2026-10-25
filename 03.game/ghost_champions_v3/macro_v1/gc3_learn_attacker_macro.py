"""One macro decision per tick before both teams move; frozen lower actions."""
import numpy as np
from iq_controller_adapter import IQAwareController
from iq_perception import build_team_position_view
from public_effects import PublicEffectReader
from ..preplant.gc3_learn_attacker_preplant import TwoPhaseAttackerController
from ..preplant.gc3_observation_attacker_history import MatchHistory, ACTION_WIDTH
from .gc3_policy_attacker_macro import MacroPolicy, MacroHistoryView, ACTIONS

MACRO_SEED = 20261022


class MacroAttackerController:
    handles_team_perception = True
    requires_public_effects = True

    def __init__(self, *, macro_policy=None, macro_checkpoint=None, preplant_policy=None,
                 guard_policy=None, preplant_checkpoint=None, guard_checkpoint=None,
                 history=None, recorder=None):
        self.macro_step = None
        if macro_checkpoint:
            from ..preplant.gc3_data_attacker_preplant import file_hash
            self.policy, metadata = MacroPolicy.load(macro_checkpoint)
            self.macro_step = metadata.get("step", 0)
            if not metadata.get("trained"):
                raise ValueError("Select a trained macro checkpoint")
            paths = dict(PREPLANT_CHECKPOINT=preplant_checkpoint, POSTPLANT_CHECKPOINT=guard_checkpoint)
            if any(path is None or metadata.get("lower_checkpoint_hashes", {}).get(key) != file_hash(path) for key,path in paths.items()):
                raise ValueError("Macro checkpoint requires the same frozen lower checkpoints used for training")
        else:
            self.policy = macro_policy if macro_policy is not None else MacroPolicy.fresh("attacker", MACRO_SEED)
        self.history = history if history is not None else MatchHistory()
        self.macro_history = MacroHistoryView(self.history)
        self.lower = TwoPhaseAttackerController(preplant_policy=preplant_policy, guard_policy=guard_policy,
            preplant_checkpoint=preplant_checkpoint, guard_checkpoint=guard_checkpoint, history=self.history)
        for model in (self.lower.preplant.policy, self.lower.postplant.policy):
            model.requires_grad_(False).eval()
        self.adapter = IQAwareController(self.lower, viewer_team="A")
        self.effects = PublicEffectReader()
        self.recorder = recorder
        self._real_game = None
        self.reset_macro()

    def reset_macro(self):
        self.current_goal = None
        self.goal_age = 0
        self.macro_decisions = self.switches = 0
        self.last_action = None
        self._token = None

    def set_game(self, game):
        # Setup supplies a public team view after the initial real binding.
        if self._real_game is None:
            self._real_game = game
            self.adapter.set_game(game)
        else:
            self.lower.set_game(game)

    def reset_round(self):
        self.reset_macro()
        self.effects.reset()
        self.adapter.reset_round()
        self.macro_history._encoded = self.macro_history._cache_policy = None

    def prepare_team_tick(self):
        game = self._real_game
        setup = bool(game.in_defender_setup_phase)
        if self.history.live:
            self.history.sync_round(game)
        elif self.history.round_id < 0:
            self.history.begin_round()
        token = (self.history.round_id, setup, game.defender_setup_phase.ticks_remaining if setup else game.battle_tick)
        if self._token == token:
            return
        own = [c for c in game.chars if c.team == "A"]
        viewer = next((c for c in own if c.is_alive and c.has_spike), next((c for c in own if c.is_alive), own[0]))
        draft = dict(grid=game.grid, chars=game.chars, is_planted=game.is_planted,
            planted_pos=game.planted_pos, spike_pos=game.spike_pos, round_timer=game.round_timer,
            detonate_timer=game.detonate_timer, battle_tick=game.battle_tick, defender_setup_active=setup,
            defender_setup_ticks_remaining=game.defender_setup_phase.ticks_remaining,
            smoke_cells=game._smoke_cells(), available_orbs=set(game.available_orbs),
            defender_defuse_info={str(c.name):(c.defuse_timer, 6) for c in game.chars if c.team=="D" and c.defuse_timer>0})
        if setup:
            view = build_team_position_view(game, "A")
            char = next(c for c in view.chars if c.name == viewer.name)
            state = dict(draft, chars=view.chars)
        else:
            engine = self.adapter.perception_engine
            view = engine.build_game_view(viewer=viewer, game=game)
            char = view.perceived_character_for(viewer)
            state = engine.build_perceived_state(viewer=viewer, game_state=draft, game_view=view)
            state["public_effects"] = self.effects.read_visible(game, "A")
        self.decide_macro(char, state)
        self._token = token

    def decide_macro(self, char, state):
        self.macro_history.sync_public_events()
        action, record = self.policy.select_macro(char, state, self.macro_history, self.current_goal, self.goal_age)
        if action not in range(len(ACTIONS)) or action == 0 and self.current_goal is None:
            raise ValueError("Invalid macro site action")
        goal = self.current_goal if action == 0 else action-1
        if self.current_goal != goal:
            self.switches += self.current_goal is not None
            self.goal_age = 0
        else:
            self.goal_age += 1
        self.current_goal = goal
        self.history.goal_sites[self.history.round_id] = goal
        command = dict(macro_action=ACTIONS[action], goal_site=("A", "B")[goal])
        self.macro_history.events.append(dict(obs=record["obs"].copy(), action=np.zeros(ACTION_WIDTH, dtype=np.float32),
            round_id=self.history.round_id, name="macro", tick=state.get("battle_tick", 0), command=command,
            goal_site=goal, macro_action=action))
        if self.recorder is not None:
            self.recorder.record("macro", record)
        self.macro_decisions += 1
        self.last_action = ACTIONS[action]
        return command

    def decide_move(self, char, state):
        if self._token is None:
            self.prepare_team_tick()
        if state.get("defender_setup_active"):
            return self.lower.decide_move(char, state)
        return self.adapter.decide_move(char, state)

    def on_round_finished(self, result):
        self.lower.on_round_finished(result)

    def attacker_snapshot(self):
        return dict(controller=type(self).__name__, macro=dict(goal_site=None if self.current_goal is None else ("A", "B")[self.current_goal],
            action=self.last_action, decisions=self.macro_decisions, switches=self.switches, selection="sampled"),
            lower=self.lower.attacker_snapshot())
