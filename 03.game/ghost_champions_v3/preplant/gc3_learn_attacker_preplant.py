"""One shared pre-plant policy for all attackers, then the existing guard policy."""
from .. import ATTACKER_SEED
from pathlib import Path
from ..gc3_policy import ScratchController
from .gc3_policy_attacker_history import HistoryPolicy, load_preplant_policy
from .gc3_observation_attacker_history import MatchHistory, encode_public, command_features

HISTORY_LOG_ROOT = Path(__file__).resolve().parents[1]/"logs"/"match_history"
SAVE_MATCH_HISTORY = True


class PreplantController(ScratchController):
    def __init__(self, *, policy=None, checkpoint=None, recorder=None, history=None):
        if policy is None:
            policy = load_preplant_policy(checkpoint)[0] if checkpoint else HistoryPolicy.fresh("attacker", ATTACKER_SEED+100)
        super().__init__(role="attacker", seed=ATTACKER_SEED+100,
                         policy=policy, checkpoint=checkpoint, recorder=recorder)
        self.history = history if history is not None else MatchHistory.live_match(HISTORY_LOG_ROOT) if SAVE_MATCH_HISTORY and recorder is None else MatchHistory()
        self._finished_count = 0

    def reset_round(self):
        super().reset_round()
        # Retain every previous round in this match, with a new boundary.
        self.finish_round_from_game()
        if self.history.live:
            self.history.sync_round(getattr(self, "game", None))
        else:
            self.history.begin_round()

    def finish_round_from_game(self):
        if self.history.live:
            self.history.sync_round(getattr(self, "game", None))
            return
        records = getattr(getattr(getattr(self, "game", None), "analytics_tracker", None), "round_records", ())
        if len(records) > self._finished_count:
            result = records[-1]
            self.history.finish_round(dict(success=result.get("winner") == "attacker",
                planted=bool(result.get("planted", False)), reason=result.get("reason", "unknown")))
            self._finished_count = len(records)

    def remember(self, char, state, command, observation=None):
        if self.history.live:
            self.history.sync_round(getattr(self, "game", None))
        if self.history.round_id < 0:
            self.history.begin_round()
        self.history.append(encode_public(char, state, self.history) if observation is None else observation,
            command_features(char, command, state), name=char.name, tick=state.get("battle_tick", 0), command=command)

    def decide_move(self, char, state):
        # A plant can finish in the middle of a tick. Do not record later
        # characters in that tick as pre-plant decisions.
        if state.get("is_planted"):
            return list(char.pos), "HOLD"
        if not isinstance(self.policy, HistoryPolicy):
            return super().decide_move(char, state)
        if not char.is_alive:
            return list(char.pos), "HOLD"
        if self.history.live:
            self.history.sync_round(getattr(self, "game", None))
        if self.history.round_id < 0:
            self.history.begin_round()
        command, record = self.policy.select(char, state, self.history)
        self.remember(char, state, command, record["obs"])
        self.last_selections[str(char.name)] = dict(record["selections"])
        if self.recorder is not None:
            self.recorder.record(str(char.name), record)
        return command

    def snapshot(self):
        self.finish_round_from_game()
        result = super().snapshot()
        result.update(history_events=len(self.history.events), history_round=self.history.round_id,
                      history_file=str(self.history.journal_path) if self.history.journal_path else None)
        return result

    def on_round_finished(self, result):
        self.history.finish_round(result)


class TwoPhaseAttackerController:
    requires_public_effects = True

    def __init__(self, *, preplant_checkpoint=None, guard_checkpoint=None,
                 preplant_policy=None, guard_policy=None, history=None):
        from ..gc3_learn_attacker_guard import GhostChampionsV3AttackerController
        self.preplant = PreplantController(policy=preplant_policy, checkpoint=preplant_checkpoint, history=history)
        self.postplant = GhostChampionsV3AttackerController(policy=guard_policy, checkpoint=guard_checkpoint)

    def set_game(self, game):
        self.preplant.set_game(game)
        self.postplant.set_game(game)

    def reset_round(self):
        self.preplant.finish_round_from_game()
        self.preplant.reset_round()
        self.postplant.reset_round()

    def decide_move(self, char, state):
        controller = self.postplant if state.get("is_planted") else self.preplant
        command = controller.decide_move(char, state)
        if controller is self.postplant and isinstance(self.preplant.policy, HistoryPolicy):
            self.preplant.remember(char, state, command)
        return command

    def attacker_snapshot(self):
        return dict(controller=type(self).__name__, preplant=self.preplant.snapshot(),
                    postplant=self.postplant.snapshot(), selection="sampled")

    def on_round_finished(self, result):
        self.preplant.on_round_finished(result)
