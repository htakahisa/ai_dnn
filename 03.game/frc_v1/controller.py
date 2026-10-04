"""One copied team decision per movement boundary, with explicit policy modes."""

from dataclasses import dataclass
from pathlib import Path

from frc_v1 import ROSTER, ABILITIES, ULTIMATES
from frc_v1.perception import FrcPerceptionBuilder
from frc_v1.memory import FrcMemory
from frc_v1.observation import FrcObservationEncoder
from frc_v1.actions import validate_action, to_game_action


@dataclass(frozen=True)
class ActionLog:
    key: tuple
    slot: int
    kind: str
    origin: tuple[int, int]
    target: tuple[int, int] | None
    ally_slot: int | None


class FrcController:
    handles_team_perception = True

    def __init__(self, side, *, actor=None, mode="learned", checkpoint=None, effects_mode=None, inference_only=False):
        self.side = side
        self.mode = mode
        self.sensor = FrcPerceptionBuilder(side)
        self.memory = FrcMemory()
        self.encoder = FrcObservationEncoder()
        self.game = None
        self.actor = actor
        if actor is None:
            if mode == "baseline":
                from frc_v1.baseline import FrcBaseline
                self.actor = FrcBaseline()
            elif mode == "learned":
                from frc_v1.model import FrcPolicy
                checkpoint = Path(checkpoint) if checkpoint else Path(__file__).parent / "checkpoints" / f"{side}_policy.pt"
                self.actor = FrcPolicy.load(checkpoint, side=side)
                self.actor.collect_statistics = not inference_only
            else:
                raise ValueError("FRC mode must be learned or baseline")
        self.effects_mode = effects_mode or getattr(self.actor, "effects_mode", "all")
        self.action_log = []
        self.reset_round()

    def set_game(self, game):
        # Setup supplies a fresh position-safe view for each character. FRC
        # builds its own public snapshot once per team tick, so those wrappers
        # must not reset the cached decision or replace its live game owner.
        owner = getattr(game, "_real", game)
        if self.game is not owner:
            self.reset_round()
        self.game = owner

    def reset_round(self):
        self.sensor.reset()
        self.memory.reset()
        self._cache_key = None
        self.snapshot = self.belief = self.observation = self.decision = None
        self._names = ()
        self._pending = []
        self._logged_slots = set()
        self.action_log.clear()
        if hasattr(self.actor, "reset"):
            self.actor.reset()

    def prepare_team_tick(self):
        if self.game is None:
            raise RuntimeError("FRC set_game must precede inference")
        setup = getattr(self.game, "defender_setup_phase", None)
        key = (int(self.game.current_round), "setup" if getattr(setup, "active", False) else "live",
               int(setup.ticks_remaining) if getattr(setup, "active", False) else int(self.game.battle_tick))
        if self._cache_key == key:
            return
        if self._cache_key is not None and self._cache_key[0] != key[0]:
            self.reset_round()
        snapshot = self.sensor.build(self.game)
        # A cast becomes team-private knowledge only after the game consumed its resource.
        for slot, kind, origin, target, tick, previous_resource in self._pending:
            ally = snapshot.allies[slot]
            resource = ally.charges if kind == "ABILITY" else ally.points
            if resource < previous_resource:
                self.memory.record_own_cast(ABILITIES[slot] if kind == "ABILITY" else ULTIMATES[slot], origin, target, tick)
        self._pending.clear()
        belief = self.memory.update(snapshot)
        observation = self.encoder.encode(snapshot, belief, effects_mode=self.effects_mode)
        decision = self.actor.act(observation, snapshot, belief)
        if len(decision.actions) != 5:
            raise ValueError("FRC actor must emit exactly five fixed slots")
        for slot, action in enumerate(decision.actions):
            validate_action(snapshot, observation.masks, slot, action)
        own = {str(getattr(c, "base_name", c.name)): c.name for c in self.game.chars if c.team == self.side}
        self._names = tuple(own[name] for name in ROSTER)
        self.snapshot, self.belief, self.observation, self.decision = snapshot, belief, observation, decision
        self._cache_key = key
        self._logged_slots.clear()

    def decide_move(self, char, game_state):
        self.prepare_team_tick()
        if char.team != self.side:
            raise ValueError("FRC character belongs to another side")
        slot = ROSTER.index(str(getattr(char, "base_name", char.name)))
        action = self.decision.actions[slot]
        # The legacy objective/utility branches return before parsing facing.
        # Apply the actor output at this execution boundary; the engine still
        # protects its forced shot-response facing in move_character.
        if not self.snapshot.allies[slot].forced_facing:
            char.facing = action.facing
        if slot not in self._logged_slots:
            ally = self.snapshot.allies[slot]
            self.action_log.append(ActionLog(self._cache_key, slot, action.kind, ally.position, action.target, action.ally_slot))
            self._logged_slots.add(slot)
            if action.kind in ("ABILITY", "ULTIMATE"):
                self._pending.append((slot, action.kind, ally.position, action.target, self.snapshot.tick,
                    ally.charges if action.kind == "ABILITY" else ally.points))
        return to_game_action(self.snapshot, slot, action, self._names)


class FrcAttackerController(FrcController):
    def __init__(self, **kwargs):
        super().__init__("A", **kwargs)


class FrcDefenderController(FrcController):
    def __init__(self, **kwargs):
        super().__init__("D", **kwargs)
