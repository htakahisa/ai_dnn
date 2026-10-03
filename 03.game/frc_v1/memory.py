"""Round-local history derived only from safe sensor snapshots."""

from dataclasses import dataclass
from game_core import BALEMOON_WARNING_TICKS, NEON_WARNING_TICKS, TUNNEL_WARNING_TICKS


@dataclass(frozen=True)
class EffectHistory:
    handle: int
    age: int
    start_known: bool
    predicted_remaining: int | None
    affiliation: str


@dataclass(frozen=True)
class FrcBelief:
    source_key: tuple
    clear: tuple[tuple[tuple[int, int], int], ...]
    last_seen: tuple[tuple[int, tuple[int, int], int], ...]
    effects: tuple[EffectHistory, ...]


class FrcMemory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.round = None
        self._last_key = None
        self._clear = {}
        self._seen = {}
        self._effects = {}
        self._own_casts = []
        self._result = None

    def record_own_cast(self, kind, origin, target, tick):
        self._own_casts.append((kind, tuple(origin), target, tick))

    def update(self, snapshot):
        if snapshot.round_number != self.round:
            self.reset()
            self.round = snapshot.round_number
        if self._last_key == snapshot.key:
            return self._result
        time = snapshot.tick if snapshot.phase == "live" else -snapshot.tick
        prior_time = None if self._last_key is None else (self._last_key[2] if self._last_key[1] == "live" else -self._last_key[2])
        continuous = prior_time is not None and time - prior_time == 1 and snapshot.phase == self._last_key[1]
        for cell in snapshot.visible_cells:
            self._clear[cell] = time
        for sighting in snapshot.sightings:
            self._seen[sighting.enemy_id] = (sighting.position, time)
        history, present = [], set()
        durations = {"BALEMOON": BALEMOON_WARNING_TICKS + 1,
                     "NEON": NEON_WARNING_TICKS + 1, "TUNNEL": TUNNEL_WARNING_TICKS + 1}
        for effect in snapshot.effects:
            present.add(effect.handle)
            previous = self._effects.get(effect.handle)
            if previous is None or previous[3] != effect.phase:
                affiliation = "unknown"
                for kind, origin, target, tick in self._own_casts:
                    direct_cast = (kind == effect.kind and tick <= time <= tick + 1 and (
                        effect.trail and effect.trail[0] == origin or
                        effect.kind == "BALEMOON" and effect.position == origin or
                        target is not None and effect.position == target))
                    ash_impact = (kind == "ASH" and effect.kind == "DESTRUCTION" and
                                  target is not None and effect.position == target and
                                  tick <= time <= tick + 5)
                    if direct_cast or ash_impact:
                        affiliation = "own"
                        break
                # A newly visible cast has already advanced once in the
                # process_battle between two movement-boundary snapshots.
                start = time - 1 if previous is None and continuous else time
                previous = (start, continuous, affiliation, effect.phase)
                self._effects[effect.handle] = previous
            start, known, affiliation, phase = previous
            remaining = max(0, durations[effect.kind] - (time - start)) if (
                known and effect.phase == "warning" and effect.kind in durations) else None
            history.append(EffectHistory(effect.handle, max(0, time - start), known, remaining, affiliation))
        self._effects = {key: value for key, value in self._effects.items() if key in present}
        self._own_casts = [cast for cast in self._own_casts if time - cast[3] <= 5]
        self._last_key = snapshot.key
        self._result = FrcBelief(snapshot.key,
            tuple(sorted((cell, max(0, time - seen)) for cell, seen in self._clear.items())),
            tuple((index, pos, max(0, time - seen)) for index, (pos, seen) in sorted(self._seen.items())), tuple(history))
        return self._result
