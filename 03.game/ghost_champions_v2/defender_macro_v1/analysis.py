"""Adapt TV4's site predictor to GC's IQ-filtered observations and rolling history."""
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch

from toruAI_v4.tv4_observer import FeatureHistory
from toruAI_v4.tv4_model import SiteModel
from toruAI_v4.tv4_scenario import Scenario
from ghost_champions_v2.geometry import axis_for, valid
from .tendency import EARLY_TICKS, summarize_tendency

HERE = Path(__file__).resolve().parent
ACTIVE_MODEL = HERE / "data" / "active" / "analysis.pt"
VERSION = 1
SIDES = ("A", "B")
HISTORY_LIMIT = 12


def public_snapshot(char, state, round_number):
    """Copy only the existing GC perception input, never inspect real characters.

    An IQ-blurred teammate position cannot certify that a region is empty.
    TV4's negative-visibility fields consequently remain unknown (zero).
    """
    grid = np.asarray(state["grid"])
    units = sorted(state.get("chars", ()), key=lambda c: str(c.name))
    own = [c for c in units if c.team == char.team][:5]
    opponents = [c for c in units if c.team != char.team][:5]
    allies, enemies, sightings = [], [], []
    for i in range(5):
        c = own[i] if i < len(own) else None
        allies.append(NS(slot=i, name=str(c.name) if c else f"missing_{i}",
                         alive=bool(c and c.is_alive), hp=float(getattr(c, "hp", 0)),
                         position=tuple(c.pos) if c else (0, 0),
                         blind=int(getattr(c, "blind_remaining", 0))))
        e = opponents[i] if i < len(opponents) else None
        enemies.append(NS(enemy_id=i, alive=bool(e and e.is_alive)))
        if (e and e.is_alive and getattr(e, "position_known", False)
                and valid(grid, tuple(e.pos))):
            sightings.append(NS(enemy_id=i, position=tuple(map(int, e.pos)), source="gc_iq"))
    return NS(tick=int(state.get("battle_tick", 0)), round_number=round_number,
              round_timer=float(state.get("round_timer", 0)), allies=tuple(allies),
              enemies=tuple(enemies), sightings=tuple(sightings), visible_cells=(),
              spike_dropped=tuple(state["spike_pos"]) if state.get("spike_pos") is not None else None)


class GCFeatureHistory(FeatureHistory):
    def __init__(self, scenario):
        super().__init__(scenario)
        self.base_fields = list(self.fields)
        self.fields += [f"recent_{i}_{key}" for i in range(1, 7) for key in
                        ("site_known", "A", "B", "no_plant", "contacts_A", "contacts_Mid", "contacts_B")]
        self.fields += ["match_round", "past_defense_rounds"]

    def reset(self):
        super().reset()
        self.max_contacts = dict.fromkeys(("A", "Mid", "B"), 0)

    def encode(self, snapshot, rounds):
        # The parent validates against its own field count.
        fields = self.fields
        self.fields = self.base_fields
        try:
            base = super().encode(snapshot, rounds[-HISTORY_LIMIT:])
        finally:
            self.fields = fields
        counts = dict.fromkeys(self.max_contacts, 0)
        for sighting in snapshot.sightings:
            counts[axis_for(sighting.position, self.scenario.grid.shape[1])] += 1
        self.max_contacts = {axis: max(self.max_contacts[axis], counts[axis]) for axis in counts}
        values = []
        for i in range(1, 7):
            row = rounds[-i] if len(rounds) >= i else {}
            contacts = row.get("contacts", {})
            values.extend([row.get("site") in ("L", "R"), row.get("site") == "L",
                           row.get("site") == "R", bool(row.get("no_plant", False)),
                           *[contacts.get(axis, 0) / 5 for axis in counts]])
        values += [min(1., snapshot.round_number / 48), min(1., len(rounds) / HISTORY_LIMIT)]
        result = np.concatenate((base, np.asarray(values, np.float32)))
        if result.shape != (len(self.fields),) or not np.isfinite(result).all():
            raise ValueError("Invalid GC defender analysis observation")
        return result


def schema(scenario):
    return dict(version=VERSION, fields=GCFeatureHistory(scenario).fields,
                map_signature=scenario.signature, sensor="gc_iq_public_v1", sites=SIDES,
                history="public_plant_and_observed_contacts_rolling12_v1")


def load_model(path, scenario):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    expected = schema(scenario)
    if payload.get("schema") != expected or payload.get("labeled_rounds", 0) < 1:
        raise ValueError(f"GC defender analysis checkpoint schema/training mismatch: {path}")
    with torch.random.fork_rng(devices=[]):
        model = SiteModel(len(expected["fields"]))
        model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


class StableDecision:
    """Require consecutive confident ticks and expire unsupported predictions."""
    def __init__(self, threshold=.75, confirm=3, expire=8):
        self.threshold, self.confirm, self.expire = threshold, confirm, expire
        self.side = self.candidate = self.last_confident = self.tick = None
        self.streak = 0

    def update(self, probabilities, tick):
        if self.tick == tick:
            return
        consecutive = self.tick is not None and tick == self.tick + 1
        self.tick = tick
        index = int(np.argmax(probabilities))
        candidate = SIDES[index] if probabilities[index] >= self.threshold else None
        self.streak = self.streak + 1 if consecutive and candidate is not None and candidate == self.candidate else int(candidate is not None)
        self.candidate = candidate
        if candidate:
            self.last_confident = tick
            if self.streak >= self.confirm:
                self.side = candidate
        elif self.last_confident is None or tick - self.last_confident >= self.expire:
            self.side = None


class AttackSiteAnalysis:
    def __init__(self, scenario=None, model=None, checkpoint=None):
        self.scenario = scenario or Scenario()
        self.model = model
        self.checkpoint = Path(checkpoint) if checkpoint else ACTIVE_MODEL
        if model is None and (checkpoint or self.checkpoint.is_file()):
            self.model, _ = load_model(self.checkpoint, self.scenario)
        self.previous_rounds = []
        self.reset_round()

    def reset_round(self):
        self.history = GCFeatureHistory(self.scenario)
        self.gate = StableDecision()
        self.probabilities = np.array([.5, .5], np.float32)
        self.frames = []
        self.public_site = None
        self.round_number = None
        self.cache = None
        self.current_contacts = dict(A=0, Mid=0, B=0)
        self.pressure_ticks = dict(A=None, B=None)
        self.rush_ticks = dict(A=None, B=None)
        self.plant_tick = None
        self.plant_announced = False

    def finish_round(self, *, winner=None, no_plant=False):
        if self.round_number is None:
            return
        public = {"L": "A", "R": "B"}.get(self.public_site)
        rushed = [s for s, tick in self.rush_ticks.items() if tick is not None]
        rush = None
        if public:
            if public in rushed or self.plant_tick is not None and self.plant_tick <= EARLY_TICKS:
                rush = public
        elif len(rushed) == 1:
            rush = rushed[0]
        counts = self.history.max_contacts
        pressure = max(("A", "B"), key=lambda s: counts[s])
        other = "B" if pressure == "A" else "A"
        pressure = pressure if counts[pressure] >= 3 and counts[pressure] > counts[other] else None
        self.previous_rounds.append(dict(site=self.public_site, winner=winner,
            no_plant=no_plant and not self.plant_announced, contacts=dict(counts), plant_tick=self.plant_tick,
            rush_site=rush, rush_ticks=dict(self.rush_ticks), pressure_site=pressure,
            enemy_sightings=[dict(enemy_id=i, position=list(p), tick=t)
                             for i, (p, t, _) in sorted(self.history.tracks.items())]))
        self.previous_rounds = self.previous_rounds[-HISTORY_LIMIT:]
        self.round_number = None

    def observe(self, char, state, round_number):
        if state.get("defender_setup_active"):
            return
        if not np.array_equal(state["grid"], self.scenario.grid):
            raise ValueError("GC defender analysis map differs from its schema")
        planted = bool(state.get("is_planted"))
        key = (round_number, int(state.get("battle_tick", 0)), planted)
        if key == self.cache:
            return
        if self.round_number is not None and self.round_number != round_number:
            self.finish_round(no_plant=self.public_site is None)
            self.reset_round()
        self.round_number, self.cache = round_number, key
        if planted:
            self.plant_announced = True
            point = state.get("planted_pos")
            if point is not None:
                self.public_site = self.scenario.site_of(point)
                if self.plant_tick is None:
                    self.plant_tick = int(state.get("battle_tick", 0))
            return  # Plant coordinates are never part of preplant training inputs.
        snapshot = public_snapshot(char, state, round_number)
        features = self.history.encode(snapshot, self.previous_rounds)
        self.current_contacts = dict(A=0, Mid=0, B=0)
        living = {e.enemy_id for e in snapshot.enemies if e.alive}
        for enemy_id, (position, seen, _) in self.history.tracks.items():
            if enemy_id in living and snapshot.tick-seen <= 8:
                self.current_contacts[axis_for(position, self.scenario.grid.shape[1])] += 1
        sightings = dict(A=0, Mid=0, B=0)
        for seen in snapshot.sightings:
            sightings[axis_for(seen.position, self.scenario.grid.shape[1])] += 1
        for side in ("A", "B"):
            if sightings[side] >= 3 and snapshot.tick <= EARLY_TICKS:
                if self.pressure_ticks[side] == snapshot.tick-1 and self.rush_ticks[side] is None:
                    self.rush_ticks[side] = snapshot.tick
                self.pressure_ticks[side] = snapshot.tick
            else:
                self.pressure_ticks[side] = None
        if self.model is not None:
            self.probabilities = self.model.probabilities(features)
        else:
            # Usable before training; require multiple recent actual contacts.
            counts = np.zeros(2, np.float32)
            for position, seen, _ in self.history.tracks.values():
                if snapshot.tick - seen <= 12:
                    index = int(self.scenario.site_dist["R"][position] < self.scenario.site_dist["L"][position])
                    counts[index] += 1
            self.probabilities = (counts + .5) / (sum(counts) + 1) if sum(counts) >= 2 else np.array([.5, .5], np.float32)
        self.gate.update(self.probabilities, snapshot.tick)
        self.frames.append(dict(tick=snapshot.tick, features=features, probabilities=self.probabilities.copy(),
                                decision=self.gate.side, sightings=len(snapshot.sightings)))

    def snapshot(self):
        return dict(source="learned" if self.model is not None else "observed_contacts",
                    probabilities=dict(zip(SIDES, map(float, self.probabilities))), site=self.gate.side,
                    history_rounds=len(self.previous_rounds), observed_contacts=dict(self.history.max_contacts),
                    current_contacts=dict(self.current_contacts), tendency=summarize_tendency(self.previous_rounds))
