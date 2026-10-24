"""Entry support from learned regional beliefs and public map geometry only."""
import math
import numpy as np

from game_core import (FLASH_SPEED_CELLS_PER_TICK, FLASH_MAX_FLIGHT_TICKS, RECON_SPEED_CELLS_PER_TICK,
                       BLIND_DURATION_TICKS, REVEAL_DURATION_TICKS, SMOKE_DURATION_TICKS)
from frc_v1.actions import KINDS

# Normal entry-support settings; shared by collection and plant action candidates.
PLACEMENT_CONFIDENCE = .65
ENTRY_LOOKAHEAD = 8
THREAT_ROUTE_DISTANCE = 4
MIN_UTILITY_COVERAGE = {"FLASH": .4, "RECON": .2, "SMOKE": .1, "ASH": .1}
UTILITY_DISTANCE = 10
REGION_TARGET_LIMIT = 8
SUPPORT_REUSE_COOLDOWN = 4
SUPPORT_REUSE_DISTANCE = 4
SUPPORT_REFRESH_TICKS = {"FLASH": FLASH_MAX_FLIGHT_TICKS + BLIND_DURATION_TICKS,
                         "RECON": FLASH_MAX_FLIGHT_TICKS + REVEAL_DURATION_TICKS,
                         "SMOKE": SMOKE_DURATION_TICKS, "ASH": BLIND_DURATION_TICKS}


def utility_schema():
    return {"version": 4, "confidence": PLACEMENT_CONFIDENCE,
            "lookahead": ENTRY_LOOKAHEAD, "distance": UTILITY_DISTANCE,
            "region_targets": REGION_TARGET_LIMIT,
            "threat_route_distance": THREAT_ROUTE_DISTANCE,
            "utility_coverage": MIN_UTILITY_COVERAGE, "sampling": "whole_unknown_region_even_samples",
            "reuse_cooldown": SUPPORT_REUSE_COOLDOWN, "reuse_distance": SUPPORT_REUSE_DISTANCE,
            "refresh_ticks": SUPPORT_REFRESH_TICKS,
            "coordination": "progress_contact_or_expired_support_v3"}


class SupportUseTracker:
    """Permit repeat support after progress/contact/effect expiry, not a round cap."""
    def __init__(self):
        self.last = {}

    @staticmethod
    def progress(snapshot):
        holder = next((a for a in snapshot.allies if a.alive and a.has_spike), None)
        return holder.position if holder else snapshot.spike_dropped

    def available(self, kind, snapshot, target):
        if kind not in SUPPORT_REFRESH_TICKS:
            return True
        if any(e.kind == kind and e.phase in ("flight", "active") for e in snapshot.effects):
            return False
        previous = self.last.get(kind)
        if previous is None:
            return True
        elapsed = snapshot.tick - previous["tick"]
        if elapsed < SUPPORT_REUSE_COOLDOWN:
            return False
        distance = lambda p, q: max(abs(p[0]-q[0]), abs(p[1]-q[1]))
        progress = self.progress(snapshot)
        advanced = progress is not None and previous["progress"] is not None and distance(progress, previous["progress"]) >= SUPPORT_REUSE_DISTANCE
        new_contact = {s.enemy_id for s in snapshot.sightings} - previous["contacts"]
        return bool(advanced or new_contact or distance(target, previous["target"]) >= SUPPORT_REUSE_DISTANCE
                    or elapsed >= SUPPORT_REFRESH_TICKS[kind])

    def record(self, snapshot, cast):
        if cast["ability"] in SUPPORT_REFRESH_TICKS:
            self.last[cast["ability"]] = {"tick": snapshot.tick, "target": cast["target"],
                                        "progress": self.progress(snapshot),
                                        "contacts": {s.enemy_id for s in snapshot.sightings}}


def entry_support_pending(snapshot, ally):
    # Recon is information gathering, not a reason for the entire team to stop.
    return any(e.kind == "FLASH" and e.phase == "flight" and e.position is not None
               and max(abs(e.position[0] - ally.position[0]), abs(e.position[1] - ally.position[1])) <= ENTRY_LOOKAHEAD
               for e in snapshot.effects)


def projectile_impact(scenario, origin, target, kind):
    path = scenario._projectile_path(origin, target)
    steps = len(path) - 1
    if steps < 1:
        return origin, 0
    speed = FLASH_SPEED_CELLS_PER_TICK if kind == "FLASH" else RECON_SPEED_CELLS_PER_TICK
    if kind == "FLASH":
        steps = min(steps, speed * FLASH_MAX_FLIGHT_TICKS)
    return path[steps], math.ceil(steps / speed)


def pending_flash_impact(scenario, snapshot, ally):
    for effect in snapshot.effects:
        if effect.kind != "FLASH" or effect.phase != "flight" or effect.position is None:
            continue
        if max(abs(effect.position[0]-ally.position[0]), abs(effect.position[1]-ally.position[1])) > ENTRY_LOOKAHEAD:
            continue
        dr, dc = getattr(effect, "direction", (0., 0.))
        if dr == dc == 0:
            return effect.position
        target = (effect.position[0] + round(dr * UTILITY_DISTANCE),
                  effect.position[1] + round(dc * UTILITY_DISTANCE))
        return projectile_impact(scenario, effect.position, target, "FLASH")[0]
    return None


def flash_entry_step(scenario, origin, destination, impact, smoke_cells=(), blocked=()):
    """Wait behind cover at the flash boundary; leave an exposed waiting cell.

    Only static geometry and public smoke/allied reservations are consulted.
    This is a collection/teacher example, never an omniscient enemy check.
    """
    smoke = set(smoke_cells)
    def exposed(p):
        return scenario.clear(p, impact) and not any(q in smoke for q in scenario._line_cells(p, impact))
    if not exposed(origin):
        return origin if exposed(destination) else destination
    covers = [p for p in scenario.neighbors(origin) if p not in blocked and not exposed(p)]
    return min(covers, key=lambda p: (max(abs(p[0]-destination[0]), abs(p[1]-destination[1])), p)) if covers else destination


def predicted_entry_utility(scenario, snapshot, ally, analysis, route, masks):
    """Return a legal cast and flight delay, without consulting hidden state.

    A region is a belief, not an exact enemy cell. Visible empty cells are
    excluded. FLASH/RECON aim is checked against the actual wall-stopped flight
    geometry rather than treating its requested target as its impact point.
    """
    kind = ally.ability_name
    if kind not in ("FLASH", "SMOKE", "RECON", "ASH") or not analysis.get("trained_rounds"):
        return None
    if not masks.kind[ally.slot, KINDS.index("ABILITY")]:
        return None
    if any(e.kind == kind and e.phase in ("flight", "active") for e in snapshot.effects):
        return None
    distance = lambda p, q: max(abs(p[0] - q[0]), abs(p[1] - q[1]))
    holder = next((a for a in getattr(snapshot, "allies", ()) if a.alive and a.has_spike), None)
    origin = holder.position if holder else route.cells[0]
    cursor = route.cells.index(origin) if origin in route.cells else 0
    entry = route.cells[cursor:cursor + ENTRY_LOOKAHEAD + 1]
    if distance(ally.position, origin) > UTILITY_DISTANCE:
        return None
    if not entry:
        return None
    visible = set(snapshot.visible_cells)
    seen = {s.position for s in snapshot.sightings}
    seen_ids = {s.enemy_id for s in snapshot.sightings}
    smoke = set(snapshot.smoke_cells)
    occupied = {a.position for a in getattr(snapshot, "allies", ()) if a.alive}
    height, width = scenario.grid.shape
    threats = []
    region_counts = {}
    for index, region in enumerate(scenario.names):
        # A sighting's one-hot region is not evidence for another hidden enemy
        # somewhere else in that region. Visible contacts use the reactive path.
        confidence = max((row.get(region, 0.) for enemy_id, row in analysis["placement"].items()
                          if enemy_id not in seen_ids), default=0.)
        if confidence < PLACEMENT_CONFIDENCE:
            continue
        # Use the full region represented by the training label, not just its marker.
        unknown = [tuple(map(int, raw)) for raw in np.argwhere((scenario.region == index) & (scenario.grid != 1))
                   if tuple(raw) not in visible and tuple(raw) not in occupied and tuple(raw) not in smoke]
        indices = np.linspace(0, len(unknown)-1, min(REGION_TARGET_LIMIT, len(unknown)), dtype=int) if unknown else ()
        sampled = [unknown[i] for i in indices]
        region_counts[index] = len(sampled)
        for p in sampled:
            if (p in visible and p not in seen) or p in smoke:
                continue
            if min(distance(p, q) for q in entry) > THREAT_ROUTE_DISTANCE or p in occupied:
                continue
            if not any(scenario.clear(q, p) for q in entry):
                continue
            threats.append((confidence, p))
    if not threats:
        return None
    legal = masks.target[ally.slot, 0].reshape(height, width)
    best = None
    for raw in np.argwhere(legal):
        target = tuple(map(int, raw))
        if distance(ally.position, target) > UTILITY_DISTANCE or (kind in ("SMOKE", "ASH") and target in occupied):
            continue
        impact, delay = target, 0
        if kind in ("FLASH", "RECON"):
            impact, delay = projectile_impact(scenario, ally.position, target, kind)
            if delay < 1:
                continue
            if delay > FLASH_MAX_FLIGHT_TICKS:
                continue
        if kind == "FLASH":
            covered = [(confidence, p) for confidence, p in threats
                       if distance(impact, p) <= ENTRY_LOOKAHEAD and scenario.clear(impact, p)
                       and not any(c in smoke for c in scenario._line_cells(impact, p))]
        else:
            covered = [(confidence, p) for confidence, p in threats
                       if distance(impact, p) <= (4 if kind == "RECON" else 1)]
        if not covered:
            continue
        # An area prediction does not mean every sample cell contains an enemy.
        # Score the fraction of each region covered, not the raw cell count.
        coverage = sum(confidence / region_counts[int(scenario.region[p])] for confidence, p in covered)
        if coverage < MIN_UTILITY_COVERAGE[kind]:
            continue
        # Prefer strong predictions close to the advancing route. Avoid smoking
        # the carrier's path when an equally useful off-path target is available.
        path_penalty = sum(distance(target, p) <= 1 for p in entry) if kind == "SMOKE" else 0
        score = (coverage, -path_penalty, -delay,
                 -distance(ally.position, target))
        if best is None or score > best[0]:
            best = score, {"ability": kind, "target": target}, delay
    return None if best is None else (best[1], best[2])
