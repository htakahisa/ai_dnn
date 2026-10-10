"""Public-only combat examples for collection and the learned plant policy."""
from dataclasses import dataclass
from game_core import FACING_VECTORS

from touyama_v3.tv3_observer import facing

LOW_HP_FRACTION = .35
# A lost sighting is still a threat: the engine can shoot beyond public vision.
# Confirmed empty cells invalidate it immediately; bound stale threats in time.
RECENT_CONTACT_TICKS = 20
MAX_ASSEMBLY_WAIT = 6
COMBAT_FEATURES = 10
CROSSFIRE_REPOSITION_STEPS = 3
CROSSFIRE_REPOSITION_MIN_GAIN = .10
VIABLE_PLANT_SURVIVORS = 3
VIABLE_PLANT_HP_FRACTION = .30
PLANT_MIN_COVERING_ALLIES = 2


def combat_schema():
    return {"version": 3, "crossfire_reposition_steps": CROSSFIRE_REPOSITION_STEPS, "crossfire_min_gain": CROSSFIRE_REPOSITION_MIN_GAIN, "low_hp": LOW_HP_FRACTION, "recent_ticks": RECENT_CONTACT_TICKS,
            "assembly_wait": MAX_ASSEMBLY_WAIT,
            "features": COMBAT_FEATURES, "viable_plant_survivors": VIABLE_PLANT_SURVIVORS,
            "viable_plant_hp_fraction": VIABLE_PLANT_HP_FRACTION,
            "plant_covering_allies": PLANT_MIN_COVERING_ALLIES,
            "combat": "local_target_stop_shoot_cover_support_v1"}


def plant_ready(ally, advice):
    if ally.blind:
        return False
    if not advice.contacts and advice.reason == "advance":
        return True
    return advice.contacts > 0 and ally.hp > ally.max_hp*LOW_HP_FRACTION and (
        advice.supporters >= max(PLANT_MIN_COVERING_ALLIES, advice.contacts))


@dataclass
class CombatAdvice:
    position: tuple
    facing: str
    reason: str
    target: tuple | None
    contacts: int
    future_contacts: int
    supporters: int
    wait_ticks: int

    def features(self, ally, shape):
        h, w = shape
        target = self.target or ally.position
        return [self.contacts/5, self.future_contacts/5, self.supporters/4,
                float(self.reason == "stop_shoot"), float(self.reason == "cover_retreat"),
                float(self.reason == "assemble"), self.wait_ticks/MAX_ASSEMBLY_WAIT,
                target[0]/(h-1), target[1]/(w-1), float(ally.blind > 0)]


class AttackerCombatCoach:
    def __init__(self, scenario):
        self.scenario = scenario
        self.waiting = {}

    def clear(self, snapshot, origin, target, slot):
        if not self.scenario.clear(origin, target):
            return False
        line = self.scenario._line_cells(origin, target)
        if len(line) > 2 and any(p in snapshot.smoke_cells for p in line):
            return False
        # A teammate in the shot line is not a shootable contact.
        occupied = {a.position for a in snapshot.allies if a.alive and a.slot != slot}
        return not any(p in occupied for p in line[1:-1])

    def contacts(self, snapshot, ally, position):
        return [s.position for s in snapshot.sightings if self.clear(snapshot, position, s.position, ally.slot)]

    def preaim_hint(self, snapshot, ally, destination, goal, analysis):
        if not analysis.get("trained_rounds"):
            return goal
        seen = {s.position for s in snapshot.sightings}
        choices = []
        for region in self.scenario.names:
            mass = sum(row.get(region, 0.) for row in analysis["placement"].values())
            if mass < .8:
                continue
            for p in self.scenario.branches[region]:
                distance = max(abs(p[0]-destination[0]), abs(p[1]-destination[1]))
                if distance <= 10 and (p not in snapshot.visible_cells or p in seen) and self.scenario.clear(destination, p):
                    choices.append((mass/(1+distance), p))
        return max(choices, key=lambda r: (r[0], r[1]))[1] if choices else goal

    def advise(self, snapshot, ally, destination, goal, tracks=None):
        origin = ally.position
        contacts = self.contacts(snapshot, ally, origin)
        future = self.contacts(snapshot, ally, destination)
        known = {s.position for s in snapshot.sightings}
        alive = {e.enemy_id for e in snapshot.enemies if e.alive}
        known.update(pos for enemy_id, (pos, tick, _) in (tracks or {}).items()
                     if enemy_id in alive and snapshot.tick-tick <= RECENT_CONTACT_TICKS)
        visible_empty = set(snapshot.visible_cells) - {s.position for s in snapshot.sightings}
        remembered = known - visible_empty
        contacts = list(dict.fromkeys(contacts + [p for p in remembered if self.clear(snapshot, origin, p, ally.slot)]))
        future = list(dict.fromkeys(future + [p for p in remembered if self.clear(snapshot, destination, p, ally.slot)]))
        target = min(contacts or future or list(known), key=lambda p: (max(abs(p[0]-origin[0]), abs(p[1]-origin[1])), p)) if contacts or future or known else None
        def facing_contact(a, p):
            dx, dy = FACING_VECTORS[a.facing]
            return (p[1]-a.position[1])*dx + (p[0]-a.position[0])*dy >= -1e-9
        supporters = sum(a.alive and a.slot != ally.slot and not a.blind and
                         any(facing_contact(a, p) and self.clear(snapshot, a.position, p, a.slot) for p in contacts or future)
                         for a in snapshot.allies)
        result = destination
        reason, waited = "advance", 0
        blocked = {a.position for a in snapshot.allies if a.alive and a.slot != ally.slot}
        exposed = lambda p: sum(self.scenario.clear(p, e) and not any(q in snapshot.smoke_cells
                               for q in self.scenario._line_cells(p, e)) for e in known)
        if known and (ally.blind or ally.hp <= ally.max_hp*LOW_HP_FRACTION or len(contacts) >= 2):
            # Search a short public route to cover; never move into more lines.
            options = [(origin, origin, 0)]
            frontier, visited = [(origin, origin)], {origin}
            for depth in range(1, 4):
                following = []
                for p, first in frontier:
                    for q in self.scenario.neighbors(p):
                        if q in blocked or q in visited:
                            continue
                        step = q if p == origin else first
                        visited.add(q)
                        if exposed(step) <= exposed(origin):
                            options.append((q, step, depth))
                            following.append((q, step))
                frontier = following
            best = min(options, key=lambda row: (exposed(row[0]), row[2], row[0]))
            if exposed(best[0]) < exposed(origin) or ally.blind:
                result, reason = best[1], "cover_retreat"
            elif contacts and not ally.blind:
                result, reason = origin, "stop_shoot"
        elif contacts and not ally.blind:
            result, reason = origin, "stop_shoot"
            # Demonstrate short, safer paths to a wider shared firing angle.
            from touyama_v3.tv3_learn_attacker_guard import crossfire_score
            teammates = [a.position for a in snapshot.allies if a.alive and a.slot != ally.slot
                         and any(self.clear(snapshot, a.position, t, a.slot) for t in contacts)]
            base = crossfire_score(self.scenario, origin, teammates, contacts)
            frontier, visited = [(origin, origin)], {origin}
            options = []
            for depth in range(1, CROSSFIRE_REPOSITION_STEPS + 1):
                following = []
                for p, first in frontier:
                    for q in self.scenario.neighbors(p):
                        if q in visited or q in blocked or exposed(q) > exposed(origin):
                            continue
                        visited.add(q)
                        step = q if p == origin else first
                        following.append((q, step))
                        if self.contacts(snapshot, ally, q):
                            score = crossfire_score(self.scenario, q, teammates, contacts)
                            if score > base + CROSSFIRE_REPOSITION_MIN_GAIN:
                                options.append((score, -depth, q, step))
                frontier = following
            if options:
                result, reason = max(options)[3], "crossfire_reposition"
        elif future and not supporters:
            previous = self.waiting.get(ally.slot)
            if previous is None or previous[0] != origin:
                previous = origin, snapshot.tick
                self.waiting[ally.slot] = previous
            waited = snapshot.tick-previous[1]
            if waited < MAX_ASSEMBLY_WAIT:
                # Allow a nearby teammate to occupy a separate shooting lane.
                result, reason = origin, "assemble"
        else:
            self.waiting.pop(ally.slot, None)
        direction = ally.facing if ally.forced_facing else facing(result, target or goal)
        return CombatAdvice(result, direction, reason, target, len(contacts), len(future), supporters, waited)


def effective_utility(scenario, snapshot, ally, cast):
    """Reject geometrically ineffective reactive throws using public sightings."""
    from touyama_v3.tv3_attacker_entry_utility import projectile_impact
    kind, target = cast.get("ability"), cast.get("target")
    if kind not in ("FLASH", "RECON", "SMOKE", "ASH") or target is None:
        return True
    if kind in ("FLASH", "RECON"):
        impact, delay = projectile_impact(scenario, ally.position, target, kind)
        if delay == 0:
            return False
        if not snapshot.sightings:
            return False  # Predicted casts are admitted separately after belief/impact checks.
        if kind == "FLASH":
            return any(scenario.clear(impact, s.position) and not any(p in snapshot.smoke_cells
                       for p in scenario._line_cells(impact, s.position)) for s in snapshot.sightings)
        return any(max(abs(impact[0]-s.position[0]), abs(impact[1]-s.position[1])) <= 4 for s in snapshot.sightings)
    if kind == "SMOKE":
        cells = {(target[0]+dr, target[1]+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        return any(any(p in cells for p in scenario._line_cells(a.position, s.position))
                   for a in snapshot.allies if a.alive for s in snapshot.sightings
                   if scenario.clear(a.position, s.position))
    return any(max(abs(target[0]-s.position[0]), abs(target[1]-s.position[1])) <= 1 for s in snapshot.sightings)
