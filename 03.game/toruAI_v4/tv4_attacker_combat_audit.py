"""Post-action diagnostics. Private engine facts are output labels, never inputs."""
from collections import Counter, deque
import math
import copy

from game_core import FACING_VECTORS, SHOOTING_SITE_DIGREE

DEATH_CONTEXT_TICKS = 4


def name(char):
    return str(getattr(char, "base_name", char.name))


def angle(origin, target, direction):
    dr, dc = target[0]-origin[0], target[1]-origin[1]
    dx, dy = FACING_VECTORS[direction]
    return math.degrees(math.acos(max(-1., min(1., (dc*dx+dr*dy) / (math.hypot(dr, dc) or 1.)))))


class CombatAudit:
    def __init__(self, side="A"):
        if side not in ("A", "D"):
            raise ValueError("Combat audit side must be A or D")
        self.side, self.enemy_side = side, "D" if side == "A" else "A"
        self.counts = Counter()
        self.context = deque(maxlen=DEATH_CONTEXT_TICKS)
        self.deaths, self.utilities, self.pending = [], [], []
        self.before_state = None
        self.previous_enemy_status = {}
        self.used_effects = set()
        self.effect_references = []

    def before(self, snapshot, actions, tracks=None):
        # Only the copied public DTO is read before actions execute.
        self.before_state = {"tick": snapshot.tick,
            "seen": {s.enemy_id: s.position for s in snapshot.sightings},
            "last_known": {i: {"position": pos, "age": snapshot.tick-tick} for i, (pos, tick, _) in (tracks or {}).items()},
            "enemies": {e.name: e.enemy_id for e in snapshot.enemies},
            "allies": {a.name: {"position": a.position, "hp": a.hp, "alive": a.alive,
                "facing": a.facing, "blind": a.blind, "charges": a.charges,
                "action": copy.deepcopy(actions.get(a.name)), "ability": a.ability_name} for a in snapshot.allies}}

    def after(self, game):
        if self.before_state is None:
            return
        state = self.before_state
        own = {name(c): c for c in game.chars if c.team == self.side}
        enemy = {name(c): c for c in game.chars if c.team == self.enemy_side}
        frame = {"tick": state["tick"], "seen": state["seen"], "last_known": state["last_known"], "actors": [], "shots": []}
        for n, before in state["allies"].items():
            c = own.get(n)
            if c is None or not before["alive"]:
                continue
            frame["actors"].append({"name": n, **before, "end_position": tuple(c.pos),
                "end_facing": c.facing, "moved": bool(c.moved_this_tick), "end_hp": c.hp,
                "end_alive": c.is_alive})
            action = before["action"]
            payload = action[1] if action and isinstance(action[1], dict) else {}
            if payload.get("ability") in ("FLASH", "RECON", "SMOKE", "ASH"):
                accepted = getattr(c, before["ability"].lower()+"_charges", 0) < before["charges"]
                record = {"tick": state["tick"], "name": n, "ability": payload["ability"],
                    "from": before["position"], "target": payload.get("target"), "accepted": accepted,
                    "resolved": False, "affected_enemies": [], "blocked_lines": 0}
                self.utilities.append(record)
                self.counts["casts"] += int(accepted)
                if accepted:
                    self.pending.append(record)
        for shot in getattr(game, "last_shots", ()):
            s, t = shot["shooter"], shot["target"]
            if s.team == self.side:
                self.counts["ally_shots"] += 1
                self.counts["ally_hits"] += int(shot["hit"])
                self.counts["moving_shots"] += int(s.moved_this_tick)
                self.counts["moving_hits"] += int(s.moved_this_tick and shot["hit"])
                self.counts["stationary_shots"] += int(not s.moved_this_tick)
                self.counts["stationary_hits"] += int(not s.moved_this_tick and shot["hit"])
            elif own.get(name(t)) is t:
                # Shots at allied drones are not damage to a team member.
                self.counts["incoming_shots"] += 1
                seen = state["enemies"].get(name(s)) in state["seen"]
                outside = angle(tuple(t.pos), tuple(s.pos), t.facing) > SHOOTING_SITE_DIGREE
                self.counts["unseen_incoming_shots"] += int(not seen)
                self.counts["outside_facing_incoming_shots"] += int(outside)
            frame["shots"].append({"shooter": name(s), "target": name(t), "side": s.team,
                "shooter_position": tuple(s.pos), "target_position": tuple(t.pos),
                # MonitorDrone is shootable but has no facing direction.
                "shooter_facing": getattr(s, "facing", None), "target_facing": getattr(t, "facing", None),
                "shooter_kind": "drone" if getattr(s, "is_ultimate_drone", False) else "character",
                "target_kind": "drone" if getattr(t, "is_ultimate_drone", False) else "character",
                "shooter_moving": bool(s.moved_this_tick), "hit": bool(shot["hit"]),
                "damage": shot["damage"], "hit_chance": shot["hit_chance"]})
        self.context.append(frame)
        for n, before in state["allies"].items():
            c = own.get(n)
            if c is None or not before["alive"] or c.is_alive:
                continue
            incoming = [s for s in frame["shots"] if s["side"] == self.enemy_side and s["target"] == n and s["damage"] > 0]
            flags = []
            if incoming:
                final = incoming[-1]
                source_id = state["enemies"].get(final["shooter"])
                if source_id not in state["seen"]:
                    flags.append("source_unseen_before_action")
                    if source_id not in state["last_known"]:
                        flags.append("source_never_seen_before_action")
                if angle(final["target_position"], final["shooter_position"], final["target_facing"]) > SHOOTING_SITE_DIGREE:
                    flags.append("source_outside_facing")
                supporters = [a for a in own.values() if a.is_alive and name(a) != n
                    and game.check_cell_line_of_sight(tuple(a.pos), final["shooter_position"], block_smoke=True)
                    and angle(tuple(a.pos), final["shooter_position"], a.facing) <= SHOOTING_SITE_DIGREE]
                if not supporters:
                    flags.append("no_supporting_ally_line")
            else:
                flags.append("no_recorded_gunshot_source")
            if c.moved_this_tick:
                flags.append("moving_on_death_tick")
            action = before["action"]
            if action and action[1] == "PLANT":
                flags.append("plant_command_on_death_tick")
            if action and action[1] == "DEFUSE":
                flags.append("defuse_command_on_death_tick")
            self.counts["deaths"] += 1
            self.counts.update(flags)
            self.deaths.append({"tick": state["tick"], "name": n, "flags": flags, "context": list(self.context)})
        for record in list(self.pending):
            kind, caster = record["ability"], record["name"]
            raw = next((e for e in getattr(game, {"FLASH": "flash_bursts", "RECON": "recon_bursts", "SMOKE": "smokes", "ASH": "destruction_areas"}[kind], ())
                        if id(e) not in self.used_effects
                        and str(e.get("owner")) in (caster, next((str(c.name) for n, c in own.items() if n == caster), caster))), None)
            if raw is None:
                continue
            record["resolved"] = True
            self.used_effects.add(id(raw))
            self.effect_references.append(raw)
            record["impact"] = raw.get("pos", raw.get("center"))
            if kind == "RECON" and record["target"] is not None:
                record["impact"] = game._projectile_path(record["from"], record["target"])[-1]
            if kind in ("FLASH", "RECON"):
                attribute = "blind_remaining" if kind == "FLASH" else "reveal_remaining"
                record["affected_enemies"] = [state["enemies"].get(n) for n, c in enemy.items()
                    if getattr(c, attribute, 0) > max(0, self.previous_enemy_status.get((n, attribute), 0)-1)]
            elif kind == "SMOKE":
                cells = set(raw.get("cells", ()))
                record["blocked_lines"] = sum(
                    game.check_cell_line_of_sight(tuple(a.pos), tuple(e.pos), block_smoke=False)
                    and any(p in cells for p in game._line_cells(tuple(a.pos), tuple(e.pos)))
                    for a in own.values() if a.is_alive for e in enemy.values() if e.is_alive)
            else:
                record["affected_enemies"] = [state["enemies"].get(n) for n, c in enemy.items() if tuple(c.pos) in set(raw.get("cells", ()))]
            self.counts["resolved_casts"] += 1
            self.counts["zero_observed_effect_casts"] += int(not record["affected_enemies"] and not record["blocked_lines"])
            self.pending.remove(record)
        self.previous_enemy_status = {(n, attribute): getattr(c, attribute, 0) for n, c in enemy.items()
                                     for attribute in ("blind_remaining", "reveal_remaining")}
        self.before_state = None

    def report(self):
        return {"counts": dict(self.counts), "deaths": self.deaths, "utilities": self.utilities,
                "note": "Post-action audit labels; flags overlap and are not proof of a sole cause."}
