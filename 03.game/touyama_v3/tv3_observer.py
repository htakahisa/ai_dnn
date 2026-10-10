"""Prediction inputs are built exclusively from copied public team snapshots."""
import heapq
import math
import numpy as np

from frc_v1.perception import FrcPerceptionBuilder
from frc_v1 import ABILITIES, ULTIMATES
from game_core import FACING_VECTORS
from touyama_v3.tv3_model import DecisionGate


class FeatureHistory:
    def __init__(self, scenario):
        self.scenario = scenario
        self.reset()
        self.fields = [f"round_{i}" for i in range(1, 13)] + ["tick", "round_time", "allies_alive", "enemies_alive"]
        self.fields += [f"past_{i}_{f}" for i in range(1, 13) for f in ("known", "L", "R", "A_win", "D_win")]
        self.fields += [f"branch_{n}_{f}" for n in scenario.names
                        for f in ("current_count", "visible_fraction", "empty_fraction", "ever_seen", "last_seen_age")]
        self.fields += [f"enemy_{i}_{f}" for i in range(5) for f in
                        ("alive", "known", "current", "row", "column", "dr", "dc", "age") + scenario.names]
        self.fields += [f"ally_{i}_{f}" for i in range(5) for f in ("alive", "hp", "row", "column", "blind")]
        self.fields += ["spike_on_ground", "spike_row", "spike_column", "spike_ever_dropped",
                        "spike_last_row", "spike_last_column", "spike_last_seen_age",
                        "spike_disappeared", "spike_disappeared_age", "spike_left_distance", "spike_right_distance"]
        self.fields += [f"spike_last_branch_{name}" for name in scenario.names]
        self.ally_capability_fields = ("max_hp", "accuracy", "hs_rate", "dodge", "reaction", "iq",
                                       "charges", "ultimate_points", "ultimate_cost")
        self.fields += [f"ally_{i}_{f}" for i in range(5) for f in
                        self.ally_capability_fields + tuple("ability_" + n for n in ABILITIES)
                        + tuple("ultimate_" + n for n in ULTIMATES)]

    def reset(self):
        self.tracks, self.branch_ticks = {}, {}
        self.spike_current = self.spike_last = self.spike_removed_tick = None

    def encode(self, snapshot, rounds):
        h, w = self.scenario.grid.shape
        tick = snapshot.tick
        current = {s.enemy_id: s.position for s in snapshot.sightings}
        for enemy_id, pos in current.items():
            old = self.tracks.get(enemy_id)
            velocity = (0., 0.)
            if old and tick > old[1]:
                dt = tick - old[1]
                velocity = ((pos[0] - old[0][0]) / dt, (pos[1] - old[0][1]) / dt)
            elif old:
                velocity = old[2]
            self.tracks[enemy_id] = pos, tick, velocity
            self.branch_ticks[self.scenario.names[int(self.scenario.region[pos])]] = tick
        values = [float(snapshot.round_number == i) for i in range(1, 13)]
        values += [min(1., tick / 100), snapshot.round_timer / 100,
                   sum(a.alive for a in snapshot.allies) / 5, sum(e.alive for e in snapshot.enemies) / 5]
        # Absolute previous round slots retain alternating-round patterns.
        for i in range(12):
            r = rounds[i] if i < len(rounds) else None
            values += [float(r is not None), float(r is not None and r["site"] == "L"),
                       float(r is not None and r["site"] == "R"),
                       float(r is not None and r["winner"] == "A"), float(r is not None and r["winner"] == "D")]
        visible, occupied = set(snapshot.visible_cells), set(current.values())
        counts = np.bincount([int(self.scenario.region[p]) for p in current.values()], minlength=len(self.scenario.names))
        for i, name in enumerate(self.scenario.names):
            cells = self.scenario.branches[name]
            values += [counts[i] / 5, sum(p in visible for p in cells) / len(cells),
                       sum(p in visible and p not in occupied for p in cells) / len(cells),
                       float(name in self.branch_ticks), min(1., (tick - self.branch_ticks[name]) / 100)
                       if name in self.branch_ticks else 1.]
        enemies = {e.enemy_id: e for e in snapshot.enemies}
        for i in range(5):
            track = self.tracks.get(i)
            pos, seen_tick, velocity = track if track else ((0, 0), tick - 100, (0., 0.))
            values += [float(enemies[i].alive), float(track is not None), float(i in current),
                       pos[0] / (h - 1), pos[1] / (w - 1), *np.clip(velocity, -1, 1), min(1., (tick - seen_tick) / 100)]
            region = int(self.scenario.region[pos]) if track else -1
            values += [float(region == j) for j in range(len(self.scenario.names))]
        for a in snapshot.allies:
            values += [float(a.alive), a.hp / 100, a.position[0] / (h - 1), a.position[1] / (w - 1), float(a.blind > 0)]
        dropped = snapshot.spike_dropped
        if dropped is not None:
            self.spike_last = dropped, tick
            self.spike_removed_tick = None
        elif self.spike_current is not None:
            # Disappearance is observable; it does not identify the new holder.
            self.spike_removed_tick = tick
        self.spike_current = dropped
        last_pos, last_tick = self.spike_last if self.spike_last else ((0, 0), tick - 100)
        pos = dropped if dropped is not None else (0, 0)
        distances = [0., 0.]
        region = -1
        if self.spike_last:
            region = int(self.scenario.region[last_pos])
            distances = [min(1., max(0, self.scenario.site_dist[side][last_pos]) / 100) for side in ("L", "R")]
        values += [float(dropped is not None), pos[0] / (h - 1), pos[1] / (w - 1), float(self.spike_last is not None),
                   last_pos[0] / (h - 1), last_pos[1] / (w - 1), min(1., (tick - last_tick) / 100),
                   float(self.spike_removed_tick is not None), min(1., (tick - self.spike_removed_tick) / 100)
                   if self.spike_removed_tick is not None else 1., *distances]
        values += [float(region == j) for j in range(len(self.scenario.names))]
        # Own capabilities are public; player names never identify learned slots.
        for ally in snapshot.allies:
            values += [ally.max_hp / 100, ally.accuracy, ally.hs_rate, ally.dodge,
                       ally.reaction / 200, ally.effective_iq / 200, ally.charges / 10,
                       ally.points / 10, ally.cost / 10]
            values += [float(ally.ability_name == n) for n in ABILITIES]
            values += [float(ally.ultimate_name == n) for n in ULTIMATES]
        features = np.asarray(values, dtype=np.float32)
        if len(features) != len(self.fields) or not np.isfinite(features).all():
            raise ValueError("Invalid public observation feature vector")
        return features


def facing(origin, target):
    dr, dc = target[0] - origin[0], target[1] - origin[1]
    return max(FACING_VECTORS, key=lambda f: FACING_VECTORS[f][0] * dc + FACING_VECTORS[f][1] * dr)


class ObserverController:
    handles_team_perception = True

    def __init__(self, scenario, model, *, threshold=.8, confirm=3, rotate=False, peek_ticks=4, hide_ticks=4):
        self.scenario, self.model = scenario, model
        self.threshold, self.confirm, self.rotate = threshold, confirm, rotate
        self.peek_ticks, self.hide_ticks = peek_ticks, hide_ticks
        self.game = None
        self.sensor = FrcPerceptionBuilder("D")
        self.history = FeatureHistory(scenario)
        self.previous_rounds = []
        self.reset_round()

    def set_game(self, game):
        # Engine setup uses a public wrapper; the sensor remains the sole
        # reader of enemy state. Planning below receives only its copied DTO.
        self.game = getattr(game, "_real", game)

    def reset_round(self):
        self.sensor.reset()
        self.history.reset()
        self.gate = DecisionGate(self.threshold, self.confirm)
        self.frames, self.events, self.actions = [], [], {}
        self.cache = None
        self.cooldown = {}
        self.was_retreating = set()
        self.active_retreats = set()
        self.peek_states = {}
        self.initial_arrived = set()

    def threats(self, snapshot):
        alive = {e.enemy_id for e in snapshot.enemies if e.alive}
        return [pos for i, (pos, tick, _) in self.history.tracks.items()
                if i in alive and snapshot.tick - tick <= 8]

    def exposed(self, pos, threats):
        return sum(self.scenario.clear(pos, enemy) for enemy in threats)

    def route_step(self, origin, goal, blocked, setup, threats):
        if origin == goal:
            return origin
        # Public known hazards only. Unknown enemy occupancy is resolved by
        # the engine, never used as an omniscient pathfinding mask.
        q = [(0., origin, origin)]
        costs = {origin: 0.}
        while q:
            cost, p, first = heapq.heappop(q)
            if cost != costs[p]:
                continue
            if p == goal:
                return first
            for n in self.scenario.neighbors(p):
                if n in blocked or setup and self.scenario.setup[n] != 0:
                    continue
                new_cost = cost + 1 + 10 * self.exposed(n, threats)
                if new_cost < costs.get(n, math.inf):
                    costs[n] = new_cost
                    heapq.heappush(q, (new_cost, n, n if p == origin else first))
        if setup and costs:
            from grid_paths import distance_map
            field = distance_map(self.scenario.grid, goal)
            reachable = [p for p in costs if field[p] >= 0]
            if reachable:
                closest = min(reachable, key=lambda p: (field[p], costs[p], p))
                # Recover the first step to the best permitted setup staging cell.
                if closest != origin:
                    return self.route_step(origin, closest, blocked, setup, threats)
        return origin

    def prepare_team_tick(self):
        snapshot = self.sensor.build(self.game)
        if snapshot.key == self.cache:
            return
        self.cache = snapshot.key
        setup = snapshot.phase == "setup"
        if not setup and not snapshot.is_planted:
            old_spike = self.history.spike_current
            features = self.history.encode(snapshot, self.previous_rounds)
            if snapshot.spike_dropped != old_spike:
                self.events.append({"tick": snapshot.tick, "type": "spike_dropped" if snapshot.spike_dropped is not None
                                    else "spike_disappeared", "position": snapshot.spike_dropped or old_spike})
            probabilities = self.model.probabilities(features)
            changed = self.gate.update(probabilities, snapshot.tick)
            self.frames.append({"tick": snapshot.tick, "features": features, "p_left": float(probabilities[0]),
                                "p_right": float(probabilities[1]), "decision": self.gate.side,
                                "decision_tick": self.gate.tick, "changes": self.gate.changes,
                                "sightings": [{"id": s.enemy_id, "position": list(s.position), "source": s.source}
                                              for s in snapshot.sightings],
                                "allies": [{"name": a.name, "position": a.position, "facing": a.facing,
                                            "hp": a.hp, "alive": a.alive} for a in snapshot.allies],
                                "allies_alive": sum(a.alive for a in snapshot.allies)})
            self.frames[-1]["spike_dropped"] = snapshot.spike_dropped
            self.frames[-1]["spike_last_known"] = self.history.spike_last[0] if self.history.spike_last else None
            if changed:
                self.events.append({"tick": snapshot.tick, "type": "decision", "side": self.gate.side,
                                    "confidence": float(max(probabilities))})
        threats = [] if setup else self.threats(snapshot)
        occupied = {a.position for a in snapshot.allies if a.alive}
        reserved = set()
        self.actions = {}
        for ally in snapshot.allies:
            post = self.scenario.post_for(ally)
            if not ally.alive:
                continue
            origin = ally.position
            if origin == post.watch:
                self.initial_arrived.add(ally.slot)
            blocked = (occupied - {origin}) | reserved
            danger = bool(threats) and (self.exposed(origin, threats) > 0 or
                     any(max(abs(origin[0] - p[0]), abs(origin[1] - p[1])) <= 3 for p in threats))
            if danger:
                self.cooldown[ally.slot] = snapshot.tick + 5
                if ally.slot not in self.active_retreats:
                    self.events.append({"tick": snapshot.tick, "type": "retreat", "role": post.role, "position": origin})
                self.was_retreating.add(ally.slot)
                self.active_retreats.add(ally.slot)
                covers = [(p, d) for p, d in self.scenario.local(origin, 5) if p not in blocked]
                goal, _ = min(covers, key=lambda item: (self.exposed(item[0], threats),
                            sum(max(0, 4 - max(abs(item[0][0] - e[0]), abs(item[0][1] - e[1]))) for e in threats),
                            item[1], self.scenario.spawn_dist[item[0]], item[0]))
            elif not setup and snapshot.tick < self.cooldown.get(ally.slot, -1):
                goal = post.retreat
            elif self.rotate and not setup and not snapshot.is_planted and self.gate.side is not None and (
                    post.role.endswith("anchor") or post.role == "mid_scout"):
                # Both outside scouts retain their lanes. Only anchors and
                # the central scout rotate; the observation-only default does not.
                targets = self.scenario.sites[self.gate.side]
                goal = min(targets, key=lambda p: abs(origin[0] - p[0]) + abs(origin[1] - p[1]))
            elif snapshot.is_planted:
                goal = post.retreat  # This trainer does not learn a retake policy.
            elif not setup and ally.slot in self.initial_arrived:
                # Stagger peeks: all lanes must not disappear simultaneously.
                cycle = self.peek_ticks + self.hide_ticks
                peeking = (snapshot.tick + ally.slot * 2) % cycle < self.peek_ticks
                goal = (post.alternate if ally.slot in self.was_retreating else post.watch) if peeking else post.retreat
                state = "peek" if peeking else "hide"
                if self.peek_states.get(ally.slot) != state:
                    self.events.append({"tick": snapshot.tick, "type": state, "role": post.role})
                    self.peek_states[ally.slot] = state
            else:
                goal = post.alternate if ally.slot in self.was_retreating and not setup else post.watch
            destination = self.route_step(origin, goal, blocked, setup, threats)
            if not danger:
                self.active_retreats.discard(ally.slot)
            payload = {"facing": ally.facing if ally.forced_facing else facing(destination, post.look)}
            if not setup and not snapshot.is_planted:
                from touyama_v3.tv3_defender_policy import PolicyEncoder
                from frc_v1.actions import build_masks, to_game_action, MOVE_STEPS
                from touyama_v3.tv3_attacker_combat import AttackerCombatCoach
                coach = AttackerCombatCoach(self.scenario)
                advice = coach.advise(snapshot, ally, destination, post.look, self.history.tracks)
                if coach.contacts(snapshot, ally, origin):
                    destination = advice.position
                    payload['facing'] = advice.facing
                inputs = PolicyEncoder(self.scenario).encode(snapshot, ally, goal, probabilities, self.history.tracks,
                                                            build_masks(snapshot))
                teacher = inputs.actions[inputs.teacher]
                if teacher.kind in ("ABILITY", "ULTIMATE"):
                    destination, payload = to_game_action(snapshot, ally.slot, teacher,
                                                         {a.slot: a.name for a in snapshot.allies})
                    destination = tuple(destination)
            if destination in blocked:
                destination = origin
            reserved.add(destination)
            self.actions[ally.name] = (list(destination), payload)

    def decide_move(self, char, game_state):
        # Freeze all five actions before any character moves this tick.
        setup = bool(getattr(self.game.defender_setup_phase, "active", False))
        key = (self.game.current_round, "setup" if setup else "live",
               self.game.defender_setup_phase.ticks_remaining if setup else self.game.battle_tick)
        if self.cache != key:
            self.prepare_team_tick()
        name = str(getattr(char, "base_name", char.name))
        return self.actions.get(name, (list(char.pos), {"facing": char.facing}))
