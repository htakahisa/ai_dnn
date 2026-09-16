"""Per-map gunfight, assist, and cover tracking."""

from collections import defaultdict
import math

from game_core import FACING_VECTORS
from analytics.tactical_classifier import TacticalRoundTracker


class CombatTracker:
    DRAW_LOS_TICKS = 3
    WINDOW_TICKS = 10

    def __init__(self):
        self.stats = defaultdict(self._new_stats)
        self.side_stats = {
            "attacker": {"rounds_played": 0, "rounds_won": 0, "plants": 0, "postplant_wins": 0, "players": defaultdict(self._new_stats)},
            "defender": {"rounds_played": 0, "rounds_won": 0, "plants_against": 0, "retakes_won": 0, "players": defaultdict(self._new_stats)},
        }
        self.gunfights = []
        self.active = []
        self._next_id = 1
        self._contributions = defaultdict(list)
        self._pending_cover = []
        self.assist_events = []
        self.cover_events = []
        self._all_chars = []
        self.round_records = []
        self._round_had_kill = False
        self._round_had_death = False
        self.tactics = None
        self._round_start_stats = {}
        self._round_start_context = {}

    @staticmethod
    def _new_stats():
        return {
            "role": "",
            "kills": 0,
            "deaths": 0,
            "gunfights_participated": 0,
            "gunfights_won": 0,
            "gunfights_lost": 0,
            "gunfights_draw": 0,
            "one_v_one_participated": 0,
            "one_v_one_won": 0,
            "one_v_one_lost": 0,
            "one_v_one_draw": 0,
            "first_kills": 0,
            "first_deaths": 0,
            "preaim_angle_sum": 0.0,
            "preaim_angle_count": 0,
            "assists": 0,
            "covers": 0,
        }

    def register_players(self, chars, team_names=None):
        self._all_chars = list(chars)
        team_names = team_names or {}
        self._round_start_stats = {
            name: dict(self.stats[name]) for name in self.stats
        }
        self._round_start_context = {
            str(char.name): {
                "team": str(team_names.get(char.team, char.team)),
                "side": "attacker" if char.team == "A" else "defender",
            }
            for char in chars
        }

    def begin_round_tactics(self, width, plant_cells, chars):
        self.tactics = TacticalRoundTracker(width, plant_cells)
        self.tactics.begin_round(chars)

    def observe_tactics(self, chars, tick):
        if self.tactics is not None:
            self.tactics.observe(chars, tick)
        for char in chars:
            row = self.stats[str(char.name)]
            row["role"] = str(getattr(char, "role", ""))
            side = "attacker" if char.team == "A" else "defender"
            side_row = self.side_stats[side]["players"][str(char.name)]
            side_row["role"] = row["role"]

    def record_round_result(self, winning_team, reason, planted, tactic=None):
        if self.tactics is not None:
            generic_tactic = self.tactics.finish(
                getattr(self, "_planted_pos", None),
                getattr(self, "_target_plant_pos", None),
            )
            # The controller snapshot is useful for fields such as the
            # defender setup, but its attacker strategy is often just the
            # controller's current macro label.  An empty/unknown label is
            # exposed by the snapshot as ``default`` and used to overwrite
            # the controller-independent movement classification here.  Keep
            # the observed classification when it is more informative.
            supplied = dict(tactic or {})
            supplied_strategy = str(
                supplied.get("attacker_strategy", "") or ""
            ).strip().lower()
            observed_strategy = str(
                generic_tactic.get("attacker_strategy", "") or ""
            ).strip().lower()
            if (
                supplied_strategy == "default"
                and observed_strategy
                and observed_strategy != "default"
            ):
                supplied.pop("attacker_strategy", None)
            tactic = {**generic_tactic, **supplied}
        self.round_records.append({
            "round_number": len(self.round_records) + 1,
            "winner": "attacker" if winning_team == "A" else "defender",
            "reason": str(reason),
            "planted": bool(planted),
            "tactic": tactic or {},
            "players": self._round_player_deltas(),
        })
        self.side_stats["attacker"]["rounds_played"] += 1
        self.side_stats["defender"]["rounds_played"] += 1
        if winning_team == "A":
            self.side_stats["attacker"]["rounds_won"] += 1
        else:
            self.side_stats["defender"]["rounds_won"] += 1
        if planted:
            self.side_stats["attacker"]["plants"] += 1
            self.side_stats["defender"]["plants_against"] += 1
            if winning_team == "A":
                self.side_stats["attacker"]["postplant_wins"] += 1
            else:
                self.side_stats["defender"]["retakes_won"] += 1

    def _round_player_deltas(self):
        result = {}
        fields = (
            "kills", "deaths", "gunfights_participated", "gunfights_won",
            "gunfights_lost", "gunfights_draw", "one_v_one_participated",
            "one_v_one_won", "one_v_one_lost", "one_v_one_draw", "assists",
            "covers", "first_kills", "first_deaths", "preaim_angle_sum",
            "preaim_angle_count",
        )
        for name, context in self._round_start_context.items():
            before = self._round_start_stats.get(name, {})
            after = self.stats.get(name, {})
            row = {field: after.get(field, 0) - before.get(field, 0) for field in fields}
            row.update(context)
            row["role"] = after.get("role", "")
            result[name] = row
        return result

    def begin_round(self):
        """Clear transient combat links while retaining map totals."""
        self.active.clear()
        self._contributions.clear()
        self._pending_cover.clear()
        self._round_had_kill = False
        self._round_had_death = False
        self.tactics = None

    def record_contribution(self, assister, victim, tick, method):
        if assister is None or victim is None or assister is victim:
            return
        self._contributions[str(victim.name)].append(
            (str(assister.name), int(tick), str(method))
        )

    def record_death(self, victim, killer, tick):
        if victim is None or killer is None or victim.team == killer.team:
            return
        self._pending_cover.append(
            (str(killer.name), str(victim.name), int(tick))
        )
        side = "attacker" if victim.team == "A" else "defender"
        self.stats[str(victim.name)]["deaths"] += 1
        self.side_stats[side]["players"][str(victim.name)]["deaths"] += 1
        if not self._round_had_death:
            self._round_had_death = True
            self.stats[str(victim.name)]["first_deaths"] += 1
            self.side_stats[side]["players"][str(victim.name)]["first_deaths"] += 1

    def record_kill(self, killer, victim, tick):
        if killer is None or victim is None:
            return
        killer_name = str(killer.name)
        victim_name = str(victim.name)
        killer_side = "attacker" if killer.team == "A" else "defender"
        self.stats[killer_name]["kills"] += 1
        self.side_stats[killer_side]["players"][killer_name]["kills"] += 1
        if not self._round_had_kill:
            self._round_had_kill = True
            self.stats[killer_name]["first_kills"] += 1
            self.side_stats[killer_side]["players"][killer_name]["first_kills"] += 1
        eligible = {}
        for assister_name, source_tick, method in self._contributions.pop(
            victim_name, ()
        ):
            delta = int(tick) - int(source_tick)
            if 0 <= delta <= self.WINDOW_TICKS and assister_name != killer_name:
                # One assist per assister per victim, even when that player
                # dealt damage and also used an ability (or hit repeatedly).
                previous = eligible.get(assister_name)
                if previous is None or delta < previous[0]:
                    eligible[assister_name] = (delta, method)

        for assister_name, (delta, method) in eligible.items():
            self.stats[assister_name]["assists"] += 1
            assister = next(
                (c for c in self._all_chars if str(c.name) == assister_name),
                None,
            )
            if assister is not None:
                side = "attacker" if assister.team == "A" else "defender"
                self.side_stats[side]["players"][assister_name]["assists"] += 1
            self.assist_events.append({
                "assister": assister_name,
                "victim": victim_name,
                "killer": killer_name,
                "tick_difference": delta,
                "method": method,
            })

        remaining = []
        for enemy_name, ally_name, death_tick in self._pending_cover:
            delta = int(tick) - int(death_tick)
            if 0 <= delta <= self.WINDOW_TICKS and enemy_name == victim_name:
                self.stats[killer_name]["covers"] += 1
                self.side_stats[killer_side]["players"][killer_name]["covers"] += 1
                self.cover_events.append({
                    "coverer": killer_name,
                    "ally": ally_name,
                    "enemy_killed": victim_name,
                    "tick_difference": delta,
                })
            elif delta <= self.WINDOW_TICKS:
                remaining.append((enemy_name, ally_name, death_tick))
        self._pending_cover = remaining

    @staticmethod
    def _components(pairs):
        graph = defaultdict(set)
        for first, second in pairs:
            a, b = str(first.name), str(second.name)
            graph[a].add(b)
            graph[b].add(a)
        result = []
        unseen = set(graph)
        while unseen:
            root = unseen.pop()
            component = {root}
            stack = [root]
            while stack:
                node = stack.pop()
                for nxt in graph[node]:
                    if nxt in unseen:
                        unseen.remove(nxt)
                        component.add(nxt)
                        stack.append(nxt)
            result.append(component)
        return result

    def _finish(self, session, chars_by_name, reason):
        participants = session["participants"]
        chars = [chars_by_name[name] for name in participants if name in chars_by_name]
        alive_by_team = {
            team: any(c.is_alive for c in chars if c.team == team)
            for team in ("A", "D")
        }
        if alive_by_team["A"] and not alive_by_team["D"]:
            result = "attacker_win"
            winner = "A"
        elif alive_by_team["D"] and not alive_by_team["A"]:
            result = "defender_win"
            winner = "D"
        else:
            result = "draw"
            winner = None

        individual = {}
        is_one_v_one = sum(char.team == "A" for char in chars) == 1 and sum(
            char.team == "D" for char in chars
        ) == 1
        for char in chars:
            if winner is None:
                outcome = "draw"
            else:
                outcome = "win" if char.team == winner else "loss"
            individual[char.name] = outcome
            row = self.stats[str(char.name)]
            row["gunfights_participated"] += 1
            row[
                "gunfights_won" if outcome == "win"
                else "gunfights_lost" if outcome == "loss"
                else "gunfights_draw"
            ] += 1
            if is_one_v_one:
                row["one_v_one_participated"] += 1
                row[
                    "one_v_one_won" if outcome == "win"
                    else "one_v_one_lost" if outcome == "loss"
                    else "one_v_one_draw"
                ] += 1
            side = "attacker" if char.team == "A" else "defender"
            side_row = self.side_stats[side]["players"][str(char.name)]
            side_row["gunfights_participated"] += 1
            side_row[
                "gunfights_won" if outcome == "win"
                else "gunfights_lost" if outcome == "loss"
                else "gunfights_draw"
            ] += 1
            if is_one_v_one:
                side_row["one_v_one_participated"] += 1
                side_row[
                    "one_v_one_won" if outcome == "win"
                    else "one_v_one_lost" if outcome == "loss"
                    else "one_v_one_draw"
                ] += 1

        self.gunfights.append({
            "gunfight_id": session["id"],
            "timestamp": session["start_tick"],
            "result": result,
            "attacker_players": [
                name for name in participants
                if name in chars_by_name and chars_by_name[name].team == "A"
            ],
            "defender_players": [
                name for name in participants
                if name in chars_by_name and chars_by_name[name].team == "D"
            ],
            "individual_results": individual,
            "end_reason": reason,
        })

    def tick(self, pairs, chars, tick):
        self._all_chars = list(chars)
        chars_by_name = {str(c.name): c for c in chars}
        components = self._components(pairs)
        matched = set()

        for component in components:
            component_chars = [
                chars_by_name[name] for name in component if name in chars_by_name
            ]
            for char in component_chars:
                angles = []
                fx, fy = FACING_VECTORS.get(char.facing, (0.0, 1.0))
                for enemy in component_chars:
                    if enemy.team == char.team:
                        continue
                    dc = float(enemy.pos[1] - char.pos[1])
                    dr = float(enemy.pos[0] - char.pos[0])
                    distance = math.hypot(dc, dr)
                    if distance <= 0:
                        continue
                    dot = max(-1.0, min(1.0, (fx * dc + fy * dr) / distance))
                    angles.append(math.degrees(math.acos(dot)))
                if angles:
                    angle = min(angles)
                    name = str(char.name)
                    self.stats[name]["preaim_angle_sum"] += angle
                    self.stats[name]["preaim_angle_count"] += 1
                    side = "attacker" if char.team == "A" else "defender"
                    side_row = self.side_stats[side]["players"][name]
                    side_row["preaim_angle_sum"] += angle
                    side_row["preaim_angle_count"] += 1
            overlaps = [
                i for i, session in enumerate(self.active)
                if session["participants"] & component
            ]
            if overlaps:
                base = overlaps[0]
                session = self.active[base]
                for index in reversed(overlaps[1:]):
                    session["participants"].update(
                        self.active[index]["participants"]
                    )
                    self.active.pop(index)
                    if index < base:
                        base -= 1
                session["participants"].update(component)
                session["no_los"] = 0
                session["last_los_tick"] = int(tick)
                matched.add(base)
            else:
                self.active.append({
                    "id": self._next_id,
                    "participants": set(component),
                    "start_tick": int(tick),
                    "last_los_tick": int(tick),
                    "no_los": 0,
                })
                self._next_id += 1
                matched.add(len(self.active) - 1)

        survivors = []
        for index, session in enumerate(self.active):
            if index not in matched:
                session["no_los"] += 1
            team_alive = {
                team: any(
                    chars_by_name[name].is_alive
                    for name in session["participants"]
                    if name in chars_by_name and chars_by_name[name].team == team
                )
                for team in ("A", "D")
            }
            if session["no_los"] >= self.DRAW_LOS_TICKS:
                self._finish(session, chars_by_name, "lost_los_3_ticks")
            elif not team_alive["A"] or not team_alive["D"]:
                self._finish(session, chars_by_name, "team_eliminated")
            else:
                survivors.append(session)
        self.active = survivors

    def end_round(self, chars):
        """Close any still-open session at the round boundary."""
        chars_by_name = {str(c.name): c for c in chars}
        for session in self.active:
            self._finish(session, chars_by_name, "round_end")
        self.active.clear()

    def export_stats(self):
        return {
            "players": {name: dict(row) for name, row in self.stats.items()},
            "side_stats": {
                side: {
                    "rounds_played": data["rounds_played"],
                    "rounds_won": data["rounds_won"],
                    "plants": data.get("plants", 0),
                    "plants_against": data.get("plants_against", 0),
                    "postplant_wins": data.get("postplant_wins", 0),
                    "retakes_won": data.get("retakes_won", 0),
                    "players": {
                        name: dict(row) for name, row in data["players"].items()
                    },
                }
                for side, data in self.side_stats.items()
            },
            "round_records": list(self.round_records),
            "gunfights": list(self.gunfights),
            "assist_events": list(self.assist_events),
            "cover_events": list(self.cover_events),
        }
