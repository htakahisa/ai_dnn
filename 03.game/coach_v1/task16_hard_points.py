"""Offline, replay-based candidate collection for the fixed map.

Replay contains referee truth. This module runs only after a match and never
feeds coordinates or labels back into either actor's observation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

from coach_v1.common.constants import WATCH_POINTS_CONFIG_PATH
from coach_v1.common.hashing import map_sha256
from coach_v1.common.watch_points import load_watch_points
from map_data import NEW_MAZE_STR


SCHEMA = "coach-v1-hard-points-v1"
MAP_HASH = map_sha256(NEW_MAZE_STR)
FACING = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
_DELTAS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def _distance(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _face(a, b):
    dr, dc = int(b[0]) - int(a[0]), int(b[1]) - int(a[1])
    if dr == dc == 0:
        return None
    return FACING[max(range(8), key=lambda i: (dr * _DELTAS[i][0] + dc * _DELTAS[i][1]) /
                  (1.41421356237 if i % 2 else 1))]


def _phase(frame, side):
    if frame.get("setup"):
        return "setup"
    if frame.get("planted"):
        return "post_plant" if side == "attacker" else "retake"
    return "pre_plant"


def _side(record, team_name, first_frame):
    sides = {value.get("side") for value in record.get("players", {}).values()
             if value.get("team") == team_name}
    if len(sides) == 1 and next(iter(sides)) in ("attacker", "defender"):
        return next(iter(sides))
    replay_side = first_frame.get("coach_side")
    if replay_side in ("attacker", "defender"):
        return replay_side
    raise ValueError("round record and replay do not identify one coach side")


def _chars(frame, team):
    return [c for c in frame["chars"] if c["team"] == team]


def collect_match(result, *, opponent_id: str, replay_id: str,
                  coach_team_name: str = "coach_v1") -> list[dict[str, Any]]:
    """Return evidence events; no event is treated as a training label."""
    if not result.summary.replay_ok:
        raise ValueError("replay integrity check failed")
    records = {int(r["round_number"]): r for r in result.round_records}
    frames = defaultdict(list)
    for index, frame in enumerate(result.replay):
        number = int(frame["round"])
        if number not in records or len(frame.get("chars", ())) != 10:
            raise ValueError("replay and round records disagree")
        frames[number].append((index, frame))
    if set(frames) != set(records):
        raise ValueError("replay is missing a round")
    events = []
    decisions = defaultdict(list)
    for raw in getattr(result, "coach_decisions", ()):
        decision = asdict(raw) if is_dataclass(raw) else raw
        side = decision["side"].value if hasattr(decision["side"], "value") else decision["side"]
        decisions[(int(decision["round_number"]), side)].append(decision)
    for sequence in decisions.values():
        sequence.sort(key=lambda value: int(value["tick"]))
    kills = defaultdict(list)
    for item in getattr(result, "kill_events", ()):
        kills[(int(item["round"]), str(item["victim"]), str(item["victim_team"]))].append(item)

    def emit(kind, position, facing, phase, side, round_number, tick, frame_index,
             loss, evidence, confidence="candidate"):
        if position is None:
            return
        events.append({
            "kind": kind, "position": [int(position[0]), int(position[1])],
            "facing": facing, "phase": phase, "side": side,
            "opponent": opponent_id, "loss": loss, "confidence": confidence,
            "evidence": evidence,
            "replay": {"id": replay_id, "frame": int(frame_index),
                       "round": int(round_number), "tick": int(tick)},
        })

    for number, sequence in sorted(frames.items()):
        record = records[number]
        side_frame = next((frame for _, frame in sequence if "coach_side" in frame),
                          sequence[0][1])
        side = _side(record, coach_team_name, side_frame)
        team = "A" if side == "attacker" else "D"
        losing = record.get("winner") != side
        unseen_streak = defaultdict(int)
        angle_seen = set()
        stale_seen = set()
        for offset, (index, frame) in enumerate(sequence):
            if frame.get("setup"):
                continue
            phase = _phase(frame, side)
            ours = _chars(frame, team)
            enemies = _chars(frame, "D" if team == "A" else "A")
            for enemy in enemies:
                name = enemy["name"]
                if enemy["alive"] and team not in enemy["visible_to"]:
                    unseen_streak[name] += 1
                else:
                    unseen_streak[name] = 0

            # Record prolonged lack of a sighting only when it culminates in
            # a nearby loss; unseen replay truth alone does not prove no clear.
            if offset:
                _, previous = sequence[offset - 1]
                previous_ours = {c["name"]: c for c in _chars(previous, team)}
                previous_enemies = _chars(previous, "D" if team == "A" else "A")
                for victim in ours:
                    before = previous_ours.get(victim["name"])
                    if not before or not before["alive"] or victim["alive"]:
                        continue
                    location = before["pos"]
                    nearby = sorted((e for e in previous_enemies if e["alive"]),
                                    key=lambda e: _distance(e["pos"], location))
                    exact = next((k for k in kills[(number, victim["name"], team)]
                                  if abs(int(k["tick"]) - int(frame["tick"])) <= 1), None)
                    enemy = (next((e for e in previous_enemies
                                   if exact and e["name"] == exact["killer"]), None)
                             if exact else None)
                    if enemy is None and nearby and _distance(nearby[0]["pos"], location) <= 8:
                        enemy = nearby[0]
                    if exact or enemy is not None:
                        enemy_pos = exact["killer_position"] if exact else enemy["pos"]
                        evidence = {"victim": victim["name"],
                                    "nearby_enemy": exact["killer"] if exact else enemy["name"],
                                    "killer_identified": bool(exact)}
                        emit("repeated_kill_location" if exact else "death_near_enemy",
                             enemy_pos, _face(location, enemy_pos), phase, side,
                             number, frame["tick"], index, "ally_death", evidence,
                             "confirmed" if exact else "candidate")
                    if enemy is not None:
                        if team not in enemy["visible_to"] and _distance(enemy["pos"], location) <= 3:
                            emit("unseen_close_incursion", enemy["pos"],
                                 _face(location, enemy["pos"]), phase, side, number,
                                 frame["tick"], index, "ally_death", evidence)
                        if unseen_streak[enemy["name"]] >= 30:
                            emit("long_unseen_before_death", enemy["pos"],
                                 _face(location, enemy["pos"]), phase, side, number,
                                 frame["tick"], index, "ally_death",
                                 {**evidence, "unseen_frames": unseen_streak[enemy["name"]]})
                        legal = [d for d in decisions[(number, side)]
                                 if int(d["tick"]) <= int(frame["tick"])]
                        if legal:
                            for point_id, point_pos, age in legal[-1]["watch_point_ages"]:
                                if (age >= 30 and point_id not in stale_seen
                                        and _distance(point_pos, enemy["pos"]) <= 2):
                                    stale_seen.add(point_id)
                                    emit("long_unconfirmed_watch_point", point_pos,
                                         _face(location, enemy["pos"]), phase, side,
                                         number, frame["tick"], index, "ally_death",
                                         {"point_id": point_id, "confirmation_age": age,
                                          "nearby_enemy": enemy["name"],
                                          "killer_identified": False})
                    if losing and side == "attacker" and not frame.get("planted"):
                        support = [c for c in previous_ours.values() if c["name"] != victim["name"]
                                   and c["alive"] and _distance(c["pos"], location) <= 4]
                        if not support:
                            emit("solo_entry_death", location, before["facing"], phase,
                                 side, number, frame["tick"], index, "ally_death",
                                 {"victim": victim["name"], "support_within_four": 0})

            # One occurrence per player pair and facing per round, avoiding
            # one candidate at every cell traversed together.
            by_facing = defaultdict(list)
            for char in ours:
                if char["alive"] and char["facing"] in FACING:
                    by_facing[char["facing"]].append(char)
            for facing, group in by_facing.items():
                for i, first in enumerate(group):
                    peers = [c for c in group[i + 1:] if _distance(c["pos"], first["pos"]) <= 4]
                    key = (first["name"], peers[0]["name"], facing) if peers else None
                    threat = min((_distance(e["pos"], first["pos"])
                                  for e in enemies if e["alive"]), default=999)
                    if peers and threat <= 8 and key not in angle_seen:
                        angle_seen.add(key)
                        emit("overlapping_facing", first["pos"], facing, phase,
                             side, number, frame["tick"], index,
                             "shared_angle_exposure", {"players": [first["name"], peers[0]["name"]],
                                                       "nearest_enemy_distance": threat})

        if losing and record.get("planted"):
            planted = next(((index, frame) for index, frame in reversed(sequence)
                            if frame.get("planted_pos") is not None), None)
            if planted:
                index, frame = planted
                phase = "post_plant" if side == "attacker" else "retake"
                emit("planted_round_loss", frame["planted_pos"], None, phase,
                     side, number, frame["tick"], index, str(record.get("reason")),
                     {"round_winner": record.get("winner")}, "confirmed")

    # ActionLog records requested abilities. A missing status change is a
    # review candidate, not proof of a failed cast or ineffective smoke.
    for raw in getattr(result, "coach_actions", ()):
        action = asdict(raw) if is_dataclass(raw) else raw
        if action.get("ability") not in ("FLASH", "RECON"):
            continue
        number, tick = int(action["round_number"]), int(action["tick"])
        if number not in frames:
            continue
        sequence = frames[number]
        matching = [(i, f) for i, f in sequence if tick <= int(f["tick"]) <= tick + 3
                    and not f.get("setup")]
        if not matching:
            continue
        action_side = action["side"].value if hasattr(action["side"], "value") else action["side"]
        team = "A" if action_side == "attacker" else "D"
        status = "blind" if action["ability"] == "FLASH" else "revealed"
        affected = any(bool(enemy.get(status)) for _, frame in matching
                       for enemy in _chars(frame, "D" if team == "A" else "A"))
        if not affected:
            index, frame = matching[0]
            emit("ability_no_observed_effect", action.get("ability_target") or action["start"],
                 action.get("facing"), _phase(frame, action_side), action_side,
                 number, tick, index, "no_enemy_status_change_within_three_ticks",
                 {"ability": action["ability"], "cast_confirmed": False})
    return events


def candidate_report(matches: Iterable[tuple[Any, str, str]]) -> dict[str, Any]:
    """Group events from (result, opponent_id, replay_id) triples."""
    grouped = defaultdict(list)
    match_count = 0
    for result, opponent, replay_id in matches:
        match_count += 1
        for event in collect_match(result, opponent_id=opponent, replay_id=replay_id):
            key = (event["kind"], tuple(event["position"]), event["facing"],
                   event["phase"], event["side"])
            grouped[key].append(event)
    candidates = []
    watch_points = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
    existing = defaultdict(list)
    for point in watch_points.points:
        for supported_side in point.sides:
            existing[(tuple(point.position), supported_side.value)].append(point.point_id)
    for key, occurrences in grouped.items():
        kind, position, facing, phase, side = key
        candidates.append({
            "kind": kind, "position": list(position), "facing": facing,
            "phase": phase, "side": side, "count": len(occurrences),
            "existing_watch_point_ids": existing[(position, side)],
            "opponents": sorted({e["opponent"] for e in occurrences}),
            "losses": sorted({e["loss"] for e in occurrences}),
            "confidence": occurrences[0]["confidence"],
            "replay_refs": [e["replay"] for e in occurrences],
            "evidence": [e["evidence"] for e in occurrences],
            "member_positions": [list(position)],
        })
    candidates.sort(key=lambda c: (-c["count"], c["kind"], c["position"], c["side"]))
    clusters = []
    for candidate in candidates:
        anchor = next((item for item in clusters
                       if item["kind"] == candidate["kind"]
                       and item["side"] == candidate["side"]
                       and item["phase"] == candidate["phase"]
                       and item["facing"] == candidate["facing"]
                       and _distance(item["position"], candidate["position"]) <= 2), None)
        if anchor is None:
            clusters.append(candidate)
            continue
        anchor["count"] += candidate["count"]
        anchor["opponents"] = sorted(set(anchor["opponents"]) | set(candidate["opponents"]))
        anchor["losses"] = sorted(set(anchor["losses"]) | set(candidate["losses"]))
        anchor["existing_watch_point_ids"] = sorted(
            set(anchor["existing_watch_point_ids"]) | set(candidate["existing_watch_point_ids"]))
        anchor["replay_refs"].extend(candidate["replay_refs"])
        anchor["evidence"].extend(candidate["evidence"])
        anchor["member_positions"].extend(candidate["member_positions"])
    clusters = [c for c in clusters
                if c["kind"] not in ("death_near_enemy", "repeated_kill_location")
                or c["count"] >= 2]
    clusters.sort(key=lambda c: (-c["count"], c["kind"], c["position"], c["side"]))
    return annotate_watch_coverage({"schema": SCHEMA, "map_sha256": MAP_HASH, "matches": match_count,
            "candidate_count": len(clusters), "candidates": clusters,
            "limits": ["Without optional referee kill events, replay proximity does not identify the killer.",
                       "A missing FLASH/RECON status change does not prove the cast succeeded.",
                       "Unseen duration does not prove that an area was never cleared.",
                       "Watch-point age uses the team's legal belief history; a nearby enemy is still an offline replay candidate.",
                       "Candidates require human review before watch-point changes or training."]})


def annotate_watch_coverage(report: dict[str, Any]) -> dict[str, Any]:
    """Mark cells covered by the existing point/jitter training distribution."""
    points = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR).points
    for candidate in report["candidates"]:
        applicable = [p for p in points if candidate["side"] in
                      {side.value for side in p.sides}]
        def covering(position):
            return [p.point_id for p in applicable
                    if _distance(position, p.position) <= p.random_radius]
        candidate["covering_watch_point_ids"] = sorted(set(covering(candidate["position"])))
        candidate["uncovered_positions"] = sorted(
            (position for position in candidate["member_positions"]
             if not covering(position)), key=lambda pos: (pos[0], pos[1]))
    return report
