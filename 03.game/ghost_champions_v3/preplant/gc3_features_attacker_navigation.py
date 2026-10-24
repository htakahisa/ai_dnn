"""Derived navigation inputs from existing public observations, never commands."""
from collections import deque
from functools import lru_cache
import numpy as np
from ..gc3_observation import ARENA_SHAPE, CELLS, CONTEXT_WIDTH, UNIT_WIDTH, BOARD_CHANNELS
from .gc3_observation_attacker_history import EXTRA_WIDTH
from .gc3_policy_attacker_history import MOVE_OFFSETS, actor_key

DISTANCE_SCALE = 12.0
IDLE_SCALE = 20.0
FEATURE_WIDTH = len(MOVE_OFFSETS)*8+12+5
DROP_FEATURE_WIDTH = 4+len(MOVE_OFFSETS)*3+5


@lru_cache(maxsize=32)
def walking_distances(wall_bytes, target_bytes):
    walls = np.frombuffer(wall_bytes, dtype=np.bool_).reshape(ARENA_SHAPE)
    targets = np.frombuffer(target_bytes, dtype=np.bool_).reshape(ARENA_SHAPE)
    distance = np.full(ARENA_SHAPE, -1, dtype=np.int32)
    queue = deque()
    for r, c in np.argwhere(targets & ~walls):
        distance[r, c] = 0
        queue.append((int(r), int(c)))
    while queue:
        r, c = queue.popleft()
        for dr, dc in MOVE_OFFSETS[1:]:
            y, x = r+dr, c+dc
            if 0 <= y < ARENA_SHAPE[0] and 0 <= x < ARENA_SHAPE[1] and not walls[y, x] and distance[y, x] < 0:
                distance[y, x] = distance[r, c]+1
                queue.append((y, x))
    distance.setflags(write=False)
    return distance


def proximity(distance):
    return 1./(1.+distance/DISTANCE_SCALE) if distance >= 0 else 0.


def public_layout(observation):
    board = observation[CONTEXT_WIDTH:CONTEXT_WIDTH+BOARD_CHANNELS*CELLS].reshape(BOARD_CHANNELS, *ARENA_SHAPE)
    units = observation[CONTEXT_WIDTH+BOARD_CHANNELS*CELLS:CONTEXT_WIDTH+BOARD_CHANNELS*CELLS+10*UNIT_WIDTH].reshape(10, UNIT_WIDTH)
    return board, units


def navigation_features(record, goal):
    """Distance/occupancy/history features; none mask or prefer an action."""
    cache = record.setdefault("_navigation_features", {})
    if goal in cache:
        return cache[goal]
    observation = record["obs"]
    board, units = public_layout(observation)
    point = tuple(np.rint(observation[:2]*ARENA_SHAPE).astype(int))
    walls = np.asarray(board[0] > .5, dtype=np.bool_)
    targets = np.asarray(board[1] > .5, dtype=np.bool_)
    if goal is None:
        targets[:] = False
    else:
        targets[:, :ARENA_SHAPE[1]//2] &= goal == 0
        targets[:, ARENA_SHAPE[1]//2:] &= goal == 1
    distances = walking_distances(walls.tobytes(), targets.tobytes())
    own, enemy = set(), set()
    for i, unit in enumerate(units):
        if unit[0] > .5 and unit[2] > .5 and unit[3] > .5 and unit[1] < .5:
            position = tuple(np.rint(unit[4:6]*ARENA_SHAPE).astype(int))
            (own if i < 5 else enemy).add(position)
    base = proximity(int(distances[point]))
    values = []
    movement = np.asarray(record["masks"]["position"]).reshape(ARENA_SHAPE)
    for dr, dc in MOVE_OFFSETS:
        p = point[0]+dr, point[1]+dc
        inside = 0 <= p[0] < ARENA_SHAPE[0] and 0 <= p[1] < ARENA_SHAPE[1]
        distance = proximity(int(distances[p])) if inside else 0.
        values.extend((not inside or walls[p], p in own, p in enemy,
                       bool(inside and board[9][p] > .5), bool(inside and movement[p]),
                       bool(inside and targets[p]), distance, distance-base))
    # Queries use only events strictly preceding this action, in this round.
    history = record["history"]
    key = actor_key(observation)
    previous = [e["obs"] for e in history.events[:record["history_index"]]
                if e["obs"][-EXTRA_WIDTH+6] == 0 and actor_key(e["obs"]) == key]
    prior_points = [tuple(np.rint(o[:2]*ARENA_SHAPE).astype(int)) for o in previous]
    idle = 0
    for prior in reversed(prior_points):
        if prior != point:
            break
        idle += 1
    last = prior_points[-1] if prior_points else point
    reversal = len(prior_points) >= 2 and point == prior_points[-2] and point != last
    flags = observation[-EXTRA_WIDTH:-EXTRA_WIDTH+5]
    carrier = next((tuple(np.rint(u[4:6]*ARENA_SHAPE).astype(int)) for u, flag in zip(units[:5], flags)
                    if u[0] > .5 and u[2] > .5 and flag > .5), None)
    carrier_distance = 0.
    if carrier is not None:
        carrier_target = np.zeros(ARENA_SHAPE, dtype=np.bool_)
        carrier_target[carrier] = True
        carrier_distance = proximity(int(walking_distances(walls.tobytes(), carrier_target.tobytes())[point]))
    values.extend((observation[11], observation[3], observation[6], observation[12], observation[14],
                   min(idle/IDLE_SCALE, 1.), point[0]-last[0], point[1]-last[1],
                   reversal, goal is not None, carrier_distance, observation[15]))
    for unit in units[:5]:
        p = tuple(np.rint(unit[4:6]*ARENA_SHAPE).astype(int))
        values.append(proximity(int(distances[p])) if unit[0] > .5 and unit[2] > .5 else 0.)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (FEATURE_WIDTH,) or not np.isfinite(result).all():
        raise ValueError("Invalid public navigation features")
    cache[goal] = result
    return result


def dropped_spike_features(record):
    """Walking distances to the perceived drop; no hidden coordinates or actions."""
    cached=record.get("_dropped_spike_features")
    if cached is not None:
        return cached
    obs=record["obs"]
    board,units=public_layout(obs)
    flags=obs[-EXTRA_WIDTH:-EXTRA_WIDTH+5]
    carried=any(flag>.5 and unit[2]>.5 for unit,flag in zip(units[:5],flags))
    dropped=bool(obs[8]>.5 and obs[2]<.5 and not carried)
    values=np.zeros(DROP_FEATURE_WIDTH,dtype=np.float32)
    if dropped:
        walls=np.asarray(board[0]>.5,dtype=np.bool_)
        drop=tuple(np.rint(obs[9:11]*ARENA_SHAPE).astype(int))
        point=tuple(np.rint(obs[:2]*ARENA_SHAPE).astype(int))
        targets=np.zeros(ARENA_SHAPE,dtype=np.bool_)
        inside=0<=drop[0]<ARENA_SHAPE[0] and 0<=drop[1]<ARENA_SHAPE[1]
        passable=bool(inside and not walls[drop])
        if passable:
            targets[drop]=True
        elif inside:
            # IQ can put the reported cell in a wall. Expose the reachable
            # neighbouring search area instead of reading the true spike cell.
            for dr,dc in MOVE_OFFSETS[1:]:
                p=drop[0]+dr,drop[1]+dc
                if 0<=p[0]<ARENA_SHAPE[0] and 0<=p[1]<ARENA_SHAPE[1] and not walls[p]:
                    targets[p]=True
        distances=walking_distances(walls.tobytes(),targets.tobytes())
        base=proximity(int(distances[point]))
        features=[1.,passable,distances[point]>=0,base]
        for dr,dc in MOVE_OFFSETS:
            p=point[0]+dr,point[1]+dc
            valid=0<=p[0]<ARENA_SHAPE[0] and 0<=p[1]<ARENA_SHAPE[1]
            near=proximity(int(distances[p])) if valid else 0.
            features.extend((bool(valid and targets[p]),near,near-base))
        for unit in units[:5]:
            p=tuple(np.rint(unit[4:6]*ARENA_SHAPE).astype(int))
            features.append(proximity(int(distances[p])) if unit[0]>.5 and unit[2]>.5 and unit[3]>.5 else 0.)
        values=np.asarray(features,dtype=np.float32)
    if values.shape!=(DROP_FEATURE_WIDTH,) or not np.isfinite(values).all():
        raise ValueError("Invalid perceived dropped-spike features")
    record["_dropped_spike_features"]=values
    return values
