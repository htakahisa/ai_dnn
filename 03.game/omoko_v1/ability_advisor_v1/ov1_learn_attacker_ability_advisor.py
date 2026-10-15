"""One classifier for all routes, opponents and fixed-point ability types."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import pickle
import warnings

import numpy as np
import torch
from torch import nn

from game_core import (FACING_DIRECTIONS, FLASH_MAX_FLIGHT_TICKS,
                       FLASH_SPEED_CELLS_PER_TICK, RECON_REVEAL_SIZE)
from grid_lines import line_cells, wall_line_cells, WALL_LOS_VERSION
OPPONENT_NAMES = {
    'Ghost Champions v2': 'ghost_champions_v2', 
    'FRC v1': 'frc_v1',
    'Fnatic v3': 'fnatic_v3', 
    'Touyama Gaming v3': 'touyama_gaming_v3',
    'ConCon v1': 'concon_v1', 
    'Toru AI v4': 'toru_ai_v4',
}

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / 'data' / 'ov1_attacker_ability_advisor.pt'
DEFAULT_HISTORY_ROUNDS = 5
MAPS = ('carry_L', 'carry_R', 'escort_L', 'escort_R')
OPPONENTS = tuple(sorted(set(OPPONENT_NAMES.values())))
ABILITIES = ('SMOKE', 'FLASH', 'RECON')
VERSION = 'omoko_fixed_ability_v1'
LABEL_VERSION = 'current_enemy_in_effect_or_smoke_firing_lane_v1'
TICK_BUCKET = 10


def group_key(scenario, opponent, ability, point):
    return f'{scenario.map_name}|{opponent}|{ability}|{point[0]},{point[1]}'


def impact_position(char, ability, aim, game):
    if ability == 'SMOKE':
        return tuple(aim)
    path = game._projectile_path(tuple(char.pos), tuple(aim))
    index = min(len(path)-1, FLASH_MAX_FLIGHT_TICKS*FLASH_SPEED_CELLS_PER_TICK) \
        if ability == 'FLASH' else len(path)-1
    return tuple(path[index]) if path else tuple(aim)


def disclosed_enemies(chars, team, shape):
    return [c for c in chars if c.team != team and c.is_alive
            and getattr(c, 'position_known', True)
            and 0 <= c.pos[0] < shape[0] and 0 <= c.pos[1] < shape[1]]


def relevant_enemy_positions(grid, chars, team, ability, impact, smoke=()):
    """Geometry for visible evidence, or separately for training-only teacher labels."""
    smoke = set(map(tuple, smoke))
    result = []
    for enemy in disclosed_enemies(chars, team, grid.shape):
        position = tuple(map(int, enemy.pos))
        dr, dc = abs(position[0]-impact[0]), abs(position[1]-impact[1])
        if ability == 'RECON':
            relevant = max(dr, dc) <= RECON_REVEAL_SIZE // 2
        elif ability == 'FLASH':
            cells = line_cells(position, impact)
            relevant = (not any(grid[p] == 1 for p in wall_line_cells(position, impact))
                        and (len(cells) <= 2 or not smoke.intersection(cells)))
        else:
            relevant = max(dr, dc) <= 1
            for ally in chars:
                if ally.team != team or not ally.is_alive:
                    continue
                target = tuple(map(int, ally.pos))
                if not (0 <= target[0] < grid.shape[0] and 0 <= target[1] < grid.shape[1]):
                    continue
                cells = line_cells(position, target)
                if (not any(grid[p] == 1 for p in wall_line_cells(position, target))
                        and (len(cells) <= 2 or not smoke.intersection(cells))
                        and any(max(abs(r-impact[0]), abs(c-impact[1])) <= 1 for r, c in cells)):
                    relevant = True
                    break
        if relevant:
            result.append(position)
    return result


class MatchMemory:
    def __init__(self):
        self.rounds = deque(maxlen=DEFAULT_HISTORY_ROUNDS)
        self.current = None
        self.opponent = None
        self.sequence = 0

    def start_round(self, map_name):
        if self.current is not None:
            self.rounds.append(self.current)
        self.sequence += 1
        self.current = dict(map_name=map_name, sightings={}, uses=set())

    def observe(self, scenario, char, state, opponent):
        if opponent != self.opponent:
            self.rounds.clear()
            self.current = None
            self.opponent = opponent
        if self.current is None:
            self.start_round(scenario.map_name)
        bucket = int(state.get('battle_tick', 0)) // TICK_BUCKET
        for enemy in disclosed_enemies(state.get('chars', []), char.team, scenario.grid.shape):
            self.current['sightings'][bucket, enemy.name] = tuple(map(int, enemy.pos))

    def features(self, point, ability, tick, shape):
        features = []
        history = list(reversed(self.rounds))
        for index in range(DEFAULT_HISTORY_ROUNDS):
            previous = history[index] if index < len(history) else None
            if previous is None:
                features.extend([0.] * 10)
                continue
            positions = [pos for (bucket, _), pos in previous['sightings'].items()
                         if abs(bucket-tick // TICK_BUCKET) <= 1]
            distance = min((math.dist(pos, point) for pos in positions), default=max(shape))
            features.extend([1.] + [float(previous['map_name'] == name) for name in MAPS]
                            + [float(bool(previous['sightings'])), min(len(positions), 5) / 5,
                               min(distance/max(shape), 1.),
                               float(any(max(abs(p[0]-point[0]), abs(p[1]-point[1])) <= 4 for p in positions)),
                               float((ability, tuple(point)) in previous['uses'])])
        return features


def feature_names():
    names = ['map:'+n for n in MAPS] + ['opponent:'+n for n in OPPONENTS] + ['ability:'+n for n in ABILITIES]
    names += ['point_r','point_c','impact_r','impact_c','caster_r','caster_c',
              'tick','charges','blind','revealed','direct_relevance']
    names += ['caster_facing:'+d for d in FACING_DIRECTIONS]
    names += [f'{team}:{i}:{f}' for team in ('ally','enemy') for i in range(5)
              for f in ('known','r','c',*FACING_DIRECTIONS)]
    names += [f'history:{i}:{f}' for i in range(DEFAULT_HISTORY_ROUNDS)
              for f in ('present',*MAPS,'observed','phase_seen','nearest','near_point','used')]
    return names


def build_features(scenario, char, state, opponent, ability, point, impact, memory):
    h, w = scenario.grid.shape
    chars = state.get('chars', [])
    direct = bool(relevant_enemy_positions(np.asarray(state['grid']), chars, char.team,
                                          ability, impact, state.get('smoke_cells', ())))
    values = ([float(scenario.map_name == n) for n in MAPS]
              + [float(opponent == n) for n in OPPONENTS] + [float(ability == n) for n in ABILITIES])
    for p in (point, impact, char.pos):
        values += [p[0]/h, p[1]/w]
    tick = int(state.get('battle_tick', 0))
    values += [min(tick,200)/200, getattr(char,ability.lower()+'_charges',0)/3,
               min(getattr(char,'blind_remaining',0),10)/10,
               min(getattr(char,'reveal_remaining',0),15)/15, float(direct)]
    values += [float(char.facing == d) for d in FACING_DIRECTIONS]
    for group in ([c for c in chars if c.team == char.team and c.is_alive],
                  disclosed_enemies(chars, char.team, (h,w))):
        ordered = sorted(group, key=lambda c:c.name)
        for i in range(5):
            if i >= len(ordered):
                values += [0.] * 11
            else:
                other = ordered[i]
                values += [1., other.pos[0]/h, other.pos[1]/w]
                values += [float(other.facing == d) for d in FACING_DIRECTIONS]
    values += memory.features(point, ability, tick, (h,w))
    if len(values) != len(feature_names()):
        raise RuntimeError('ability adviser feature dimensions changed')
    return np.asarray(values,dtype=np.float32), direct


class AbilityJudge(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(len(feature_names()),96),nn.ReLU(),
                                    nn.Linear(96,48),nn.ReLU(),nn.Linear(48,1))

    def forward(self, features):
        return self.layers(features).squeeze(-1)


@dataclass(frozen=True)
class Advice:
    use: bool
    reason: str
    score: float | None = None


class FixedAbilityAdvisor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.memory = MatchMemory()
        self.model = self.checkpoint = self.last_advice = None
        self.pending = {}
        self.route_hashes = {}
        self.decisions = {'use':0,'save':0}
        self.last_match_round = None
        if model_path is not None and Path(model_path).is_file():
            try:
                checkpoint = torch.load(model_path,map_location='cpu',weights_only=False)
                if (checkpoint['version'] != VERSION or checkpoint['label_version'] != LABEL_VERSION
                        or checkpoint['wall_los_version'] != WALL_LOS_VERSION
                        or checkpoint['feature_names'] != feature_names()
                        or not isinstance(checkpoint.get('scenarios'),dict)
                        or not isinstance(checkpoint.get('groups'),dict)
                        or not 0 < checkpoint.get('skip_threshold',0) < 1):
                    raise ValueError('adviser checkpoint contract changed')
                if (any(not isinstance(v,dict) or not isinstance(v.get('signature'),str)
                        or not isinstance(v.get('carry_sha256'),str) for v in checkpoint['scenarios'].values())
                        or any(not isinstance(v,dict) for v in checkpoint['groups'].values())):
                    raise ValueError('invalid adviser group metadata')
                with torch.random.fork_rng(devices=[]):
                    model = AbilityJudge()
                model.load_state_dict(checkpoint['model_state_dict'])
                self.model,self.checkpoint = model.eval(),checkpoint
            except (OSError,ValueError,KeyError,RuntimeError,TypeError,EOFError,pickle.UnpicklingError) as error:
                warnings.warn(f'Fixed ability adviser unavailable; using fixed abilities: {error}')

    def start_round(self,map_name,round_number=None):
        if (round_number is not None and self.last_match_round is not None
                and round_number <= self.last_match_round):
            self.memory = MatchMemory()
        self.last_match_round = round_number
        self.pending.clear()
        self.memory.start_round(map_name)
        self.memory.current['round_number'] = round_number
        if round_number is not None:
            while (self.memory.rounds and self.memory.rounds[0].get('round_number') is not None
                   and self.memory.rounds[0]['round_number'] < round_number-DEFAULT_HISTORY_ROUNDS):
                self.memory.rounds.popleft()

    def bind_route(self,scenario,model_hash):
        self.route_hashes[scenario.map_name] = model_hash

    def observe(self,scenario,char,state,opponent):
        if opponent != self.memory.opponent:
            self.pending.clear()
        self.memory.observe(scenario,char,state,opponent)
        self.memory.current['round_number'] = self.last_match_round
        request = self.pending.get(char.name)
        if request is not None:
            ability,point,charges,tick = request
            if getattr(char,ability.lower()+'_charges',0) < charges:
                self.memory.current['uses'].add((ability,point))
                self.pending.pop(char.name,None)
            elif int(state.get('battle_tick',0)) > tick:
                self.pending.pop(char.name,None)

    def advise(self,scenario,char,state,opponent,ability,point,aim,game):
        impact = impact_position(char,ability,aim,game)
        features,direct = build_features(scenario,char,state,opponent,ability,point,impact,self.memory)
        advice = Advice(True,'no_validated_model')
        if direct:
            advice = Advice(True,'observed_enemy_relevance',1.)
        elif self.model is not None:
            metadata = self.checkpoint['scenarios'].get(scenario.map_name)
            report = self.checkpoint['groups'].get(group_key(scenario,opponent,ability,point),{})
            if (metadata is not None and metadata['signature'] == scenario.signature
                    and metadata['carry_sha256'] == self.route_hashes.get(scenario.map_name)
                    and report.get('validated',False)):
                with torch.no_grad():
                    score = float(torch.sigmoid(self.model(torch.from_numpy(features))).item())
                if math.isfinite(score):
                    advice = Advice(score >= self.checkpoint['skip_threshold'],'model',score)
        self.last_advice = advice
        self.decisions['use' if advice.use else 'save'] += 1
        if advice.use:
            self.pending[char.name] = (ability,tuple(point),getattr(char,ability.lower()+'_charges',0),int(state.get('battle_tick',0)))
        return advice


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
