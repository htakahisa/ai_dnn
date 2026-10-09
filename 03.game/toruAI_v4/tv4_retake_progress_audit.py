"""Post-action labels for arrival, defuse starts, cancellations and timeouts."""
from collections import Counter
from grid_paths import distance_map
from game_core import DEFUSE_REQUIRED_TICKS


def defuse_distances(scenario, plant):
    # The engine accepts all eight adjacent cells, including diagonals.
    goals = [(plant[0]+dr, plant[1]+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if 0 <= plant[0]+dr < scenario.grid.shape[0] and 0 <= plant[1]+dc < scenario.grid.shape[1]
             and scenario.grid[plant[0]+dr, plant[1]+dc] != 1]
    import numpy as np
    fields = [distance_map(scenario.grid, p) for p in goals]
    return np.minimum.reduce([np.where(f >= 0, f, np.inf) for f in fields])


class RetakeProgressAudit:
    def __init__(self, scenario):
        self.scenario = scenario
        self.frames, self.interruptions = [], []
        self.counts = Counter()
        self.before_state = None
        self.first_in_range = self.first_start = None
        self.max_progress = 0
        self.plant, self.field = None, None

    def before(self, snapshot, inputs, plans, launched):
        if self.plant != snapshot.spike_planted:
            self.plant = snapshot.spike_planted
            self.field = defuse_distances(self.scenario, self.plant)
        field = self.field
        actors = []
        for ally in snapshot.allies:
            if not ally.alive:
                continue
            data = inputs.get(ally.name)
            planned = plans.get(ally.name)
            action = data.actions[planned[1]].kind if data and planned else None
            distance = float(field[ally.position])
            actors.append(dict(name=ally.name, slot=ally.slot, position=ally.position, hp=ally.hp,
                progress=ally.defuse_progress, distance=distance if distance != float('inf') else None,
                action=action, defuse_legal=bool(data.mask[-2]) if data else False,
                contact=bool(data.combat_contact) if data else False,
                teacher_action=data.actions[data.teacher].kind if data else None,
                goal=data.goal if data else None,
                designated=bool(getattr(data, 'retake_context', {}).get('defuser'))))
        self.before_state = dict(tick=snapshot.tick, remaining=snapshot.detonate_timer,
                                 launched=launched, actors=actors)

    def after(self, game):
        if self.before_state is None:
            return
        frame = self.before_state
        actors = {str(getattr(c, 'base_name', c.name)): c for c in game.chars if c.team == 'D'}
        for a in frame['actors']:
            c = actors[a['name']]
            effective_progress = c.defuse_timer if c.is_alive else 0
            a.update(end_position=tuple(c.pos), end_progress=effective_progress, alive=c.is_alive)
            in_range = a['distance'] == 0
            self.counts['in_range_player_ticks'] += int(in_range)
            self.counts['in_range_without_defuse_ticks'] += int(in_range and a['action'] != 'DEFUSE')
            self.counts['defuse_commands'] += int(a['action'] == 'DEFUSE')
            if in_range and self.first_in_range is None:
                self.first_in_range = frame['remaining']
            if c.defuse_timer > 0 and a['progress'] == 0:
                self.counts['starts'] += 1
                if self.first_start is None:
                    self.first_start = frame['remaining']
            if a['progress'] > 0 and effective_progress < a['progress'] and not game.is_defused:
                cause = 'death' if not c.is_alive else 'different_action' if a['action'] != 'DEFUSE' else 'execution_failed'
                self.counts['interruptions'] += 1
                self.counts['interrupted_'+cause] += 1
                self.interruptions.append(dict(tick=frame['tick'], remaining=frame['remaining'],
                    name=a['name'], lost_progress=a['progress'], action=a['action'], cause=cause))
            self.max_progress = max(self.max_progress, int(c.defuse_timer), int(a['progress']))
        frame['feasible'] = any(a['distance'] is not None and a['distance'] +
            DEFUSE_REQUIRED_TICKS-a['progress'] <= frame['remaining'] for a in frame['actors'])
        frame['any_in_range'] = any(a['distance'] == 0 for a in frame['actors'])
        frame['any_defusing'] = any(a['action'] == 'DEFUSE' for a in frame['actors'])
        self.counts['feasible_ticks'] += int(frame['feasible'])
        self.counts['available_without_defuse_ticks'] += int(frame['any_in_range'] and not frame['any_defusing'])
        self.frames.append(frame)
        self.before_state = None

    def report(self):
        return dict(counts=dict(self.counts), first_in_range_remaining=self.first_in_range,
                    first_start_remaining=self.first_start, max_progress=self.max_progress,
                    interruptions=self.interruptions, frames=self.frames)
