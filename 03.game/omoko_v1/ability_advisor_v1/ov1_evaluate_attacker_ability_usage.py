"""Evaluation-only hooks: measure actual consumption, never expose teacher state to AI."""

from collections import defaultdict

from omoko_v1.ability_advisor_v1.ov1_learn_attacker_ability_advisor import (
    ABILITIES, impact_position, relevant_enemy_positions,
)


class AbilityUsageMonitor:
    def __init__(self, advisor):
        self.advisor = advisor
        self.casts = []
        self.projectiles = {}
        self.installed = False

    def observe_tick(self, game, when):
        if self.installed or when != 'before':
            return
        self.installed = True
        original = game.execute_ai_ability

        def execute(owner, action):
            ability = action.get('ability') if isinstance(action, dict) else None
            if owner.team != 'A' or ability not in ABILITIES or 'target' not in action:
                return original(owner, action)
            target = tuple(action['target'])
            impact = impact_position(owner, ability, target, game)
            needed = bool(relevant_enemy_positions(game.grid, game.chars, owner.team,
                                                   ability, impact, game._smoke_cells()))
            phase = ('guard' if game.is_planted else 'carry' if owner.has_spike else
                     'escort' if any(c.team == owner.team and c.is_alive and c.has_spike
                                     for c in game.chars) else 'retrieve')
            pending = self.advisor.pending.get(owner.name)
            advised = bool(pending and pending[0] == ability and tuple(pending[1]) == target)
            before = getattr(owner, ability.lower() + '_charges', 0)
            outcome = original(owner, action)
            if getattr(owner, ability.lower() + '_charges', 0) < before:
                record = dict(phase=phase, ability=ability, target=list(target),
                              advised_fixed=advised, teacher_unnecessary=not needed,
                              round=game.current_round, tick=game.battle_tick,
                              resolved=False, affected_enemies=None)
                self.casts.append(record)
                if ability in ('FLASH', 'RECON'):
                    projectiles = getattr(game, ability.lower() + '_projectiles')
                    # Keep the projectile alive until resolution or monitor teardown;
                    # a discarded round must not let Python reuse its id for an enemy shot.
                    self.projectiles[id(projectiles[-1])] = (projectiles[-1], record)
            return outcome

        game.execute_ai_ability = execute
        for ability in ('FLASH', 'RECON'):
            name = '_explode_' + ability.lower()
            explode = getattr(game, name)

            def resolved(projectile, impact=None, *, original=explode, ability=ability):
                entry = self.projectiles.pop(id(projectile), None)
                if entry is not None:
                    record = entry[1]
                    point = impact or projectile['path'][min(projectile['progress'], len(projectile['path']) - 1)]
                    record['resolved'] = True
                    record['affected_enemies'] = len(relevant_enemy_positions(
                        game.grid, game.chars, projectile['team'], ability, point, game._smoke_cells()))
                return original(projectile, impact)

            setattr(game, name, resolved)

    def report(self):
        summary = defaultdict(lambda: dict(casts=0, teacher_unnecessary_casts=0,
                                          resolved_projectiles=0, zero_enemy_effect_projectiles=0,
                                          unresolved_projectiles=0))
        for cast in self.casts:
            keys = ['all', cast['phase'] + '|' + cast['ability'],
                    cast['phase'] + '|' + cast['ability'] + '|' + ','.join(map(str, cast['target']))]
            if cast['advised_fixed']:
                keys += ['advised_fixed']
            for key in keys:
                stats = summary[key]
                stats['casts'] += 1
                stats['teacher_unnecessary_casts'] += int(cast['teacher_unnecessary'])
                if cast['ability'] in ('FLASH', 'RECON'):
                    stats['resolved_projectiles'] += int(cast['resolved'])
                    stats['zero_enemy_effect_projectiles'] += int(cast['resolved'] and cast['affected_enemies'] == 0)
                    stats['unresolved_projectiles'] += int(not cast['resolved'])
        return dict(summary)
