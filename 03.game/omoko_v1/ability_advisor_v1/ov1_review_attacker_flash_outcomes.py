"""Diagnose flash overlap, combat conversion and other-ability effects without training."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from game_core import SHOOTING_SITE_DIGREE
from omoko_v1.ability_advisor_v1.ov1_learn_attacker_ability_advisor import (
    ROOT, DEFAULT_MODEL_PATH, FixedAbilityAdvisor, sha256_file,
)
from omoko_v1.ability_advisor_v1.ov1_collect_attacker_ability_advisor import play_match
from omoko_v1.ability_advisor_v1.ov1_evaluate_attacker_ability_usage import AbilityUsageMonitor
from omoko_v1.ability_advisor_v1.ov1_opponent_processes import run_opponent_tasks

DEFAULT_CANDIDATE = ROOT / 'data' / 'ov1_attacker_ability_advisor_candidate.pt'
DEFAULT_OUTPUT = ROOT / 'logs' / 'ov1_attacker_flash_outcomes.json'
DEFAULT_OPPONENT = 'frc_v1'
DEFAULT_SEEDS = tuple(range(100020, 100030))  # Exact seeds of the saved FRC comparison.
DEFAULT_ROUNDS = 12
DEFAULT_WORKERS = 4
DEFAULT_ARMS = ('current', 'candidate', 'flash_only', 'other_only')
DEFAULT_FOLLOWUP_ARMS = ('flash_smoke', 'flash_recon')
DEFAULT_FOLLOWUP_OUTPUT = ROOT / 'logs' / 'ov1_attacker_flash_ability_interaction.json'
MAX_OUTPUT_BYTES = 99_000_000


class HybridAdvisor(FixedAbilityAdvisor):
    """Select the existing or candidate adviser per ability; keep shared match memory."""
    def __init__(self, arm):
        super().__init__(DEFAULT_MODEL_PATH)
        candidate = FixedAbilityAdvisor(DEFAULT_CANDIDATE)
        self.arm = arm
        self.reference = self.model, self.checkpoint
        self.candidate = candidate.model, candidate.checkpoint

    def advise(self, scenario, char, state, opponent, ability, point, aim, game):
        use_candidate = (self.arm == 'candidate' or
                         self.arm == 'flash_only' and ability == 'FLASH' or
                         self.arm == 'other_only' and ability != 'FLASH' or
                         self.arm == 'flash_smoke' and ability in ('FLASH', 'SMOKE') or
                         self.arm == 'flash_recon' and ability in ('FLASH', 'RECON'))
        self.model, self.checkpoint = self.candidate if use_candidate else self.reference
        try:
            return super().advise(scenario, char, state, opponent, ability, point, aim, game)
        finally:
            self.model, self.checkpoint = self.reference


class CombatMonitor:
    def __init__(self, advisor):
        self.usage = AbilityUsageMonitor(advisor)
        self.game = None
        self.events = []
        self.round_stats = defaultdict(Counter)
        self.rounds = []
        self.score = 0

    def observe(self, game, when):
        self.usage.observe_tick(game, when)
        if self.game is None:
            self.game = game
            self.install(game)
        stats = self.round_stats[game.current_round]
        if when == 'after':
            stats['blind_enemy_ticks'] += sum(c.is_alive and c.team == 'D' and c.blind_remaining > 0
                                              for c in game.chars)
            if game.round_over:
                won = game.attacker_wins > self.score
                self.score = game.attacker_wins
                reason = ('defused' if game.is_defused else
                          'timeout' if not game.is_planted and game.round_timer <= 0 else
                          'detonated' if won and game.is_planted and game.detonate_timer <= 0 else
                          'defenders_eliminated' if not any(c.team == 'D' and c.is_alive for c in game.chars) else
                          'detonated' if won and game.is_planted else
                          'attackers_eliminated' if not any(c.team == 'A' and c.is_alive for c in game.chars) else
                          'timeout')
                self.rounds.append(dict(round=game.current_round, won=won, planted=game.is_planted,
                                        reason=reason, tick=game.battle_tick, stats=dict(stats)))

    def install(self, game):
        original_flash = game._explode_flash

        def explode(projectile, impact=None):
            if projectile.get('team') != 'A':
                return original_flash(projectile, impact)
            point = impact or projectile['path'][min(projectile['progress'], len(projectile['path']) - 1)]
            targets = [c for c in game.chars if c.team == 'D' and c.is_alive
                       and game.check_cell_line_of_sight(tuple(c.pos), point, block_smoke=True)]
            before = {c.name: c.blind_remaining for c in targets}
            supported = [c for c in targets if any(
                a.team == 'A' and a.is_alive and not a.plant_timer and not a.defuse_timer
                and not getattr(a, 'collecting_orb_this_tick', False)
                and game.check_shot_line_of_sight(a, c)
                and game._facing_angle_diff(a, c) <= SHOOTING_SITE_DIGREE
                for a in game.chars)]
            result = original_flash(projectile, impact)
            gains = [max(0, c.blind_remaining-before[c.name]) for c in targets]
            event = dict(round=game.current_round, tick=game.battle_tick, impact=list(point),
                         target_names=list(map(lambda c:str(c.name), targets)),
                         targets=len(targets), fresh=sum(before[c.name] == 0 for c in targets),
                         refreshed=sum(before[c.name] > 0 for c in targets),
                         added_ticks=sum(gains), zero_gain=bool(targets) and not any(gains),
                         supported_targets=len(supported))
            self.events.append(event)
            stats = self.round_stats[game.current_round]
            stats['flash_bursts'] += 1
            stats['reaching_bursts'] += int(bool(targets))
            stats['zero_gain_bursts'] += int(event['zero_gain'])
            stats['fresh_targets'] += event['fresh']
            stats['refreshed_targets'] += event['refreshed']
            stats['added_blind_ticks'] += event['added_ticks']
            stats['reaching_without_support'] += int(bool(targets) and not supported)
            return result

        game._explode_flash = explode
        original_shots = game._resolve_all_shots

        def shots(*args, **kwargs):
            blind = {c.name:c.blind_remaining for c in game.chars}
            result = original_shots(*args, **kwargs)
            stats = self.round_stats[game.current_round]
            for shot in game.last_shots:
                shooter, target = shot['shooter'], shot['target']
                if getattr(target, 'is_ultimate_drone', False):
                    continue
                prefix = 'attacker' if shooter.team == 'A' else 'defender'
                stats[prefix+'_shots'] += 1
                stats[prefix+'_hits'] += int(shot['hit'])
                stats[prefix+'_damage'] += shot['damage']
                if blind.get(target.name, 0) > 0:
                    stats[prefix+'_shots_at_blind'] += 1
                    stats[prefix+'_damage_to_blind'] += shot['damage']
                if blind.get(shooter.name, 0) > 0:
                    stats[prefix+'_shots_while_blind'] += 1
                    stats[prefix+'_damage_while_blind'] += shot['damage']
            return result

        game._resolve_all_shots = shots
        original_kill = game._kill_character

        def kill(shooter, target, *args, **kwargs):
            stats = self.round_stats[game.current_round]
            stats['attacker_kills' if shooter.team == 'A' else 'defender_kills'] += 1
            if shooter.team == 'A' and target.blind_remaining > 0:
                stats['attacker_kills_blind'] += 1
            return original_kill(shooter, target, *args, **kwargs)

        game._kill_character = kill


def run_seed(task):
    seed, selected_arms = task
    torch.set_num_threads(1)
    arms = {}
    for arm in selected_arms:
        advisor = HybridAdvisor(arm)
        monitor = CombatMonitor(advisor)
        outcome = play_match(DEFAULT_OPPONENT, seed, DEFAULT_ROUNDS, advisor, tick_observer=monitor.observe)
        arms[arm] = dict(outcome=outcome, usage=monitor.usage.report(),
                         rounds=monitor.rounds, flash_events=monitor.events)
        print(f'seed={seed} arm={arm} wins={outcome["attacker_wins"]}', flush=True)
    return dict(seed=seed, arms=arms)


def summarize(runs, selected_arms=DEFAULT_ARMS):
    results = {}
    for arm in selected_arms:
        rounds = [r for run in runs for r in run['arms'][arm]['rounds']]
        stats = Counter()
        for record in rounds:
            stats.update(record['stats'])
        results[arm] = dict(wins=sum(r['won'] for r in rounds), rounds=len(rounds),
                            planted=sum(r['planted'] for r in rounds),
                            losses=dict(Counter(r['reason'] for r in rounds if not r['won'])),
                            stats=dict(stats))
    return results


def main(selected_arms=DEFAULT_ARMS, output=DEFAULT_OUTPUT):
    hashes = dict(reference=sha256_file(DEFAULT_MODEL_PATH), candidate=sha256_file(DEFAULT_CANDIDATE))
    runs = run_opponent_tasks(run_seed, [(seed, selected_arms) for seed in DEFAULT_SEEDS], DEFAULT_WORKERS)
    if hashes != dict(reference=sha256_file(DEFAULT_MODEL_PATH), candidate=sha256_file(DEFAULT_CANDIDATE)):
        raise RuntimeError('adviser weights changed during diagnosis')
    report = dict(model_hashes=hashes, opponent=DEFAULT_OPPONENT, seeds=DEFAULT_SEEDS,
                  limitation='Same initial seeds; later random draws and states diverge after policy changes. '
                             'Flash overlap and damage are measured; arm differences do not identify '
                             'individual counterfactual decisions or statistical significance.',
                  summary=summarize(runs, selected_arms), runs=runs)
    data = json.dumps(report, ensure_ascii=False, indent=2).encode('utf-8')
    if len(data) > MAX_OUTPUT_BYTES:
        raise ValueError('diagnosis output exceeds configured limit')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
