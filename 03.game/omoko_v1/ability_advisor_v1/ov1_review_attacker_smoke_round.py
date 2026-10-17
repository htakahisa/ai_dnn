"""Trace one identical-start round with candidate versus reference smoke advice."""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omoko_v1.ability_advisor_v1.ov1_review_attacker_flash_outcomes import HybridAdvisor, CombatMonitor
from omoko_v1.ability_advisor_v1.ov1_collect_attacker_ability_advisor import play_match
from omoko_v1.ability_advisor_v1.ov1_learn_attacker_ability_advisor import ROOT

DEFAULT_SEED = 100029
DEFAULT_OPPONENT = 'frc_v1'
DEFAULT_ROUNDS = 1
DEFAULT_ARMS = ('candidate', 'flash_recon')
DEFAULT_OUTPUT = ROOT / 'logs' / 'ov1_attacker_smoke_round.json'
MAX_OUTPUT_BYTES = 99_000_000


def trace(arm):
    advisor = HybridAdvisor(arm)
    monitor = CombatMonitor(advisor)
    actions, advice, frames = [], [], []
    installed = False
    previous_tick = None
    original_advice = advisor.advise

    def advise(scenario, char, state, opponent, ability, point, aim, game):
        result = original_advice(scenario, char, state, opponent, ability, point, aim, game)
        if ability == 'SMOKE':
            advice.append(dict(tick=game.battle_tick, caster=str(char.name), pos=list(char.pos),
                               point=list(point), use=result.use, reason=result.reason, score=result.score))
        return result

    advisor.advise = advise

    def observe(game, when):
        nonlocal installed, previous_tick
        monitor.observe(game, when)
        if not installed:
            installed = True
            original = game.execute_ai_ability

            def execute(owner, action):
                ability = action.get('ability') if isinstance(action, dict) else None
                before = getattr(owner, str(ability).lower()+'_charges', 0)
                result = original(owner, action)
                if getattr(owner, str(ability).lower()+'_charges', 0) < before:
                    actions.append(dict(tick=game.battle_tick, team=owner.team, caster=str(owner.name),
                                        pos=list(owner.pos), ability=ability, target=action.get('target')))
                return result

            game.execute_ai_ability = execute
        if when == 'after' and previous_tick != game.battle_tick:
            previous_tick = game.battle_tick
            frames.append(dict(tick=game.battle_tick, planted=game.is_planted,
                               chars=[dict(name=str(c.name), team=c.team, pos=list(c.pos),
                                           facing=c.facing, alive=c.is_alive, hp=c.hp,
                                           blind=c.blind_remaining) for c in game.chars],
                               smoke=sorted(map(list, game._smoke_cells())),
                               shots=[dict(shooter=str(s['shooter'].name), target=str(s['target'].name),
                                           hit=s['hit'], damage=s['damage'], hit_chance=s['hit_chance'],
                                           shooter_moved=s['shooter'].moved_this_tick,
                                           shooter_blind=s['shooter'].blind_remaining,
                                           shooter_smoke_vision=getattr(s['shooter'], 'sees_through_smoke', False),
                                           target_revealed=s['target'].reveal_remaining) for s in game.last_shots]))

    outcome = play_match(DEFAULT_OPPONENT, DEFAULT_SEED, DEFAULT_ROUNDS, advisor, tick_observer=observe)
    return dict(outcome=outcome, rounds=monitor.rounds, actions=actions, advice=advice, frames=frames)


def main():
    torch.set_num_threads(1)
    report = dict(seed=DEFAULT_SEED, arms={arm:trace(arm) for arm in DEFAULT_ARMS})
    data = json.dumps(report, ensure_ascii=False, indent=2).encode('utf-8')
    if len(data) > MAX_OUTPUT_BYTES:
        raise ValueError('trace exceeds configured size limit')
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_bytes(data)
    print(json.dumps({arm:r['rounds'] for arm,r in report['arms'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
