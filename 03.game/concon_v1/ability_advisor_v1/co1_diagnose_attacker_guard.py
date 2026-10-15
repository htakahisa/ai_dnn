"""Audit actual postplant counter utility in complete, IQ-filtered attacking matches."""

import argparse
from collections import Counter
import contextlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

import torch
from concon_v1.ability_advisor_v1.co1_collect_attacker_ability_advisor import play_match
from concon_v1.ability_advisor_v1.co1_learn_attacker_ability_advisor import (
    ROOT, MAPS, ABILITIES, DEFAULT_MODEL_PATH, FixedAbilityAdvisor, sha256_file,
)
from concon_v1.co1_learn_guard import ConconGuardController
from concon_v1.co1_guard_common import decode_action
from concon_v1.co1_guard_scenarios import get_scenario
from abilities_los import AbilityLosMixin
from battle_logic import BattleLogicMixin
from concon_v1.ability_advisor_v1.co1_opponent_processes import run_opponent_tasks

DEFAULT_MATCHES = 10
DEFAULT_ROUNDS = 12
DEFAULT_OPPONENT = 'omoko_v1'
DEFAULT_SEED = 100030  # Same omoko seeds as the initial six-opponent comparison.
DEFAULT_MODEL = DEFAULT_MODEL_PATH
DEFAULT_OUTPUT = ROOT/'logs'/'co1_attacker_guard_diagnosis.json'
MAX_EXAMPLES = 25
DEFAULT_WORKERS = 2  # Compare baseline/advised in separate processes.


def decision_kind(action,position,context):
    _,payload = decode_action(action,position,context['targets'],context['ultimate_actions'])
    return payload.get('ability',payload.get('ultimate','move' if action//8 < 4 else 'wait'))


class GuardAudit:
    def __init__(self):
        self.counts = Counter()
        self.scopes = {}
        self.examples = []
        self.plants = []
        self.ends = []
        self.casts = []
        self.plant_ticks = {}
        self.last_decisions = {}
        self.recorded_plants = set()
        self.game = None
        self.before = None

    def observe_tick(self,game,phase):
        self.game = game
        if phase == 'before':
            self.before = (game.current_round,game.is_planted,game.round_over)
            return
        round_number,planted,already_over = self.before
        if game.current_round != round_number:
            return
        if game.is_planted and not planted:
            self.record_plant(game)

    def record_plant(self,game):
        if game.current_round in self.recorded_plants:
            return
        self.recorded_plants.add(game.current_round)
        self.plant_ticks[game.current_round] = game.battle_tick
        self.plants.append(dict(round=game.current_round,charges={a:sum(
            getattr(c,a.lower()+'_charges',0) for c in game.chars if c.team=='A' and c.is_alive)
            for a in ABILITIES}))

    def record_decision(self,controller,char,observation,mask,context,action):
        # Some teammates act during the same tick in which the carrier plants.
        self.record_plant(self.game)
        kind = decision_kind(action,tuple(char.pos),context)
        self.last_decisions[(self.game.current_round,char.name)] = dict(
            tick=context['tick'],tap=bool(context['tap']),fireable=bool(context['fireable']),
            blind=bool(context['blind']),revealed=bool(context['revealed']))
        ability = str(getattr(char,'ability_name','')).upper()
        available = any(mask[i] and decision_kind(i,tuple(char.pos),context) in ABILITIES
                        for i in range(40,len(mask)))
        smoke_tap = bool(context['tap'] and self.game and
                         tuple(context['spike']) in self.game._smoke_cells())
        scopes = ['all'] + [name for name,condition in (
            ('blind',context['blind']),('revealed',context['revealed']),
            ('impaired',context['impaired']),('tap',context['tap']),('smoke_tap',smoke_tap)) if condition]
        for scope in scopes:
            counts = self.scopes.setdefault(scope,Counter())
            counts['decisions'] += 1
            counts['utility_available'] += int(available)
            counts['selected:'+kind] += 1
            if available:
                counts['available:'+ability] += 1
                counts['available_selected:'+kind] += 1
        if context['impaired'] and available and kind not in ABILITIES and len(self.examples)<MAX_EXAMPLES:
            device = next(controller.model.parameters()).device
            with torch.no_grad():
                values = controller.model(torch.as_tensor(observation,device=device).unsqueeze(0))[0]
            candidates = [i for i in range(40,len(mask)) if mask[i] and
                          decision_kind(i,tuple(char.pos),context) in ABILITIES]
            best = max(candidates,key=lambda i:float(values[i]))
            self.examples.append(dict(round=self.game.current_round,tick=context['tick'],
                site=controller.scenario.map_name,actor=char.name,position=list(char.pos),
                blind=context['blind'],revealed=context['revealed'],tap=context['tap'],
                ability=ability,charges=getattr(char,ability.lower()+'_charges',0),
                selected=kind,best_utility=decision_kind(best,tuple(char.pos),context),
                selected_q=float(values[action]),utility_q=float(values[best]),
                targets=context['targets']))

    @contextlib.contextmanager
    def instrument(self):
        original_choose = ConconGuardController.choose_action
        original_cast = AbilityLosMixin.execute_ai_ability
        original_flash = AbilityLosMixin._explode_flash
        original_recon = AbilityLosMixin._explode_recon
        original_end = BattleLogicMixin.check_match_winner
        audit = self

        def end_round(game):
            # Headless matches immediately replace the terminal characters.
            if game.is_planted:
                audit.record_plant(game)
                audit.ends.append(dict(round=game.current_round,defused=bool(game.is_defused),
                    unused={a:sum(getattr(c,a.lower()+'_charges',0) for c in game.chars if c.team=='A')
                            for a in ABILITIES}))
            return original_end(game)

        def choose(controller,char,observation,mask,context):
            action = original_choose(controller,char,observation,mask,context)
            audit.record_decision(controller,char,observation,mask,context,action)
            return action

        def cast(game,owner,payload):
            guard = bool(game.is_planted and owner.team=='A')
            counter = guard and (owner.blind_remaining>0 or owner.reveal_remaining>0
                                or getattr(owner,'electric_remaining',0)>0)
            ability = payload.get('ability','').upper() if isinstance(payload,dict) else ''
            result = original_cast(game,owner,payload)
            if guard and ability in ABILITIES:
                audit.counts['attempt:'+ability] += 1
                audit.counts[('cast:' if result else 'rejected:')+ability] += 1
                if counter and result:
                    audit.counts['counter_cast:'+ability] += 1
                if result:
                    decision = audit.last_decisions.get((game.current_round,owner.name),{})
                    audit.casts.append(dict(round=game.current_round,tick=game.battle_tick,
                        after_plant_ticks=game.battle_tick-audit.plant_ticks.get(game.current_round,game.battle_tick),
                        actor=owner.name,ability=ability,target=payload.get('target'),
                        counter=counter,decision=decision))
                if result and ability in ('FLASH','RECON'):
                    projectiles = getattr(game,ability.lower()+'_projectiles')
                    if projectiles and projectiles[-1].get('owner') == owner.name:
                        projectiles[-1]['guard_audit_counter'] = counter
                        projectiles[-1]['guard_audit_cast'] = True
            return result

        def explode(original,ability,game,projectile,impact):
            previous = game.analytics_tracker
            hit_count = [0]
            class HitRecorder:
                def record_contribution(self,owner,target,tick,kind):
                    if game.is_planted:
                        if projectile.get('team')=='D':
                            audit.counts['incoming_hit_targets:'+ability] += 1
                        elif projectile.get('guard_audit_cast'):
                            hit_count[0] += 1
                            audit.counts['hit_targets:'+ability] += 1
                            if projectile.get('guard_audit_counter'):
                                audit.counts['counter_hit_targets:'+ability] += 1
                    if previous is not None:
                        previous.record_contribution(owner,target,tick,kind)
            game.analytics_tracker = HitRecorder()
            try:
                result = original(game,projectile,impact)
                if hit_count[0]:
                    audit.counts['hit_casts:'+ability] += 1
                    if projectile.get('guard_audit_counter'):
                        audit.counts['counter_hit_casts:'+ability] += 1
                return result
            finally:
                game.analytics_tracker = previous

        with patch.object(ConconGuardController,'choose_action',choose), \
             patch.object(BattleLogicMixin,'check_match_winner',end_round), \
             patch.object(AbilityLosMixin,'execute_ai_ability',cast), \
             patch.object(AbilityLosMixin,'_explode_flash',
                          lambda game,projectile,impact=None:explode(original_flash,'FLASH',game,projectile,impact)), \
             patch.object(AbilityLosMixin,'_explode_recon',
                          lambda game,projectile,impact=None:explode(original_recon,'RECON',game,projectile,impact)):
            yield


def diagnose_arm(task):
    arm,matches,rounds,opponent,seed,model_path = task
    audit = GuardAudit()
    results = []
    for trial in range(matches):
        audit.plant_ticks.clear()
        audit.recorded_plants.clear()
        audit.last_decisions.clear()
        advisor = FixedAbilityAdvisor(model_path if arm=='advised' else None)
        if arm=='advised' and advisor.model is None:
            raise ValueError('adviser model could not be loaded')
        with audit.instrument():
            results.append(play_match(opponent,seed+trial,rounds,advisor,MAPS,
                                      tick_observer=audit.observe_tick))
        print(f'diagnose {opponent} {arm} match={trial+1}/{matches}',flush=True)
    result = dict(matches=results,counts=dict(audit.counts),
        scopes={k:dict(v) for k,v in audit.scopes.items()},examples=audit.examples,
        plants=audit.plants,ends=audit.ends,casts=audit.casts)
    return arm,result


def diagnose(matches=DEFAULT_MATCHES,rounds=DEFAULT_ROUNDS,opponent=DEFAULT_OPPONENT,
             seed=DEFAULT_SEED,model_path=DEFAULT_MODEL,output=DEFAULT_OUTPUT,workers=DEFAULT_WORKERS):
    if matches < 1 or not 1<=rounds<=12:
        raise ValueError('positive matches and 1..12 rounds required')
    report = dict(opponent=opponent,seed=seed,model_sha256=sha256_file(model_path),
                  guard_sha256={site:sha256_file(get_scenario(site).model_path) for site in ('L','R')},arms={})
    tasks = [(arm,matches,rounds,opponent,seed,Path(model_path).resolve()) for arm in ('baseline','advised')]
    report['arms'] = dict(run_opponent_tasks(diagnose_arm,tasks,workers))
    if sha256_file(model_path)!=report['model_sha256']:
        raise RuntimeError('adviser weights changed during diagnosis')
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'report={output}',flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches',type=int,default=DEFAULT_MATCHES)
    parser.add_argument('--rounds',type=int,default=DEFAULT_ROUNDS)
    parser.add_argument('--opponent',default=DEFAULT_OPPONENT)
    parser.add_argument('--seed',type=int,default=DEFAULT_SEED)
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--workers',type=int,default=DEFAULT_WORKERS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    diagnose(args.matches,args.rounds,args.opponent,args.seed,args.model,args.output,args.workers)


if __name__=='__main__':
    main()
