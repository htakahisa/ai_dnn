"""Find covered approaches and corner utility using saved public knowledge; never select live actions."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))

INPUT_TRACE=HERE/'logs'/'attacker_route_risk_review'/'frozen_804000042_r6_True.jsonl'
CASE_TICK=33
SITE='R'
STAGING_DISTANCE=3
OUTPUT_PATH=HERE/'logs'/'attacker_route_risk_review'/'safe_entry_33.json'
MAX_RESULTS=12

import json
from collections import deque
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_learn_attacker_analysis import branch_sequence,TIME_MARGIN_TICKS
from touyama_v3.tv3_attacker_entry_utility import projectile_impact
from touyama_v3.tv3_review_attacker_right import json_default
from game_core import PLANT_REQUIRED_TICKS,FLASH_MAX_FLIGHT_TICKS


def covered_paths(scenario,origin,site,threat_positions,staging_distance=STAGING_DISTANCE):
    """Wall cover only: do not rely on transient smoke, ally bodies, or enemy facing."""
    blocked=set(scenario.sites['L'])|set(scenario.sites['R'])
    exposure={}
    def unsafe(p):
        if p not in exposure:
            exposure[p]=any(scenario.clear(p,t) for t in threat_positions)
        return exposure[p]
    queue=deque([origin]); parents={origin:None}; distance={origin:0}
    while queue:
        p=queue.popleft()
        for q in scenario.neighbors(p):
            if q not in parents and q not in blocked and not unsafe(q):
                parents[q]=p; distance[q]=distance[p]+1; queue.append(q)
    rows=[]
    for p,steps in distance.items():
        entry_steps=int(scenario.site_dist[site][p])
        if p==origin or not 0<entry_steps<=staging_distance:
            continue
        path=[]; cursor=p
        while cursor is not None:
            path.append(cursor); cursor=parents[cursor]
        path.reverse()
        rows.append(dict(staging=p,path=tuple(path),steps=steps,entry_steps=entry_steps,
                         exposed_start=unsafe(origin),branches=branch_sequence(scenario,path)))
    return rows


def corner_flash_options(scenario,origin,site,threat_positions,smoke=()):
    """Geometric possible hits, not predictions that the enemy will remain at this position."""
    from grid_lines import line_cells
    candidates=set(scenario.sites[site])|set(threat_positions)
    results={}
    for aim in sorted(candidates):
        impact,delay=projectile_impact(scenario,origin,aim,'FLASH')
        if not 0<delay<=FLASH_MAX_FLIGHT_TICKS:
            continue
        hits=[t for t in threat_positions if scenario.clear(impact,t)
              and not set(line_cells(impact,t)).intersection(smoke)]
        if hits:
            results.setdefault((impact,delay),dict(aim=aim,impact=impact,delay=delay,covered_positions=hits))
    return sorted(results.values(),key=lambda r:(-len(r['covered_positions']),r['delay'],r['aim']))


def main():
    rows=[json.loads(line) for line in INPUT_TRACE.open(encoding='utf-8')]
    rank=next(r for r in rows if r['event']=='ranking' and r['tick']==CASE_TICK)
    decision=next(r for r in rows if r['event']=='decision' and r['tick']==CASE_TICK)
    scenario=Scenario()
    known=[tuple(t['position']) for t in rank['tracks'] if t['alive'] and t['valid_combat_memory']
           and not t.get('visible_empty',False)]
    if not known:
        raise ValueError('No valid public threat; cannot claim a known covered approach')
    holder=next(a for a in decision['actors'] if a['carrier'])
    approaches=covered_paths(scenario,tuple(holder['position']),SITE,known)
    for approach in approaches:
        approach['flash_options']=corner_flash_options(scenario,approach['staging'],SITE,known,decision.get('smoke_cells',()))
        # Optimistic lower bound: no traffic, no waiting, and the same enemy positions.
        approach['minimum_completion_ticks']=approach['steps']+approach['entry_steps']+PLANT_REQUIRED_TICKS
        approach['time_feasible_lower_bound']=approach['minimum_completion_ticks']+TIME_MARGIN_TICKS<=rank['timer']
    approaches.sort(key=lambda r:(not r['time_feasible_lower_bound'],not r['flash_options'],r['steps'],r['entry_steps'],r['staging']))
    actor_options=[]
    for actor in decision['actors']:
        if actor.get('ability')!='FLASH':
            continue
        options=corner_flash_options(scenario,tuple(actor['position']),SITE,known,decision.get('smoke_cells',()))
        offered=actor.get('ability_candidates',[])
        actor_options.append(dict(slot=actor['slot'],position=actor['position'],charges=actor.get('charges'),
            possible_corner_throws=options,currently_offered=offered))
    result=dict(scope='public_known_enemy_fixed_position_geometry_not_match_success',tick=CASE_TICK,
        source_trace=str(INPUT_TRACE),site=SITE,known_positions=known,
        known_positions_all_in_site=all(p in scenario.sites[SITE] for p in known),
        staging_candidates=len(approaches),feasible_staging_candidates=sum(r['time_feasible_lower_bound'] for r in approaches),
        approaches=approaches[:MAX_RESULTS],current_flash_actors=actor_options,
        caveats=['Unseen enemies and enemy movement remain uncertain',
                 'Completion time is a lower bound, not a promise',
                 'Future smoke, traffic and ability charges are not guaranteed'],optimizer_updates=0)
    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result,default=json_default,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Covered staging:',len(approaches),'feasible:',result['feasible_staging_candidates'])
    for row in approaches[:3]:
        print(row['staging'],'approach steps:',row['steps'],'entry steps:',row['entry_steps'],'flash options:',len(row['flash_options']))
    print('Saved:',OUTPUT_PATH)


if __name__=='__main__':
    main()
