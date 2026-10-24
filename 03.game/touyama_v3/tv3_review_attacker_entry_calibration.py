"""Compare first public-contact route predictions with actual round completion; no learning."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))

INPUT_SUMMARY=HERE/'logs'/'attacker_right_review'/'baseline_summary.json'
OUTPUT_PATH=HERE/'logs'/'attacker_route_risk_review'/'entry_prediction_calibration.json'
SITE='R'
OUTER_ENTRY=(11,40)

import json
from collections import defaultdict
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_learn_attacker_analysis import route_entry_positions,Route


def main():
    baseline=json.loads(INPUT_SUMMARY.read_text(encoding='utf-8'))
    scenario=Scenario()
    terminal={(r['seed'],r['round']):r for r in baseline['rounds']}
    first_contact={}
    unmatched=[]
    for line in Path(baseline['trace']).open(encoding='utf-8'):
        row=json.loads(line)
        if row['event']!='decision' or not row.get('sightings') or not row.get('route_candidates'):
            continue
        key=row['seed'],row['round']
        if key in first_contact:
            continue
        holder=next((a for a in row['actors'] if a['carrier']),None)
        if holder is None:
            continue
        path=row['route']
        if holder['position'] not in path:
            unmatched.append(dict(seed=key[0],round=key[1],tick=row['tick'],reason='carrier_off_selected_path'))
            continue
        suffix=path[path.index(holder['position']):]
        matches=[c for c in row['route_candidates'] if c['cells']==suffix and c['site']==row['site']]
        if not matches:
            unmatched.append(dict(seed=key[0],round=key[1],tick=row['tick'],reason='selected_suffix_not_scored'))
            continue
        score=row['route_decision']['score']
        chosen=min(matches,key=lambda c:abs(c['score']-score))
        if abs(chosen['score']-score)>1e-5:
            unmatched.append(dict(seed=key[0],round=key[1],tick=row['tick'],reason='cannot_identify_selected_mode'))
            continue
        outside,inside=route_entry_positions(scenario,Route(row['site'],tuple(map(tuple,suffix)),()))
        outcome=terminal[key]
        first_contact[key]=dict(seed=key[0],round=key[1],tick=row['tick'],site=row['site'],
            outside=outside,inside=inside,mode=chosen['mode'],
            predicted_success=chosen['plant_probability'],score=chosen['score'],
            any_site_success=outcome['planted'],selected_site_success=outcome['planted'] and outcome['site']==row['site'],
            reason=outcome['reason'],alive=outcome['alive'],remaining_hp=outcome['remaining_hp'])
    groups=defaultdict(list)
    for row in first_contact.values():
        if row['site']==SITE:
            groups['right_contact'].append(row)
            groups['outer_C_contact' if tuple(row['outside'])==OUTER_ENTRY else 'other_right_entry_contact'].append(row)
            if tuple(row['outside'])==OUTER_ENTRY and row['predicted_success']>=.9:
                groups['outer_C_high_confidence_contact'].append(row)
    summaries={name:dict(rounds=len(rows),mean_prediction=sum(r['predicted_success'] for r in rows)/len(rows),
        any_site_plants=sum(r['any_site_success'] for r in rows),same_site_plants=sum(r['selected_site_success'] for r in rows),
        brier=sum((r['predicted_success']-r['any_site_success'])**2 for r in rows)/len(rows))
        for name,rows in groups.items()}
    output=dict(scope='first_scored_public_contact_per_round_adaptive_policy_not_fixed_entry_success',
        source_hashes=baseline['source_hashes'],groups=summaries,rows=list(first_contact.values()),
        unmatched=unmatched,optimizer_updates=0)
    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2))
    print('Matched rounds:',len(first_contact),'unmatched decision points:',len(unmatched))
    print('Saved:',OUTPUT_PATH)


if __name__=='__main__':
    main()
