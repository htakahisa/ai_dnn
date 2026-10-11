"""Aggregate existing controlled-route and paired teacher logs; no simulation or learning."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
LOG_DIRECTORY = HERE / 'logs' / 'attacker_approach_review'
ROUTE_PREFIXES = ('routes_042', 'routes_142', 'routes_242')
TEACHER_PAIRS = (
    ('exposure_control_r10_teacher_v11', 'exposure_anticipate_teacher_exposure'),
    ('exposure_control_r6_teacher_v11', 'exposure_anticipate_r6_teacher_exposure'),
    ('exposure_control_r8_teacher_v11', 'exposure_anticipate_r8_teacher_exposure'),
    ('exposure_control_042_teacher_v11', 'exposure_anticipate_042_teacher_exposure'),
)
OUTPUT_FILENAME = 'approach_findings.json'
EXPOSURE_TRIAL_PREFIXES = (
    'support_teacher_scoped', 'exposure_teacher_36', 'exposure_bounded_pilot',
    'exposure_no_wait_pilot', 'exposure_before_plan_pilot', 'exposure_geometry_pilot',
    'exposure_retreat_pilot', 'exposure_asymmetric_36', 'exposure_deadline_36',
)
EXPOSURE_TRIAL_DIRECTORY = HERE / 'logs' / 'attacker_right_review'


def read(prefix):
    return json.loads((LOG_DIRECTORY / f'{prefix}_summary.json').read_text(encoding='utf-8'))


def outcome(result):
    r = result['record']
    return dict(reason=r['reason'], planted=r['planted'], tick=r['tick'],
                alive=r['alive'], remaining_hp=r['remaining_hp'],
                actual_branches=r['actual_branches'], actual_site_entries=r['actual_site_entries'],
                shot_labels=result['shot_labels'])


def main():
    sources = [read(p + '_learned') for p in ROUTE_PREFIXES]
    reference = sources[0]
    grouped = {}
    for data in sources:
        for field in ('source_hashes', 'scenario', 'mode', 'target_round', 'optimizer_updates'):
            if data[field] != reference[field]:
                raise RuntimeError(f'Route comparison differs: {field}')
        initial = None
        for result in data['results']:
            initial = initial or result['initial_state_hash']
            if initial != result['initial_state_hash']:
                raise RuntimeError('Route comparison initial state/RNG differs')
            index = result['route_index']
            key = 'natural' if index is None else str(index)
            grouped.setdefault(key, []).append(dict(seed=result['seed'],
                requested_branches=result['requested_branches'], **outcome(result)))
    routes = {key:dict(planted=sum(r['planted'] for r in rows), count=len(rows), cases=rows)
              for key, rows in grouped.items()}
    pairs = []
    for before_prefix, after_prefix in TEACHER_PAIRS:
        before, after = read(before_prefix), read(after_prefix)
        for field in ('source_hashes', 'opponent_hashes', 'scenario', 'target_round', 'optimizer_updates'):
            if before[field] != after[field]:
                raise RuntimeError(f'Teacher comparison differs: {field}')
        a, b = before['results'][0], after['results'][0]
        for field in ('seed', 'route_index', 'initial_state_hash'):
            if a[field] != b[field]:
                raise RuntimeError(f'Teacher comparison differs: {field}')
        pairs.append(dict(seed=a['seed'], round=before['target_round'],
                          initial_state_hash=a['initial_state_hash'],
                          before_prefix=before_prefix, after_prefix=after_prefix,
                          before=outcome(a), after=outcome(b)))
    trials=[]
    for prefix in EXPOSURE_TRIAL_PREFIXES:
        data=json.loads((EXPOSURE_TRIAL_DIRECTORY/f'{prefix}_summary.json').read_text(encoding='utf-8'))
        if data['source_hashes'] != reference['source_hashes'] or data['scenario'] != reference['scenario']:
            raise RuntimeError('Exposure trial model/scenario differs')
        records=data['rounds']
        trials.append(dict(prefix=prefix,rounds=len(records),plants=sum(r['planted'] for r in records),
            timeouts=sum(r['reason']=='timeout' for r in records),
            eliminations=sum(r['reason']=='attacker_eliminated' for r in records),
            exposure_probe_hash=data.get('exposure_probe_hash'),
            seed_804000042_plants=sum(r['planted'] for r in records if r['seed']==804000042),
            by_target_site={site:dict(rounds=sum(r['target_site']==site for r in records),
                plants=sum(r['target_site']==site and r['planted'] for r in records)) for site in ('L','R')}))
    output = dict(scope='diagnostic_geometry_and_teacher_interventions_not_learned_win_rates',
                  optimizer_updates=0, source_hashes=reference['source_hashes'],
                  scenario=reference['scenario'], routes=routes, teacher_pairs=pairs,
                  historical_teacher_pair_timing='after_first_plan_with_repreparation; not adoption evidence',
                  exposure_trials=trials,
                  unresolved=['Learned observation/action representation for exposure and detours',
                              'Broader regression and right-site success evaluation',
                              'Counterfactual route labels and analysis calibration'])
    path = LOG_DIRECTORY / OUTPUT_FILENAME
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    for key, data in routes.items():
        print('Route', key, 'plants', data['planted'], '/', data['count'])
    for pair in pairs:
        print(pair['seed'], pair['round'], pair['before']['reason'], '->', pair['after']['reason'])
    print('Saved:', path)


if __name__ == '__main__':
    main()
