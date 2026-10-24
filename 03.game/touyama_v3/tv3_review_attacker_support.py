"""Summarize frozen public traces and separately labelled teacher diagnostics."""
from pathlib import Path
from collections import Counter, defaultdict
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
LOG_DIRECTORY = HERE / 'logs' / 'attacker_right_review'
EVALUATION_PREFIXES = ('frozen_support_verified', 'legacy_teacher', 'support_teacher_scoped')
OUTPUT_PREFIX = 'support_findings'
CASE_SEED = 804000142
CASE_ROUND = 6


def draw_geometry(case, scenario_signature):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from touyama_v3.tv3_scenario import Scenario
    from touyama_v3.tv3_learn_attacker_analysis import shortest_path
    scenario = Scenario()
    if scenario.signature != scenario_signature:
        raise ValueError('Current geometry differs from the audited scenario')
    row = next(r for r in case if r['tick']==55)
    helper = next(a for a in row['actors'] if a['slot']==4)
    plan = helper['support_path']
    path = shortest_path(scenario,tuple(helper['position']),(tuple(plan['goal']),))
    fig,ax = plt.subplots(figsize=(7,7))
    for r in range(6,17):
        for c in range(37,44):
            color = '#344054' if scenario.grid[r,c]==1 else '#dbeafe' if scenario.grid[r,c]==2 else '#f8fafc'
            ax.add_patch(Rectangle((c-.5,r-.5),1,1,facecolor=color,edgecolor='#cbd5e1',linewidth=.6))
    start,target = helper['position'],plan['target']
    ax.plot([start[1],target[1]],[start[0],target[0]],'--',color='#dc2626',label='Current helper line: wall blocked')
    ax.plot([p[1] for p in path],[p[0] for p in path],color='#16a34a',linewidth=3,label=f"Support path: {plan['steps']} steps")
    ax.plot([plan['goal'][1],target[1]],[plan['goal'][0],target[0]],':',color='#16a34a',linewidth=2,label='Reachable helper firing line')
    for c,r,label,color in [(target[1],target[0],'Public enemy','#dc2626'),(start[1],start[0],'Helper','#16a34a'),
                             (40,12,'Carrier cover','#2563eb'),(40,11,'Repeated peek','#2563eb')]:
        ax.scatter([c],[r],s=90,color=color,zorder=5)
        ax.annotate(label,(c,r),xytext=(8,8),textcoords='offset points',fontsize=9)
    ax.annotate('',xy=(40,11),xytext=(40,12),arrowprops=dict(arrowstyle='<->',color='#2563eb',linewidth=2))
    ax.set(xlim=(36.5,45),ylim=(16.5,5.5),xlabel='Column',ylabel='Row',
           title=f'Seed {CASE_SEED} / round {CASE_ROUND} / tick 55\nPublic support geometry')
    ax.set_aspect('equal')
    ax.legend(loc='lower left',fontsize=8)
    fig.tight_layout()
    fig.savefig(LOG_DIRECTORY/'support_geometry.png',dpi=180)
    plt.close(fig)


def analyze_trace(path):
    counts = Counter()
    examples = []
    edge_counts = defaultdict(Counter)
    for line in path.open(encoding='utf-8'):
        row = json.loads(line)
        key = row['seed'], row['round']
        if row['event'] == 'executed':
            for actor in row['actors']:
                if actor['moved']:
                    edge = tuple(sorted((tuple(actor['before']), tuple(actor['after']))))
                    edge_counts[key+(actor['slot'],)][edge] += 1
            continue
        carrier = next((a for a in row['actors'] if a['carrier']), None)
        helpers = [a for a in row['actors'] if a.get('support_path') and a['support_path']['steps'] > 0]
        counts['helper_stay_with_reachable_support_path'] += sum(a['action']=='STAY' for a in helpers)
        if carrier:
            counts['unsupported_carrier_retreat'] += int(carrier['combat']=='cover_retreat' and carrier['supporters']==0)
            counts['unsupported_carrier_retreat_with_reachable_helper'] += int(
                carrier['combat']=='cover_retreat' and carrier['supporters']==0 and bool(helpers))
            for move in carrier['moves']:
                counts['carrier_edge_mask_ticks'] += int('recent_edge_limit' in move.get('excluded_by',()))
                if move.get('unfiltered_q') is not None and carrier['q'] is not None:
                    counts['carrier_preferred_move_removed_by_edge_mask'] += int(
                        'recent_edge_limit' in move.get('excluded_by',()) and move['unfiltered_q'] > carrier['q'])
        if key == (CASE_SEED,CASE_ROUND) and row['tick'] in (54,55,56,62,63,94,95):
            examples.append(dict(tick=row['tick'], site=row['site'], planner_progress_tick=row.get('planner_progress_tick'),
                actors=[{k:a[k] for k in ('slot','position','goal','action','teacher','combat','supporters','support_path','shot_geometry')}
                        for a in row['actors'] if a['carrier'] or a['slot']==4]))
    repeated = [dict(seed=key[0],round=key[1],slot=key[2],edge=edge,traversals=n)
                for key,edges in edge_counts.items() for edge,n in edges.items() if n >= 6]
    return dict(counts=dict(counts), repeated_edges=repeated, case=examples)


def main():
    reports = []
    reference = None
    for prefix in EVALUATION_PREFIXES:
        summary_path = LOG_DIRECTORY / (prefix+'_summary.json')
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        provenance = summary['source_hashes'], summary['scenario'], summary['seeds']
        if reference is not None and provenance != reference:
            raise ValueError('Evaluation sources/scenario/seeds differ; cannot compare these runs')
        reference = provenance
        metrics = summary['evaluation']
        reports.append(dict(prefix=prefix,scope=summary['scope'],plant_version=summary['plant_version'],
            optimizer_updates=summary['optimizer_updates'],teacher_probability=summary['teacher_probability'],
            plants=metrics['plants'],rounds=metrics['rounds'],timeouts=metrics['timeouts'],
            outcomes=dict(Counter(r['reason'] for r in summary['rounds'])),
            by_final_target={site:{k:metrics['by_site'][site][k] for k in ('rounds','plants','plant_rate')}
                             for site in ('L','R')},
            initial_right_routes=dict(Counter('/'.join(r['route_decisions'][0]['branches'])
                for r in summary['rounds'] if r['route_decisions'][0]['site']=='R')),
            trace_analysis=analyze_trace(Path(summary['trace']))))
    output = LOG_DIRECTORY / (OUTPUT_PREFIX+'.json')
    output.write_text(json.dumps(dict(source_hashes=reference[0],scenario=reference[1],seeds=reference[2],
        warning='Teacher diagnostics are not trained-model performance; final target site is not initial commitment.',
        runs=reports),ensure_ascii=False,indent=2),encoding='utf-8')
    draw_geometry(reports[0]['trace_analysis']['case'],reference[1])
    for report in reports:
        print(report['prefix'],report['scope'],f"plants={report['plants']}/{report['rounds']}",report['outcomes'])
    print('Saved:',output)


if __name__ == '__main__':
    main()
