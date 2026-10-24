"""Audit physical entrance connectivity without traversing plant floors; no learning."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SITE = 'R'
ORIGIN_BRANCHES = ('b', 'd', 'h')
REMAINING_TICKS = 67
KNOWN_ENEMY_POSITIONS = ((2, 40),)  # Fixed public contact from the recorded tick33 case.
OUTPUT_PATH = HERE/'logs'/'attacker_route_risk_review'/'entry_connectivity.json'

import json
from game_core import PLANT_REQUIRED_TICKS
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_learn_attacker_analysis import shortest_path, branch_sequence, TIME_MARGIN_TICKS


def entrance_audit(scenario, origin, outside, site, timer, threats=()):
    blocked = set(scenario.sites['L']) | set(scenario.sites['R'])
    approach = shortest_path(scenario, origin, (outside,), blocked)
    unrestricted = shortest_path(scenario, origin, (outside,))
    entry = shortest_path(scenario, outside, scenario.sites[site],
                          scenario.sites['R' if site == 'L' else 'L'])
    reachable = bool(approach and entry)
    steps = len(approach)-1 if approach else None
    minimum = steps + len(entry)-1 + PLANT_REQUIRED_TICKS if reachable else None
    return dict(origin=origin, outside=outside, approach=approach, entry=entry,
                reachable_outside_sites=reachable, approach_steps=steps,
                unrestricted_steps=len(unrestricted)-1 if unrestricted else None,
                unrestricted_crosses_plant_floor=bool(set(unrestricted) & blocked),
                branches=branch_sequence(scenario, approach),
                minimum_completion_ticks=minimum,
                time_feasible=reachable and minimum + TIME_MARGIN_TICKS <= timer,
                known_wall_exposure=[p for p in approach if any(scenario.clear(p,t) for t in threats)],
                enemy_positions=threats)


def main():
    scenario = Scenario()
    rows = []
    for branch in ORIGIN_BRANCHES:
        cells = scenario.branches[branch]
        origin = cells[len(cells)//2]
        for marker, positions in scenario.retake_entries[SITE].items():
            for outside in positions:
                rows.append(dict(branch=branch, entrance_marker=marker.upper(),
                    **entrance_audit(scenario, origin, outside, SITE, REMAINING_TICKS, KNOWN_ENEMY_POSITIONS)))
    result = dict(scope='static_connectivity_not_policy_or_success_prediction', site=SITE,
                  scenario=scenario.signature, remaining_ticks=REMAINING_TICKS,
                  rows=rows, optimizer_updates=0)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for row in rows:
        print(row['branch'], row['entrance_marker'], 'steps=', row['approach_steps'],
              'reachable=', row['reachable_outside_sites'],
              'exposed_cells=', len(row['known_wall_exposure']))
    print('Saved:', OUTPUT_PATH)


if __name__ == '__main__':
    main()
