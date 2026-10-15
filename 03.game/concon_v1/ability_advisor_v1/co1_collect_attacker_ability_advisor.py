"""Collect frozen A1-A4 games with IQ inputs, five-round history, and teacher labels."""

import argparse
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import uuid
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from concon_v1.ability_advisor_v1.co1_learn_attacker_ability_advisor import (
    ROOT, MAPS, OPPONENTS, VERSION, LABEL_VERSION, WALL_LOS_VERSION, Advice,
    FixedAbilityAdvisor, build_features, feature_names, group_key, impact_position,
    relevant_enemy_positions, sha256_file,
)
from concon_v1.ability_advisor_v1.co1_opponent_processes import run_opponent_tasks

DEFAULT_MATCHES_PER_OPPONENT = 40
DEFAULT_ROUNDS_PER_MATCH = 12  # Stop before side swap: preserve Gorigons' attacking role.
DEFAULT_OPPONENTS = OPPONENTS
DEFAULT_MAPS = MAPS
DEFAULT_SEED = 0
DEFAULT_OUTPUT_DIR = ROOT / 'data' / 'datasets'
DEFAULT_WORKERS = 6  # One independent process per opponent; 1 runs serially.
MAX_STEPS_PER_ROUND = 500


def route_metadata(map_names=MAPS):
    from concon_v1.co1_attacker_scenarios import get_scenario
    result = {}
    for name in map_names:
        scenario = get_scenario(name)
        path = scenario.model_path
        if not path.is_file():
            path = scenario.save_dir / scenario.checkpoint_filename('latest')
        result[name] = dict(signature=scenario.signature, carry_sha256=sha256_file(path))
    return result


class RecordingAdvisor(FixedAbilityAdvisor):
    """Teacher state is isolated here; it never enters features or match memory."""
    def __init__(self, stream, dataset_id, match_id, opponent):
        super().__init__(model_path=None)
        self.stream,self.dataset_id,self.match_id,self.opponent = stream,dataset_id,match_id,opponent
        self.teacher_game = None
        self.groups = Counter()

    def advise(self,scenario,char,state,opponent,ability,point,aim,game):
        impact = impact_position(char,ability,aim,game)
        features,_ = build_features(scenario,char,state,self.opponent,ability,point,impact,self.memory)
        teacher = self.teacher_game
        label = bool(relevant_enemy_positions(teacher.grid,teacher.chars,char.team,
                                              ability,impact,teacher._smoke_cells()))
        key = group_key(scenario,self.opponent,ability,point)
        row = dict(event='sample',dataset_id=self.dataset_id,match_id=self.match_id,
                   round_number=teacher.current_round,group=key,map_name=scenario.map_name,
                   opponent=self.opponent,ability=ability,point=list(point),
                   tick=int(state.get('battle_tick',0)),features=features.tolist(),label=int(label))
        self.stream.write(json.dumps(row,separators=(',',':'))+'\n')
        self.groups[key] += 1
        self.decisions['use'] += 1
        # Always use the existing fixed plan while collecting its baseline.
        self.pending[char.name] = (ability,tuple(point),getattr(char,ability.lower()+'_charges',0),row['tick'])
        return Advice(True,'collection_baseline')


def play_match(opponent, seed, rounds, advisor, map_names=MAPS):
    from concon_v1.co1_battle_training import OPPONENTS as TEAMS, _run_from_project_root

    @_run_from_project_root
    def run():
        with contextlib.redirect_stdout(io.StringIO()):
            from run_game import VisualFPSBattle, _build_team_ai
        from team_ai import DualRoleTeamAI
        from controllers import DefaultDefenderController
        from party_presets import get_preset
        from concon_v1.co1_attacker_controller import ConconRoundAttackerController
        from concon_v1.co1_attacker_scenarios import GAME_MAZE_STR, CONCON_ATTACKER_POSTPLANT_MODELS
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        schedule = list(map_names)
        random.Random(seed+104729).shuffle(schedule)
        attackers,defenders = get_preset('Gorigons'),get_preset(TEAMS[opponent][1])
        attacker_ai = DualRoleTeamAI('ConCon ability collection',
            attacker_factory=lambda:ConconRoundAttackerController(
                map_names=map_names,seed=seed,selection_path=None,route_schedule=schedule,
                postplant_factories=CONCON_ATTACKER_POSTPLANT_MODELS,ability_advisor=advisor),
            defender_factory=DefaultDefenderController,use_iq_perception=True)
        with contextlib.redirect_stdout(io.StringIO()):
            game = VisualFPSBattle(GAME_MAZE_STR,attacker_ai,_build_team_ai(TEAMS[opponent][0]),
                headless=True,attacker_roster=list(attackers.players),defender_roster=list(defenders.players),
                spike_holder_name=attackers.spike_holder,defender_spike_holder_name=defenders.spike_holder,
                attacker_igl_name=attackers.igl,defender_igl_name=defenders.igl,
                attacker_team_name=attackers.name,defender_team_name=defenders.name,disable_side_swap=True)
            game.analytics_tracker = None
            if isinstance(advisor,RecordingAdvisor):
                advisor.teacher_game = game
            plants = []
            for _ in range(rounds*MAX_STEPS_PER_ROUND):
                if game.current_round > rounds or game.match_over:
                    break
                was_planted = game.is_planted
                game.step_tick()
                if game.is_planted and not was_planted:
                    plants.append({ability:sum(getattr(c,ability.lower()+'_charges',0)
                                               for c in game.chars if c.team == 'A' and c.is_alive)
                                   for ability in ('SMOKE','FLASH','RECON')})
            else:
                raise RuntimeError('match exceeded tick limit')
        return dict(attacker_wins=game.attacker_wins,defender_wins=game.defender_wins,
                    rounds=game.attacker_wins+game.defender_wins,
                    plants=len(plants),charges_at_plant=plants,
                    decisions=advisor.decisions,history_rounds=len(advisor.memory.rounds))
    return run()


def collect_opponent(task):
    opponent,index,opponent_count,matches,rounds,map_names,seed,metadata,path = task
    counts = Counter()
    with Path(path).open('x',encoding='utf-8') as stream:
        for match in range(matches):
            match_seed = seed+match*opponent_count+index
            match_id = f'{opponent}:seed={match_seed}:maps={",".join(map_names)}'
            advisor = RecordingAdvisor(stream,metadata['dataset_id'],match_id,opponent)
            result = play_match(opponent,match_seed,rounds,advisor,map_names)
            if any(advisor.route_hashes.get(n) != metadata['scenarios'][n]['carry_sha256'] for n in map_names):
                raise RuntimeError('carry weights changed during collection; collect a new dataset')
            counts.update(advisor.groups)
            stream.flush()
            print(f'collect {opponent} match={match+1}/{matches} samples={sum(advisor.groups.values())} '
                  f'wins={result["attacker_wins"]}/{result["rounds"]}',flush=True)
    return str(path),counts


def collect(matches=DEFAULT_MATCHES_PER_OPPONENT,rounds=DEFAULT_ROUNDS_PER_MATCH,
            opponents=DEFAULT_OPPONENTS,map_names=DEFAULT_MAPS,seed=DEFAULT_SEED,
            output_dir=DEFAULT_OUTPUT_DIR,workers=DEFAULT_WORKERS):
    if workers < 1:
        raise ValueError('workers must be positive')
    if matches < 1 or not 1 <= rounds <= 12:
        raise ValueError('positive match count and 1..12 attacking rounds required')
    if not opponents or any(n not in OPPONENTS for n in opponents):
        raise ValueError('select known opponents')
    if not map_names or any(n not in MAPS for n in map_names):
        raise ValueError('select A1-A4 routes')
    dataset_id = uuid.uuid4().hex
    directory = Path(output_dir)
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/(dataset_id+'.jsonl')
    metadata = dict(event='metadata',version=VERSION,label_version=LABEL_VERSION,
                    wall_los_version=WALL_LOS_VERSION,feature_names=feature_names(),
                    scenarios=route_metadata(map_names),dataset_id=dataset_id,
                    matches_per_opponent=matches,rounds_per_match=rounds,seed=seed)
    parts = directory/(dataset_id+'.parts')
    parts.mkdir()
    tasks = [(opponent,index,len(opponents),matches,rounds,tuple(map_names),seed,
              metadata,parts/f'{index}.part') for index,opponent in enumerate(opponents)]
    # Workers only write their own shards. The parent publishes one complete dataset.
    results = run_opponent_tasks(collect_opponent,tasks,workers)
    counts = Counter()
    temporary = directory/(dataset_id+'.jsonl.partial')
    with temporary.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(metadata)+'\n')
        for shard,groups in results:
            counts.update(groups)
            with Path(shard).open(encoding='utf-8') as source:
                shutil.copyfileobj(source,stream)
        stream.write(json.dumps(dict(event='complete',samples=sum(counts.values())))+'\n')
    temporary.replace(path)
    for shard,_ in results:
        Path(shard).unlink()
    parts.rmdir()
    print(f'dataset={path} samples={sum(counts.values())} groups={len(counts)}',flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches',type=int,default=DEFAULT_MATCHES_PER_OPPONENT)
    parser.add_argument('--rounds',type=int,default=DEFAULT_ROUNDS_PER_MATCH)
    parser.add_argument('--opponents',nargs='+',choices=OPPONENTS,default=DEFAULT_OPPONENTS)
    parser.add_argument('--maps',nargs='+',choices=MAPS,default=DEFAULT_MAPS)
    parser.add_argument('--seed',type=int,default=DEFAULT_SEED)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--workers',type=int,default=DEFAULT_WORKERS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    collect(args.matches,args.rounds,args.opponents,args.maps,args.seed,args.output_dir,args.workers)


if __name__ == '__main__':
    main()
