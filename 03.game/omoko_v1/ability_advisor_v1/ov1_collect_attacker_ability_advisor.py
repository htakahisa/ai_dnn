"""Collect frozen Omoko carry/escort games with IQ inputs, five-round history, and teacher labels."""

import argparse
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from omoko_v1.ability_advisor_v1.ov1_learn_attacker_ability_advisor import (
    ROOT, MAPS, OPPONENTS, VERSION, LABEL_VERSION, WALL_LOS_VERSION, Advice,
    FixedAbilityAdvisor, build_features, feature_names, group_key, impact_position,
    relevant_enemy_positions, sha256_file,
)
from omoko_v1.ability_advisor_v1.ov1_opponent_processes import run_opponent_tasks

DEFAULT_MATCHES_PER_OPPONENT = 40
DEFAULT_ROUNDS_PER_MATCH = 12  # Stop before side swap: preserve Omoko' attacking role.
DEFAULT_OPPONENTS = OPPONENTS
DEFAULT_MAPS = MAPS
DEFAULT_SEED = 0
DEFAULT_OUTPUT_DIR = ROOT / 'data' / 'datasets'
DEFAULT_WORKERS = 6  # One independent process per opponent; 1 runs serially.
DEFAULT_DELETE_INCOMPATIBLE_DATASETS = True
MAX_DATASET_FILE_BYTES = 80_000_000  # Decimal bytes; applies to worker and published files.
MAX_STEPS_PER_ROUND = 500


class CappedJsonlWriter:
    """Rotate UTF-8 records before the byte limit, including metadata and footer."""
    def __init__(self, directory, prefix, suffix, metadata=None, max_bytes=None):
        self.directory, self.prefix, self.suffix = Path(directory), prefix, suffix
        self.metadata = metadata
        self.max_bytes = MAX_DATASET_FILE_BYTES if max_bytes is None else max_bytes
        self.paths, self.stream = [], None
        self.header = (json.dumps(metadata) + '\n') if metadata is not None else ''
        self.reserve = 100 if metadata is not None else 0
        self.size, self.samples = 0, 0

    def __enter__(self):
        return self

    def _finish(self, complete=True):
        if self.stream is not None:
            if self.metadata is not None and complete:
                self.stream.write(json.dumps(dict(event='complete', samples=self.samples)) + '\n')
            self.stream.close()
            self.stream = None

    def write(self, record):
        for line in record.splitlines(keepends=True):
            if not line.endswith('\n'):
                raise ValueError('dataset records must end with a newline')
            length = len(line.encode('utf-8'))
            header_size = len(self.header.encode('utf-8'))
            if header_size + length + self.reserve > self.max_bytes:
                raise ValueError('one dataset record exceeds the file size limit')
            if self.stream is not None and self.size + length + self.reserve > self.max_bytes:
                self._finish()
            if self.stream is None:
                path = self.directory / f'{self.prefix}_{len(self.paths):04d}{self.suffix}'
                self.stream = path.open('x', encoding='utf-8', newline='')
                self.paths.append(path)
                self.stream.write(self.header)
                self.size, self.samples = header_size, 0
            self.stream.write(line)
            self.size += length
            self.samples += 1

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def __exit__(self, exc_type, exc, traceback):
        self._finish(complete=exc_type is None)


def route_metadata(map_names=MAPS):
    from omoko_v1.ability_advisor_v1.ov1_advisor_attacker_context import (
        scenario, policy_digest, DEFAULT_POLICY_PATHS,
    )
    digest = policy_digest(DEFAULT_POLICY_PATHS)
    contexts = {name: scenario(name) for name in map_names}
    return {name: dict(signature=context.signature, carry_sha256=digest,
                       manifest=context.manifest)
            for name, context in contexts.items()}


def delete_incompatible_datasets(directory):
    """Delete obsolete collection files, retaining compatible data and other files."""
    if not DEFAULT_DELETE_INCOMPATIBLE_DATASETS:
        return []
    current = route_metadata()
    deleted = []
    for path in sorted(Path(directory).glob('*.jsonl')):
        # Never follow a link outside the explicitly selected dataset directory.
        if path.is_symlink() or not path.is_file():
            continue
        with path.open(encoding='utf-8') as stream:
            try:
                header = json.loads(stream.readline())
            except ValueError:
                continue
        if not isinstance(header, dict) or header.get('event') != 'metadata':
            continue
        scenarios = header.get('scenarios')
        contract_changed = (
            header.get('version') != VERSION
            or header.get('label_version') != LABEL_VERSION
            or header.get('wall_los_version') != WALL_LOS_VERSION
            or header.get('feature_names') != feature_names()
        )
        scenario_changed = (
            not isinstance(scenarios, dict) or not scenarios
            or any(name not in current or metadata != current[name]
                   for name, metadata in scenarios.items())
        )
        if contract_changed or scenario_changed:
            path.unlink()
            deleted.append(path)
            reason = 'feature/label contract changed' if contract_changed else 'policy/scenario changed'
            print(f'[DELETE] incompatible dataset ({reason}): {path}', flush=True)
    return deleted


class RecordingAdvisor(FixedAbilityAdvisor):
    """Teacher state is isolated here; it never enters features or match memory."""
    def __init__(self, stream, dataset_id, match_id, opponent):
        super().__init__(model_path=None, enabled=False)
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


def play_match(opponent, seed, rounds, advisor, map_names=MAPS, *, tick_observer=None):
    from omoko_v1.ov1_evaluate_attacker_carry_routes import legacy_resources, OPPONENTS as ROSTERS
    if tuple(map_names) != MAPS:
        raise ValueError('Omoko collection uses all carry/escort site contexts together')
    with legacy_resources(), contextlib.redirect_stdout(io.StringIO()):
        from run_game import VisualFPSBattle, _build_team_ai
        from map_data import NEW_MAZE_STR
        from party_presets import get_preset
        from run_competition_manager import TeamPlayerKey, _key_for_name
        from omoko_v1.ov1_attacker_controller import Ov1AttackerController
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        our, other = get_preset('Omoko Gaming'), get_preset(ROSTERS[opponent])
        our_keys = [TeamPlayerKey(n, f'attacker:{our.name}') for n in our.players]
        other_keys = [TeamPlayerKey(n, f'defender:{other.name}') for n in other.players]
        attacker_ai = _build_team_ai('omoko_gaming_v1', device='cpu')
        attacker_ai.attacker_factory = lambda: Ov1AttackerController(device='cpu', ability_advisor=advisor)
        game = VisualFPSBattle(NEW_MAZE_STR, attacker_ai, _build_team_ai(opponent, device='cpu'),
            headless=True, attacker_roster=our_keys, defender_roster=other_keys,
            spike_holder_name=_key_for_name(our_keys, our.spike_holder),
            defender_spike_holder_name=_key_for_name(other_keys, other.spike_holder),
            attacker_igl_name=_key_for_name(our_keys, our.igl),
            defender_igl_name=_key_for_name(other_keys, other.igl),
            attacker_team_name=our.name, defender_team_name=other.name, disable_side_swap=True)
        game.analytics_tracker = None
        game.record_replay = False
        game.stop_after_round = True
        if isinstance(advisor, RecordingAdvisor):
            advisor.teacher_game = game
        plants = []
        for round_index in range(rounds):
            for _ in range(MAX_STEPS_PER_ROUND):
                was_planted = game.is_planted
                if tick_observer is not None:
                    tick_observer(game, 'before')
                game.step_tick()
                if tick_observer is not None:
                    tick_observer(game, 'after')
                if game.is_planted and not was_planted:
                    plants.append({ability: sum(getattr(c, ability.lower()+'_charges', 0)
                        for c in game.chars if c.team == 'A' and c.is_alive)
                        for ability in ('SMOKE', 'FLASH', 'RECON')})
                if game.round_over or game.match_over:
                    break
            else:
                raise RuntimeError('round exceeded tick limit')
            if game.match_over or round_index + 1 == rounds:
                break
            game.current_round += 1
            game.init_round()
        return dict(attacker_wins=game.attacker_wins, defender_wins=game.defender_wins,
                    rounds=game.attacker_wins+game.defender_wins,
                    plants=len(plants), charges_at_plant=plants,
                    decisions=advisor.decisions, history_rounds=len(advisor.memory.rounds))


def collect_opponent(task):
    opponent,index,opponent_count,matches,rounds,map_names,seed,metadata,path = task
    counts = Counter()
    with CappedJsonlWriter(Path(path).parent, Path(path).stem, '.part') as stream:
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
    return [str(p) for p in stream.paths],counts


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
        raise ValueError('select Omoko role/site contexts')
    dataset_id = uuid.uuid4().hex
    directory = Path(output_dir)
    directory.mkdir(parents=True,exist_ok=True)
    delete_incompatible_datasets(directory)
    metadata = dict(event='metadata',version=VERSION,label_version=LABEL_VERSION,
                    wall_los_version=WALL_LOS_VERSION,feature_names=feature_names(),
                    scenarios=route_metadata(map_names),dataset_id=dataset_id,
                    matches_per_opponent=matches,rounds_per_match=rounds,seed=seed)
    parts = directory/(dataset_id+'.parts')
    parts.mkdir()
    tasks = [(opponent,index,len(opponents),matches,rounds,tuple(map_names),seed,
              metadata,parts/f'{index}.part') for index,opponent in enumerate(opponents)]
    # Bound worker shards as well as published files; never concatenate into one huge file.
    results = run_opponent_tasks(collect_opponent,tasks,workers)
    counts = Counter()
    with CappedJsonlWriter(directory, dataset_id, '.jsonl.partial', metadata) as stream:
        for shards,groups in results:
            counts.update(groups)
            for shard in shards:
                with Path(shard).open(encoding='utf-8') as source:
                    for line in source:
                        stream.write(line)
    if not stream.paths:
        raise ValueError('collection produced no samples')
    paths = []
    for temporary in stream.paths:
        published = temporary.with_suffix('')
        temporary.replace(published)
        paths.append(published)
    for shards,_ in results:
        for shard in shards:
            Path(shard).unlink()
    parts.rmdir()
    print(f'dataset={dataset_id} files={len(paths)} samples={sum(counts.values())} groups={len(counts)}',flush=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches',type=int,default=DEFAULT_MATCHES_PER_OPPONENT)
    parser.add_argument('--rounds',type=int,default=DEFAULT_ROUNDS_PER_MATCH)
    parser.add_argument('--opponents',nargs='+',choices=OPPONENTS,default=DEFAULT_OPPONENTS)
    parser.add_argument('--seed',type=int,default=DEFAULT_SEED)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--workers',type=int,default=DEFAULT_WORKERS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    collect(args.matches,args.rounds,args.opponents,DEFAULT_MAPS,args.seed,args.output_dir,args.workers)


if __name__ == '__main__':
    main()
