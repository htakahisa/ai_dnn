# Task 06 training scenarios

`ScenarioGenerator` samples private enemy positions for the current fixed map.
Pass the actor team's current legal visibility mask and occupied allied cells
when generating a scenario. The generator removes visible, occupied, walled,
unreachable, and too-far-from-enemy-spawn cells. Five sampled enemies must also
be assignable to five distinct enemy spawn cells. `elapsed_ticks` is the number
of movement ticks since the enemy could leave its spawn. At tick zero, only
enemy spawn cells can be selected. An enemy may move one orthogonal cell per
tick. The actor side and situation select eligible watch points; the enemy side
selects the spawn cells for reachability.

The initial category weights are 70% exact watch point, 20% jitter, and 10%
legal random. Exact points are weighted by importance. Jitter uses 1 to the
point's configured radius of **walking steps** and excludes other exact watch
point cells. At early ticks, categories without legal candidates are removed
and the remaining weights are renormalized. The emitted `category` records the
actual source for distribution audits.

`Scenario.enemies` is **training-only ground truth**. Place these enemies in
the training game, then obtain actor input through `TeamPerceptionBuilder`,
`BeliefMemory`, and the appropriate actor encoder. Do not pass `Scenario` or
`EnemyPlacement` to an actor. Later training tasks own game-state construction
and critic inputs.

```python
from coach_v1.training import (
    ScenarioGenerator, write_scenarios_jsonl, write_distribution_report,
)

generator = ScenarioGenerator(seed=42)
scenarios = [
    generator.generate(
        actor_side="attacker", situation="carry", elapsed_ticks=40,
        currently_visible=team_snapshot.currently_visible,
        occupied_positions=(ally.position for ally in team_snapshot.allies),
    )
    for _ in range(100)
]
write_scenarios_jsonl("coach_v1/logs/scenarios.jsonl", scenarios)
write_distribution_report("coach_v1/reports/distribution.json", scenarios)
```

The JSONL log contains true enemy positions and must remain on the training
side. The report records target and actual percentages, including any change
caused by early-tick eligibility or occupied cells.

## Task 08 character trainer

`CharacterTrainer` accepts `CharacterTrainingExample` records. Each record has
one `CharacterObservation` returned by `CharacterEnvironment.prepare`, a legal
facing label, an ability-use label, an optional legal target, and an optional
outcome flag for ability-effectiveness reporting. Labels can be derived from a
training simulator, but the model receives only the observation's copied grid
and vector. The trainer checks the fixed map and watch-point hashes, slot,
version, shapes, finite values, and action masks before training. Keep training
and validation examples separate by scenario or round when collecting real
rollouts.

```python
from coach_v1.train_character_gorimaru import train_gorimaru
from coach_v1.training import CharacterTrainingExample

# train_examples and validation_examples are sequences of safe, labeled samples.
trainer = train_gorimaru(
    train_examples, validation_examples, seed=42, epochs=20,
)
```

The trainer writes `checkpoints/characters/gorimaru/latest.pt` after every
epoch and `best.pt` when validation loss improves. It saves metadata, network
and optimizer states, epoch, best loss, and metric history. `resume=True`
restores the latest optimizer and step. `GorimaruPolicy` loads the best model
with strict character/map/watch-point/interface checks and selects only legal
facing, use, and target actions. It has no movement output. The remaining
character checkpoints are produced by the Task 09 curriculum below; full
round rollout integration remains a later task.

Validation metrics are facing accuracy, ability-use accuracy, accuracy on
examples labeled "do not use", target accuracy on use examples, and the
observed effectiveness rate for examples carrying an outcome label. That last
rate describes the supplied rollouts; it is not a counterfactual estimate of
the current policy. A missing target/outcome group is reported as `None`.

## Task 09 staged character curriculum

`python -m coach_v1.train_task09` generates a deterministic 1v1/2v1 supervised
curriculum for each fixed-roster character, trains five independent checkpoints,
and writes `reports/task09_character_curriculum.json`. The train and validation
seeds differ. The current run uses 360 training examples, 80 validation examples,
and 80 epochs per character. The private enemy placement uses `ScenarioGenerator`'s 70/20/10
point/jitter/random sampler; a staged actor is placed near that enemy. The
training-only game goes through `TeamPerceptionBuilder`, `BeliefMemory`, and
`CharacterEnvironment.prepare` before an example is supplied to the model.
Facing and ability labels use only the resulting legal sighting, watch-point
metadata, coach intent, and action mask. The enemy's private location is never
included in the model input or label calculation after staging.

The four new `train_character_*.py` and `learning_character_*.py` pairs use
their own slot and checkpoint. `gongon` learns facing only because HUNT has no
active ability. The report separates validation at exact watch points, jitter
cells, and legal random cells. These staged examples do not simulate ability
outcomes or full rounds; `ability_effective_rate` is therefore `null`. Check
the report's positive ability-use recall as well as overall accuracy, since
the use label is less common than the no-use label.

## Task 09-A Gorimaru follow-up

After Task 10 found weak real-game facing, Gorimaru alone was fine tuned with
actor-safe 5v5 rollout observations. `gorimaru_rollout.py` obtains these through
the normal coordinator and labels facing from legal shared sightings only.
`experiment_task09a_gorimaru_rollout.py` reproduces the earlier selected
checkpoint from the archived pre-update baseline. See `HANDOFF-09-A.md` for paired
real-game results and limits. Running the older `train_task09.py` recreates the
staged baseline and would replace the promoted Gorimaru checkpoint.

## Task 09-A facing label and evaluation follow-up

`experiment_task09a_gorimaru_labels.py` trains a separate Gorimaru candidate.
For a current legal team sighting, any facing within 45 degrees of a reported
enemy is accepted. The trainer sums the probability of all accepted directions;
the existing single-label behavior remains the default for other characters.
70% or more of the candidate's examples come from the 70/20/10 watch-point
curriculum. Supplemental 5v5 observations cover near, west, east, crossfire,
and natural encounters. Train, validation, and holdout use distinct seeds.

`evaluate_task09a_gorimaru_labels.py` compares saved candidates in the game
loop. Its primary facing measure is slot 0's `unforced_aligned_45` divided by
`unforced_sighting_actions`; forced-facing actions are reported separately.
Run with `PYTHONHASHSEED=0` in the process environment for the recorded setup.
The candidates from that experiment remain under `checkpoints/experiments/`.

## Task 09-A attacker adjustment

`analyze_task09a_gorimaru_attacker.py` records only legal team report offsets
for evaluation. `experiment_task09a_gorimaru_attacker_adjustment.py` compares
two separate fine tunes: one removes facing loss when the team has no current
sighting, and the other also increases attacker near-encounter examples. Both
retain at least 70% staged watch-point curriculum examples. The no-sighting
change is opt-in, so previous experiment scripts keep their original labels.

`evaluate_task09a_gorimaru_attacker_adjustment.py` compares paired 20-seed
game blocks. The selected `near_priority/epoch90_latest.pt` is now the official
Gorimaru `best.pt` and `latest.pt`; the prior official checkpoint is archived
as `task09a_gorimaru_rollout/epoch70_latest.pt`. Use `PYTHONHASHSEED=0` for the
recorded game comparisons. See `HANDOFF-09-A.md` for the measured results and
the remaining smoke-target limitation.

## Task 09-A smoke evaluation

`python -m coach_v1.evaluate_task09a_gorimaru_smoke` replays the selected and
prior Gorimaru policies on the fixed map with a STAY/HOLD coach. The report
counts successful throws and uses the game's shot LOS function to measure
directed shooting lanes removed by Gorimaru's smoke. It temporarily removes
only that smoke for each measurement and restores the original game state.
`--shadow-prior` also asks the prior policy for an action on the selected
policy's exact actor-safe observation and measures the prior target in the
same positions and facings without applying it to the match. Use
`PYTHONHASHSEED=0` to reproduce the recorded 10-seed comparisons. These are
potential shot lanes, not a measured change in kills or match wins.
