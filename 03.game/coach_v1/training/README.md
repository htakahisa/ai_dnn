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
