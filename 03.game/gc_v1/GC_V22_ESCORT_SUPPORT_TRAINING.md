# GC Carry/Escort joint movement (v22 files, updated in place)

The original Escort movement-only run was stopped at EP2000. Its 300-round
evaluation did not improve pre-entry carrier death or no-entry; do not promote
those weights. A later 60-round diagnostic used the same seeds but newer roster
data, so its absolute success rates are not directly comparable with the
original evaluation. It found that 11 of 13 first-threat events without an
eligible screener involved a `FAKE_WAIT` Spike carrier. In six of the seven
cases with another living `FAKE_WAIT` teammate, that teammate was 11-29 cells
away. The front-screen metric is intentionally inactive during the fake's wait
stage, so it cannot be used alone to judge that stage's protection.

The updated run now trains both Carry and Escort movement together. Guard,
ability, ultimate and independent-facing weights remain frozen. Carry may
change only its movement output rows. Escort may change its movement output
rows plus the six FAKE_WAIT-support input columns described below. Existing
input columns and every non-movement action row remain unchanged.
For a `FAKE_WAIT` carrier, a same-role waiting Escort receives training-only
demonstrations and reward for approaching and staying within four path cells
of the carrier before contact. Fake sellers keep their Macro assignments.
No production movement override was added. This addresses the distant waiting
teammate cases; it cannot fix rounds where no waiting teammate survives.

The run through EP2500 exposed an input-training bug: the six new FAKE_WAIT
bodyguard observations were present in the v12 Escort model, but their first
layer weights remained exactly zero. Movement-only training froze every input
weight, so the model could not use the new role or pursuit-direction context.
The trainer now updates only those six new input columns and the Escort
movement output rows. Existing input columns, ability/ultimate rows, and the
independent facing head stay frozen. The selector verifies this boundary.

The EP2000 retry did learn nonzero input weights, but stopped on the carrier
death/plant guardrail and did not beat its EP1500 source. Further inspection
found conflicting supervision: after a `FAKE_WAIT` bodyguard approached within
four path cells, the support teacher stopped labeling it and the generic Macro
waypoint teacher/reward could pull it away again. The current trainer aims
for two path cells of separation, allowing up to four when movement is blocked,
and steps aside when adjacent,
and samples these role-specific demonstrations in half of Escort's supervised
batch. This is training-only guidance, not an inference movement override.
`fake_wait_demo` in training logs and `fake_wait_threat_close_rate` in
evaluations show whether the correction is taking effect.

The next invocation snapshots the current evaluation winner (EP1500 at the
time of this fix) under `attacker_gc_escort_support_current/source` before
overwriting `training`. This prevents the rejected EP2500 training checkpoint
from becoming the next starting point. If no current winner exists, the script
falls back to the older EP1250 source. The selector evaluates the source and
trained branches on the same current roster and holdout seeds.

The joint run is capped at 1000 additional episodes (EP2500 when starting at
EP1500). Because two policies now receive updates, the initial learning rate is
reduced from `2e-5` to `1e-5`. It remains constant through 250 additional
episodes, then decays linearly to `2e-6` over the next 500 episodes:

- EP1500-1750: `1e-5`
- EP2000: approximately `6e-6`
- EP2250 and later: `2e-6`

Evaluation remains every 250 episodes. Three evaluations without a new safe
winner stop the run, and two consecutive safety-guardrail violations stop it
earlier. The safety checks cover timeout, quiet Carry stalls, no-entry,
pre-entry carrier death and plant regression.

Run the updated experiment only when ready for a full training run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_escort_support_v22.ps1
```

Stop any currently running training process before invoking the script again.
Repeated runs overwrite `gc_v1/data/attacker_gc_escort_support_current/training`
and `selected` in place; `source` holds the previous winner for safe restart.
The stopped timestamped run remains untouched because promotion scripts still
reference it. Do not create a v23 copy. Checkpoint
selection first rejects safety regressions, then prioritizes entry-quality,
planting, carrier survival and win rate. Holdout seeds report generalization
but do not select the checkpoint. Existing artifact names such as
`best_escort_support_ab` are retained for compatibility; no v23 directory or
script is created.
