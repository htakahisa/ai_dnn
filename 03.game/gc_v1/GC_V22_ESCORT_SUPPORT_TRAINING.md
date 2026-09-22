# GC Escort support (v22, updated in place)

The original Escort movement-only run was stopped at EP2000. Its 300-round
evaluation did not improve pre-entry carrier death or no-entry; do not promote
those weights. A later 60-round diagnostic used the same seeds but newer roster
data, so its absolute success rates are not directly comparable with the
original evaluation. It found that 11 of 13 first-threat events without an
eligible screener involved a `FAKE_WAIT` Spike carrier. In six of the seven
cases with another living `FAKE_WAIT` teammate, that teammate was 11-29 cells
away. The front-screen metric is intentionally inactive during the fake's wait
stage, so it cannot be used alone to judge that stage's protection.

The updated training run keeps Carry, Guard, ability, ultimate and facing
weights frozen. Only Escort's UP/DOWN/LEFT/RIGHT/STAY output rows may change.
For a `FAKE_WAIT` carrier, a same-role waiting Escort receives training-only
demonstrations and reward for approaching and staying within four path cells
of the carrier before contact. Fake sellers keep their Macro assignments.
No production movement override was added. This addresses the distant waiting
teammate cases; it cannot fix rounds where no waiting teammate survives.

The source is reconstructed from the stopped run: its EP1250 Escort facing
checkpoint is the untrained Escort baseline, while the frozen Carry and Guard
weights come from its best-by-evaluation bundle. The current roster data has
changed since those source checkpoints, so the selector explicitly records the
source fingerprint and evaluates both branches against the current data.

Run the updated experiment only when ready for a full training run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_escort_support_v22.ps1
```

Repeated runs overwrite `gc_v1/data/attacker_gc_escort_support_current/training`
and `selected` in place. The stopped timestamped run remains untouched because
promotion scripts still reference it. Do not create a v23 copy. Checkpoint
selection prioritizes pre-entry carrier death after entry-quality pass/fail;
no-entry, timeout, and plant regressions remain safety checks. Holdout seeds
report generalization but do not select the checkpoint. The packaged evaluator
kind is `best_escort_support_ab`.
