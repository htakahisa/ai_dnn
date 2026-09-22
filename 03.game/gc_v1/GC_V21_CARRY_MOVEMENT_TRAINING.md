# GC v21 isolated Carry movement training

v21 starts from the selected v20 models and trains only Carry's five final
advantage rows for STAY/UP/DOWN/LEFT/RIGHT. Escort, Guard, observation/value
layers, facing heads, and the ability/PLANT/ULT rows stay bit-identical to v20.
No route-control rule is added to inference.

The run uses conservative updates (`lr=2e-5`, at most one movement TD update
and four navigation-demonstration updates per episode). Evaluation runs every
250 episodes with exploration and teacher intervention disabled.

Training stops and retains `best_by_eval` when either condition is met:

- six consecutive evaluations do not improve the Carry movement score;
- two consecutive evaluations exceed the timeout limit or raise quiet-stall
  by more than five percentage points over the episode-zero baseline.

The stop decision is written to `training_stop.json`. A timestamped output
directory is created for every invocation; the script never deletes an older
run. The previously rejected reset branch is not retrained.

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_carry_movement_v21.ps1
```

Each run produces:

```text
gc_v1/data/attacker_gc_curriculum_v21_carry_movement_<timestamp>/
  warm/       # training history, latest, best_by_eval, and training holdout
  selected/   # baseline-vs-warm winner and final unseen-seed holdout
```

The deployable checkpoint kind under `selected` is
`best_carry_movement_ab`. Selection uses only the evaluation seeds; the final
holdout is not used to choose the winner. The selector also rejects the run if
Escort, Guard, a facing tensor, or a non-movement Carry row changed.
