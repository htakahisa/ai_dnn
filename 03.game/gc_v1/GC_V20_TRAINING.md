# GC v20 independent facing training

v20 keeps the selected v19 movement policy bit-for-bit frozen and trains only
new Carry/Escort facing networks.  Each facing network has its own observation
encoder, so facing learning cannot alter or depend on the frozen movement
feature encoder.

Training labels are confidence weighted: visible enemies are strongest, recent
team sightings and Escort outward screening are medium confidence, and movement
or hold directions are weak fallback labels.  Carry and Escort facing heads are
selected independently, subject to entry, timeout, plant, and formation safety
limits.  The selected composite is saved as `best_phase_facing`.

The run also fingerprints `character_stats.py`, `player_combos.py`, and
`awakening_events.py` and stops if any of them changes mid-run.  The evaluator
also rejects a changed data revision by default, avoiding silent cross-roster
comparisons.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_phase_facing_v20_independent.ps1
```

Evaluate the independently selected composite with:

```powershell
D:\git\python\python.exe -X utf8 .\gc_v1\evaluate_gc_curriculum_checkpoint.py `
  --models-dir .\gc_v1\data\attacker_gc_curriculum_v20_independent_facing `
  --kind best_phase_facing `
  --seeds 9026092000 10026092000 11026092000 `
  --episodes 50
```

If movement is frozen, use phase-isolated A/B selection before deployment.  It
changes only one facing head at a time and treats no-entry/timeout as relative
regression guards rather than objectives the facing model is expected to fix:

```powershell
D:\git\python\python.exe -X utf8 .\gc_v1\select_gc_phase_facing_ab.py `
  --models-dir .\gc_v1\data\attacker_gc_curriculum_v20_independent_facing `
  --eval-seeds 3026091700 5026091700 6026091700 4026091700 7026091700 8026091700 `
  --holdout-seeds 9026092000 10026092000 11026092000
```

The resulting checkpoint kind is `best_phase_facing_ab`.
If roster data was intentionally changed after training, add
`--allow-data-revision-mismatch`; the report records both fingerprints and marks
the result as a cross-revision evaluation.
