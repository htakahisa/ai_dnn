# GC early screen and balanced ultimate training v17

v16 was stopped at episode 250. Exact screen guidance improved, but the
designated Escort stopped after first reaching the screen point. Carrier then
caught up: worst-seed no-entry rose to 63.33%, worst registered plant fell to
16.67%, and only 9 of 59 ultimate uses were tactical.

v17 begins exact screen formation up to 40 route cells from the plant goal.
After reaching the three-cell lead, the observation requests another forward
step whenever Carrier closes the gap. This remains an observation, teacher,
and reward change; inference does not force a movement action.

Every executable ultimate decision is also stored as a tactical-use or save
example. Training samples both classes equally and learns a margin between the
ULT action and the best other legal action. This gives rare tactical windows
positive supervision while explicitly lowering ULT outside those windows.

Checkpoint selection now includes worst-seed registered-plant rate before win
rate once entry failures and pre-entry Carrier deaths are compared. Training
starts from the preserved v12 episode 0 models. Only coordination-feature
columns and the ULT output row may change.

Run from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_early_screen_ultimates_v17.ps1
```

At episode 250, stop unless all of these hold against episode 0:

- worst `carry_no_entry_rate` does not rise;
- worst `registered_plant_rate` does not fall;
- `first_threat_screen_ready_rate` or mean first-threat support ahead improves;
- plant and round-win rates do not materially fall;
- tactical ULT rate reaches at least 0.50 overall, with both positive and
  negative classification examples present in the training history.

For an interrupted run, resume into a new directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\resume_gc_early_screen_ultimates_v17.ps1
```
