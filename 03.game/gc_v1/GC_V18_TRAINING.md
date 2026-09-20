# GC phase-specific facing heads v18

Carry and Escort keep their existing base action spaces and each learn a separate
eight-direction facing head from their own shared feature encoder. The selected
base action is an input to the facing head. Current facing is appended to the
observation, and training-only observable labels prefer a visible enemy, then
Carry team sighting or the phase navigation goal. No teacher logic runs during
evaluation or inference.

Start a clean run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_phase_facing_v18.ps1
```

Resume into a new output directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\resume_gc_phase_facing_v18.ps1
```

Only checkpoints containing `facing_head_version=1` enable learned facing at
runtime. Older checkpoints remain compatible and retain their previous facing
behavior.

## Stage 2 from the selected episode 250 checkpoint

The first run selected episode 250. Stage 2 lowers facing supervision from
`0.5` to `0.25`, unfreezes Carry/Escort at a lower `0.00001` learning rate,
keeps Guard restricted, and strengthens training-only quiet-stall/timeout
penalties:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_phase_facing_v18_stage2.ps1
```

## Recorded result (2026-09-20)

The initial v18 run selected episode 250.  Its fixed six-seed, 180-round
evaluation was:

- round win rate: `0.2833` (worst seed `0.2000`)
- registered plant rate: `0.3556` (worst seed `0.2667`)
- worst Carry no-entry rate: `0.4333`
- worst timeout rate: `0.1000`
- Carry facing teacher match rate: `0.5719`
- Escort facing teacher match rate: `0.5923`

Stage 2 was evaluated at episodes 500 and 750.  Episode 500 improved aggregate
win/plant rates (`0.3111` / `0.4333`) but worsened the worst-seed no-entry rate
to `0.4667`.  Episode 750 regressed further (`0.2611` win, `0.3667`
registered plant, `0.5333` worst no-entry), so the run was stopped after that
evaluation.  The stage-2 `best_by_eval` bundle therefore intentionally remains
episode 250 and has tensor-identical model weights to the initial v18 selected
bundle.  Do not promote the stage-2 `latest` checkpoints (episode 750/760).

The selected v18 model is retained as an experiment, not promoted to the
default runtime model directories, because it does not pass the configured
worst-seed no-entry target of `0.25`.

An independent three-seed, 150-round holdout of the selected episode 250
checkpoint produced:

- round win rate: `0.2733` (worst seed `0.2400`)
- registered plant rate: `0.3733` (worst seed `0.3200`)
- worst Carry no-entry rate: `0.4400`
- worst timeout rate: `0.0800`
- Carry / Escort facing match: `0.5623` / `0.5739`

The holdout is stored in `data/attacker_gc_curriculum_v18/holdout_evaluation.json`.
It confirms that learned phase-specific facing generalizes to unseen seeds, but
also confirms that Carry route reliability remains below the deployment gate.

## v19 facing-head-only correction

The follow-up v19 mode freezes every Carry/Escort movement tensor, including
the shared feature encoder, value head, and action-advantage head.  It skips
DQN, navigation-demonstration, and ultimate-classification updates for those
phases and applies only the facing cross-entropy loss to `facing_head`.
Guard is frozen as well.  This makes the movement Q-values bit-for-bit stable
while improving facing:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_phase_facing_v19_head_only.ps1
```
