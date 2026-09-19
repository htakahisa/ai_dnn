# GC feature-isolated entry coordination v12

> Retired: episode 250 increased designated-Escort route blocking and quiet
> Carrier stalls. Use `GC_V13_TRAINING.md` and the v13 training script.

v11 bounded the staging wait and recovered plant rate, but a rare coordination
label still changed the shared network's ordinary states.  Quiet stalls rose
from 10.74% to 14.50%, while win rate and first-contact readiness fell.

v12 adds one explicit Carry observation for the two-tick synchronization window.
Carry may learn only the first-layer column connected to that new observation.
Escort may learn only its four existing formation-input columns. Every bias,
hidden layer, output layer, and old input column is frozen. Therefore an input
with coordination features equal to zero produces exactly the episode 0 output,
even after training. Guard remains fully frozen. This isolates coordination
learning from the proven ordinary movement policy without adding an action
override to inference.

Start from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_feature_isolated_v12.ps1
```

At episode 250, ordinary behavior should remain close to episode 0. Stop if
`carry_quiet_stall_tick_rate` rises by more than 0.02 or plant rate falls by more
than 0.03. Continue to episode 500 only when entry formation/contact metrics
improve without worsening worst-seed no-entry and Carrier-death rates.

Resume an interrupted run into a new directory with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\resume_gc_feature_isolated_v12.ps1
```
