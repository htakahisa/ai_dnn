# GC route-clearance Escort v13

> Retired: one blocking bit could not distinguish which direction clears the
> route. Use `GC_V14_TRAINING.md` and the v14 training script.

The v12 trace comparison reproduced the regression. From episode 0 to episode
250, quiet Carrier stays increased from 69 to 146 ticks. Cases where Carrier's
preferred move was occupied by an ally increased from 21 to 60; the designated
Escort was that blocker in 48 cases, up from 13. Only 7 of those 48 cases were
combat states. Macro target changes and an isolated Escort were not the cause.

v13 starts again from the preserved v12 episode 0 bundle. Carry and Guard are
fully frozen. Escort receives one new observation that is 1 only when the
designated Escort occupies an adjacent cell that advances along Carrier's
route. Training penalizes that state and labels a legal clearing move. Only the
first-layer weight column connected to this new observation can change; all old
Escort input columns, biases, hidden/output layers, Carry, and Guard remain
exactly unchanged. The production controller contains no forced clearing move.

Start from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_route_clearance_v13.ps1
```

At episode 250, compare against the episode 0 evaluation printed at startup.
Continue only if `designated_route_block_rate` falls without
`carry_quiet_stall_tick_rate` rising by more than 0.02, plant rate falling by
more than 0.03, or `worst_carry_no_entry_rate` increasing. A lower blocking
rate alone is insufficient if entry and planting regress.

Resume an interrupted run into a new directory with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\resume_gc_route_clearance_v13.ps1
```
