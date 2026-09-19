# GC directional route clearance v14

v13 reduced designated-Escort route blocking from 8.04% to 7.13% by episode
750, but plant rate fell from 30.00% to 27.78% and the worst-seed no-entry rate
rose from 53.33% to 60.00%. The single blocking bit could only add one fixed
action preference to every blocking state, so labels requiring different
clearing directions conflicted.

v14 adds four observations in movement-action order: up, down, left, and right.
A value of 1 means that move is legal and would clear the designated Escort from
Carrier's adjacent progress cells. The training teacher selects only one of
these clearing moves. Reward distinguishes moving clear, remaining blocked, and
staying blocked.

Carry, Guard, every existing Escort input, all biases, and every hidden/output
layer remain frozen. Only the four new first-layer input columns can change.
Inference receives learned observations and contains no forced movement rule.
Training starts from the preserved v12 episode 0 checkpoint, whose deployed
behavior is identical to v13 episode 0.

Stop the running v13 process with `Ctrl+C`, then run from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_directional_clearance_v14.ps1
```

At episode 250, compare with the episode 0 evaluation. A useful checkpoint must
increase `designated_route_clear_rate` and reduce
`designated_route_block_rate`, while keeping `carry_quiet_stall_tick_rate`
within +0.02, plant rate within -0.03, and worst-seed no-entry no higher than
episode 0. Continue to episode 500 only when all of these conditions hold.

Resume an interrupted v14 run into a new directory with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\resume_gc_directional_clearance_v14.ps1
```
