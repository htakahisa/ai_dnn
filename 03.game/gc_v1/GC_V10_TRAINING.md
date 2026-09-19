# GC entry synchronization v10

> Retired at episode 250 because repeated synchronization doubled quiet stalls
> and sharply reduced plants and wins. Use `retrain_gc_bounded_entry_sync_v11.ps1`.

v9 raised quiet-approach formation readiness but did not preserve that screen at
contact.  Its Carry checkpoint came from before the new formation inputs were
trained, so the Carrier continued forward while its designated Escort stopped
to shoot.  At episode 500, win rate fell from 19.44% to 16.11%, plant rate fell
from 30.00% to 25.56%, and worst-seed no-entry rose from 53.33% to 63.33%.

v10 starts from v9's retained episode 0 checkpoint and trains Carry and Escort
together.  In the final eight route cells, Carry is taught to hold briefly when
the designated Escort is close enough to establish its three-cell lead.  The
rule is excluded when the Escort is too far away or the round clock is below 15
ticks.  Guard remains frozen.  Runtime still chooses actions from the learned
models; this synchronization is only a training label and reward.

Start from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_entry_sync_v10.ps1
```

Compare episode 250 and 500 with the run's episode 0 row:

- `entry_sync_hold_rate` should rise, showing that Carry learned the bounded
  synchronization state.
- `entry_formation_ready_rate` and `first_threat_screen_ready_rate` should rise.
- `worst_carry_no_entry_rate`, `worst_timeout_rate`, and
  `worst_carrier_preentry_death_rate` must not rise together.
- Plant and round-win rates must recover rather than trading outcomes for a
  cosmetic formation score.

Stop at episode 500 if synchronization rises but plant/no-entry results worsen;
that means waiting is being learned without successful entry.

Resume an interrupted run into a new directory with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\resume_gc_entry_sync_v10.ps1
```
