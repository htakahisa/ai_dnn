# GC bounded entry synchronization v11

> Retired at episode 250 because coordination updates still generalized into
> ordinary Carry states. Use `retrain_gc_feature_isolated_v12.ps1`.

v10 taught the synchronization condition, but repeated it at every entry cell.
At episode 250 its hold rate rose from 37.23% to 82.88%, quiet stalls doubled,
plant rate fell from 30.00% to 18.89%, and win rate fell from 19.44% to 11.11%.

v11 limits synchronization to the eight-cell entry boundary. Carry may wait for
at most two ticks there while a nearby designated Escort establishes its lead.
After two no-progress ticks the teacher resumes normal forward navigation, and
no additional synchronization wait is taught deeper in the site approach. The
existing observation already contains no-progress time, so this remains learned
behavior rather than a runtime movement override.

Start from v10's retained episode 0 checkpoint:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_bounded_entry_sync_v11.ps1
```

At episode 250, reject the checkpoint immediately if `carry_quiet_stall_tick_rate`
exceeds episode 0 by more than 0.03, or if plant rate falls by more than 0.03.
Otherwise continue to episode 500 and require improvement in
`entry_formation_ready_rate`, `first_threat_screen_ready_rate`, and worst-seed
no-entry results. The selected checkpoint still uses round outcomes and entry
quality, so an attractive synchronization metric cannot replace actual results.

Resume after interruption with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\resume_gc_bounded_entry_sync_v11.ps1
```
