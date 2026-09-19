# GC entry screening v9

> Retired after episode 500: approach formation improved, but contact readiness
> stayed flat and entry outcomes worsened. Use `retrain_gc_entry_sync_v10.ps1`.

v8's best checkpoint remained episode 0 through episode 750.  Its designated
Escort received a formation label on the open approach, but its ordinary Macro
route reward competed with that label, and the label stopped at the last eight
route cells where first contact usually happens.

v9 trains only Escort from v8's preserved episode 0 synchronized checkpoint.
While an Escort is the designated screener, its formation target replaces its
ordinary Macro route target.  The target remains active from 24 route cells out
until the Carrier is within two cells of the site.  This keeps Macro's selected
Rush, MID_TO_B, Split, Default or Fake group while teaching one member of that
group to enter in front of the Carrier.  Carry and Guard remain frozen.

Start from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_entry_screener_v9.ps1
```

The run writes to `gc_v1/data/attacker_gc_curriculum_v9`.  At episode 250 and
500, compare these fields with the episode 0 row:

- `entry_formation_ready_rate`: screener readiness in the last eight route
  cells before the site.
- `first_threat_screen_ready_rate`: readiness at the Carrier's first threat.
- `worst_carrier_preentry_death_rate`: worst seed's Carrier death rate before
  entry.
- `worst_carry_no_entry_rate`: worst seed's failure to enter.

Stop after episode 500 if the first two fields have not increased while the
last two have stayed level or fallen.  The run retains the best evaluated
checkpoint independently of the latest checkpoint.

After Ctrl+C, resume from the latest synchronized checkpoint into a fresh
directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\resume_gc_entry_screener_v9.ps1
```
