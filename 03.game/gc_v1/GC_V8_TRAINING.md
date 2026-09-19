# GC entry screening v8

> Retired after the episode 750 evaluation failed to beat its episode 0
> baseline.  Use `retrain_gc_entry_screener_v9.ps1`.

v7 failed because it stopped the Carrier whenever nobody happened to be ahead,
while the Escort teacher stopped helping as soon as any teammate was briefly in
front.  This increased quiet stalls without keeping a screen at first contact.

v8 keeps the proven v6 episode 750 Carry and Guard weights frozen.  It trains
only Escort.  For each Macro route it selects one eligible teammate and teaches
that teammate to maintain a position three route cells ahead of the Carrier.
The selection respects Rush, MID_TO_B, Split, Default and Fake groups; Fake
sellers are not recalled while selling.  The selected teammate stays assigned
until the strategy/group changes or the teammate dies.

Start a fresh run from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_designated_screener_v8.ps1
```

Output is written to `gc_v1/data/attacker_gc_curriculum_v8`.  The new evaluation
fields are:

- `formation_ready_approach_rate`: fraction of approach ticks where the selected
  Escort actually held the intended formation.
- `first_threat_screen_ready_rate`: fraction of first Carrier contacts with that
  formation ready.
- `death_screen_ready_rate`: fraction of pre-entry Carrier deaths where the
  formation was ready immediately before the fight.
- `worst_no_entry_carrier_death_rate`: worst seed block for the failure v8 is
  intended to reduce.

Compare these with the episode 0 row from the same run.  A higher generic
`screened_approach_rate` alone is not evidence of improvement.  The checkpoint
selector now uses worst-seed no-entry Carrier deaths before win rate when both
checkpoints fail the entry-quality gate.

If Ctrl+C interrupts the run, continue from its latest synchronized checkpoint
into a new directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\resume_gc_designated_screener_v8.ps1
```
