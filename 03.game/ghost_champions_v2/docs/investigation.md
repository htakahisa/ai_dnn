# GC attacker v2: T0 investigation and implementation plan

## Scope and status

2026-10-07. T0 was investigated without code changes and its findings and plan
were presented before beginning T1. The v1 source and checkpoints present at
the start of this task are the baseline, including the preceding roster
observation work. Do not reset those changes. Defender behavior and other
teams' implementations remain fixed. The v2 rule layer has since been implemented;
see `implementation.md` for its validation and current evaluation status. RL has
not started. The user's subsequent instruction changes commands to `py`.

## Confirmed implementation

* The inspected workspace contains no C# files or `unity_env.py`. The engine is
  Python: `run_game.VisualFPSBattle`, `BattleLogicMixin`, `AbilityLosMixin`.
* `_build_team_ai` in `run_game.py` registers v1 using the attacker and defender
  classes in `ghost_champions_v1_macro.py`. The attacker subclasses the phase
  router in `ghost_champions_v1.py`: Carry / Escort / Retrieve / Guard, with a
  learned macro coordinator in `gc_v1/learning_attacker_macro_gc_runtime.py`.
* Add a lazy `ghost_champions_v2` branch in `_build_team_ai`, with a new attacker
  class under `ghost_champions_v2/` and the exact existing v1 defender factory.
  Add a selection entry to `roster_select.TEAM_AI_OPTIONS` and
  `run_competition_manager.CONTROLLER_OPTIONS`. Leave v1 aliases and the preset's
  default AI unchanged. `team_controllers` maps team names to controller keys.
* `run_competition_manager.run_series_core` is the actual BO3 API (`maps_to_win=2`).
  It carries fatigue and tactical memory between maps and alternates initial
  attack side by map number. The older `run_headless_series.py` hardcodes one AI
  and exports fewer fields, so it is unsuitable for this evaluation.
* `analytics/combat_tracker.py` creates round records. `VisualFPSBattle` creates
  replay frames. `CompetitionApp._execute_job` combines series results with
  controller selections and leaderboards. `save_json` writes to
  `competition_results/`; the supplied `logs/` directory does not exist.
* Side swap occurs when round 13 starts. Overtime swaps every round from 25.
  Although overtime behavior is now confirmed, exclude it to retain the requested
  denominator. Prefer each record's player `side` for identifying GC attack.

## Rules affecting the new policy

Verified in `abilities_los.py`, `game_core.py`, and `battle_logic.py`:

| Ability | Actual effect |
|---|---|
| SMOKE | 3×3 walkable cells, 25 ticks. Blocks normal LOS when either endpoint or any intermediate cell is inside smoke. Same/adjacent cells are exempt. Does not stop movement or defuse. Recon-revealed targets can be shot through smoke; walls and living body blockers still apply. |
| FLASH | Travels 3 cells/tick along the aimed direction, until a wall/map edge or 5 flight ticks. On explosion, every living enemy with unobstructed LOS to the explosion is blinded for 10 ticks. No radial distance limit or facing test. Smoke and walls block the flash, allies are exempt. |
| RECON | 2 initial charges per seeker. Travels 3 cells/tick along the aimed direction to the wall/map edge, **not** to the supplied target cell. Reveals enemies in the impact-centered 9×9 square for 15 ticks, without a wall-LOS test for that square. |
| DANCE | 3 charges; heals another living teammate by 50 HP, up to the applicable cap (normally 100). No cast-distance or LOS check in the engine. |
| ASH | 3 charges; aimed cell must be within Euclidean radius 8. Creates a 3×3 destruction area lasting 10 ticks, applying a 5-tick life contract (10 damage/tick, including maximum HP loss after shield handling). |
| HUNT | Passive, not a cast action. A credited kill heals 50 HP, capped at normal maximum HP (100). |

Defuse requires 6 consecutive ticks, one defender at a time, within Chebyshev
distance 1 of the spike. Completion is checked **after shooting**, so killing a
defuser on the completion tick can interrupt it. Spike detonation constant is
55 ticks; the first recorded planted frame can show 54 after the tick update.
Do not change the engine constant based on that recorded value.

Movement affects shooting accuracy. The engine automatically chooses and fires
at a valid opponent, prioritizing visible defusers; the controller must supply
position and facing rather than invent a new explicit shooting action.

## Observation and public information

`battle_logic.py` supplies grid, characters, spike state/positions, round timer,
battle tick, detonation timer, smoke cells, and `defender_defuse_info`. The latter
is an intentional public cross-team defuse tap notification, even without LOS.
Each friendly character exposes its own ability-specific remaining charges.

`IQAwareController` passes a `PerceivedGameView`: unknown enemy coordinates are
`(-1,-1)`, some enemy entries may be omitted, and visible positions, allied
positions and the planted spike position can have IQ noise. Team decisions must
not inspect hidden objects, omniscient replay frames or analytics-derived
defender setup. Public full roster metadata (`enemy_roster`) is available and
is appended as a 265-value suffix to the phase observations from the preceding
task. It describes identity and static abilities/stats, not the actual defensive
deployment. Opponent identity can be inferred from a matching preset roster.
Treat unknown/custom rosters as unknown; do not assume their deployment.

The existing guard already observes defuse notification, remaining detonation
time and ability availability. These fields do not themselves enforce a hold
formation. There is no current `observed_defenders_by_axis` state.

## Evaluation plan

Existing scripts include `gc_v1/evaluate_real_match_gc.py`,
`gc_v1/evaluate_real_series_gc.py`, and real curriculum/checkpoint evaluations.
GC tests live in `test/test_gc_*.py` and `gc_v1/test/`. The current training
opponent pool covers five teams and omits Gorigons; the new evaluation explicitly
uses all six controller keys from the source logs:

| Code | Preset | Controller |
|---|---|---|
| TYG | Touyama Gaming | touyama_gaming_v2 |
| OMG | Omoko Gaming | omoko_gaming_v1 |
| FRC | Furina Classic | frc_v1 |
| FNC | Fnatic2023 | fnatic_v3 |
| GG | Gorigons | concon_v1 |
| SPS | SUPES | toru_ai_v3.1 |

1. T1: copy the user's diagnostic script to `tools/`; verify the source eight
   series, then evaluate v1 in 10 BO3s per opponent and save full compatible
   series logs under `eval_logs/baseline_v1/`.
2. T2: add configurable PostPlantHold, nearest-two defuse interruption, stable
   walkable LOS assignments with crossfire/trade preferences, and no chasing
   beyond the leash. Preserve public information boundaries and handle blocked
   routes. Compare with T1 using the same seed schedule.
3. T3: add early recon casts using simulated static projectile endpoints, record
   only actually revealed defenders by axis, and evaluate independently.
4. T4: add roster-based profile selection and observation-based site decisions.
   A lack of sightings is not proof of an empty site. Keep TYG's existing A
   preference and evaluate FRC/FNC deployment-dependent decisions explicitly.
5. T5: add entry/trade discipline and validate first contact and preplant wipes.
6. T6: only after reporting T2–T5 results, decide whether attacker-only RL is
   needed. If needed, collect new imitation data, retrain BC with versioned
   observations, then use the specified PFSP and annealed potential shaping.
7. T7: use the common evaluator during T1 onward and finish the cumulative
   per-opponent comparison report. It must not select favorable seeds.

Each configuration stage requires at least 100 GC attacker rounds per opponent.
Run additional series if 10 BO3s do not supply enough. Report raw counts and
Wilson 95% intervals for attack and postplant wins; distinguish inconclusive
samples from target attainment. Distances use Euclidean cells, matching the
supplied +0/+10 results; report the surviving sample size at each offset because
early round endings censor the later offsets.

Commands run from `D:\git\ai_dnn\03.game`:

```powershell
py tools/gc_attack_report.py "competition_results/series_Ghost_Champions_vs_*202610*.json"
py -m unittest discover -s test -p test_gc_eval_tools.py -v
py tools/run_eval.py --controller ghost_champions_v1 --series-count 10 --workers 3 --output eval_logs/baseline_v1 --stage baseline
```

The first attempted run used Python 3.10, which lacked Tcl/Tk and TensorBoard.
TensorBoard was installed into that Python. A temporary GUI import workaround
was removed following the user's instruction to use ordinary `py`. Two completed
3.10 series are retained under `eval_logs/baseline_v1/interrupted_py310/` and are
excluded from the new baseline. The normal environment is Python 3.14.7 and
Torch 2.11.0+cu128; the baseline manifest records the actual interpreter.

## Source-log verification and discrepancies

All eight source series are under `competition_results/`: the six series dated
20261007, plus the two FRC series dated 20261002. They contain 166 GC attacker
rounds and no overtime. The copied diagnostic reproduces:

| Opponent | Attack wins | Plants | Postplant wins | Defuse losses | Recon used/available |
|---|---|---|---|---|---|
| TYG | 19/22 | 22/22 | 19/22 | 2 | 37/88 |
| FNC | 15/36 | 23/36 | 15/23 | 8 | 33/144 |
| FRC | 14/52 | 18/52 | 14/18 | 4 | 33/180 |
| OMG | 1/21 | 18/21 | 1/18 | 17 | 29/84 |
| GG | 3/16 | 13/16 | 3/13 | 10 | 21/64 |
| SPS | 4/19 | 18/19 | 4/18 | 14 | 42/76 |

Central-heavy opponents jointly reproduce 49 plants, 8 postplant wins and 41
defuse losses across 56 attacker rounds. FRC reproduces 30 preplant attacker
wipes, and attacked sites with 2 defenders 17 times and 3 defenders 28 times.

Discrepancies with the supplied prose:

* The five-player FRC deployment in these logs is `0-0-5` (5 rounds), not `5-0-0`.
* The supplied script sums recon to **195/636 = 30.7%**, rather than 27%. It
  starts inventory accounting at the first non-setup frame. The evaluator also
  uses recorded setup inventory, allowing it to count casts made before that
  first active frame: FRC changes from **33/180** to **57/208**, and the overall
  corrected inventory comparison is **219/664 = 33.0%**. Keep the original-script
  and corrected-inventory values separate; neither reproduces 27%.
* Central-heavy mean Euclidean spike distance at +0/+10 is **11.81/11.21**
  (49/49 rounds), matching the supplied table. At +20/+30, exact existing frame
  matches give **10.45/10.00** (45/36 rounds), rather than 11.8/9.8. The evaluator
  explicitly excludes terminal frames and publishes offset sample counts.
  Values from differing terminal-frame/censoring conventions are not comparable.

These are source-log observations, not results of the new 60-series baseline.
