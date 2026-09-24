# Coach actor observation v1

`CoachObservationEncoder.encode(team_snapshot, belief_snapshot, situation=...)` returns
two copied `float32` NumPy arrays. The encoder accepts only Task 03's
`TeamPerceptionSnapshot` and Task 04's `BeliefSnapshot`; it never receives live
game state or critic data. The caller must supply the current situation:
`carry`, `retrieve`, `guard` for attacker, or `search`, `retake` for defender.
The encoder does not choose a situation or a move.

## Grid: `[27, 26, 44]`

| Index | Channel | Encoding |
| --- | --- | --- |
| 0–5 | `walkable`, `wall`, `plantable`, `attacker_spawn`, `defender_spawn`, `orb` | Fixed map binary masks. `walkable` means any map code except `1`. |
| 6 | `watch_importance` | Applicable side and situation points, importance / 5. |
| 7–8 | `watch_facing_row`, `watch_facing_column` | Applicable point's facing delta, each in `[-1, 1]`. |
| 9–10 | `watch_confirmed`, `watch_confirmation_age` | Confirmation presence and elapsed tick / 256, clipped to 1. |
| 11–12 | `currently_visible`, `visible_viewer_count` | Legal shared visibility binary mask and count / 5. |
| 13–14 | `clear_known`, `clear_age` | Last legal clear presence and elapsed tick / 256, clipped to 1. |
| 15 | `smoke` | Public smoke cell binary mask. |
| 16 | `current_enemy_sighting` | Current legal reported sighting count / 5. |
| 17–19 | `last_seen_enemy_count`, `last_seen_known`, `last_seen_age` | Historical reported count / 5, presence, and youngest elapsed tick / 256. |
| 20–21 | `spike_dropped`, `spike_planted` | Public spike positions. |
| 22–26 | `ally_slot_0` … `ally_slot_4` | Alive ally position binary masks, fixed roster order. Dead slots stay zero. |

Unknown age values are zero in the age channel and zero in their separate
presence channel. A known age of zero has presence one and age zero. Age values
remain continuous with elapsed ticks; the encoder makes no safety threshold.

## Vector: `[84]`

Indices 0–13 contain side one-hot (2), situation one-hot in the order
`carry, retrieve, guard, search, retake` (5), defender setup flag (1), allied
spike carrier / dropped / planted flags (3), and living ally / living enemy /
currently sighted enemy counts divided by 5 (3).

Each subsequent slot contributes 14 values in fixed roster order: alive,
HP clipped to 0–100 then divided by 100, normal ability available, has spike,
row / 25, column / 43, then facing one-hot in the order
`N, NE, E, SE, S, SW, W, NW`. All 14 values are zero for a dead slot.

The `COACH_GRID_CHANNELS` and `COACH_VECTOR_FIELDS` constants specify exact
field names and order for checkpoint consumers. A change to shape, order,
meaning, or normalization requires a new `COACH_OBSERVATION_VERSION` and
retraining. The encoder validates fixed map and watch-point hashes against the
current files at construction; `validate_checkpoint(metadata, side=...)` checks
those hashes, the observation version, fixed roster, and coach side against a
checkpoint.

Input belief must contain precisely the watch points applicable to its side,
for example `BeliefMemory(config.for_side(side))`. The `BeliefMemory` result
must come from the same team snapshot and tick. Public round and detonation
timers are not present in the current Task 03 DTO, so they have no v1 channel.
