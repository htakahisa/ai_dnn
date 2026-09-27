# Character actor contract v1

`CharacterObservationEncoder.encode(snapshot, belief, situation, slot, instruction)`
uses only the same copied team perception and belief DTOs as the coach encoder.
`instruction` is the external coach move, objective, and tactical intent. The
character model has no move output. The actor receives no game, Character,
scenario ground truth, or critic data.

The grid is `float32 [29, 26, 44]`: the first 27 channels have exactly the
`COACH_GRID_CHANNELS` order, followed by `self_position` and
`coach_destination`. The destination marker is empty if that cell is outside
the map or a wall; it does not alter the coach's instruction. The vector is
`float32 [106]`: the first 84 fields have `COACH_VECTOR_FIELDS` order, followed
by self slot one-hot (5), coach movement one-hot (5), objective one-hot (3),
and tactical intent one-hot (9). All arrays are read-only.

`CharacterAction` has only `facing` (`N, NE, E, SE, S, SW, W, NW`),
`use_ability`, and an optional `(row, column)` target. The action mask has
facing `[8]`, use `[2]` (`no`, `yes`), and target `[26, 44]` boolean arrays.
For a living character all eight facings are legal. Use is masked after its
one round charge is spent, for the HUNT character, during coach objective
actions, and when no legal target remains. Smoke targets are all walkable map
cells. Flash and recon targets must be walkable, differ from the owner's cell,
and launch a projectile past its first map cell according to the game's
Bresenham rule. No tactical target preference is encoded in the mask.

`CharacterEnvironment.prepare` accepts an optional `1v1` or `2v1` curriculum
stage and checks the living allied and enemy counts. It does not place enemies
or advance a round. `resolve` rejects masked actions and returns the existing
controller tuple: `(coach_destination, {"facing": ...})`,
`(current_position, {"ability": ..., "target": ..., "facing": ...})`, or
`(current_position, "PLANT"/"DEFUSE")`. Ability use suppresses coach movement.
The game currently ignores a facing value in an ability payload because its
ability branch returns early; facing-only and movement-plus-facing actions do
apply facing. This limitation should be handled during Task 10 runtime
integration if simultaneous ability/facing is required. No core file changed.

Any change to shape, order, normalization, action indices, or mask semantics
requires the corresponding character interface version to change and the
character checkpoints to be retrained.
