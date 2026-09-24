import dataclasses
import inspect
import unittest

from coach_v1.common.types import Facing, Side
from coach_v1.common.watch_points import WatchPoint
from coach_v1.perception import (
    AllyPerception,
    BeliefInputError,
    BeliefMemory,
    EnemyPublicState,
    EnemySighting,
    PerceptionTick,
    SightingSource,
    SpikeSharedInfo,
    TeamPerceptionSnapshot,
    normalize_age,
)


def _grid(value, rows=3, columns=4):
    return tuple(tuple(value for _ in range(columns)) for _ in range(rows))


def _snapshot(
    tick,
    *,
    round_number=1,
    phase="live",
    visible=(),
    sightings=(),
    enemy_alive=True,
):
    visible_set = frozenset(visible)
    currently_visible = tuple(
        tuple((row, column) in visible_set for column in range(4))
        for row in range(3)
    )
    viewer_count = tuple(
        tuple(int(cell) for cell in row) for row in currently_visible
    )
    allies = tuple(
        AllyPerception(
            slot=slot,
            character_id=f"ally_{slot}",
            position=((1, 0) if slot == 0 else (0, slot - 1)),
            facing=Facing.E,
            is_alive=slot == 0,
            hp=100 if slot == 0 else 0,
            normal_ability_charges=0,
            has_spike=False,
        )
        for slot in range(5)
    )
    return TeamPerceptionSnapshot(
        side=Side.ATTACKER,
        tick=PerceptionTick(round_number, phase, tick),
        allies=allies,
        enemies=(EnemyPublicState("enemy", enemy_alive),),
        sightings=tuple(sightings),
        currently_visible=currently_visible,
        visible_viewer_count=viewer_count,
        smoke_cells=(),
        spike=SpikeSharedInfo(None, None, False, None),
    )


def _sighting(position=(1, 2)):
    return EnemySighting("enemy", position, 0, SightingSource.NORMAL)


def _watch_point(point_id="middle", position=(1, 2)):
    return WatchPoint(
        point_id=point_id,
        position=position,
        importance=3,
        facing=Facing.E,
        sides=(Side.ATTACKER,),
        situations=("carry",),
        random_radius=1,
        tags=("corner",),
    )


class CoachV1Task04BeliefMemoryTest(unittest.TestCase):
    def test_clear_age_increases_by_tick_delta_and_recheck_resets_it(self):
        memory = BeliefMemory()
        first = memory.update(_snapshot(10, visible={(1, 1)}))
        older = memory.update(_snapshot(13))
        refreshed = memory.update(_snapshot(15, visible={(1, 1)}))

        self.assertEqual(0, first.clear_age[1][1])
        self.assertEqual(3, older.clear_age[1][1])
        self.assertEqual(0, refreshed.clear_age[1][1])
        self.assertEqual(5, refreshed.last_clear_tick[1][1])
        self.assertIsNone(refreshed.clear_age[2][3])

    def test_enemy_rediscovery_updates_legal_position_age_and_direction(self):
        memory = BeliefMemory()
        first = memory.update(_snapshot(2, sightings={_sighting((1, 2))}))
        hidden = memory.update(_snapshot(6))
        rediscovered = memory.update(
            _snapshot(7, sightings={_sighting((2, 0))})
        )

        self.assertEqual((1, 2), first.enemy_for("enemy").last_seen_position)
        self.assertEqual(Facing.E, first.enemy_for("enemy").last_seen_direction)
        self.assertEqual(4, hidden.enemy_for("enemy").last_seen_age)
        self.assertEqual((1, 2), hidden.enemy_for("enemy").last_seen_position)
        enemy = rediscovered.enemy_for("enemy")
        self.assertEqual((2, 0), enemy.last_seen_position)
        self.assertEqual(0, enemy.last_seen_age)
        self.assertEqual(Facing.S, enemy.last_seen_direction)

    def test_last_seen_maps_keep_newest_age_when_reports_share_a_cell(self):
        memory = BeliefMemory()
        seen = memory.update(_snapshot(1, sightings={_sighting()}))
        hidden = memory.update(_snapshot(4))

        self.assertEqual(1, seen.last_seen_enemy_count[1][2])
        self.assertEqual(0, seen.last_seen_age[1][2])
        self.assertEqual(3, hidden.last_seen_age[1][2])
        self.assertIsNone(hidden.last_seen_age[0][0])

    def test_watch_point_history_updates_only_from_normal_clear_mask(self):
        memory = BeliefMemory([_watch_point()])
        unseen = memory.update(_snapshot(1, sightings={_sighting()}))
        cleared = memory.update(_snapshot(3, visible={(1, 2)}))
        aging = memory.update(_snapshot(8))

        self.assertIsNone(unseen.watch_point_for("middle").last_confirmed_tick)
        self.assertEqual(0, cleared.watch_point_for("middle").confirmation_age)
        self.assertEqual(5, aging.watch_point_for("middle").confirmation_age)

    def test_new_round_resets_all_clear_sighting_and_watch_point_history(self):
        memory = BeliefMemory([_watch_point()])
        memory.update(
            _snapshot(12, visible={(1, 2)}, sightings={_sighting()})
        )
        reset = memory.update(_snapshot(0, round_number=2))

        self.assertEqual(0, reset.memory_tick)
        self.assertIsNone(reset.clear_age[1][2])
        self.assertIsNone(reset.enemy_for("enemy").last_seen_position)
        self.assertIsNone(reset.watch_point_for("middle").last_confirmed_tick)

    def test_dead_enemy_loses_stale_sighting_and_no_longer_marks_map(self):
        memory = BeliefMemory()
        memory.update(_snapshot(1, sightings={_sighting()}))
        dead = memory.update(_snapshot(2, enemy_alive=False))

        enemy = dead.enemy_for("enemy")
        self.assertFalse(enemy.is_alive)
        self.assertIsNone(enemy.last_seen_position)
        self.assertEqual(0, dead.last_seen_enemy_count[1][2])
        self.assertIsNone(dead.last_seen_age[1][2])

    def test_hidden_enemy_truth_has_no_input_path_and_cannot_be_followed(self):
        memory_a = BeliefMemory()
        memory_b = BeliefMemory()
        legal_history = _snapshot(1, sightings={_sighting((1, 2))})
        memory_a.update(legal_history)
        memory_b.update(legal_history)

        # These identical actor-safe snapshots can correspond to arbitrary,
        # different hidden enemy positions; neither position exists in input.
        hidden_a = memory_a.update(_snapshot(5))
        hidden_b = memory_b.update(_snapshot(5))

        self.assertEqual(hidden_a, hidden_b)
        self.assertEqual((1, 2), hidden_a.enemy_for("enemy").last_seen_position)
        parameters = inspect.signature(BeliefMemory.update).parameters
        self.assertEqual(("self", "snapshot"), tuple(parameters))
        self.assertNotIn("game", inspect.getsource(BeliefMemory.update))
        self.assertNotIn("character", inspect.getsource(BeliefMemory.update).lower())

    def test_duplicate_tick_is_idempotent_but_changed_content_is_rejected(self):
        memory = BeliefMemory()
        snapshot = _snapshot(3, visible={(0, 0)})
        first = memory.update(snapshot)

        self.assertIs(first, memory.update(snapshot))
        with self.assertRaisesRegex(BeliefInputError, "different content"):
            memory.update(_snapshot(3, visible={(0, 1)}))

    def test_setup_countdown_and_live_transition_have_monotonic_memory_age(self):
        memory = BeliefMemory()
        memory.update(_snapshot(3, phase="defender_setup", visible={(0, 0)}))
        setup = memory.update(_snapshot(1, phase="defender_setup"))
        live = memory.update(_snapshot(2, phase="live"))

        self.assertEqual(2, setup.clear_age[0][0])
        self.assertEqual(5, live.clear_age[0][0])

    def test_malformed_or_backwards_history_is_rejected(self):
        memory = BeliefMemory()
        memory.update(_snapshot(5))
        with self.assertRaisesRegex(BeliefInputError, "move forwards"):
            memory.update(_snapshot(4))
        with self.assertRaises(TypeError):
            memory.update(object())

        next_round_other_side = dataclasses.replace(
            _snapshot(0, round_number=2), side=Side.DEFENDER
        )
        with self.assertRaisesRegex(BeliefInputError, "mix team sides"):
            memory.update(next_round_other_side)

    def test_snapshots_are_immutable_and_explicit_reset_discards_history(self):
        memory = BeliefMemory()
        state = memory.update(_snapshot(1, visible={(0, 0)}))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.memory_tick = 99

        memory.reset()
        self.assertIsNone(memory.state)
        fresh = memory.update(_snapshot(20))
        self.assertEqual(0, fresh.memory_tick)
        self.assertIsNone(fresh.clear_age[0][0])

    def test_age_normalization_is_bounded_and_keeps_unknown_explicit(self):
        self.assertEqual(0.0, normalize_age(0, 100))
        self.assertEqual(0.25, normalize_age(25, 100))
        self.assertEqual(1.0, normalize_age(101, 100))
        self.assertEqual(1.0, normalize_age(None, 100))
        with self.assertRaises(ValueError):
            normalize_age(-1, 100)
        with self.assertRaises(ValueError):
            normalize_age(1, 0)


if __name__ == "__main__":
    unittest.main()
