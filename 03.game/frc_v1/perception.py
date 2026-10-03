"""The only FRC sensor allowed to read live game/character objects."""

from dataclasses import dataclass

from public_effects import DisplayEffect, PublicEffectReader
from frc_v1 import ROSTER, ABILITIES, ULTIMATES, FACING, FACING_STEPS


@dataclass(frozen=True)
class AllyState:
    slot: int
    name: str
    position: tuple[int, int]
    facing: str
    alive: bool
    hp: float
    max_hp: float
    charges: int
    points: int
    cost: int
    has_spike: bool
    plant_progress: int
    defuse_progress: int
    orb_progress: int
    blind: int
    reveal: int
    contract: int
    max_hp_lost: int
    movement_disabled: int
    warping: bool
    effective_iq: float
    accuracy: float
    hs_rate: float
    dodge: float
    reaction: float
    forced_facing: bool
    ramp_blocked: bool


@dataclass(frozen=True)
class EnemyState:
    enemy_id: int
    name: str
    role: str
    alive: bool


@dataclass(frozen=True)
class Sighting:
    enemy_id: int
    position: tuple[int, int]
    source: str


@dataclass(frozen=True)
class FrcTeamSnapshot:
    side: str
    round_number: int
    phase: str
    tick: int
    grid: tuple[tuple[int, ...], ...]
    allies: tuple[AllyState, ...]
    enemies: tuple[EnemyState, ...]
    sightings: tuple[Sighting, ...]
    visible_cells: tuple[tuple[int, int], ...]
    effects: tuple[DisplayEffect, ...]
    smoke_cells: tuple[tuple[int, int], ...]
    orbs: tuple[tuple[int, int], ...]
    setup_cells: tuple[tuple[int, int], ...]
    spike_dropped: tuple[int, int] | None
    spike_planted: tuple[int, int] | None
    is_planted: bool
    round_timer: float
    detonate_timer: float
    defuse_notified: bool

    @property
    def key(self):
        return self.round_number, self.phase, self.tick


def in_front(origin, position, facing):
    dr, dc = FACING_STEPS[FACING.index(facing)]
    return dr * (position[0] - origin[0]) + dc * (position[1] - origin[1]) >= -1e-9


def _pos(value):
    return None if value is None else tuple(map(int, value))


class FrcPerceptionBuilder:
    def __init__(self, side):
        if side not in ("A", "D"):
            raise ValueError("FRC side must be A or D")
        self.side = side
        self.effects = PublicEffectReader()

    def reset(self):
        self.effects.reset()

    def build(self, game):
        own = [c for c in game.chars if c.team == self.side]
        by_name = {str(getattr(c, "base_name", c.name)): c for c in own}
        if len(own) != 5 or set(by_name) != set(ROSTER):
            raise ValueError("FRC requires Furina, Lisa, Lohen, Jean, Arlecchino on its own side")
        own = [by_name[name] for name in ROSTER]
        allies = []
        for slot, c in enumerate(own):
            if c.ability_name != ABILITIES[slot] or c.ultimate_name != ULTIMATES[slot]:
                raise ValueError("FRC roster abilities changed; update schema and retrain")
            charge_attr = {"DANCE": "dance_charges", "SMOKE": "smoke_charges", "HUNT": "hunt_charges",
                           "RECON": "recon_charges", "ASH": "ash_charges"}[ABILITIES[slot]]
            forced = getattr(c, "forced_facing_next_tick", None)
            facing = forced if forced in FACING else c.facing
            allies.append(AllyState(slot, ROSTER[slot], _pos(c.pos), facing, bool(c.is_alive),
                float(c.hp), float(c.max_hp), int(getattr(c, charge_attr, 0)),
                int(c.ultimate_points), int(c.ultimate_cost), bool(c.has_spike),
                int(c.plant_timer), int(c.defuse_timer), int(getattr(c, "orb_collect_timer", 0)),
                int(c.blind_remaining), int(c.reveal_remaining), int(getattr(c, "life_contract_remaining", 0)),
                int(getattr(c, "contract_max_hp_lost", 0)), int(getattr(c, "movement_disabled_remaining", 0)),
                any(p.get("owner") == c.name for p in getattr(game, "escape_portals", ())),
                float(getattr(c, "effective_iq", c.iq)), float(c.accuracy), float(c.hs_rate),
                float(c.dodge_rate), float(c.reaction), bool(forced in FACING or getattr(c, "facing_forced_this_tick", False)),
                bool(game._ramp_blocks_movement(c))))
        grid = tuple(tuple(map(int, row)) for row in game.grid)
        smoke = set(game._smoke_cells())
        visible = set()
        active_viewers = [ally for ally in allies if ally.alive and ally.blind == 0]
        for r, row in enumerate(grid):
            for col, cell in enumerate(row):
                if cell == 1 or (r, col) in smoke:
                    continue
                if any(in_front(v.position, (r, col), v.facing) and
                       game.check_cell_line_of_sight(v.position, (r, col), block_smoke=True)
                       for v in active_viewers):
                    visible.add((r, col))
        enemies, sightings = [], []
        for index, enemy in enumerate(c for c in game.chars if c.team != self.side):
            enemies.append(EnemyState(index, str(getattr(enemy, "base_name", enemy.name)), str(enemy.role), bool(enemy.is_alive)))
            if not enemy.is_alive:
                continue
            # The engine applies LOS reveal after resolving shots. Geometry
            # alone must not expose the enemy on that first contact tick.
            recon_revealed = int(getattr(enemy, "reveal_remaining", 0)) > 0
            if recon_revealed or bool(getattr(enemy, "los_revealed", False)):
                sightings.append(Sighting(index, _pos(enemy.pos), "reveal" if recon_revealed else "normal"))
        setup = getattr(game, "defender_setup_phase", None)
        during_setup = bool(getattr(setup, "active", False))
        allowed = ()
        if during_setup:
            allowed = tuple((r, c) for r, row in enumerate(grid) for c, value in enumerate(row)
                            if value != 1 and setup.can_move_to(self.side, r, c))
        return FrcTeamSnapshot(self.side, int(getattr(game, "current_round", 1)),
            "setup" if during_setup else "live", int(setup.ticks_remaining) if during_setup else int(game.battle_tick),
            grid, tuple(allies), tuple(enemies), tuple(sightings), tuple(sorted(visible)), self.effects.read(game),
            tuple(sorted(smoke)), tuple(sorted(_pos(p) for p in getattr(game, "available_orbs", ()))), allowed,
            _pos(game.spike_pos), _pos(game.planted_pos), bool(game.is_planted),
            float(game.round_timer), float(game.detonate_timer), bool(getattr(game, "active_defuser_name", None)))
