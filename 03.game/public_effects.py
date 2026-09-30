"""Copied display geometry shared by rendering and team perception.

This boundary deliberately does not export owners, future projectile paths,
cast targets or simulation countdowns. The opaque handle identifies a displayed
object across frames; it does not identify its caster.
"""

from dataclasses import dataclass
import math


def displayed_projectile_path(projectile):
    path = projectile.get("path", ())
    if not path:
        return ()
    progress = max(0, min(int(projectile.get("progress", 0)), len(path) - 1))
    return tuple((int(r), int(c)) for r, c in path[:progress + 1])


def displayed_area_cells(effect):
    return tuple(sorted((int(r), int(c)) for r, c in effect.get("cells", ())))


@dataclass(frozen=True)
class DisplayEffect:
    handle: int
    kind: str
    phase: str
    position: tuple[int, int] | None
    cells: tuple[tuple[int, int], ...] = ()
    trail: tuple[tuple[int, int], ...] = ()
    direction: tuple[float, float] = (0.0, 0.0)
    level: int = 0
    blinking: bool = False
    drawn_this_frame: bool = True
    displayed_hp: int = 0


class PublicEffectReader:
    def __init__(self):
        self.reset()

    def reset(self):
        self._handles = {}
        self._objects = {}
        self._serial = 0

    def read(self, game):
        effects, present = [], set()

        def add(raw, kind, phase, position=None, **fields):
            key = (kind, id(raw))
            present.add(key)
            if key not in self._handles:
                self._serial += 1
                self._handles[key] = self._serial
            # Retain identities inside the sensor only until disappearance.
            # This prevents Python id reuse from continuing an old effect's age.
            self._objects[key] = raw
            pos = None if position is None else tuple(map(int, position))
            effects.append(DisplayEffect(self._handles[key], kind, phase, pos, **fields))

        for kind in ("FLASH", "RECON", "ASH"):
            for raw in getattr(game, kind.lower() + "_projectiles", ()):
                trail = displayed_projectile_path(raw)
                if not trail:
                    continue
                direction = (0.0, 0.0)
                if len(trail) > 1:
                    dr, dc = trail[-1][0] - trail[0][0], trail[-1][1] - trail[0][1]
                    length = math.hypot(dr, dc)
                    direction = (dr / length, dc / length) if length else direction
                add(raw, kind, "flight", trail[-1], trail=trail, direction=direction)
        for attr, kind in (("tunnel_bursts", "TUNNEL"), ("neon_bursts", "NEON"),
                           ("balemoon_warnings", "BALEMOON"),
                           ("destruction_areas", "DESTRUCTION"),
                           ("recon_bursts", "RECON"), ("flash_bursts", "FLASH")):
            for raw in getattr(game, attr, ()):
                phase = "warning" if kind == "BALEMOON" else str(raw.get("phase", "active"))
                add(raw, kind, phase, raw.get("pos"), cells=displayed_area_cells(raw),
                    level=int(raw.get("level", 0)))
        for raw in getattr(game, "smokes", ()):
            # Only the visible blink state is exposed, never the countdown.
            blinking = int(raw.get("remaining_ticks", 0)) <= 3
            add(raw, "SMOKE", "active", raw.get("center"), cells=displayed_area_cells(raw),
                blinking=blinking,
                drawn_this_frame=not (blinking and int(getattr(game, "battle_tick", 0)) % 2 == 0))
        for raw in getattr(game, "escape_portals", ()):
            add(raw, "ESCAPE", "active", raw.get("pos"))
        for raw in getattr(game, "ultimate_trails", ()):
            # RAID colour is explicitly driven by the remaining display time.
            colour_phase = "fresh" if raw.get("remaining_ticks", 1) >= 3 else (
                "fading" if raw.get("remaining_ticks", 1) == 2 else "faint")
            add(raw, "RAID", colour_phase, raw.get("end"), cells=displayed_area_cells(raw),
                direction=tuple(map(float, raw.get("direction", (0, 0)))))
        for raw in getattr(game, "monitor_drones", ()):
            if raw.is_alive:
                add(raw, "MONITOR", "active", raw.pos, displayed_hp=int(raw.hp))
        self._handles = {key: value for key, value in self._handles.items() if key in present}
        self._objects = {key: value for key, value in self._objects.items() if key in present}
        return tuple(effects)
