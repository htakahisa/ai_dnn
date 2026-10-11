"""Confirm empty cells despite the engine's one-tick LOS display delay."""
from dataclasses import dataclass,fields,replace
from frc_v1.perception import FrcTeamSnapshot
from frc_v1.actions import MOVE_STEPS


@dataclass(frozen=True)
class TacticalEvidenceSnapshot(FrcTeamSnapshot):
    confirmed_empty_cells: tuple = ()
    temporary_body_cover: bool = False


def confirmed_empty_cells(snapshot):
    return getattr(snapshot,'confirmed_empty_cells',snapshot.visible_cells)


def planned_team_snapshot(snapshot, action_kinds):
    """Diagnostic projection of public chosen moves, not executed positions."""
    allies=[]
    for ally in snapshot.allies:
        dr,dc=MOVE_STEPS.get(action_kinds.get(ally.slot,'STAY'),(0,0)) if ally.alive else (0,0)
        allies.append(replace(ally,position=(ally.position[0]+dr,ally.position[1]+dc)))
    return replace(snapshot,allies=tuple(allies))


class PublicEmptyEvidence:
    def __init__(self):
        self.key = None
        self.visible = set()
        self.seen = set()
        self.empty = set()

    def tactical_snapshot(self, snapshot, *, temporary_body_cover=False):
        if snapshot.key != self.key:
            visible = set(snapshot.visible_cells)
            seen = {s.position for s in snapshot.sightings}
            consecutive = (self.key is not None and self.key[:2] == snapshot.key[:2]
                           and self.key[2] + 1 == snapshot.tick)
            self.empty = visible.intersection(self.visible) - self.seen - seen if consecutive else set()
            self.key, self.visible, self.seen = snapshot.key, visible, seen
        # This copy is used only by tactical threat validation. The sensor and
        # ordinary observation retain all geometrically visible cells.
        return TacticalEvidenceSnapshot(**{f.name:getattr(snapshot,f.name) for f in fields(FrcTeamSnapshot)},
                                        confirmed_empty_cells=tuple(sorted(self.empty)),
                                        temporary_body_cover=temporary_body_cover)


def empty_evidence_schema():
    return dict(version=2, empty='two_consecutive_visible_ticks_without_current_or_previous_sighting',
                source='public_only', base_observation='unchanged_geometric_visibility',
                reset='round_phase_or_tick_gap', selection='learned_actions')
