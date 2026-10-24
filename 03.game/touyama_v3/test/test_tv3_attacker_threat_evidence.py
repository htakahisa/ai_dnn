"""Public threat validation across delayed LOS display and repeated actor calls."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from frc_v1.perception import FrcPerceptionBuilder,Sighting
from touyama_v3.test.tv3_fixtures import attacker_world
from touyama_v3.tv3_attacker_threat_evidence import PublicEmptyEvidence,planned_team_snapshot
from touyama_v3.tv3_learn_attacker_fire_support import public_threats
from touyama_v3.tv3_learn_attacker_plant import policy_schema,observation_dim
from touyama_v3.tv3_scenario import Scenario


class ThreatEvidenceTests(unittest.TestCase):
    def test_potential_incoming_does_not_treat_mobile_ally_as_wall_cover(self):
        from types import SimpleNamespace
        from touyama_v3.tv3_learn_attacker_recon_cover import incoming_clear
        base=FrcPerceptionBuilder('A').build(attacker_world())
        own=replace(base.allies[0],position=(4,3),reveal=0,alive=True)
        shield=replace(base.allies[1],position=(4,5),alive=True,movement_disabled=0,warping=False)
        threat=Sighting(0,(4,7),'normal')
        snapshot=replace(base,allies=(own,shield)+tuple(replace(a,alive=False) for a in base.allies[2:]),
                         sightings=(threat,),smoke_cells=())
        scenario=SimpleNamespace(clear=lambda *args:True)
        legacy=PublicEmptyEvidence().tactical_snapshot(snapshot)
        candidate=PublicEmptyEvidence().tactical_snapshot(snapshot,temporary_body_cover=True)
        self.assertFalse(incoming_clear(scenario,legacy,own,own.position,threat))
        self.assertTrue(incoming_clear(scenario,candidate,own,own.position,threat))
        from touyama_v3.tv3_learn_attacker_fire_support import shot_clear
        self.assertFalse(shot_clear(scenario,candidate,own.position,threat.position,own.slot))
        self.assertTrue(shot_clear(scenario,candidate,own.position,threat.position,own.slot,potential_body_cover=True))
        fixed=replace(candidate,allies=(own,replace(shield,movement_disabled=2))+candidate.allies[2:])
        self.assertFalse(incoming_clear(scenario,fixed,own,own.position,threat))
        warping=replace(fixed,allies=(own,replace(shield,movement_disabled=2,warping=True))+candidate.allies[2:])
        self.assertTrue(incoming_clear(scenario,warping,own,own.position,threat))

    def test_potential_body_exposure_is_not_an_unmarked_indefinite_hold(self):
        import numpy as np
        from types import SimpleNamespace
        from touyama_v3.tv3_learn_attacker_recon_cover import recon_cover_plan
        from frc_v1.actions import KINDS
        base=FrcPerceptionBuilder('A').build(attacker_world())
        own=replace(base.allies[0],position=(3,3),reveal=0,alive=True)
        shield=replace(base.allies[1],position=(4,5),alive=True,movement_disabled=0,warping=False)
        snapshot=replace(base,allies=(own,shield)+tuple(replace(a,alive=False) for a in base.allies[2:]),
            sightings=(Sighting(0,(4,7),'normal'),),smoke_cells=(),grid=tuple(map(tuple,np.zeros((9,9),int))))
        scenario=SimpleNamespace(grid=np.zeros((9,9),int),clear=lambda origin,target:origin!=(3,3) and target!=(3,3))
        masks=SimpleNamespace(kind=np.ones((5,len(KINDS)),bool))
        self.assertIsNone(recon_cover_plan(scenario,snapshot,own,{},masks,(4,3)))
        candidate=PublicEmptyEvidence().tactical_snapshot(snapshot,temporary_body_cover=True)
        self.assertIsNone(recon_cover_plan(scenario,candidate,own,{},masks,(4,3)))
        own=replace(own,reveal=5)
        plan=recon_cover_plan(scenario,candidate,own,{},masks,(4,3))
        self.assertEqual(plan['step'],own.position)
        self.assertEqual(plan['incoming'],0)
        self.assertEqual(plan['future_incoming'],1)

    def test_chosen_ally_move_can_remove_temporary_body_cover(self):
        import numpy as np
        from types import SimpleNamespace
        from touyama_v3.tv3_learn_attacker_recon_cover import incoming_clear
        base=FrcPerceptionBuilder('A').build(attacker_world())
        own=replace(base.allies[0],position=(4,3),reveal=0,alive=True)
        shield=replace(base.allies[1],position=(4,5),alive=True)
        allies=(own,shield)+tuple(replace(a,alive=False) for a in base.allies[2:])
        threat=Sighting(0,(4,7),'normal')
        snapshot=replace(base,allies=allies,sightings=(threat,),smoke_cells=())
        scenario=SimpleNamespace(clear=lambda *args:True,grid=np.zeros((9,9),int))
        self.assertFalse(incoming_clear(scenario,snapshot,own,own.position,threat))
        projected=planned_team_snapshot(snapshot,{1:'N'})
        self.assertTrue(incoming_clear(scenario,projected,own,own.position,threat))
        self.assertEqual(snapshot.allies[1].position,(4,5))
        self.assertEqual(projected.allies[1].position,(3,5))

    def setUp(self):
        self.snapshot=replace(FrcPerceptionBuilder('A').build(attacker_world()),
            tick=10,phase='live',sightings=(),visible_cells=((7,38),))

    def test_first_visible_tick_preserves_known_enemy_until_empty_is_confirmed(self):
        evidence=PublicEmptyEvidence()
        tracks={4:((7,38),8,(0,0))}
        tactical=evidence.tactical_snapshot(self.snapshot)
        self.assertEqual(len(public_threats(self.snapshot,tracks)),0)
        self.assertEqual(len(public_threats(tactical,tracks)),1)
        self.assertEqual(self.snapshot.visible_cells,((7,38),))
        self.assertEqual(tactical.visible_cells,self.snapshot.visible_cells)
        # Five actors in one tick must share the same evidence, not five updates.
        for _ in range(5):
            self.assertEqual(evidence.tactical_snapshot(self.snapshot).confirmed_empty_cells,())
        next_tick=evidence.tactical_snapshot(replace(self.snapshot,tick=11))
        self.assertEqual(public_threats(next_tick,tracks),())

    def test_previous_sighting_prevents_immediate_empty_claim(self):
        evidence=PublicEmptyEvidence()
        evidence.tactical_snapshot(replace(self.snapshot,sightings=(Sighting(4,(7,38),'normal'),)))
        self.assertEqual(evidence.tactical_snapshot(replace(self.snapshot,tick=11)).confirmed_empty_cells,())
        self.assertEqual(evidence.tactical_snapshot(replace(self.snapshot,tick=12)).confirmed_empty_cells,((7,38),))

    def test_round_phase_and_missing_tick_reset_empty_confirmation(self):
        for changed in (dict(tick=12),dict(tick=11,round_number=2),dict(tick=11,phase='setup')):
            evidence=PublicEmptyEvidence()
            evidence.tactical_snapshot(self.snapshot)
            self.assertEqual(evidence.tactical_snapshot(replace(self.snapshot,**changed)).confirmed_empty_cells,())

    def test_candidate_schema_changes_semantics_without_enabling_rejected_edge_escape(self):
        scenario=Scenario()
        self.assertEqual(observation_dim(15),observation_dim(13))
        self.assertIn('empty_evidence',policy_schema(scenario,15))
        self.assertNotIn('recent_edge_progress_exception',policy_schema(scenario,15))
        self.assertNotIn('empty_evidence',policy_schema(scenario,13))
        self.assertIn('temporary_body_cover',policy_schema(scenario,16))
        self.assertNotIn('recent_edge_progress_exception',policy_schema(scenario,16))


if __name__=='__main__':
    unittest.main()
