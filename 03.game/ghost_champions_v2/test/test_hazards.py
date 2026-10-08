"""Visibility boundary, avoidance decisions, credit and checkpoint compatibility."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from public_effects import PublicEffectReader,DisplayEffect
from ghost_champions_v2.hazards import PublicHazards,avoidance_reward,EFFECT_SLOTS,EFFECT_WIDTH
from ghost_champions_v2.config import load_config
from ghost_champions_v2.tactics import AttackerTactics
from ghost_champions_v2.rl.observation import ObservationEncoder,OBS_DIM,BLOCKS,LEGACY_V2_BLOCKS
from ghost_champions_v2.rl.actions import ACTIONS,candidates
from ghost_champions_v2.rl.policy import ResidualPolicy
from ghost_champions_v2.rl.rollout import Recorder
from ghost_champions_v2.tools.upgrade_observation_checkpoint import upgraded_policy,upgrade


class HazardTests(unittest.TestCase):
    def setUp(self):
        self.char=NS(name="self",team="A",pos=[5,5],facing="E",is_alive=True,hp=100,
                     has_spike=False,plant_timer=0,blind_remaining=0,ability_name="HUNT")
        self.game=NS(grid=np.zeros((14,14),int),chars=[self.char],battle_tick=10,smokes=[])
        self.state=dict(grid=self.game.grid,chars=[self.char],battle_tick=10,is_planted=False,
                        smoke_cells=set(),enemy_roster=[])
        self.tactics=AttackerTactics(load_config())

    def observe(self):
        self.state["public_effects"]=PublicEffectReader().read_visible(self.game,"A")
        self.tactics.observe(self.char,self.state)
        return ObservationEncoder().encode(self.char,self.state,self.tactics,([5,5],"HOLD"))

    def test_future_path_owner_and_private_countdown_cannot_change_input(self):
        raw=dict(path=[(5,7),(5,8),(5,9),(5,10)],progress=1,owner="hidden",team="D",ticks_alive=1)
        self.game.flash_projectiles=[raw]
        self.game.neon_bursts=[dict(cells={(5,5),(5,6)},pos=(5,6),phase="warning",remaining_ticks=9)]
        before=self.observe()
        raw.update(path=[(5,7),(5,8),(1,1),(2,1)],owner="different",team="A",ticks_alive=4)
        self.game.neon_bursts[0]["remaining_ticks"]=1
        after=self.observe()
        np.testing.assert_array_equal(before,after)
        self.game.neon_bursts[0]["phase"]="active"
        self.assertFalse(np.array_equal(after,self.observe()))
        self.assertEqual(after.shape,(OBS_DIM,))

    def test_wall_smoke_blindness_and_partial_warning_do_not_reveal_hidden_geometry(self):
        self.game.grid[:,7]=1
        self.game.neon_bursts=[dict(cells={(5,6),(5,8)},pos=(5,8),phase="warning")]
        self.game.flash_projectiles=[dict(path=[(5,6),(5,8)],progress=1)]
        reader=PublicEffectReader()
        effects=reader.read_visible(self.game,"A")
        self.assertEqual(len(effects),1)
        self.assertEqual(effects[0].cells,((5,6),))
        self.assertIsNone(effects[0].position)
        self.game.smokes=[dict(cells={(5,6)},center=(5,6),remaining_ticks=5)]
        self.assertFalse(any(e.kind=="NEON" for e in reader.read_visible(self.game,"A")))
        self.char.blind_remaining=5
        self.assertEqual(reader.read_visible(self.game,"A"),())

    def test_projectile_direction_uses_only_visible_trail(self):
        self.game.flash_projectiles=[dict(path=[(5,3),(5,4),(5,5),(5,6)],progress=3)]
        e=PublicEffectReader().read_visible(self.game,"A")[0]
        self.assertEqual(e.trail,((5,5),(5,6)))
        self.assertEqual(e.direction,(0.,1.))
        self.assertFalse(hasattr(e,"owner"))
        self.assertFalse(hasattr(e,"remaining_ticks"))

    def test_warning_escape_can_take_multiple_steps_and_unlocks_plant_mask(self):
        self.char.has_spike=True
        self.char.plant_timer=2
        self.state["grid"][5,5]=2
        area=tuple((r,c) for r in range(4,7) for c in range(4,7))
        self.state["public_effects"]=(DisplayEffect(1,"NEON","warning",(5,5),cells=area),)
        self.tactics.observe(self.char,self.state)
        self.tactics.plant_commit=dict(holder="self",target=(5,5),reason="safe_plant")
        proposal=self.tactics.dodge(self.char,self.state)
        self.assertIsNotNone(proposal)
        self.assertEqual(sum(abs(proposal[0][i]-self.char.pos[i]) for i in (0,1)),1)
        choices,mask=candidates(self.char,self.state,self.tactics,proposal)
        self.assertGreater(mask.sum(),1)
        self.assertTrue(any(mask[i] for i in range(4,8)))
        self.assertEqual(self.tactics.plant_commit["holder"],"self")
        self.state["public_effects"]=()
        self.tactics.observe(self.char,self.state)
        self.assertIsNone(self.tactics.dodge(self.char,self.state))
        _,mask=candidates(self.char,self.state,self.tactics,([5,5],"PLANT"))
        self.assertEqual(np.flatnonzero(mask).tolist(),[3])

    def test_flying_flash_seeks_cover_and_does_not_cancel_active_plant(self):
        # Reach cover behind a wall that screens the displayed projectile
        # and all of its short public-direction extrapolation.
        self.game.grid[4,5:9]=1
        self.state["public_effects"]=(DisplayEffect(1,"FLASH","flight",(5,8),
                                                 trail=((5,9),(5,8)),direction=(0.,-1.)),)
        self.tactics.observe(self.char,self.state)
        proposal=self.tactics.dodge(self.char,self.state)
        self.assertIsNotNone(proposal)
        self.assertNotEqual(tuple(proposal[0]),tuple(self.char.pos))
        self.char.plant_timer=2
        self.assertIsNone(self.tactics.dodge(self.char,self.state))

    def test_flash_risk_has_no_distance_attenuation_like_the_engine(self):
        self.state["public_effects"]=(DisplayEffect(1,"FLASH","flight",(5,12),direction=(0.,0.)),)
        hazards=PublicHazards()
        hazards.update(self.state)
        self.assertEqual(hazards.risk((5,1)),hazards.risk((5,11)))

    def test_slot_overflow_still_marks_all_local_warning_cells(self):
        effects=[DisplayEffect(i,"NEON","warning",(5,5),cells=((5,5),))
                 for i in range(EFFECT_SLOTS+3)]
        effects.append(DisplayEffect(100,"TUNNEL","warning",None,cells=((5,6),)))
        hazards=PublicHazards()
        hazards.update(dict(self.state,public_effects=effects))
        values=hazards.features(self.char)
        local=values[EFFECT_SLOTS*EFFECT_WIDTH:-8].reshape(7,7,3)
        self.assertAlmostEqual(float(local[3,4,0]),.8)
        self.assertGreater(values[-7],0)
        self.assertEqual(values.shape,(363,))

    def test_own_cast_requires_actual_consumption_and_never_uses_hidden_team(self):
        self.char.flash_charges=1
        self.state["public_effects"]=(DisplayEffect(1,"FLASH","flight",(5,6),
                                                 trail=((5,5),(5,6)),direction=(0.,1.)),)
        hazards=PublicHazards()
        hazards.note_action(self.char,([5,5],{"ability":"FLASH","target":(5,10)}),10)
        hazards.update(self.state)
        self.assertGreater(hazards.risk(self.char.pos),0)
        self.assertFalse(hazards.known_own)
        self.char.flash_charges=0
        hazards.update(self.state)
        self.assertEqual(hazards.known_own,{1})
        self.assertEqual(hazards.risk(self.char.pos),0.)
        self.assertEqual(hazards.features(self.char)[25],1.)
        # An unexecuted proposal replaced by a move must not classify a cast.
        hazards.reset()
        self.char.flash_charges=1
        hazards.note_action(self.char,([5,5],{"ability":"FLASH","target":(5,10)}),10)
        hazards.note_action(self.char,([5,6],"MOVE"),10)
        self.char.flash_charges=0
        hazards.update(self.state)
        self.assertFalse(hazards.known_own)

    def test_safe_character_does_not_walk_back_into_an_active_area(self):
        self.state["public_effects"]=(DisplayEffect(1,"NEON","active",(5,6),cells=((5,6),)),)
        self.tactics.observe(self.char,self.state)
        proposed=([5,6],"MOVE")
        result=self.tactics.avoid_entry(self.char,self.state,proposed)
        self.assertEqual(tuple(result[0]),(5,5))

    def test_real_engine_warning_reaches_policy_and_teacher_evades_before_damage(self):
        import contextlib
        from functools import partial
        import io
        from controllers import DefaultAttackerController
        from ghost_champions_v1_macro import GhostChampionsV1DefenderController
        from ghost_champions_v2.rl.controller import LearnedAttackerController
        from map_data import NEW_MAZE_STR
        from party_presets import get_preset
        from run_game import VisualFPSBattle
        from simulation_runtime import cpu_inference
        from team_ai import DualRoleTeamAI
        class Stay:
            def decide_move(self,char,state): return list(char.pos)
        for iq in (True,False):
            recorder=Recorder()
            gc,enemy=get_preset("Ghost Champions"),get_preset("Omoko Gaming")
            with self.subTest(iq=iq),contextlib.redirect_stdout(io.StringIO()),cpu_inference():
                team=DualRoleTeamAI("warning test",partial(LearnedAttackerController,mode="teacher",recorder=recorder),
                                    GhostChampionsV1DefenderController,use_iq_perception=iq)
                game=VisualFPSBattle(NEW_MAZE_STR,team,DualRoleTeamAI("stationary",DefaultAttackerController,Stay),
                    headless=True,attacker_roster=list(gc.players),defender_roster=list(enemy.players),
                    spike_holder_name=gc.spike_holder,defender_spike_holder_name=enemy.spike_holder,
                    attacker_igl_name=gc.igl,defender_igl_name=enemy.igl,disable_side_swap=True)
                game.stop_after_round=True
                game.defender_setup_phase.finish()
                holder=next(c for c in game.chars if c.team=="A" and c.has_spike)
                holder.pos=[10,40]
                holder.facing="N"
                holder.plant_timer=2
                for c in game.chars:
                    if c.team=="D": c.pos=[1,18+sum(d.team=="D" and d.name<c.name for d in game.chars)]
                owner=next(c for c in game.chars if c.team=="D")
                game.neon_bursts=[dict(cells={(10,40),(9,40)},pos=(10,40),phase="warning",
                                      remaining_ticks=0,owner=owner.name,team="D")]
                original=game.attacker_controller.decide_move
                game.attacker_controller.decide_move=lambda c,state:original(c,state) if c is holder else list(c.pos)
                hp=holder.hp
                game.step_tick()
                self.assertNotIn(tuple(holder.pos),{(10,40),(9,40)})
                self.assertEqual(holder.hp,hp)
                row=recorder.by_agent[str(holder.name)][0]
                self.assertEqual(row["obs"].shape,(OBS_DIM,))
                self.assertGreater(row["obs"][-8:].sum(),0.)
                self.assertEqual(row["teacher"],0)
                self.assertGreater(row["mask"].sum(),1)

    def test_observed_age_starts_on_first_sighting_and_reset_clears_it(self):
        hazards=PublicHazards()
        self.state["public_effects"]=(DisplayEffect(7,"NEON","warning",(5,5),cells=((5,5),)),)
        hazards.update(self.state)
        self.assertEqual(hazards.features(self.char)[21],0.)
        hazards.update(dict(self.state,battle_tick=12))
        self.assertAlmostEqual(float(hazards.features(self.char)[21]),.1)
        hazards.reset()
        hazards.update(dict(self.state,battle_tick=100))
        self.assertEqual(hazards.features(self.char)[21],0.)

    def test_danger_reward_telescopes_and_credits_only_the_affected_agent(self):
        risks=[0.,.8,.8,0.]
        gamma=.99
        rewards=[avoidance_reward((0,0),(0,0),a,b,gamma=gamma,terminal=i==2,
                                  weight=.05,hit_weight=.03,kinds=set())
                 for i,(a,b) in enumerate(zip(risks,risks[1:]))]
        self.assertAlmostEqual(sum(gamma**i*r for i,r in enumerate(rewards)),0.)
        hit=avoidance_reward((0,0),(10,0),0.,0.,gamma=gamma,terminal=False,
                             weight=.05,hit_weight=.03,kinds={"FLASH"})
        self.assertAlmostEqual(hit,-.03)
        recorder=Recorder()
        for name in ("hit","safe"):
            recorder.record(name,np.zeros(OBS_DIM),np.ones(len(ACTIONS)),0,0.,0.,0)
        recorder.reward(0.,gamma,False,{"hit":hit})
        self.assertAlmostEqual(recorder.by_agent["hit"][0]["reward"],-.03)
        self.assertEqual(recorder.by_agent["safe"][0]["reward"],0.)

    def test_legacy_upgrade_preserves_old_logits_but_is_untrained_and_never_overwrites(self):
        schema=dict(version=2,dim=448,blocks=LEGACY_V2_BLOCKS,enemy_roster_version=1)
        old=ResidualPolicy()
        weights=old.state_dict()
        weights["features.0.weight"]=weights["features.0.weight"][:,:448].clone()
        with tempfile.TemporaryDirectory() as folder:
            source,output=Path(folder)/"old.pt",Path(folder)/"new.pt"
            torch.save(dict(schema=schema,schema_hash=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest(),
                            hidden=old.hidden,actions=ACTIONS,model_state_dict=weights,verified=True),source)
            original=source.read_bytes()
            with self.assertRaises(ValueError): ResidualPolicy.load(source)
            model,metadata=upgraded_policy(source)
            self.assertFalse(metadata["hazard_trained"])
            self.assertFalse(metadata["verified"])
            x=torch.randn(2,OBS_DIM)
            expected=torch.tanh(torch.nn.functional.linear(x[:,:448],weights["features.0.weight"],weights["features.0.bias"]))
            expected=old.features[2:](expected)
            torch.testing.assert_close(model.features(x),expected)
            upgrade(source,output)
            _,payload=ResidualPolicy.load(output)
            self.assertFalse(payload["verified"])
            with self.assertRaises(ValueError): upgrade(source,source)
            with self.assertRaises(ValueError): upgrade(source,output)
            self.assertEqual(source.read_bytes(),original)


if __name__=="__main__":
    unittest.main()
