import contextlib
import copy
import io
import math
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ghost_champions_v2.config import load_config
from ghost_champions_v2.controller import GhostChampionsV2AttackerController
from ghost_champions_v1_macro import (GhostChampionsV1AttackerController,
                                       GhostChampionsV1DefenderController)


class RuntimeTests(unittest.TestCase):
    def test_registration_keeps_existing_defender_and_adds_both_selectors(self):
        from run_game import _build_team_ai
        from roster_select import TEAM_AI_OPTIONS
        from run_competition_manager import CONTROLLER_OPTIONS
        team=_build_team_ai("ghost_champions_v2")
        self.assertIs(team.attacker_factory,GhostChampionsV2AttackerController)
        self.assertIs(team.defender_factory,GhostChampionsV1DefenderController)
        self.assertEqual(TEAM_AI_OPTIONS["Ghost Champions v2"],"ghost_champions_v2")
        self.assertEqual(CONTROLLER_OPTIONS["Ghost Champions v2"],"ghost_champions_v2")

    def test_disabled_features_and_tyg_return_exact_v1_decision(self):
        from party_presets import get_preset
        config=load_config()
        char=NS(name="Absol",pos=[20,18],team="A",is_alive=True,has_spike=True)
        state=dict(grid=np.zeros((26,44),int),chars=[char],battle_tick=1,
                   is_planted=False,enemy_roster=[dict(base_name=str(n))
                   for n in get_preset("Touyama Gaming").players])
        sentinel=([19,18],"MOVE",{"facing":"N"})
        with patch.object(GhostChampionsV1AttackerController,"__init__",return_value=None), \
             patch.object(GhostChampionsV1AttackerController,"decide_move",return_value=sentinel) as base:
            for stage in ("hold","recon","profiles","entry"):
                c=GhostChampionsV2AttackerController(stage=stage)
                self.assertIs(c.decide_move(char,state),sentinel)
            config["flags"]=dict.fromkeys(config["flags"],False)
            state["enemy_roster"]=[]
            c=GhostChampionsV2AttackerController(config=config)
            self.assertIs(c.decide_move(char,state),sentinel)
            self.assertEqual(base.call_count,5)

    def test_planting_is_not_interrupted_by_early_recon(self):
        char=NS(name="seeker",pos=[2,2],team="A",is_alive=True,has_spike=True,
                ability_name="RECON",recon_charges=2,plant_timer=1)
        state=dict(grid=np.zeros((6,6),int),chars=[char],enemy_roster=[],
                   battle_tick=5,is_planted=False)
        with patch.object(GhostChampionsV1AttackerController,"__init__",return_value=None), \
             patch.object(GhostChampionsV1AttackerController,"decide_move",return_value=([2,2],"PLANT")):
            c=GhostChampionsV2AttackerController(stage="recon")
            self.assertEqual(c.decide_move(char,state),([2,2],"PLANT"))

    def test_real_engine_postplant_uses_legal_moves_and_returns_near_spike(self):
        import torch
        from controllers import DefaultAttackerController
        from map_data import NEW_MAZE_STR
        from party_presets import get_preset
        from run_game import VisualFPSBattle
        from team_ai import DualRoleTeamAI
        from ghost_champions_v2 import build_team_ai
        from simulation_runtime import cpu_inference

        class Stay:
            def decide_move(self,char,state): return list(char.pos)

        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        gc,enemy=get_preset("Ghost Champions"),get_preset("Omoko Gaming")
        output=io.StringIO()
        with contextlib.redirect_stdout(output), cpu_inference(), patch.dict(os.environ,{"GC_V2_STAGE":"hold"}):
            game=VisualFPSBattle(NEW_MAZE_STR,build_team_ai(),
                DualRoleTeamAI("Stationary",DefaultAttackerController,Stay),headless=True,
                attacker_roster=list(gc.players),defender_roster=list(enemy.players),
                spike_holder_name=gc.spike_holder,defender_spike_holder_name=enemy.spike_holder,
                attacker_igl_name=gc.igl,defender_igl_name=enemy.igl,disable_side_swap=True)
            game.defender_setup_phase.finish()
            game.is_planted=True
            game.planted_pos=(8,3)
            game.spike_pos=None
            game.detonate_timer=55
            attackers=[c for c in game.chars if c.team=="A"]
            positions=[(11,7),(12,6),(11,6),(13,3),(12,2)]
            for char,point in zip(attackers,positions):
                self.assertNotEqual(game.grid[point],1)
                char.pos=list(point)
                char.has_spike=False
            controller=game.attacker_controller.inner
            controller.tactics.plant_attempt=((8,3),0)
            for _ in range(20):
                game._simulate_tick()
                self.assertEqual(game.current_round,1)
                for char in attackers:
                    self.assertNotEqual(game.grid[tuple(char.pos)],1)
                    self.assertTrue(0<=char.pos[0]<game.height and 0<=char.pos[1]<game.width)
            survivors=[c for c in attackers if c.is_alive]
            self.assertTrue(survivors)
            self.assertLessEqual(sum(math.dist(c.pos,game.planted_pos) for c in survivors)/len(survivors),6)
            self.assertEqual(controller.tactics.spike,(8,3))
            self.assertEqual(game.replay_frames[-1]["gc_attacker_v2"]["spike"],[8,3])
        self.assertNotIn("load failed",output.getvalue())

    def test_real_engine_entry_reaches_plant_and_spends_recon(self):
        import torch
        from controllers import DefaultAttackerController
        from map_data import NEW_MAZE_STR
        from party_presets import get_preset
        from run_game import VisualFPSBattle
        from team_ai import DualRoleTeamAI
        from ghost_champions_v2 import build_team_ai
        from simulation_runtime import cpu_inference

        class Stay:
            def decide_move(self,char,state): return list(char.pos)

        random.seed(124)
        np.random.seed(124)
        torch.manual_seed(124)
        gc,enemy=get_preset("Ghost Champions"),get_preset("Omoko Gaming")
        with contextlib.redirect_stdout(io.StringIO()), cpu_inference(), patch.dict(os.environ,{"GC_V2_STAGE":"entry"}):
            game=VisualFPSBattle(NEW_MAZE_STR,build_team_ai(),
                DualRoleTeamAI("Stationary",DefaultAttackerController,Stay),headless=True,
                attacker_roster=list(gc.players),defender_roster=list(enemy.players),
                spike_holder_name=gc.spike_holder,defender_spike_holder_name=enemy.spike_holder,
                attacker_igl_name=gc.igl,defender_igl_name=enemy.igl,disable_side_swap=True)
            controller=game.attacker_controller.inner
            for _ in range(150):
                game._simulate_tick()
                if game.defender_setup_phase.active:
                    self.assertFalse(controller.tactics.recon_sequences)
                if game.is_planted or game.round_over or game.current_round != 1:
                    break
            attackers=[c for c in game.chars if c.team=="A"]
            self.assertTrue(game.is_planted,[(c.name,c.pos,c.is_alive) for c in attackers])
            self.assertEqual(controller.tactics.opponent,"OMG")
            self.assertLessEqual(sum(c.recon_charges for c in attackers),1)
            self.assertLess(game.planted_pos[1],game.width/2)
            self.assertEqual(game.planted_pos,(8,3))
            self.assertIn("Mid",[e["axis"] for e in controller.tactics.events])
            self.assertEqual(game._analytics_tactic_snapshot()["final_attack_site"],"A")

    def test_real_engine_frc_recon_can_unlock_the_two_defender_site(self):
        self._assert_frc_recon_site([(7,1),(10,2),(7,38),(3,39),(2,40)],"A")

    def test_real_engine_frc_recon_identifies_b_when_a_has_three(self):
        self._assert_frc_recon_site([(7,1),(10,2),(8,8),(2,40),(7,38)],"B")

    def _assert_frc_recon_site(self,positions,expected):
        import torch
        from controllers import DefaultAttackerController
        from map_data import NEW_MAZE_STR
        from party_presets import get_preset
        from run_game import VisualFPSBattle
        from team_ai import DualRoleTeamAI
        from ghost_champions_v2 import build_team_ai
        from simulation_runtime import cpu_inference

        class Stay:
            def decide_move(self,char,state): return list(char.pos)

        random.seed(125)
        np.random.seed(125)
        torch.manual_seed(125)
        gc,enemy=get_preset("Ghost Champions"),get_preset("Furina Classic")
        with contextlib.redirect_stdout(io.StringIO()), cpu_inference(), patch.dict(os.environ,{"GC_V2_STAGE":"entry"}):
            game=VisualFPSBattle(NEW_MAZE_STR,build_team_ai(),
                DualRoleTeamAI("Stationary",DefaultAttackerController,Stay),headless=True,
                attacker_roster=list(gc.players),defender_roster=list(enemy.players),
                spike_holder_name=gc.spike_holder,defender_spike_holder_name=enemy.spike_holder,
                attacker_igl_name=gc.igl,defender_igl_name=enemy.igl,disable_side_swap=True)
            game.defender_setup_phase.finish()
            defenders=[c for c in game.chars if c.team=="D"]
            for char,point in zip(defenders,positions):
                self.assertNotEqual(game.grid[point],1)
                char.pos=list(point)
                char.facing="N"
            controller=game.attacker_controller.inner
            selected=None
            for _ in range(100):
                game._simulate_tick()
                if controller.tactics.selected_site and selected is None:
                    selected=(controller.tactics.selected_site,
                              controller.observed_defenders_by_axis.copy())
                if game.is_planted or game.round_over or game.current_round != 1:
                    break
            self.assertIsNotNone(selected,controller.attacker_snapshot())
            self.assertEqual(selected[0],expected)
            self.assertEqual(selected[1][expected],2)


if __name__=="__main__": unittest.main()
