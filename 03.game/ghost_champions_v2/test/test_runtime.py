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

    def test_disabled_features_return_v1_decision_for_every_roster(self):
        from party_presets import get_preset
        config=load_config()
        char=NS(name="Absol",pos=[20,18],team="A",is_alive=True,has_spike=True)
        state=dict(grid=np.zeros((26,44),int),chars=[char],battle_tick=1,
                   is_planted=False,enemy_roster=[dict(base_name=str(n))
                   for n in get_preset("Touyama Gaming").players])
        sentinel=([19,18],"MOVE",{"facing":"N"})
        with patch.object(GhostChampionsV1AttackerController,"__init__",return_value=None), \
             patch.object(GhostChampionsV1AttackerController,"decide_move",return_value=sentinel) as base:
            config["flags"]=dict.fromkeys(config["flags"],False)
            for roster in (state["enemy_roster"],[]):
                state["enemy_roster"]=roster
                c=GhostChampionsV2AttackerController(config=config)
                self.assertIs(c.decide_move(char,state),sentinel)
            self.assertEqual(base.call_count,2)

    def test_v2_disables_inherited_tyg_site_override_and_v1_keeps_default(self):
        from gc_v1.opponent_site_gc import forced_attack_site,attack_plant_cells
        from party_presets import get_preset
        game=NS(defender_roster=list(get_preset("Touyama Gaming").players))
        self.assertEqual(forced_attack_site(game),"A")
        with patch.object(GhostChampionsV1AttackerController,"__init__",return_value=None), \
             patch.object(GhostChampionsV1AttackerController,"set_game"):
            controller=GhostChampionsV2AttackerController()
            controller.set_game(game)
        self.assertIsNone(forced_attack_site(game))
        grid=np.zeros((26,44),int)
        grid[8,3]=grid[7,40]=2
        self.assertEqual(set(attack_plant_cells(game,grid)),{(8,3),(7,40)})
        self.assertTrue(all(controller.config["flags"].values()))
        self.assertEqual(forced_attack_site(NS(defender_roster=game.defender_roster)),"A")

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
            controller.tactics.initial_site=controller.tactics.selected_site="A"
            controller.tactics.opening_strategy="DEFAULT"
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
            self.assertEqual(game.grid[game.planted_pos],2)
            self.assertIn("Mid",[e["axis"] for e in controller.tactics.events])
            self.assertEqual(game._analytics_tactic_snapshot()["final_attack_site"],
                             "A" if game.planted_pos[1]<game.width/2 else "B")

    def test_real_engine_frc_recon_can_unlock_the_two_defender_site(self):
        self._assert_frc_recon_site([(7,1),(10,2),(7,38),(3,39),(2,40)],"A")

    def test_real_engine_can_plant_b_against_tyg(self):
        from controllers import DefaultAttackerController
        from map_data import NEW_MAZE_STR
        from party_presets import get_preset
        from run_game import VisualFPSBattle
        from team_ai import DualRoleTeamAI
        from ghost_champions_v2 import build_team_ai
        from simulation_runtime import cpu_inference
        from gc_v1.opponent_site_gc import forced_attack_site

        class Stay:
            def decide_move(self,char,state): return list(char.pos)

        gc,enemy=get_preset("Ghost Champions"),get_preset("Touyama Gaming")
        with contextlib.redirect_stdout(io.StringIO()), cpu_inference(), patch.dict(os.environ,{"GC_V2_STAGE":"entry"}):
            game=VisualFPSBattle(NEW_MAZE_STR,build_team_ai(),
                DualRoleTeamAI("Stationary",DefaultAttackerController,Stay),headless=True,
                attacker_roster=list(gc.players),defender_roster=list(enemy.players),
                spike_holder_name=gc.spike_holder,defender_spike_holder_name=enemy.spike_holder,
                attacker_igl_name=gc.igl,defender_igl_name=enemy.igl,disable_side_swap=True)
            game.defender_setup_phase.finish()
            attackers=[c for c in game.chars if c.team=="A"]
            holder=next(c for c in attackers if c.has_spike)
            holder.pos=[7,40]
            cells=sorted((tuple(map(int,p)) for p in np.argwhere(game.grid!=1)
                          if 0<math.dist(p,holder.pos)<=4),key=lambda p:(math.dist(p,holder.pos),p))
            for char,point in zip((c for c in attackers if c is not holder),cells):
                char.pos=list(point)
            controller=game.attacker_controller.inner
            controller.tactics.initial_site=controller.tactics.selected_site="B"
            controller.tactics.opening_strategy="DEFAULT"
            controller.tactics.utility_used=True
            self.assertIsNone(forced_attack_site(game))
            self.assertIsNone(controller.macro_controller.env.forced_target_site)
            for _ in range(10):
                game._simulate_tick()
                if game.is_planted: break
            self.assertTrue(game.is_planted)
            self.assertEqual(game.planted_pos,(7,40))
            self.assertEqual(controller.tactics.opponent,"TYG")

    def test_live_launcher_runs_trained_policy_with_distinct_openings_without_saving_models(self):
        from ghost_champions_v2.tools.watch_match import build_match
        from ghost_champions_v2.rl.controller import LearnedAttackerController
        from simulation_runtime import cpu_inference
        checkpoint=Path(__file__).resolve().parents[1]/"data/attacker_rl_defuse_20261007_eval2/last_passed_guardrail.pt"
        if not checkpoint.is_file():
            self.skipTest("Local trained checkpoint is not installed")
        before=checkpoint.read_bytes()
        traces={}
        with contextlib.redirect_stdout(io.StringIO()),cpu_inference():
            for opening in ("DEFAULT","RUSH","SPLIT"):
                game=build_match(checkpoint,"OMG",seed=1462,opening=opening,headless=True,legacy_checkpoint=True)
                game.stop_after_round=True
                game.defender_setup_phase.finish()
                controller=game.attacker_controller.inner
                self.assertIsInstance(controller,LearnedAttackerController)
                self.assertIsNotNone(controller.policy)
                trace=[]
                for _ in range(24):
                    if not game.step_tick(): break
                    trace.append(tuple((str(c.name),tuple(c.pos)) for c in game.chars if c.team=="A"))
                self.assertEqual(controller.tactics.opening_strategy,opening)
                self.assertIsNotNone(controller.last_policy_action)
                self.assertTrue(game.replay_frames)
                traces[opening]=trace
        self.assertNotEqual(traces["DEFAULT"],traces["RUSH"])
        self.assertNotEqual(traces["RUSH"],traces["SPLIT"])
        self.assertEqual(checkpoint.read_bytes(),before)

    def test_live_log_preserves_the_actual_watched_state_and_round_records(self):
        import json
        import tempfile
        from ghost_champions_v2.tools.watch_match import save_match_log
        frames=[dict(round=2,tick=10,chars=[dict(name="carrier",pos=[10,40])])]
        records=[dict(round=1,winner="defender",reason="time_up")]
        game=NS(current_round=2,match_over=False,replay_frames=frames,
                analytics_tracker=NS(round_records=records))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"logs/live/match.json"
            save_match_log(game,path,dict(seed=123,opponent="FRC"))
            saved=json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["replay_frames"],frames)
            self.assertEqual(saved["round_records"],records)
            self.assertEqual(saved["seed"],123)
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_learned_policy_cannot_veto_favorable_or_last_second_plant_in_real_engine(self):
        import torch
        from game_core import PLANT_REQUIRED_TICKS
        from ghost_champions_v2.tools.watch_match import build_match
        from simulation_runtime import cpu_inference
        checkpoint=Path(__file__).resolve().parents[1]/"data/attacker_rl_defuse_20261007_eval2/last_passed_guardrail.pt"
        if not checkpoint.is_file(): self.skipTest("Local trained checkpoint is not installed")
        with contextlib.redirect_stdout(io.StringIO()),cpu_inference():
            for opening in ("DEFAULT","RUSH","SPLIT"):
                for urgent in (False,True):
                    with self.subTest(opening=opening,urgent=urgent):
                        game=build_match(checkpoint,"FRC",seed=1462,opening=opening,headless=True,legacy_checkpoint=True)
                        game.stop_after_round=True
                        game.defender_setup_phase.finish()
                        attackers=[c for c in game.chars if c.team=="A"]
                        defenders=[c for c in game.chars if c.team=="D"]
                        holder=next(c for c in attackers if c.has_spike)
                        holder.pos=[10,40]  # Legal B cell, away from the old (7,40) anchor.
                        self.assertEqual(game.grid[tuple(holder.pos)],2)
                        for char,point in zip((c for c in attackers if c is not holder),
                                              ((10,39),(20,18),(20,19),(20,20))):
                            char.pos=list(point)
                        for i,char in enumerate(defenders):
                            char.pos=[1,18+i]
                            char.is_alive=i<3  # Five attackers versus three defenders.
                        # Hold everyone else so distant teammates never satisfy the old cohort gate.
                        original_decide=game.attacker_controller.decide_move
                        game.attacker_controller.decide_move=lambda char,state: (
                            original_decide(char,state) if char.has_spike else list(char.pos))
                        game.defender_controller.decide_move=lambda char,state:list(char.pos)
                        controller=game.attacker_controller.inner
                        with torch.no_grad():
                            controller.policy.actor.weight.zero_()
                            controller.policy.actor.bias.zero_()
                            controller.policy.actor.bias[8]=100  # Always prefer facing over planting.
                        if urgent:
                            holder.pos=[10,39]  # One move plus the exact four plant ticks remain.
                            next(c for c in attackers if c is not holder).pos=[20,17]
                            game.round_timer=PLANT_REQUIRED_TICKS+1
                            controller.tactics.initial_site=controller.tactics.selected_site="A"
                        for _ in range(PLANT_REQUIRED_TICKS+1):
                            game.step_tick()
                            if game.is_planted or game.round_over: break
                        self.assertTrue(game.is_planted,controller.attacker_snapshot())
                        self.assertEqual(game.planted_pos,(10,40))
                        self.assertFalse(game.round_over)

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
            controller.tactics.opening_strategy="DEFAULT"
            for _ in range(100):
                game._simulate_tick()
                if controller.tactics.site_reason=="observed_lower_count" and selected is None:
                    selected=(controller.tactics.selected_site,
                              controller.observed_defenders_by_axis.copy())
                if game.is_planted or game.round_over or game.current_round != 1:
                    break
            self.assertIsNotNone(selected,controller.attacker_snapshot())
            self.assertEqual(selected[0],expected)
            self.assertEqual(selected[1][expected],2)


if __name__=="__main__": unittest.main()
