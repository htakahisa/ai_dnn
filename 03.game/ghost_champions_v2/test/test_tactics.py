import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ghost_champions_v2.config import load_config, active_flags, validate_config
from ghost_champions_v2.geometry import (axis_for, los, projectile_path, route, angle_at, watch_cells)
from ghost_champions_v2.tactics import AttackerTactics


def unit(name, point, team="A", ability="HUNT", known=True, spike=False):
    return NS(name=name,pos=list(point),team=team,is_alive=True,position_known=known,
              ability_name=ability,recon_charges=2 if ability=="RECON" else 0,
              flash_charges=1 if ability=="FLASH" else 0,smoke_charges=1 if ability=="SMOKE" else 0,
              has_spike=spike,plant_timer=0,reveal_remaining=0)


def state(chars, planted=True, grid=None, tick=1):
    return dict(chars=chars, grid=np.zeros((21,21),int) if grid is None else grid,
                is_planted=planted,planted_pos=(10,10) if planted else None,
                battle_tick=tick,smoke_cells=set(),enemy_roster=[],defender_defuse_info={})


class TacticsTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config()
        self.policy=AttackerTactics(self.config)

    def test_frc_changes_preserve_other_profiles_and_global_parameters(self):
        previous=load_config(Path(__file__).resolve().parents[1]/"configs/pre_frc_fix.json")
        current=copy.deepcopy(self.config)
        del previous["profiles"]["FRC"]
        del current["profiles"]["FRC"]
        self.assertEqual(current,previous)

    def test_frc_carrier_queue_clears_the_narrow_corridor_without_a_solo_move(self):
        from map_data import NEW_MAZE_STR
        from party_presets import get_preset
        grid=np.array([[int(c) for c in row] for row in NEW_MAZE_STR.strip().splitlines()])
        chars=[unit("SyouTa",(15,16)),unit("Absol",(15,15),spike=True),
               unit("eKo",(15,17)),unit("SugarZ3ro",(15,14))]
        s=state(chars,planted=False,grid=grid,tick=40)
        s["enemy_roster"]=[dict(name=n) for n in get_preset("Furina Classic").players]
        self.policy.selected_site="B"
        self.policy.utility_used=True
        holder=chars[1]
        before=len(route(grid,tuple(holder.pos),[(7,40)]))
        for tick in range(40,65):
            s["battle_tick"]=tick
            for char in chars:
                profile,flags=self.policy.observe(char,s)
                result=self.policy.preplant(char,s,profile,flags)
                destination=tuple(result[0])
                occupied={tuple(c.pos) for c in chars if c is not char}
                if destination!=tuple(char.pos) and destination not in occupied:
                    self.assertLessEqual(math.dist(char.pos,destination),1)
                    if char is holder:
                        self.assertTrue(any(math.dist(char.pos,c.pos)<=4 for c in chars if c is not char))
                    char.pos=list(destination)
        self.assertLess(len(route(grid,tuple(holder.pos),[(7,40)])),before-10)

    def test_config_and_stage_ablation_respect_global_disable(self):
        self.assertFalse(active_flags(self.config,"hold",{})["early_recon"])
        self.assertTrue(active_flags(self.config,"recon",{})["early_recon"])
        self.config["flags"]["post_plant_hold"]=False
        self.assertFalse(active_flags(self.config,"entry",{"post_plant_hold":True})["post_plant_hold"])
        self.assertFalse(any(active_flags(self.config,"entry",{"preserve_v1":True}).values()))
        self.config["hold"]["radius"]=99
        with self.assertRaises(ValueError): validate_config(self.config)

    def test_hold_cells_are_distinct_visible_tradeable_and_have_crossfire(self):
        chars=[unit(str(i),p) for i,p in enumerate(((15,10),(10,15),(5,10),(10,5),(16,10)))]
        s=state(chars)
        self.policy.observe(chars[0],s)
        self.policy.postplant(chars[0],s)
        points=list(self.policy.hold_assignments.values())
        self.assertEqual(len(points),5)
        self.assertEqual(len(set(points)),5)
        for p in points:
            self.assertLessEqual(math.dist(p,(10,10)),5)
            self.assertTrue(any(los(s["grid"],p,q) for q in watch_cells(s["grid"],(10,10))))
            self.assertTrue(any(p!=q and math.dist(p,q)<=4 for q in points))
        self.assertTrue(any(angle_at((10,10),p,q)>=60 for p in points for q in points))

    def test_remote_enemy_is_not_chased_and_two_step_move_is_limited(self):
        char=unit("a",(10,17))
        s=state([char,unit("enemy",(10,20),"D")])
        self.policy.observe(char,s)
        result=self.policy.postplant(char,s)
        self.assertLess(math.dist(result[0],(10,10)),7)
        self.assertEqual(result[2]["move_step_limit"],1)

    def test_hidden_defuse_tap_routes_without_hidden_enemy_position(self):
        char=unit("a",(10,15))
        friend=unit("b",(15,10))
        hidden=unit("hidden",(-1,-1),"D",known=False)
        s=state([char,friend,hidden])
        s["defender_defuse_info"]={"hidden":(2,6)}
        self.policy.observe(char,s)
        result=self.policy.postplant(char,s)
        self.assertLess(math.dist(result[0],(10,10)),5)
        self.assertEqual(self.policy.observed_defenders_by_axis,{"A":0,"Mid":0,"B":0})

    def test_planter_position_memory_is_used_instead_of_noisy_spike(self):
        char=unit("planter",(10,10),spike=True)
        s=state([char],planted=False,tick=10)
        self.policy.observe(char,s)
        self.policy.remember_result(char,(list(char.pos),"PLANT"))
        s.update(is_planted=True,planted_pos=(12,12),battle_tick=11)
        self.policy.observe(char,s)
        self.assertEqual(self.policy.planted_position(s),(10,10))
        self.policy.reset_round()
        self.assertIsNone(self.policy.spike)

    def test_defuse_tap_overrides_an_iq_noisy_enemy_aim_position(self):
        char=unit("a",(11,10))
        friend=unit("b",(10,11))
        noisy=unit("defuser",(11,15),"D")
        s=state([char,friend,noisy])
        s["defender_defuse_info"]={"defuser":(1,6)}
        self.policy.observe(char,s)
        result=self.policy.postplant(char,s)
        self.assertEqual(result[0],[11,10])
        self.assertEqual(result[1]["facing"],"N")

    def test_noisy_or_missing_enemy_is_not_counted_as_a_known_defender(self):
        char=unit("a",(10,10))
        enemies=[unit("hidden",(10,2),"D",known=False),unit("seen",(10,18),"D")]
        s=state([char,*enemies])
        self.policy.observe(char,s)
        self.assertEqual(self.policy.observed_defenders_by_axis,{"A":0,"Mid":0,"B":1})
        s.update(chars=[char],battle_tick=30)
        self.policy.observe(char,s)
        self.assertEqual(self.policy.observed_defenders_by_axis,{"A":0,"Mid":0,"B":0})

    def test_recon_cadence_deadline_and_successful_use_memory(self):
        char=unit("seeker",(10,10),ability="RECON")
        self.policy.config["recon"]["aims"]["A"]=[10,1]
        s=state([char],planted=False)
        self.policy.observe(char,s)
        result=self.policy.recon(char,s)
        self.assertEqual(result[1]["ability"],"RECON")
        self.assertFalse(self.policy.utility_used)
        self.assertIsNone(self.policy.recon(char,s))
        char.recon_charges-=1
        s["battle_tick"]=2
        self.policy.observe(char,s)
        self.assertTrue(self.policy.utility_used)
        self.assertNotIn("A",self.policy.surveyed_axes)
        s["battle_tick"]=5
        self.policy.observe(char,s)
        self.assertIn("A",self.policy.surveyed_axes)
        s["battle_tick"]=31
        self.policy.observe(char,s)
        self.assertIsNone(self.policy.recon(char,s))

    def test_setup_does_not_consume_recon_sequence_or_deadline(self):
        char=unit("seeker",(10,10),ability="RECON")
        s=state([char],planted=False,tick=0)
        s["defender_setup_active"]=True
        for _ in range(50):
            self.policy.observe(char,s)
            self.assertIsNone(self.policy.recon(char,s))
        self.assertFalse(self.policy.recon_sequences)
        self.assertFalse(self.policy.last_cast)
        s.update(defender_setup_active=False,battle_tick=1)
        self.policy.observe(char,s)
        self.assertEqual(self.policy.recon(char,s)[1]["ability"],"RECON")
        self.assertEqual(self.policy.last_recon_request["seeker"]["axis"],"A")

    def test_frc_site_gate_and_fnatic_four_player_avoidance(self):
        profile=self.config["profiles"]["FRC"]
        self.assertIsNone(self.policy.choose_site(profile))
        self.policy.observed_defenders_by_axis={"A":2,"Mid":0,"B":3}
        self.assertEqual(self.policy.choose_site(profile),"A")
        self.policy.observed_defenders_by_axis={"A":4,"Mid":0,"B":1}
        self.assertEqual(self.policy.choose_site(self.config["profiles"]["FNC"]),"B")
        self.policy.observed_defenders_by_axis={"A":4,"Mid":1,"B":0}
        self.assertIsNone(self.policy.choose_site(self.config["profiles"]["FNC"]))
        self.policy.surveyed_axes.add("B")
        self.assertEqual(self.policy.choose_site(self.config["profiles"]["FNC"]),"B")

    def test_smoke_never_shields_the_spike_or_assigned_fire_lines(self):
        char=unit("smoker",(10,12),ability="SMOKE")
        enemy=unit("defuser",(10,11),"D")
        s=state([char,enemy])
        self.policy.observe(char,s)
        self.policy._assign_hold(char,s,(10,10))
        self.assertIsNone(self.policy._smoke(char,s,(10,10),[enemy]))

    def test_expired_third_defender_does_not_turn_a_crowded_site_into_two(self):
        char=unit("a",(10,10))
        enemies=[unit(str(i),(i+1,2),"D") for i in range(3)]
        s=state([char,*enemies],planted=False)
        self.policy.observe(char,s)
        self.assertEqual(self.policy.max_observed_defenders_by_axis["A"],3)
        s.update(chars=[char,*enemies[:2]],battle_tick=30)
        self.policy.observe(char,s)
        self.assertEqual(self.policy.observed_defenders_by_axis["A"],2)
        self.assertIsNone(self.policy.choose_site(self.config["profiles"]["FRC"]))
        self.policy.reset_round()
        self.assertEqual(self.policy.max_observed_defenders_by_axis["A"],0)

    def test_frc_scout_keeps_b_assignment_when_a_scout_dies(self):
        from party_presets import get_preset
        first=unit("a",(16,4),ability="RECON")
        second=unit("b",(7,23),ability="RECON")
        s=state([first,second],planted=False)
        s["enemy_roster"]=[dict(base_name=n) for n in get_preset("Furina Classic").players]
        self.policy.observe(second,s)
        self.policy.recon_sequences["b"]=1
        self.assertEqual(self.policy.scout_axis(second,self.policy.allies(second,s)),"B")
        first.is_alive=False
        s["battle_tick"]=2
        self.policy.observe(second,s)
        self.assertEqual(self.policy.scout_axis(second,self.policy.allies(second,s)),"B")

    def test_frc_contact_does_not_spend_last_recon_before_reaching_vantage(self):
        from party_presets import get_preset
        grid=np.zeros((26,44),int)
        grid[8,3]=grid[7,40]=2
        first=unit("a",(16,4),ability="RECON")
        second=unit("b",(8,23),ability="RECON")
        carrier=unit("carrier",(20,18),spike=True)
        support=unit("support",(8,22))
        enemy=unit("enemy",(8,28),"D")
        s=state([first,second,carrier,support,enemy],planted=False,grid=grid,tick=20)
        s["enemy_roster"]=[dict(base_name=n) for n in get_preset("Furina Classic").players]
        profile,flags=self.policy.observe(second,s)
        self.policy.recon_sequences["b"]=1
        self.policy.config["profiles"]["FRC"]["recon"]["secondary_waypoints"]["B"]=[7,23]
        action=self.policy.preplant(second,s,profile,flags)
        self.assertEqual(action[0],[7,23])
        self.assertEqual(action[1],"MOVE")
        self.assertEqual(self.policy.recon_sequences["b"],1)

    def test_frc_escort_keeps_its_scout_after_the_other_scout_finishes(self):
        from party_presets import get_preset
        grid=np.zeros((26,44),int)
        grid[8,3]=grid[7,40]=2
        first=unit("a",(16,4),ability="RECON")
        second=unit("b",(12,30),ability="RECON")
        carrier=unit("carrier",(16,9),spike=True)
        left=unit("left",(16,5))
        right=unit("right",(13,30))
        s=state([first,second,carrier,left,right],planted=False,grid=grid,tick=20)
        s["enemy_roster"]=[dict(base_name=n) for n in get_preset("Furina Classic").players]
        profile,flags=self.policy.observe(right,s)
        first.recon_charges=0
        self.policy.recon_sequences["b"]=1
        action=self.policy.preplant(right,s,profile,flags)
        self.assertLessEqual(math.dist(action[0],second.pos),2)
        self.assertEqual(self.policy.scout_supports["b"],"right")

    def test_frc_b_recon_covers_upper_and_lower_site_lane(self):
        from party_presets import get_preset
        from map_data import NEW_MAZE_STR
        grid=np.array([[int(c) for c in row] for row in NEW_MAZE_STR.strip().splitlines()])
        first=unit("a",(16,4),ability="RECON")
        second=unit("b",(9,29),ability="RECON")
        s=state([first,second],planted=False,grid=grid,tick=20)
        s["enemy_roster"]=[dict(base_name=n) for n in get_preset("Furina Classic").players]
        self.policy.observe(second,s)
        self.policy.recon_sequences["b"]=1
        action=self.policy.recon(second,s,scouting=True)
        impact=projectile_path(grid,tuple(second.pos),action[1]["target"])[-1]
        self.assertTrue(all(max(abs(impact[i]-p[i]) for i in (0,1))<=4 for p in ((2,40),(7,38))))

    def test_b_recon_moves_to_the_required_vantage_before_casting(self):
        from map_data import NEW_MAZE_STR
        grid=np.array([[int(c) for c in row] for row in NEW_MAZE_STR.strip().splitlines()])
        first=unit("a",(16,4),ability="RECON")
        seeker=unit("b",(8,23),ability="RECON")
        self.policy.recon_sequences["b"]=1
        s=state([first,seeker],planted=False,grid=grid,tick=10)
        self.policy.observe(seeker,s)
        action=self.policy.recon(seeker,s,scouting=True)
        self.assertEqual(action[0],[7,23])
        self.assertNotIsInstance(action[1],dict)
        seeker.pos=action[0]
        s["battle_tick"]=11
        self.policy.observe(seeker,s)
        action=self.policy.recon(seeker,s,scouting=True)
        self.assertEqual(action[1]["ability"],"RECON")
        impact=projectile_path(grid,tuple(seeker.pos),action[1]["target"])[-1]
        self.assertTrue(all(max(abs(impact[i]-p[i]) for i in (0,1))<=4
                            for p in ((7,38),(3,39))))

    def test_smoke_alone_does_not_satisfy_flash_or_recon_before_entry(self):
        char=unit("smoker",(10,12),ability="SMOKE")
        s=state([char],planted=False)
        self.policy.observe(char,s)
        char.smoke_charges-=1
        s["battle_tick"]=2
        self.policy.observe(char,s)
        self.assertFalse(self.policy.utility_used)

    def test_dance_support_heals_a_public_wounded_ally_in_the_real_engine(self):
        from abilities_los import AbilityLosMixin
        healer=unit("healer",(10,10),ability="DANCE")
        healer.dance_charges=3
        friend=unit("friend",(18,18))
        friend.hp,friend.max_hp=30,100
        s=state([healer,friend])
        self.policy.observe(healer,s)
        action=self.policy.postplant(healer,s)
        engine=AbilityLosMixin()
        engine.chars=[healer,friend]
        engine.battle_tick=1
        self.assertTrue(engine.execute_ai_ability(healer,action[1]))
        self.assertEqual(friend.hp,80)
        self.assertEqual(healer.dance_charges,2)
        healer.plant_timer=1
        self.assertIsNone(self.policy._heal(healer,s))

    def test_occupied_route_detours_and_never_steps_outside_leash(self):
        grid=np.zeros((9,9),int)
        path=route(grid,(4,3),[(4,5)],[(4,4)],(4,4),2)
        self.assertEqual(path[-1],(4,5))
        self.assertTrue(all(math.dist(p,(4,4))<=2 for p in path))

    def test_waiting_carrier_yields_a_narrow_scout_route(self):
        carrier=unit("carrier",(10,8),spike=True)
        scout=unit("seeker",(10,15),ability="RECON")
        self.policy.config["recon"]["waypoints"]["A"]=[10,2]
        s=state([carrier,scout],planted=False)
        self.policy.observe(carrier,s)
        action=self.policy.wait_carrier(carrier,s,(2,2),self.policy.allies(carrier,s))
        self.assertNotEqual(action[0],carrier.pos)
        self.assertNotEqual(action[0][0],10)

    def test_preplant_waits_instead_of_taking_a_long_mid_detour(self):
        grid=np.zeros((7,7),int)
        grid[1:6,2]=1
        grid[3,2]=0
        char=unit("seeker",(3,1),ability="RECON")
        blocker=unit("friend",(3,2))
        s=state([char,blocker],planted=False,grid=grid)
        self.policy.observe(char,s)
        path=self.policy.preplant_route(char,s,[(3,5)])
        self.assertEqual(len(path),5)
        self.assertEqual(path[1],(3,2))  # IQ adapter/engine wait until it clears.

    def test_plant_cohort_wait_does_not_interrupt_an_active_plant(self):
        grid=np.zeros((21,21),int)
        grid[2,2]=2
        carrier=unit("carrier",(2,2),spike=True)
        s=state([carrier,unit("friend",(18,18))],planted=False,grid=grid)
        profile=self.config["profiles"]["OMG"]
        flags=active_flags(self.config,"entry",profile)
        self.policy.observe(carrier,s)
        self.assertNotEqual(self.policy.preplant(carrier,s,profile,flags)[1],"PLANT")
        carrier.plant_timer=1
        self.assertEqual(self.policy.preplant(carrier,s,profile,flags)[1],"PLANT")

    def test_directional_recon_matches_actual_engine_through_walls(self):
        from abilities_los import AbilityLosMixin
        grid=np.zeros((12,12),int)
        grid[3,6]=1
        engine=AbilityLosMixin()
        engine.grid=grid
        engine.height,engine.width=grid.shape
        for aim in ((3,4),(8,8),(2,10),(10,3)):
            self.assertEqual(projectile_path(grid,(3,3),aim),engine._projectile_path((3,3),aim))
        for r,c in np.argwhere(grid!=1):
            self.assertEqual(projectile_path(grid,(3,3),(int(r),int(c))),
                             engine._projectile_path((3,3),(int(r),int(c))))
        self.assertNotEqual(projectile_path(grid,(3,3),(3,4))[-1],(3,4))
        self.assertEqual(axis_for((0,4),12),"Mid")
        self.assertEqual(axis_for((0,8),12),"Mid")


if __name__=="__main__": unittest.main()
