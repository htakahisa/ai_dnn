"""Audit actual accepted casts and engine-reported hits without changing policy."""

from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from abilities_los import AbilityLosMixin
from battle_logic import BattleLogicMixin
from concon_v1.ability_advisor_v1.co1_diagnose_attacker_guard import GuardAudit


class GuardAuditTests(unittest.TestCase):
    def test_cast_acceptance_counter_tag_and_actual_hit_are_separate(self):
        audit = GuardAudit()
        game = NS(is_planted=True,current_round=1,battle_tick=30,flash_projectiles=[],analytics_tracker=None)
        actor = NS(team='A',name='caster',blind_remaining=2,reveal_remaining=0)
        audit.plant_ticks[1] = 20
        def original_cast(game,owner,payload):
            if payload['target']==(0,0):
                return False
            game.flash_projectiles.append(dict(team=owner.team,owner=owner.name))
            return True
        def original_hit(game,projectile,impact=None):
            game.analytics_tracker.record_contribution(actor,NS(team='D'),30,'flash')
            return 'engine result'
        with patch.object(AbilityLosMixin,'execute_ai_ability',original_cast), \
             patch.object(AbilityLosMixin,'_explode_flash',original_hit):
            with audit.instrument():
                self.assertFalse(AbilityLosMixin.execute_ai_ability(game,actor,dict(ability='FLASH',target=(0,0))))
                self.assertTrue(AbilityLosMixin.execute_ai_ability(game,actor,dict(ability='FLASH',target=(2,2))))
                projectile = game.flash_projectiles[-1]
                self.assertTrue(projectile['guard_audit_counter'])
                self.assertEqual(AbilityLosMixin._explode_flash(game,projectile),'engine result')
                self.assertIsNone(game.analytics_tracker)
        self.assertEqual(audit.counts['attempt:FLASH'],2)
        for key in ('rejected:FLASH','cast:FLASH','counter_cast:FLASH',
                    'hit_casts:FLASH','counter_hit_casts:FLASH','counter_hit_targets:FLASH'):
            self.assertEqual(audit.counts[key],1,key)
        self.assertEqual(audit.casts[0]['after_plant_ticks'],10)

    def test_incoming_hit_is_not_counted_as_guard_counter_hit(self):
        audit = GuardAudit()
        game = NS(is_planted=True,analytics_tracker=None)
        def original_hit(game,projectile,impact=None):
            game.analytics_tracker.record_contribution(None,NS(team='A'),1,'recon')
        with patch.object(AbilityLosMixin,'_explode_recon',original_hit),audit.instrument():
            AbilityLosMixin._explode_recon(game,dict(team='D'))
        self.assertEqual(audit.counts['incoming_hit_targets:RECON'],1)
        self.assertEqual(audit.counts['counter_hit_targets:RECON'],0)
        self.assertIsNone(game.analytics_tracker)

    def test_terminal_inventory_is_captured_before_headless_round_reset(self):
        audit = GuardAudit()
        actor = NS(team='A',is_alive=True,smoke_charges=1,flash_charges=0,recon_charges=2)
        game = NS(is_planted=True,current_round=1,battle_tick=10,is_defused=True,chars=[actor])
        def reset_round(game):
            game.current_round += 1
            game.is_planted = False
            game.chars = []
        with patch.object(BattleLogicMixin,'check_match_winner',reset_round),audit.instrument():
            BattleLogicMixin.check_match_winner(game)
        self.assertEqual(audit.ends,[dict(round=1,defused=True,
            unused={'SMOKE':1,'FLASH':0,'RECON':2})])
        self.assertEqual(audit.plants[0]['charges'],audit.ends[0]['unused'])
        self.assertEqual(game.current_round,2)
