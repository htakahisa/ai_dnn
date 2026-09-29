"""Evaluation-only ability audit contract before Task 17 comparisons."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from coach_v1.ability_effect_audit import AbilityEffectAudit, summarize_ability_events


class AbilityAuditTest(unittest.TestCase):
    def _game(self):
        coach = object()
        ally = SimpleNamespace(name="caster", team="A", is_alive=True,
                               blind_remaining=0, reveal_remaining=0)
        enemy = SimpleNamespace(name="target", team="D", is_alive=True,
                                blind_remaining=0, reveal_remaining=0)
        game = SimpleNamespace(
            current_attacker_team_ai=coach, current_round=1, battle_tick=4,
            chars=[ally, enemy], flash_projectiles=[], recon_projectiles=[],
            smokes=[], replay_frames=[],
        )

        def execute(owner, action):
            if action.get("target") is None:
                return False
            ability = action["ability"]
            if ability == "FLASH":
                game.flash_projectiles.append({"owner": owner.name, "team": owner.team})
            elif ability == "RECON":
                game.recon_projectiles.append({"owner": owner.name, "team": owner.team})
            elif ability == "SMOKE":
                game.smokes.append({"owner": owner.name, "team": owner.team})
            else:
                return False
            return True

        game.execute_ai_ability = execute
        game._explode_flash = lambda projectile: setattr(enemy, "blind_remaining", 3)
        game._explode_recon = lambda projectile: setattr(enemy, "reveal_remaining", 3)
        game._record_replay_frame = lambda: game.replay_frames.append({"round_over": True})
        return game, coach, ally, enemy

    def test_successful_flash_and_recon_attribute_effect_to_their_cast(self):
        game, coach, ally, enemy = self._game()
        audit = AbilityEffectAudit(game, coach).install()
        self.assertFalse(game.execute_ai_ability(ally, {"ability": "FLASH"}))
        self.assertTrue(game.execute_ai_ability(ally, {"ability": "FLASH", "target": (5, 5)}))
        self.assertTrue(game.execute_ai_ability(ally, {"ability": "RECON", "target": (6, 6)}))
        game._explode_flash(game.flash_projectiles[0])
        game._explode_recon(game.recon_projectiles[0])
        result = audit.summary()
        self.assertEqual((2, 1, 1, 1),
                         tuple(result["FLASH"][key] for key in
                               ("requests", "successful_casts", "resolved_casts", "effective_casts")))
        self.assertEqual(1.0, result["RECON"]["effective_rate"])
        self.assertEqual(["target"], audit.events[1]["affected_enemies"])
        self.assertEqual(["target"], audit.events[2]["affected_enemies"])
        self.assertEqual(3, enemy.blind_remaining)

    def test_smoke_counts_net_blocked_lanes_without_changing_game(self):
        game, coach, ally, _ = self._game()
        audit = AbilityEffectAudit(
            game, coach,
            smoke_measure=lambda game, team, smoke: {
                "enemy_lanes_blocked": 2, "ally_lanes_blocked": 1,
            },
        ).install()
        self.assertTrue(game.execute_ai_ability(ally, {"ability": "SMOKE", "target": (5, 5)}))
        smoke = game.smokes[0]
        game._record_replay_frame()
        event = audit.events[0]
        self.assertTrue(event["resolved"])
        self.assertEqual(2, event["enemy_lane_pair_ticks_blocked"])
        self.assertEqual(1, event["ally_lane_pair_ticks_blocked"])
        self.assertEqual(1.0, audit.summary()["SMOKE"]["effective_rate"])
        self.assertIs(smoke, game.smokes[0])

    def test_unresolved_cast_is_excluded_from_denominator(self):
        game, coach, ally, _ = self._game()
        audit = AbilityEffectAudit(game, coach).install()
        self.assertTrue(game.execute_ai_ability(ally, {"ability": "FLASH", "target": (5, 5)}))
        self.assertEqual(0, audit.summary()["FLASH"]["resolved_casts"])
        self.assertIsNone(audit.summary()["FLASH"]["effective_rate"])
        self.assertEqual(audit.summary(), summarize_ability_events(audit.events))

    def test_projectiles_are_attributed_by_identity_even_if_impact_order_reverses(self):
        game, coach, ally, enemy = self._game()
        audit = AbilityEffectAudit(game, coach).install()
        self.assertTrue(game.execute_ai_ability(ally, {"ability": "FLASH", "target": (5, 5)}))
        self.assertTrue(game.execute_ai_ability(ally, {"ability": "FLASH", "target": (6, 6)}))
        game._explode_flash(game.flash_projectiles[1])
        self.assertFalse(audit.events[0]["resolved"])
        self.assertTrue(audit.events[1]["resolved"])
        self.assertEqual(["target"], audit.events[1]["affected_enemies"])
        game._explode_flash(game.flash_projectiles[0])
        self.assertEqual([], audit.events[0]["affected_enemies"])
        self.assertEqual(0.5, audit.summary()["FLASH"]["effective_rate"])


if __name__ == "__main__":
    unittest.main()
