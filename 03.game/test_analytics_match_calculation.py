import unittest

from analytics.match_calculation import calculate_match_data_from_original


class MatchCalculationTests(unittest.TestCase):
    def test_post_plant_and_retake_use_all_planted_rounds(self):
        def round_record(number, winner, reason, planted):
            return {
                "round_number": number,
                "winner": winner,
                "reason": reason,
                "planted": planted,
                "tactic": {},
                "players": {},
            }

        original = {
            "team1": "Alpha",
            "team2": "Bravo",
            "team1_score": 0,
            "team2_score": 0,
            "maps": [
                {
                    "initial_attacker": "Alpha",
                    "player_stats": [],
                    "round_records": [
                        round_record(1, "defender", "defused", True),
                        round_record(2, "attacker", "detonated", True),
                        round_record(13, "attacker", "defender_wipe", True),
                        round_record(14, "defender", "attacker_wipe", True),
                        round_record(15, "defender", "attacker_wipe", False),
                    ],
                }
            ],
        }

        result = calculate_match_data_from_original(original)
        alpha = result["map_aggregate"]["Alpha"]
        bravo = result["map_aggregate"]["Bravo"]

        for team in (alpha, bravo):
            self.assertEqual(team["plant_count"], 2)
            self.assertEqual(team["post_plant_wins"], 1)
            self.assertEqual(team["post_plant_winrate"], 0.5)
            self.assertEqual(team["retake_attempts"], 2)
            self.assertEqual(team["retake_wins"], 1)
            self.assertEqual(team["retake_winrate"], 0.5)

        self.assertEqual(alpha["defuses"], 0)
        self.assertEqual(bravo["defuses"], 1)
        self.assertEqual(alpha["attacker_rounds"], 3)
        self.assertEqual(alpha["attacker_wins"], 2)
        self.assertEqual(alpha["attacker_winrate"], 2 / 3)
        self.assertEqual(alpha["defender_rounds"], 2)
        self.assertEqual(alpha["defender_wins"], 2)
        self.assertEqual(alpha["defender_winrate"], 1.0)
        self.assertEqual(
            [r["retake_attempted"] for r in result["round_features"]],
            [True, True, True, True, False],
        )


if __name__ == "__main__":
    unittest.main()
