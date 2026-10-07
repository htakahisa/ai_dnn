import copy
import json
from pathlib import Path
import tempfile
import unittest

from ghost_champions_v2.tools.analyze_postplant import analyze, classify


def frame(tick, alive=True, charges=1):
    return dict(round=13, tick=tick, planted=True, planted_pos=[8, 3], setup=False,
        chars=[dict(name="GC", team="A", alive=alive, pos=[7, 3], ability="FLASH", ability_charges=charges),
               dict(name="enemy", team="D", alive=True, pos=[7, 4], visible_to=["A", "D"])], smokes=[])


class PostplantAnalysisTests(unittest.TestCase):
    def test_defuse_after_wipe_is_separate_from_live_defuse(self):
        record=dict(winner="defender", reason="defused")
        wiped=classify(record,[frame(10), frame(11,False), frame(20,False)])
        live=classify(record,[frame(10), frame(20)])
        self.assertEqual(wiped["category"],"defused_after_attacker_wipe")
        self.assertEqual(wiped["ticks_dead_before_terminal"],9)
        self.assertEqual(len(wiped["unused_flash_deaths"]),1)
        self.assertEqual(live["category"],"defused_with_attackers_alive")

    def test_duplicate_frames_do_not_duplicate_flash_or_death(self):
        frames=[frame(10), frame(11,False,0), frame(11,False,0)]
        result=classify(dict(winner="defender",reason="defused"),frames)
        self.assertEqual(result["postplant_flash_used"],1)
        self.assertEqual(result["unused_flash_deaths"],[])

    def test_contact_requires_team_visibility_and_smoke_has_no_owner_claim(self):
        first=frame(10)
        first["chars"][1]["visible_to"]=["D"]
        contact=frame(11)
        contact["smokes"]=[dict(cells=[[8,3]])]
        result=classify(dict(winner="attacker",reason="detonated"),[first,contact])
        self.assertEqual(result["first_contact_tick"],11)
        self.assertTrue(result["spike_smoked_at_first_contact"])

    def test_actual_side_and_overtime_filter_and_missing_replay(self):
        record=dict(round_number=13,winner="defender",reason="defused",planted=True,
                    players={"GC":dict(team="Ghost Champions",side="attacker")})
        defense=copy.deepcopy(record)
        defense.update(round_number=1,players={"GC":dict(team="Ghost Champions",side="defender")})
        overtime=copy.deepcopy(record)
        overtime["round_number"]=25
        data=dict(maps=[dict(initial_attacker="opponent",round_records=[defense,record,overtime])])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"series_OMG_000.json"
            path.write_text(json.dumps(data),encoding="utf-8")
            result=analyze([path])
        self.assertEqual(result["planted_rounds"],1)
        self.assertEqual(result["categories"],{"missing_planted_replay":1})


if __name__ == "__main__":
    unittest.main()
