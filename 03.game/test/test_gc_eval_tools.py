import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("gc_run_eval", Path(__file__).resolve().parents[1]/"tools"/"run_eval.py")
eval_tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_tools)


class EvaluationTests(unittest.TestCase):
    def test_runtime_snapshot_keeps_the_original_definitions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"source"
            root.mkdir()
            for name in eval_tools.RUNTIME_FILES:
                (root/name).write_text("original "+name,encoding="utf-8")
            snapshot=Path(folder)/"snapshot"
            with patch.object(eval_tools,"ROOT",root):
                eval_tools.runtime_snapshot(snapshot)
                (root/"character_stats.py").write_text("edited during evaluation",encoding="utf-8")
                eval_tools.runtime_snapshot(snapshot)
            self.assertEqual((snapshot/"character_stats.py").read_text(encoding="utf-8"),"original character_stats.py")

    def test_incomplete_existing_runtime_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                eval_tools.runtime_snapshot(Path(folder))

    def test_pilot_baseline_uses_only_matching_series_indices(self):
        def series(winner):
            return dict(team1="Ghost Champions",team2="Furina Classic",maps=[dict(
                initial_attacker="Ghost Champions",round_records=[dict(round_number=1,
                    winner=winner,planted=False,reason="attacker_wipe",players={},tactic={})])])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"series_FRC_000.json").write_text(json.dumps(series("defender")),encoding="utf-8")
            (root/"series_FRC_001.json").write_text(json.dumps(series("attacker")),encoding="utf-8")
            result=eval_tools.matched_baseline([Path("pilot/series_FRC_000.json")],root)
            self.assertEqual((result["Furina Classic"]["wins"],result["Furina Classic"]["n"]),(0,1))
            with self.assertRaises(ValueError):
                eval_tools.matched_baseline([Path("series_FRC_002.json")],root)

    def test_wilson_edges_and_reference(self):
        self.assertIsNone(eval_tools.wilson(0, 0))
        low, high = eval_tools.wilson(50, 100)
        self.assertAlmostEqual(low, 0.40383153, places=7)
        self.assertAlmostEqual(high, 0.59616847, places=7)
        self.assertAlmostEqual(eval_tools.wilson(0, 100)[0], 0.0)
        self.assertAlmostEqual(eval_tools.wilson(100, 100)[1], 1.0)

    def test_actual_side_overtime_recon_and_live_distance(self):
        def record(rn, side, winner="attacker", planted=True):
            return dict(round_number=rn, winner=winner, planted=planted, reason="detonated",
                tactic=dict(defender_initial_setup="2-0-3", final_attack_site="A"),
                players={"GC": dict(team="Ghost Champions", side=side, first_deaths=0)})
        def frame(tick, charges, setup=False, over=False):
            return dict(round=13, tick=tick, setup=setup, round_over=over, planted=not setup,
                planted_pos=[2, 2], chars=[dict(name="GC", ability="RECON", ability_charges=charges,
                    team="A", pos=[2, 5], alive=True)])
        # GC attacks the second half. Setup inventory catches a first-tick cast.
        data = dict(team1="Ghost Champions", team2="Furina Classic", maps=[dict(
            initial_attacker="Furina Classic", round_records=[record(1,"defender"),
                record(13,"attacker"), record(25,"attacker")],
            replay_frames=[frame(0,2,True), frame(1,1), frame(11,0), frame(21,0,over=True)])])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"series.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            row = eval_tools.summarize([path])["Furina Classic"]
        self.assertEqual((row["wins"], row["n"], row["overtime_excluded"]), (1,1,1))
        self.assertEqual((row["recon_used"], row["recon_available"]), (2,2))
        self.assertEqual((row["distance_10"], row["distance_10_n"]), (3,1))
        self.assertIsNone(row["distance_20"])
        self.assertEqual(row["site_two"], 1)

    def test_late_recon_is_not_counted_as_early_usage(self):
        frames=[dict(round=1,tick=tick,setup=setup,planted=False,
                     chars=[dict(name="GC",ability="RECON",ability_charges=charges,team="A")])
                for tick,charges,setup in ((0,2,True),(30,1,False),(31,0,False))]
        data=dict(team1="Ghost Champions",team2="Furina Classic",maps=[dict(
            initial_attacker="Ghost Champions",replay_frames=frames,round_records=[dict(
                round_number=1,winner="defender",planted=False,reason="time_expired",
                players={},tactic=dict(final_attack_site=None))])])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"series.json"
            path.write_text(json.dumps(data),encoding="utf-8")
            row=eval_tools.summarize([path])["Furina Classic"]
        self.assertEqual((row["recon_used"],row["recon_used_by_30"],row["recon_available"]),(2,1,2))
        self.assertEqual(row["early_recon_rate"],.5)
        self.assertEqual((row["preplant_timeouts"],row["site_selected"]),(1,0))


if __name__ == "__main__":
    unittest.main()
