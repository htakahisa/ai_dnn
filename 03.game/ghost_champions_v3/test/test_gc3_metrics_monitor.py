"""Concurrent append and live-monitor snapshot checks."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from ghost_champions_v3.tools.gc3_metrics_monitor_data import read_complete_lines, log_signature, snapshot, bc_snapshot

def write(path, value):
    Path(path).write_text(json.dumps(value),encoding="utf-8")

class MonitorDataTests(unittest.TestCase):
    def test_append_in_progress_is_completed_on_next_read(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/"metrics.jsonl"
            path.write_bytes(b'{"step":1}\n{"step":')
            self.assertEqual(read_complete_lines(path),[{"step":1}])
            with path.open("ab") as handle:
                handle.write(b'2}\n')
            self.assertEqual(read_complete_lines(path),[{"step":1},{"step":2}])

    def test_partial_utf8_at_end_does_not_break_completed_records(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/"metrics.jsonl"
            path.write_bytes(b'{"step":1}\n{"label":"'+bytes([0xe3,0x81]))
            self.assertEqual(read_complete_lines(path),[{"step":1}])

    def test_new_episode_invalidates_previous_complete_status(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            write(root/"status.json",dict(step=5,complete=True))
            write(root/"final_evaluation.json",dict(success_rate=.25,rounds=[]))
            (root/"episodes.jsonl").write_text(json.dumps(dict(decisions=5,success=False))+"\n",encoding="utf-8")
            previous=log_signature(root)
            old=snapshot(root)
            self.assertTrue(old["complete"])
            self.assertIsNotNone(old["final"])
            with (root/"episodes.jsonl").open("a",encoding="utf-8") as handle:
                handle.write(json.dumps(dict(decisions=5,success=True))+"\n")
            current=snapshot(root)
            self.assertEqual(current["step"],10)
            self.assertEqual(current["rolling_success"],[0.,50.])
            self.assertFalse(current["complete"])
            self.assertIsNone(current["final"])
            self.assertNotEqual(previous,log_signature(root))

    def test_new_run_without_evaluations_or_metrics_is_supported(self):
        with TemporaryDirectory() as temp:
            data=snapshot(temp)
            self.assertEqual(data["step"],0)
            self.assertEqual(data["metrics"],[])
            self.assertEqual(data["evaluations"],[])
            self.assertIsNone(data["measured_from"])
            self.assertFalse(data["complete"])

    def test_corrupt_completed_record_is_reported(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/"metrics.jsonl"
            path.write_bytes(b'{"step":1}\nBROKEN\n')
            with self.assertRaises(ValueError):
                read_complete_lines(path)

    def test_bc_snapshot_reads_only_complete_epochs_and_updates_signature(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            before=log_signature(root)
            row=dict(epoch=1,step=100,train_loss=1.4,validation_loss=1.6)
            (root/"bc_metrics.jsonl").write_text(json.dumps(row)+"\n"+'{"epoch":2',encoding="utf-8")
            data=bc_snapshot(root)
            self.assertEqual(data["metrics"],[row])
            self.assertEqual(data["epoch"],1)
            self.assertEqual(data["step"],100)
            self.assertNotEqual(before,log_signature(root))

    def test_bc_new_epoch_invalidates_old_complete_status(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            write(root/"status.json",dict(epoch=1,step=100,complete=True))
            row=dict(epoch=2,step=200,train_loss=1.,validation_loss=1.2)
            (root/"bc_metrics.jsonl").write_text(json.dumps(row)+"\n",encoding="utf-8")
            self.assertFalse(bc_snapshot(root)["complete"])

    def test_bc_chart_displays_measured_closed_loop_results(self):
        from matplotlib.figure import Figure
        from ghost_champions_v3.tools.gc3_preplant_metrics_plot import render_metrics
        from ghost_champions_v3.tools.gc3_watch_attacker_preplant_metrics import configure_fonts
        configure_fonts()
        with TemporaryDirectory() as temp:
            bc, rl = bc_snapshot(temp), snapshot(temp)
            bc["metrics"] = [dict(epoch=1, train_loss=1.2, validation_loss=1.5, plant_rate=.25, success_rate=.5)]
            figure = Figure()
            render_metrics(figure, bc, rl)
            self.assertEqual(len(figure.axes), 5)
            self.assertEqual([list(line.get_ydata()) for line in figure.axes[-1].lines], [[25.], [50.]])

    def test_preplant_chart_does_not_invent_successes_or_bc_metrics(self):
        from matplotlib.figure import Figure
        from ghost_champions_v3.tools.gc3_preplant_metrics_plot import render_metrics
        from ghost_champions_v3.tools.gc3_watch_attacker_preplant_metrics import configure_fonts
        configure_fonts()
        with TemporaryDirectory() as temp:
            bc=bc_snapshot(temp)
            rl=snapshot(temp)
            rl["evaluations"]=[dict(step=0,success_rate=0.,plant_rate=0.)]
            figure=Figure()
            render_metrics(figure,bc,rl)
            self.assertEqual(len(figure.axes[0].lines),0)
            self.assertEqual(len(figure.axes[1].lines),0)
            self.assertEqual(len(figure.axes[2].lines),0)
            for line in figure.axes[3].lines:
                self.assertEqual(list(line.get_ydata()),[0.])

if __name__=="__main__":
    unittest.main()
