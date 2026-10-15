"""After evaluation completes, export a compact prior JSON and Markdown table.

Working directory: omoko_v1. Run: python ov1_summarize_attacker_carry_routes.py
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

AI_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AI_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ov1_attacker_carry_route_priors import SUMMARY_SCHEMA_VERSION, map_signature
from ov1_map_data_carry import NEW_MAZE_STR
from party_presets import get_preset

INPUT_PATH = AI_DIR / "data/carry_route_evaluation/route_win_rates_12round_sets.json"
OUTPUT_PATH = AI_DIR / "data/carry_route_evaluation/route_priors_summary.json"
TABLE_PATH = AI_DIR / "data/carry_route_evaluation/route_win_rates_summary.md"


def summarize(report):
    scenario = report.get("scenario", {})
    if (report.get("schema_version") != 2 or report.get("rounds_per_set") != 12
            or report.get("evaluation") != "12round_attack_set_full_engine_fixed_carry_route_v2"):
        raise ValueError("Expected twelve-round-set evaluation data")
    # Reader-side integration may have changed while the fixed-route evaluator
    # was running. Check maps/rosters, rather than requiring identical reader code.
    for name in ("omoko_v1/ov1_map_data_carry.py", "map_data.py", "party_presets.py", "grid_lines.py"):
        actual = hashlib.sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
        if scenario.get("source_hashes", {}).get(name) != actual:
            raise ValueError(f"Evaluation input changed: {name}")
    expected_routes = {route["id"] for route in scenario["routes"]}
    opponents = {}
    table = ["# Carry経路別勝率", "", "| 相手AI | 経路 | 勝数 / ラウンド数 | 勝率 |",
             "|---|---|---:|---:|"]
    for key, entry in report.get("opponents", {}).items():
        rows = entry.get("routes", {})
        target = entry.get("target_rounds_per_route", 0)
        if (entry.get("status") != "complete" or target <= 0 or target % 12
                or set(rows) != expected_routes
                or any(row.get("rounds", 0) < target for row in rows.values())):
            raise ValueError(f"Evaluation is still incomplete: {key}")
        preset = get_preset(entry["roster"])
        if preset is None:
            raise ValueError(f"Unknown roster: {entry['roster']}")
        rates = {}
        for route, row in rows.items():
            rounds, wins = row["rounds"], row["wins"]
            if not isinstance(rounds, int) or not isinstance(wins, int) or not 0 <= wins <= rounds:
                raise ValueError(f"Invalid counts: {key}/{route}")
            rates[route] = wins / rounds
            table.append(f"| {key} | {route} | {wins} / {rounds} | {rates[route]:.1%} |")
        opponents[key] = {"roster": entry["roster"], "players": list(preset.players), "win_rates": rates}
    if not opponents or report.get("worker_errors"):
        raise ValueError("No completed, error-free evaluation data")
    summary = {"schema_version": SUMMARY_SCHEMA_VERSION,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "map_signature": map_signature(NEW_MAZE_STR),
               "wall_los_version": scenario["wall_los_version"],
               "model_hashes": report["model_hashes"], "opponents": opponents}
    return summary, "\n".join(table) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--table", type=Path, default=TABLE_PATH)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    summary, table = summarize(report)
    for path, text in ((args.output, json.dumps(summary, ensure_ascii=False, indent=2)), (args.table, table)):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    print(f"Saved {len(summary['opponents'])} opponent priors: {args.output}")
    print(f"Win-rate table: {args.table}")


if __name__ == "__main__":
    main()
