"""Replay saved series seeds with actual controllers and cumulative fatigue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluate_series(
    source, carry_model=None, guard_model=None, escort_model=None, save_replay=False
):
    import ghost_champions_v1 as gc

    previous = {attr: getattr(gc, attr) for attr in ("CARRY", "GUARD", "ESCORT")}
    try:
        return _evaluate_series(
            source, carry_model, guard_model, escort_model, save_replay
        )
    finally:
        for attr, paths in previous.items():
            setattr(gc, attr, paths)


def extract_team_metrics(players, side):
    team_players = [p for p in players.values() if p.get("side") == side]
    if not team_players:
        return {}
    total_gunfights = sum(p.get("gunfights_participated", 0) for p in team_players)
    total_gunfight_wins = sum(p.get("gunfights_won", 0) for p in team_players)
    avg_gunfight_winrate = total_gunfight_wins / max(1, total_gunfights)
    total_1v1 = sum(p.get("one_v_one_participated", 0) for p in team_players)
    total_1v1_wins = sum(p.get("one_v_one_won", 0) for p in team_players)
    avg_one_v_one_winrate = total_1v1_wins / max(1, total_1v1)
    total_first_kills = sum(p.get("first_kills", 0) for p in team_players)
    total_first_deaths = sum(p.get("first_deaths", 0) for p in team_players)
    avg_first_kill_efficiency = total_first_kills / max(
        1, total_first_kills + total_first_deaths
    )
    preaim_angles = []
    for p in team_players:
        if p.get("preaim_angle_count", 0) > 0:
            avg_preaim = p.get("preaim_angle_sum", 0) / p.get("preaim_angle_count", 1)
            preaim_angles.append(avg_preaim)
    avg_preaim_consistency = (
        sum(preaim_angles) / len(preaim_angles) if preaim_angles else 0.0
    )
    total_assists = sum(p.get("assists", 0) for p in team_players)
    total_covers = sum(p.get("covers", 0) for p in team_players)
    metrics = {
        "avg_gunfight_winrate": round(avg_gunfight_winrate, 2),
        "avg_first_kill_efficiency": round(avg_first_kill_efficiency, 2),
        "avg_one_v_one_winrate": round(avg_one_v_one_winrate, 2),
        "timeout_frequency": 0.0,
        "avg_preaim_consistency": round(avg_preaim_consistency / 360, 2),
        "total_assists": total_assists,
        "total_covers": total_covers,
    }
    if side == "defender":
        metrics["first_kills_scored"] = total_first_kills
    return metrics


def extract_tactic_metrics(tactic):
    return {
        "attacker_strategy": tactic.get("attacker_strategy", "unknown"),
        "defender_strategy": tactic.get("defender_strategy", "unknown"),
        "attacker_spread": tactic.get("max_attacker_spread", 0),
        "rush_entry_tick": tactic.get("rush_entry_tick"),
        "rush_entry_spread": tactic.get("rush_entry_spread"),
        "fake_executed": tactic.get("fake_or_rotate", False),
        "fake_effectiveness": round(tactic.get("fake_effect_rate", 0.0) / 100, 2),
    }


def extract_entry_metrics(tactic, round_record):
    return {
        "defender_initial_setup": tactic.get("defender_initial_setup", "0-0-0"),
        "defenders_at_fake_entry": tactic.get("defenders_at_fake_entry", 0),
        "defenders_at_final_entry": tactic.get("defenders_at_final_entry", 0),
        "defenders_displaced_from_final": tactic.get(
            "defenders_displaced_from_final", 0
        ),
        "site_entry_success": round_record.get("winner", "") == "attacker",
        "site_entry_duration_ticks": tactic.get("final_site_entry_tick") or 0,
    }


def extract_individual_efficiency(players):
    all_players = list(players.values())
    if not all_players:
        return {}
    total_kills = sum(p.get("kills", 0) for p in all_players)
    total_deaths = sum(p.get("deaths", 0) for p in all_players)
    avg_kill_death_ratio = total_kills / max(1, total_deaths)
    avg_damage_per_kill = 15.0
    total_assists = sum(p.get("assists", 0) for p in all_players)
    avg_assists_per_player = total_assists / len(all_players)
    avg_kills = total_kills / len(all_players)
    high_performers = sum(1 for p in all_players if p.get("kills", 0) > avg_kills)
    underperformers = sum(1 for p in all_players if p.get("kills", 0) < avg_kills * 0.5)
    return {
        "avg_kill_death_ratio": round(avg_kill_death_ratio, 2),
        "avg_damage_per_kill": round(avg_damage_per_kill, 1),
        "avg_assists_per_player": round(avg_assists_per_player, 1),
        "high_performers_count": high_performers,
        "underperformers_count": underperformers,
    }


def calculate_side_outcomes(all_rounds):
    postplant_attempts = sum(1 for r in all_rounds if r.get("planted", False))
    postplant_wins = sum(
        1
        for r in all_rounds
        if r.get("planted", False) and r.get("winner", "") == "attacker"
    )
    retake_attempts = sum(
        1
        for r in all_rounds
        if r.get("planted", False) and r.get("winner", "") == "defender"
    )
    retake_wins = retake_attempts
    return {
        "postplant_wins_total": postplant_wins,
        "postplant_attempts_total": postplant_attempts,
        "postplant_winrate": round(postplant_wins / max(1, postplant_attempts), 2),
        "retakes_won_total": retake_wins,
        "retakes_attempted_total": retake_attempts,
        "retake_winrate": round(retake_wins / max(1, retake_attempts), 2),
    }


def generate_inference_input(series_data, output_dir=None):
    team1_name = series_data["team1"]
    team2_name = series_data["team2"]
    series_score = (
        f"{series_data.get('team1_score', 0)}-{series_data.get('team2_score', 0)}"
    )
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    match_id = f"series_{team1_name.replace(' ', '_')}_vs_{team2_name.replace(' ', '_')}_{series_score}_{current_time}"
    match_metadata = {
        "match_id": match_id,
        "map_number": 1,
        "team_attacker": team1_name,
        "team_defender": team2_name,
    }
    all_rounds = []
    total_attacker_wins = 0
    total_defender_wins = 0
    total_plants = 0
    total_postplant_wins = 0
    total_retakes_won = 0
    all_attacker_gunfights = []
    all_defender_gunfights = []
    mvp_attacker = ""
    mvp_defender = ""
    player_total_kills = defaultdict(int)
    for map_data in series_data["maps"]:
        for round_record in map_data.get("round_records", []):
            for player_name, player_stats in round_record.get("players", {}).items():
                player_total_kills[player_name] += player_stats.get("kills", 0)
    attackers = []
    defenders = []
    for player_name in player_total_kills:
        for map_data in series_data["maps"]:
            for round_record in map_data.get("round_records", []):
                if player_name in round_record.get("players", {}):
                    side = round_record["players"][player_name].get("side")
                    if side == "attacker" and player_name not in attackers:
                        attackers.append(player_name)
                    elif side == "defender" and player_name not in defenders:
                        defenders.append(player_name)
    if attackers:
        mvp_attacker = max(attackers, key=lambda p: player_total_kills[p])
    if defenders:
        mvp_defender = max(defenders, key=lambda p: player_total_kills[p])
    global_rounds = [r for m in series_data["maps"] for r in m.get("round_records", [])]
    global_side_metrics = calculate_side_outcomes(global_rounds)
    for map_data in series_data["maps"]:
        for round_record in map_data.get("round_records", []):
            round_number = len(all_rounds) + 1
            round_outcome = (
                "attacker_win"
                if round_record.get("winner", "") == "attacker"
                else "defender_win"
            )
            if round_outcome == "attacker_win":
                total_attacker_wins += 1
            else:
                total_defender_wins += 1
            if round_record.get("planted", False):
                total_plants += 1
                if round_outcome == "attacker_win":
                    total_postplant_wins += 1
                else:
                    total_retakes_won += 1
            players = round_record.get("players", {})
            attacker_metrics = extract_team_metrics(players, "attacker")
            defender_metrics = extract_team_metrics(players, "defender")
            if attacker_metrics.get("avg_gunfight_winrate"):
                all_attacker_gunfights.append(attacker_metrics["avg_gunfight_winrate"])
            if defender_metrics.get("avg_gunfight_winrate"):
                all_defender_gunfights.append(defender_metrics["avg_gunfight_winrate"])
            tactic = round_record.get("tactic", {})
            tactic_metrics = extract_tactic_metrics(tactic)
            entry_metrics = extract_entry_metrics(tactic, round_record)
            individual_metrics = extract_individual_efficiency(players)
            round_features = {
                "attacker_team_metrics": attacker_metrics,
                "defender_team_metrics": defender_metrics,
                "tactic_metrics": tactic_metrics,
                "side_outcome_metrics": global_side_metrics,
                "entry_metrics": entry_metrics,
                "individual_efficiency": individual_metrics,
            }
            all_rounds.append(
                {
                    "round_number": round_number,
                    "round_outcome": round_outcome,
                    "features": round_features,
                }
            )
    total_rounds = len(all_rounds)
    avg_gunfight_attacker = (
        sum(all_attacker_gunfights) / len(all_attacker_gunfights)
        if all_attacker_gunfights
        else 0
    )
    avg_gunfight_defender = (
        sum(all_defender_gunfights) / len(all_defender_gunfights)
        if all_defender_gunfights
        else 0
    )
    map_aggregate = {
        "total_rounds": total_rounds,
        "attacker_rounds_won": total_attacker_wins,
        "defender_rounds_won": total_defender_wins,
        "total_plants": total_plants,
        "total_postplant_wins": total_postplant_wins,
        "total_retakes_won": total_retakes_won,
        "avg_timeout_frequency": 0.05,
        "avg_gunfight_efficiency_attacker": round(avg_gunfight_attacker, 2),
        "avg_gunfight_efficiency_defender": round(avg_gunfight_defender, 2),
        "most_valuable_attacker": mvp_attacker,
        "most_valuable_defender": mvp_defender,
    }
    inference_input = {
        "inference_input": {
            "match_metadata": match_metadata,
            "rounds": all_rounds,
            "map_aggregate": map_aggregate,
        }
    }
    if output_dir is None:
        output_dir = ROOT / "series_data" / match_id
    else:
        output_dir = Path(output_dir) / match_id
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_file = output_dir / f"{match_id}_inference.json"
    inference_file.write_text(
        json.dumps(inference_input, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    original_file = output_dir / f"{match_id}_original.json"
    series_clean = series_data.copy()
    for m in series_clean.get("maps", []):
        if "replay_frames" in m:
            del m["replay_frames"]
    original_file.write_text(
        json.dumps(series_clean, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[Auto-Organized Series] Created directory: {output_dir}")
    print(f"  - Inference analysis: {inference_file.name}")
    print(f"  - Original series data: {original_file.name}")
    print(
        f"  - Total rounds: {total_rounds}, Final score: {total_attacker_wins}-{total_defender_wins}\n"
    )
    return output_dir


def _evaluate_series(
    source, carry_model=None, guard_model=None, escort_model=None, save_replay=False
):
    import torch

    torch.set_num_threads(1)
    import ghost_champions_v1 as gc
    from party_presets import get_preset
    from run_competition_manager import play_map
    from gc_v1.positioning_gc import REGISTERED_PLANT_CELLS

    for attr, path in (
        ("CARRY", carry_model),
        ("GUARD", guard_model),
        ("ESCORT", escort_model),
    ):
        if path is not None:
            setattr(gc, attr, (Path(path).resolve(),))
    models = {}
    for phase in ("carry", "guard", "escort"):
        path = getattr(gc, phase.upper())[0]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        models[phase] = {
            "path": str(path),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "positioning_version": checkpoint.get("positioning_version", 0),
            "episode": checkpoint.get("episode"),
        }
    original = json.loads(Path(source).read_text(encoding="utf-8"))
    team1, team2 = get_preset(original["team1"]), get_preset(original["team2"])
    controllers = original["team_controllers"]
    wins = [0, 0]
    fatigue = {}
    maps = []
    start = time.monotonic()
    for entry in original["maps"]:
        number = entry["number"]
        result = play_map(
            team1,
            team2,
            number,
            entry["seed"],
            False,
            controllers[team1.name],
            controllers[team2.name],
            wins[0],
            wins[1],
            original["maps_to_win"],
            fatigue,
        )
        wins[0 if result.winner == team1.name else 1] += 1
        rows = [
            r
            for r in result.round_records
            if any(
                p.get("team") == "Ghost Champions" and p.get("side") == "attacker"
                for p in r.get("players", {}).values()
            )
        ]
        positions = {}
        for frame in result.replay_frames:
            if frame.get("planted") and frame.get("planted_pos") is not None:
                positions.setdefault(frame["round"], tuple(frame["planted_pos"]))
        row = {
            "number": number,
            "seed": entry["seed"],
            "score": [result.score1, result.score2],
            "attack_rounds": len(rows),
            "attack_wins": sum(r["winner"] == "attacker" for r in rows),
            "gc_round_wins": (
                result.score1 if team1.name == "Ghost Champions" else result.score2
            ),
            "total_rounds": result.score1 + result.score2,
            "plants": sum(r["planted"] for r in rows),
            "registered_plants": sum(
                r["planted"]
                and positions.get(r["round_number"]) in REGISTERED_PLANT_CELLS
                for r in rows
            ),
            "round_records": result.round_records,
        }
        if save_replay:
            row["replay_frames"] = result.replay_frames
        maps.append(row)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k not in {"round_records", "replay_frames"}
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    attacks = sum(m["attack_rounds"] for m in maps)
    plants = sum(m["plants"] for m in maps)
    attack_wins = sum(m["attack_wins"] for m in maps)
    round_wins = sum(m["gc_round_wins"] for m in maps)
    total_rounds = sum(m["total_rounds"] for m in maps)
    result = {
        "source": str(source),
        "evaluation_context": "saved_series_seeds",
        "cumulative_fatigue": True,
        "models": models,
        "maps": maps,
        "attack_rounds": attacks,
        "plants": plants,
        "attack_wins": attack_wins,
        "attack_win_rate": attack_wins / max(1, attacks),
        "gc_round_wins": round_wins,
        "total_rounds": total_rounds,
        "round_win_rate": round_wins / max(1, total_rounds),
        "plant_rate": plants / max(1, attacks),
        "registered_plants": sum(m["registered_plants"] for m in maps),
        "series_score": wins,
        "seconds": round(time.monotonic() - start, 1),
    }
    if save_replay:
        generate_inference_input(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=Path, nargs="+", required=True)
    parser.add_argument("--carry-model", type=Path)
    parser.add_argument("--guard-model", type=Path)
    parser.add_argument("--escort-model", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--save-replay", action="store_true")
    args = parser.parse_args()
    results = [
        evaluate_series(
            source,
            args.carry_model,
            args.guard_model,
            args.escort_model,
            args.save_replay,
        )
        for source in args.series
    ]
    if len(results) == 1:
        result = results[0]
    else:
        attacks = sum(r["attack_rounds"] for r in results)
        plants = sum(r["plants"] for r in results)
        result = {
            "evaluation_context": "multiple_saved_series_seeds",
            "series": results,
            "attack_rounds": attacks,
            "plants": plants,
            "plant_rate": plants / max(1, attacks),
            "attack_wins": sum(r["attack_wins"] for r in results),
            "gc_round_wins": sum(r["gc_round_wins"] for r in results),
            "total_rounds": sum(r["total_rounds"] for r in results),
            "registered_plants": sum(r["registered_plants"] for r in results),
        }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"maps", "series"}},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
