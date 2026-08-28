from __future__ import annotations

import contextlib
import io
import json
import queue
import random
import secrets
import threading
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

from map_data import NEW_MAZE_STR
from party_presets import all_preset_names, get_preset
from run_game import VisualFPSBattle, _build_team_ai
from game_core import PLAYER_COMBOS, get_character_combat_stats

CONTROLLER_OPTIONS = {
    "Toru AI v3.1": "toru_ai_v3.1",
    "Touyama Gaming v2": "touyama_gaming_v2",
    "Fnatic v2": "fnatic_2",
    "Fnatic v1": "fnatic_v1",
    "Toru AI v3.1": "toru_ai_v3.1",
    "Touyama Gaming v1": "touyama_gaming_v1",
    "Ghost Champions v1": "ghost_champions_v1",
    "AI v1": "learning_v1",
    "ロジック": "default",
    "ユーザー操作": "user",
}
CONTROLLER_KEY_TO_DISPLAY = {key: label for label, key in CONTROLLER_OPTIONS.items()}
DEFAULT_CONTROLLER_DISPLAY = "Toru AI v3.1"

RESULT_DIR = Path("competition_results")
RATING_FILE = RESULT_DIR / "team_ratings.json"
DEFAULT_TEAM_RATING = 2500.0
RATING_K_FACTOR = 40.0
UNUSED = "（未使用）"
MAX_TEAM_SLOTS = 16


class TeamPlayerKey(str):
    """
    画面上では通常の選手名として表示されるが、辞書キーとしては
    所属チームごとに区別される文字列。

    run_game.py の match_stats が選手名だけをキーにしているため、
    両チームに同名選手がいる場合の上書きを防ぐ。
    """

    def __new__(cls, display_name: str, team_id: str):
        obj = str.__new__(cls, display_name)
        obj.team_id = str(team_id)
        return obj

    def __hash__(self) -> int:
        # 通常文字列と同じ名前検索を維持しつつ、同名TeamPlayerKey同士は
        # __eq__で所属チームまで比較する。
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TeamPlayerKey):
            return str(self) == str(other) and self.team_id == other.team_id
        return str(self) == str(other)

    def __reduce__(self):
        return TeamPlayerKey, (str(self), self.team_id)


@dataclass
class PStat:
    name: str
    kills: int
    deaths: int


@dataclass
class PlayerMapStat:
    name: str
    team: str
    kills: int
    deaths: int


@dataclass
class MResult:
    number: int
    seed: int
    team1: str
    team2: str
    score1: int
    score2: int
    winner: str
    mvp1: PStat
    mvp2: PStat
    initial_attacker: str
    overtime: bool
    player_stats: list[PlayerMapStat]


@dataclass
class SeriesResult:
    team1: str
    team2: str
    maps_to_win: int
    team1_wins: int
    team2_wins: int
    winner: str
    loser: str
    maps: list[MResult]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def validate_preset(preset: Any) -> None:
    if preset is None or len(preset.players) != 5:
        raise ValueError("無効なチームプリセットです")
    if preset.igl not in preset.players:
        raise ValueError(f"{preset.name}: IGLが選手一覧に含まれていません")
    if preset.spike_holder not in preset.players:
        raise ValueError(f"{preset.name}: スパイク担当が選手一覧に含まれていません")


def validate_maps_to_win(value: Any) -> int:
    try:
        need = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("勝利に必要なマップ数は整数で入力してください") from exc
    if need < 1:
        raise ValueError("勝利に必要なマップ数は1以上にしてください")
    return need


def validate_seed_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in {"fixed", "random"}:
        raise ValueError("Seedモードが不正です")
    return mode


def validate_base_seed(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("固定Seedは整数で入力してください") from exc


def generate_map_seed(
    seed_mode: str,
    base_seed: int | None,
    sequence_index: int,
) -> int:
    if validate_seed_mode(seed_mode) == "random":
        return secrets.randbits(32)
    if base_seed is None:
        raise ValueError("固定Seedが必要です")
    return int(base_seed) + int(sequence_index)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(prefix: str, data: dict[str, Any]) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"{safe_name(prefix)}_{timestamp()}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def mvp_from_match_stats(
    match_stats: dict[Any, dict[str, Any]],
    player_keys: list[TeamPlayerKey],
) -> PStat:
    """
    チーム固有キーで通算match_statsを読み、MVPを決定する。

    Characterはラウンドごとに再生成されるため、終了時のgame.charsではなく
    通算match_statsを使用する。TeamPlayerKeyにより同名選手も区別される。
    """
    values: list[PStat] = []

    for key in player_keys:
        stats = match_stats.get(key, {})
        values.append(
            PStat(
                name=str(key),
                kills=int(stats.get("kills", 0)),
                deaths=int(stats.get("deaths", 0)),
            )
        )

    if not values:
        raise RuntimeError("MVP対象の選手統計が見つかりません")

    order = {str(key): index for index, key in enumerate(player_keys)}
    return max(
        values,
        key=lambda item: (
            item.kills,
            -item.deaths,
            -order.get(item.name, 999),
        ),
    )


def player_stats_from_match_stats(
    match_stats: dict[Any, dict[str, Any]],
    player_keys: list[TeamPlayerKey],
    team_name: str,
) -> list[PlayerMapStat]:
    return [
        PlayerMapStat(
            name=str(key),
            team=team_name,
            kills=int(match_stats.get(key, {}).get("kills", 0)),
            deaths=int(match_stats.get(key, {}).get("deaths", 0)),
        )
        for key in player_keys
    ]


def build_player_leaderboards(
    series_list: list[SeriesResult],
) -> dict[str, Any]:
    totals: dict[tuple[str, str], dict[str, Any]] = {}

    for series in series_list:
        for map_result in series.maps:
            for stat in map_result.player_stats:
                key = (stat.team, stat.name)
                row = totals.setdefault(
                    key,
                    {
                        "player": stat.name,
                        "team": stat.team,
                        "kills": 0,
                        "deaths": 0,
                        "maps": 0,
                        "mvps": 0,
                    },
                )
                row["kills"] += stat.kills
                row["deaths"] += stat.deaths
                row["maps"] += 1

            for team_name, mvp in (
                (map_result.team1, map_result.mvp1),
                (map_result.team2, map_result.mvp2),
            ):
                key = (team_name, mvp.name)
                row = totals.setdefault(
                    key,
                    {
                        "player": mvp.name,
                        "team": team_name,
                        "kills": 0,
                        "deaths": 0,
                        "maps": 0,
                        "mvps": 0,
                    },
                )
                row["mvps"] += 1

    players = []
    for base in totals.values():
        row = dict(base)
        row["kd"] = (
            round(row["kills"] / row["deaths"], 3)
            if row["deaths"]
            else float(row["kills"])
        )
        row["kills_per_map"] = (
            round(row["kills"] / row["maps"], 3) if row["maps"] else 0.0
        )
        players.append(row)

    return {
        "kd_top5": sorted(
            players,
            key=lambda r: (-r["kd"], -r["kills"], r["deaths"]),
        )[:5],
        "mvp_top5": sorted(
            players,
            key=lambda r: (-r["mvps"], -r["kd"], -r["kills"]),
        )[:5],
        "kills_top5": sorted(
            players,
            key=lambda r: (-r["kills"], -r["kd"], r["deaths"]),
        )[:5],
        "kills_per_map_top5": sorted(
            players,
            key=lambda r: (
                -r["kills_per_map"],
                -r["kd"],
                -r["kills"],
            ),
        )[:5],
        "all_players": sorted(
            players,
            key=lambda r: (-r["kd"], -r["kills"]),
        ),
    }


def _key_for_name(
    keys: list[TeamPlayerKey],
    name: str,
) -> TeamPlayerKey:
    for key in keys:
        if str(key) == str(name):
            return key
    raise ValueError(f"選手キーが見つかりません: {name}")


def original_scores(
    game: VisualFPSBattle, team1_players: list[str], team2_players: list[str]
) -> tuple[int, int]:
    current_a = {str(name) for name in (game.attacker_roster or [])}
    current_d = {str(name) for name in (game.defender_roster or [])}
    team1_set = {str(name) for name in team1_players}
    team2_set = {str(name) for name in team2_players}

    if current_a == team1_set and current_d == team2_set:
        return int(game.attacker_wins), int(game.defender_wins)
    if current_a == team2_set and current_d == team1_set:
        return int(game.defender_wins), int(game.attacker_wins)
    raise RuntimeError("試合終了後のチームとスコアを対応付けられません")


def play_map(
    team1: Any,
    team2: Any,
    map_number: int,
    seed: int,
    render: bool,
    team1_controller_key: str = "fnatic_v1",
    team2_controller_key: str = "fnatic_v1",
    team1_series_wins: int = 0,
    team2_series_wins: int = 0,
    series_maps_to_win: int = 1,
) -> MResult:
    seed_all(seed)

    team1_controller_key = str(team1_controller_key or "fnatic_v1")
    team2_controller_key = str(team2_controller_key or "fnatic_v1")

    if "user" in {team1_controller_key, team2_controller_key} and not render:
        raise ValueError("ユーザー操作を使う試合は描画が必要です")

    ai1 = _build_team_ai(team1_controller_key)
    ai2 = _build_team_ai(team2_controller_key)

    team1_keys = [TeamPlayerKey(name, f"team1:{team1.name}") for name in team1.players]
    team2_keys = [TeamPlayerKey(name, f"team2:{team2.name}") for name in team2.players]

    if map_number % 2 == 1:
        attacker, defender = team1, team2
        attacker_keys, defender_keys = team1_keys, team2_keys
        attacker_ai, defender_ai = ai1, ai2
    else:
        attacker, defender = team2, team1
        attacker_keys, defender_keys = team2_keys, team1_keys
        attacker_ai, defender_ai = ai2, ai1

    attacker_spike = _key_for_name(
        attacker_keys,
        attacker.spike_holder,
    )
    defender_spike = _key_for_name(
        defender_keys,
        defender.spike_holder,
    )
    attacker_igl = _key_for_name(attacker_keys, attacker.igl)
    defender_igl = _key_for_name(defender_keys, defender.igl)

    if attacker is team1:
        attacker_series_wins = int(team1_series_wins)
        attacker_series_losses = int(team2_series_wins)
        defender_series_wins = int(team2_series_wins)
        defender_series_losses = int(team1_series_wins)
    else:
        attacker_series_wins = int(team2_series_wins)
        attacker_series_losses = int(team1_series_wins)
        defender_series_wins = int(team1_series_wins)
        defender_series_losses = int(team2_series_wins)

    series_context = {
        "maps_played": max(0, int(map_number) - 1),
        "maps_to_win": max(1, int(series_maps_to_win)),
        "attacker_maps_won": attacker_series_wins,
        "attacker_maps_lost": attacker_series_losses,
        "defender_maps_won": defender_series_wins,
        "defender_maps_lost": defender_series_losses,
    }

    output_context = (
        contextlib.nullcontext()
        if render
        else contextlib.redirect_stdout(io.StringIO())
    )
    with output_context:
        game = VisualFPSBattle(
            NEW_MAZE_STR,
            attacker_ai,
            defender_ai,
            headless=not render,
            attacker_roster=list(attacker_keys),
            defender_roster=list(defender_keys),
            spike_holder_name=attacker_spike,
            defender_spike_holder_name=defender_spike,
            attacker_igl_name=attacker_igl,
            defender_igl_name=defender_igl,
            attacker_team_name=attacker.name,
            defender_team_name=defender.name,
            disable_side_swap=False,
            series_context=series_context,
        )

        if render:
            # 管理画面は隠さず、観戦画面と同時に表示する。
            # VisualFPSBattle側のmainloopはネストされるが、
            # 同じTkイベント系の管理画面も更新され続ける。
            # destroy()で観戦用Tkを破棄すると、大会管理画面側の
            # Tcl/Tkイベント処理まで止まる環境がある。
            # quit()でこの観戦画面のmainloopだけを抜け、
            # ウィンドウ自体はwithdraw()で非表示にする。
            closing_started = False

            def finish_rendered_match() -> None:
                nonlocal closing_started
                if closing_started:
                    return
                closing_started = True
                try:
                    game.root.withdraw()
                except tk.TclError:
                    pass
                try:
                    game.root.quit()
                except tk.TclError:
                    pass

            def watch_match_end() -> None:
                try:
                    if bool(getattr(game, "match_over", False)):
                        # 最終スコアを少し表示してから次のマップへ進む。
                        game.root.after(
                            1200,
                            finish_rendered_match,
                        )
                        return
                    game.root.after(100, watch_match_end)
                except tk.TclError:
                    return

            # 手動で閉じた場合も大会自体は継続する。
            game.root.protocol(
                "WM_DELETE_WINDOW",
                finish_rendered_match,
            )
            game.root.after(100, watch_match_end)

        game.run()

        if render:
            # run()内のmainloopから戻ったことを保証したうえで非表示化。
            try:
                game.root.withdraw()
                game.root.update_idletasks()
            except tk.TclError:
                pass

    score1, score2 = original_scores(
        game,
        list(team1.players),
        list(team2.players),
    )
    winner = team1.name if score1 > score2 else team2.name

    return MResult(
        number=map_number,
        seed=int(seed),
        team1=team1.name,
        team2=team2.name,
        score1=score1,
        score2=score2,
        winner=winner,
        mvp1=mvp_from_match_stats(game.match_stats, team1_keys),
        mvp2=mvp_from_match_stats(game.match_stats, team2_keys),
        initial_attacker=attacker.name,
        overtime=bool(getattr(game, "overtime", False)),
        player_stats=(
            player_stats_from_match_stats(game.match_stats, team1_keys, team1.name)
            + player_stats_from_match_stats(game.match_stats, team2_keys, team2.name)
        ),
    )


def run_series_core(
    team1_name: str,
    team2_name: str,
    maps_to_win: int,
    seed_mode: str,
    base_seed: int | None,
    seed_offset: int,
    render: bool | Callable[[], bool],
    emit: Callable[[tuple[Any, ...]], None],
    team_controllers: dict[str, str] | None = None,
    context_label: str = "",
) -> SeriesResult:
    if team1_name == team2_name:
        raise ValueError("異なる2チームを選んでください")

    need = validate_maps_to_win(maps_to_win)
    seed_mode = validate_seed_mode(seed_mode)
    team1 = get_preset(team1_name)
    team2 = get_preset(team2_name)
    validate_preset(team1)
    validate_preset(team2)

    controller_map = team_controllers or {}
    controller1 = str(controller_map.get(team1.name, "fnatic_v1"))
    controller2 = str(controller_map.get(team2.name, "fnatic_v1"))

    wins1 = 0
    wins2 = 0
    maps: list[MResult] = []

    while wins1 < need and wins2 < need:
        map_number = len(maps) + 1
        prefix = f"{context_label} / " if context_label else ""
        emit(
            (
                "status",
                f"{prefix}{team1.name} vs {team2.name} / MAP {map_number} 実行中…",
            )
        )
        map_seed = generate_map_seed(
            seed_mode,
            base_seed,
            seed_offset + map_number - 1,
        )
        emit(("log", f"{prefix}MAP {map_number} Seed: {map_seed}\n"))
        emit(
            (
                "log",
                f"{prefix}Controller: "
                f"{team1.name}="
                f"{CONTROLLER_KEY_TO_DISPLAY.get(controller1, controller1)}"
                f" / {team2.name}="
                f"{CONTROLLER_KEY_TO_DISPLAY.get(controller2, controller2)}\n",
            )
        )
        user_match = "user" in {controller1, controller2}
        current_render = bool(render()) if callable(render) else bool(render)
        if user_match:
            current_render = True

        emit(
            (
                "log",
                f"{prefix}Render: "
                f"{'ON' if current_render else 'OFF'}"
                + ("（ユーザー操作のため強制ON）" if user_match else "")
                + "\n",
            )
        )
        if hasattr(render, "play_map"):
            result = render.play_map(
                team1,
                team2,
                map_number,
                map_seed,
                controller1,
                controller2,
                wins1,
                wins2,
                need,
            )
        else:
            result = play_map(
                team1,
                team2,
                map_number,
                map_seed,
                current_render,
                controller1,
                controller2,
                wins1,
                wins2,
                need,
            )
        maps.append(result)

        if result.winner == team1.name:
            wins1 += 1
        else:
            wins2 += 1

        emit(("map", result, context_label))
        emit(("series_score", team1.name, wins1, wins2, team2.name, context_label))

    winner = team1.name if wins1 > wins2 else team2.name
    loser = team2.name if winner == team1.name else team1.name
    return SeriesResult(
        team1=team1.name,
        team2=team2.name,
        maps_to_win=need,
        team1_wins=wins1,
        team2_wins=wins2,
        winner=winner,
        loser=loser,
        maps=maps,
    )


def choose_pairings(
    active: list[str], records: dict[str, dict[str, int]], played: set[frozenset[str]]
) -> tuple[list[tuple[str, str]], str | None]:
    """成績の近い相手を優先し、可能なら再戦を避ける2敗脱落用ペアリング。"""
    ordered = sorted(
        active,
        key=lambda name: (
            records[name]["losses"],
            -records[name]["wins"],
            name.lower(),
        ),
    )

    bye = None
    if len(ordered) % 2 == 1:
        # 試合数が少なく、かつ敗戦数が少ないチームをbyeにする。
        bye = min(
            ordered,
            key=lambda name: (
                records[name]["matches"],
                records[name]["losses"],
                -records[name]["wins"],
                name.lower(),
            ),
        )
        ordered.remove(bye)

    pairings: list[tuple[str, str]] = []
    while ordered:
        team = ordered.pop(0)
        opponent_index = None
        for index, candidate in enumerate(ordered):
            if frozenset((team, candidate)) not in played:
                opponent_index = index
                break
        if opponent_index is None:
            opponent_index = 0
        opponent = ordered.pop(opponent_index)
        pairings.append((team, opponent))

    return pairings, bye


def _next_power_of_two(value: int) -> int:
    size = 1
    while size < value:
        size *= 2
    return size


def _pair_nodes(
    nodes: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(nodes) % 2 != 0:
        raise RuntimeError(f"ブラケットノード数が偶数ではありません: {len(nodes)}")
    return [(nodes[index], nodes[index + 1]) for index in range(0, len(nodes), 2)]


def _spread_match_positions(
    match_count: int,
    count: int,
) -> list[int]:
    """指定数の位置をブラケット全体へできるだけ均等に分散する。"""
    if count <= 0:
        return []
    if count >= match_count:
        return list(range(match_count))
    if count == 1:
        return [0]

    positions: list[int] = []
    for index in range(count):
        pos = round(index * (match_count - 1) / (count - 1))
        while pos in positions and pos + 1 < match_count:
            pos += 1
        while pos in positions and pos - 1 >= 0:
            pos -= 1
        positions.append(pos)
    return positions


def build_seeded_bracket_slots(
    team_names: list[str],
    seed_count: int,
    seed_method: str,
    manual_seeds: list[str] | None,
    rating_snapshot: dict[str, float] | None,
) -> tuple[list[str | None], list[str]]:
    """
    シードを分散配置し、BYEを優先的にシードへ割り当てる。

    12チームの場合:
      bracket_size=16 / bye_count=4
      seed_count=4なら、4シードすべてがRound 1 BYE。
    """
    if len(set(team_names)) != len(team_names):
        raise ValueError("同じチームを複数設定できません")

    allowed_counts = {0, 2, 4, 8}
    if seed_count not in allowed_counts:
        raise ValueError("シード数は0・2・4・8から選んでください")
    if seed_count > len(team_names):
        raise ValueError(
            f"シード数{seed_count}は参加チーム数{len(team_names)}を超えています"
        )

    method = str(seed_method).strip().lower()
    if method not in {"rating", "manual", "random"}:
        raise ValueError("シード決定方法はrating・manual・randomのいずれかです")

    if seed_count == 0:
        seeded_teams: list[str] = []
    elif method == "rating":
        ratings = rating_snapshot or {}
        seeded_teams = sorted(
            team_names,
            key=lambda name: (
                -float(ratings.get(name, DEFAULT_TEAM_RATING)),
                team_names.index(name),
            ),
        )[:seed_count]
    elif method == "manual":
        candidates = [name for name in (manual_seeds or []) if name and name != UNUSED]
        if len(candidates) != seed_count:
            raise ValueError(f"手動シードを{seed_count}チームすべて設定してください")
        if len(set(candidates)) != len(candidates):
            raise ValueError("手動シードに同じチームが重複しています")
        missing = [name for name in candidates if name not in team_names]
        if missing:
            raise ValueError(
                "参加チームではない手動シードがあります: " + ", ".join(missing)
            )
        seeded_teams = candidates
    else:
        seeded_teams = secrets.SystemRandom().sample(
            team_names,
            seed_count,
        )

    bracket_size = _next_power_of_two(len(team_names))
    match_count = bracket_size // 2
    bye_count = bracket_size - len(team_names)

    OPEN = object()
    slots: list[Any] = [OPEN] * bracket_size

    # シード同士が初戦で当たりにくいように、別々の試合へ均等配置。
    seed_match_positions = _spread_match_positions(
        match_count,
        len(seeded_teams),
    )

    # BYEはシード位置へ優先配分し、残りもブラケット全体へ分散。
    bye_match_positions: list[int] = []
    for pos in seed_match_positions:
        if len(bye_match_positions) >= bye_count:
            break
        bye_match_positions.append(pos)

    if len(bye_match_positions) < bye_count:
        for pos in _spread_match_positions(match_count, bye_count):
            if pos not in bye_match_positions:
                bye_match_positions.append(pos)
            if len(bye_match_positions) >= bye_count:
                break

    for match_pos in bye_match_positions:
        slots[match_pos * 2 + 1] = None

    for team, match_pos in zip(seeded_teams, seed_match_positions):
        preferred = match_pos * 2
        alternate = preferred + 1
        if slots[preferred] is OPEN:
            slots[preferred] = team
        elif slots[alternate] is OPEN:
            slots[alternate] = team
        else:
            raise RuntimeError("シード配置先を確保できませんでした")

    non_seeded = [team for team in team_names if team not in seeded_teams]
    open_indices = [index for index, value in enumerate(slots) if value is OPEN]
    if len(open_indices) != len(non_seeded):
        raise RuntimeError(
            "ブラケット空き枠数とノンシード数が一致しません: "
            f"open={len(open_indices)} teams={len(non_seeded)}"
        )

    for index, team in zip(open_indices, non_seeded):
        slots[index] = team

    return list(slots), seeded_teams



def build_double_elimination_ranking(
    team_names: list[str],
    bracket_matches: list[dict[str, Any]],
    champion: str,
    runner_up: str,
    records: dict[str, dict[str, int]],
) -> list[str]:
    """
    ダブルエリミネーションの最終順位を実際の敗退順から作る。

    1位はGrand Final勝者、2位はGrand Final敗者。
    3位以下は、2敗目を喫して大会から脱落した時点が遅い順に並べる。
    これによりLower Final敗者が必ず3位になる。
    """
    loss_counts = {name: 0 for name in team_names}
    elimination_index: dict[str, int] = {}
    latest_loss_index: dict[str, int] = {}

    for index, match in enumerate(bracket_matches):
        if match.get("status") != "finished":
            continue

        loser = match.get("loser")
        if not loser or loser not in loss_counts:
            continue

        loss_counts[loser] += 1
        latest_loss_index[loser] = index

        if loss_counts[loser] >= 2:
            elimination_index[loser] = index

    fixed = [champion]
    if runner_up and runner_up != champion:
        fixed.append(runner_up)

    remaining = [
        name
        for name in team_names
        if name not in fixed
    ]

    # 原則は2敗目を喫した時点が遅い順。
    # 万一2敗目が記録されていないチームがあれば、最後に敗れた時点、
    # それもなければ従来の戦績を補助基準として使う。
    remaining.sort(
        key=lambda name: (
            -elimination_index.get(
                name,
                latest_loss_index.get(name, -1),
            ),
            -int(records[name].get("wins", 0)),
            -(
                int(records[name].get("map_wins", 0))
                - int(records[name].get("map_losses", 0))
            ),
            name.lower(),
        )
    )

    return fixed + remaining


def run_double_elimination(
    team_names: list[str],
    normal_maps_to_win: int,
    lower_final_maps_to_win: int,
    grand_final_maps_to_win: int,
    tournament_seed_config: dict[str, Any],
    seed_mode: str,
    base_seed: int | None,
    render: bool | Callable[[], bool],
    emit: Callable[[tuple[Any, ...]], None],
    team_controllers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """固定Slot式ダブルエリミネーション。"""
    if len(team_names) < 4:
        raise ValueError("ダブルエリミネーションには4チーム以上必要です")
    if len(set(team_names)) != len(team_names):
        raise ValueError("同じチームを複数のSlotへ設定できません")

    for name in team_names:
        validate_preset(get_preset(name))

    normal_need = validate_maps_to_win(normal_maps_to_win)
    lower_final_need = validate_maps_to_win(lower_final_maps_to_win)
    grand_final_need = validate_maps_to_win(grand_final_maps_to_win)

    seed_count = int(tournament_seed_config.get("seed_count", 0))
    seed_method = str(tournament_seed_config.get("seed_method", "rating"))
    manual_seeds = list(tournament_seed_config.get("manual_seeds", []))
    rating_snapshot = {
        str(name): float(value)
        for name, value in tournament_seed_config.get(
            "rating_snapshot",
            {},
        ).items()
    }

    slots, resolved_seeds = build_seeded_bracket_slots(
        team_names,
        seed_count,
        seed_method,
        manual_seeds,
        rating_snapshot,
    )
    bracket_size = len(slots)

    emit(
        (
            "log",
            "  Resolved seeds: "
            + (
                " / ".join(
                    f"{index + 1}.{name}" for index, name in enumerate(resolved_seeds)
                )
                if resolved_seeds
                else "none"
            )
            + "\n"
            + "  Initial bracket slots: "
            + " | ".join(value if value is not None else "BYE" for value in slots)
            + "\n"
            + "-" * 78
            + "\n",
        )
    )

    records = {
        name: {
            "wins": 0,
            "losses": 0,
            "matches": 0,
            "map_wins": 0,
            "map_losses": 0,
        }
        for name in team_names
    }
    completed_series: list[SeriesResult] = []
    bracket_matches: list[dict[str, Any]] = []
    series_counter = 0

    def emit_match(result: dict[str, Any]) -> None:
        bracket_matches.append(result)
        emit(("bracket_match", dict(result)))

    def play_match(
        match_id: str,
        bracket: str,
        round_number: int,
        match_number: int,
        left: dict[str, Any],
        right: dict[str, Any],
        maps_to_win: int,
        special: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal series_counter
        team1 = left.get("team")
        team2 = right.get("team")

        if team1 is None or team2 is None:
            winner = team1 or team2
            result = {
                "id": match_id,
                "bracket": bracket,
                "round": round_number,
                "match": match_number,
                "team1": team1,
                "team2": team2,
                "winner": winner,
                "loser": None,
                "team1_wins": 0,
                "team2_wins": 0,
                "maps": [],
                "status": "bye",
                "special": special,
                "source1": left.get("source"),
                "source2": right.get("source"),
            }
            emit_match(result)
            return (
                {"team": winner, "source": match_id},
                {"team": None, "source": match_id},
            )

        series_counter += 1
        if special == "lower_final":
            context = "LOWER FINAL"
        elif special == "grand_final":
            context = "GRAND FINAL"
        else:
            side = "WINNERS" if bracket == "W" else "LOSERS"
            context = f"{side} R{round_number} M{match_number}"

        pending = {
            "id": match_id,
            "bracket": bracket,
            "round": round_number,
            "match": match_number,
            "team1": team1,
            "team2": team2,
            "winner": None,
            "loser": None,
            "team1_wins": 0,
            "team2_wins": 0,
            "maps": [],
            "status": "playing",
            "special": special,
            "source1": left.get("source"),
            "source2": right.get("source"),
        }
        emit(("bracket_match", dict(pending)))

        series = run_series_core(
            team1_name=team1,
            team2_name=team2,
            maps_to_win=maps_to_win,
            seed_mode=seed_mode,
            base_seed=base_seed,
            seed_offset=series_counter * 100000,
            render=render,
            emit=emit,
            team_controllers=team_controllers,
            context_label=context,
        )
        completed_series.append(series)

        records[series.winner]["wins"] += 1
        records[series.loser]["losses"] += 1
        for team in (series.team1, series.team2):
            records[team]["matches"] += 1
        records[series.team1]["map_wins"] += series.team1_wins
        records[series.team1]["map_losses"] += series.team2_wins
        records[series.team2]["map_wins"] += series.team2_wins
        records[series.team2]["map_losses"] += series.team1_wins

        final = {
            **pending,
            "winner": series.winner,
            "loser": series.loser,
            "team1_wins": series.team1_wins,
            "team2_wins": series.team2_wins,
            "maps": [asdict(item) for item in series.maps],
            "status": "finished",
        }
        emit_match(final)
        emit(("series_done", series, context))
        return (
            {"team": series.winner, "source": match_id},
            {"team": series.loser, "source": match_id},
        )

    winners_nodes = [
        {"team": team, "source": f"SLOT-{index + 1}"}
        for index, team in enumerate(slots)
    ]
    winners_round = 1
    lower_round = 0
    lower_survivors: list[dict[str, Any]] = []

    while len(winners_nodes) > 1:
        next_winners: list[dict[str, Any]] = []
        current_losers: list[dict[str, Any]] = []

        for match_number, (left, right) in enumerate(_pair_nodes(winners_nodes), 1):
            winner_node, loser_node = play_match(
                f"W{winners_round}M{match_number}",
                "W",
                winners_round,
                match_number,
                left,
                right,
                normal_need,
            )
            next_winners.append(winner_node)
            if loser_node.get("team") is not None:
                current_losers.append(loser_node)

        if winners_round == 1:
            lower_round = 1
            working = list(reversed(current_losers))
            if len(working) % 2 == 1:
                lower_survivors.append(working.pop(0))
            for match_number, pair in enumerate(_pair_nodes(working), 1):
                winner_node, _ = play_match(
                    f"L{lower_round}M{match_number}",
                    "L",
                    lower_round,
                    match_number,
                    pair[0],
                    pair[1],
                    normal_need,
                )
                lower_survivors.append(winner_node)
        else:
            incoming = list(reversed(current_losers))

            # BYEを含む12チーム大会などでは、Winners側から落ちる人数と
            # Losers側の生存者数が一致しない場合がある。
            #
            # 既存Losers側が多い場合はLosers内で予備ラウンドを行い、
            # Winners流入側が多い場合は流入チーム同士で予備ラウンドを
            # 行ってから、両グループを同数にして合流させる。
            while len(lower_survivors) > len(incoming):
                lower_round += 1
                reduced: list[dict[str, Any]] = []
                working = list(lower_survivors)

                # 奇数なら先頭をこの予備ラウンドのBYEとして残す。
                if len(working) % 2 == 1:
                    reduced.append(working.pop(0))

                for match_number, pair in enumerate(_pair_nodes(working), 1):
                    winner_node, _ = play_match(
                        f"L{lower_round}M{match_number}",
                        "L",
                        lower_round,
                        match_number,
                        pair[0],
                        pair[1],
                        normal_need,
                    )
                    reduced.append(winner_node)

                lower_survivors = reduced

            while len(incoming) > len(lower_survivors):
                lower_round += 1
                reduced_incoming: list[dict[str, Any]] = []
                working = list(incoming)

                # シードBYEによる人数差では、流入側にもBYEが必要になる。
                # 末尾ではなく先頭を残し、毎回同じ側だけが有利に
                # なりにくいよう、current_losersは事前にreverse済み。
                if len(working) % 2 == 1:
                    reduced_incoming.append(working.pop(0))

                for match_number, pair in enumerate(_pair_nodes(working), 1):
                    winner_node, _ = play_match(
                        f"L{lower_round}M{match_number}",
                        "L",
                        lower_round,
                        match_number,
                        pair[0],
                        pair[1],
                        normal_need,
                    )
                    reduced_incoming.append(winner_node)

                incoming = reduced_incoming

            if len(lower_survivors) != len(incoming):
                raise RuntimeError(
                    "Losersブラケットの人数調整に失敗しました: "
                    f"survivors={len(lower_survivors)} "
                    f"incoming={len(incoming)}"
                )

            lower_round += 1
            merged: list[dict[str, Any]] = []

            for match_number, (survivor, dropped) in enumerate(
                zip(lower_survivors, incoming), 1
            ):
                is_lower_final = len(next_winners) == 1 and len(lower_survivors) == 1
                match_id = (
                    "LOWER_FINAL"
                    if is_lower_final
                    else f"L{lower_round}M{match_number}"
                )
                winner_node, _ = play_match(
                    match_id,
                    "L",
                    lower_round,
                    match_number,
                    survivor,
                    dropped,
                    lower_final_need if is_lower_final else normal_need,
                    "lower_final" if is_lower_final else "",
                )
                merged.append(winner_node)

            lower_survivors = merged

        winners_nodes = next_winners
        winners_round += 1

    winners_champion = winners_nodes[0]

    while len(lower_survivors) > 1:
        lower_round += 1
        next_lower: list[dict[str, Any]] = []
        pairs = _pair_nodes(lower_survivors)
        for match_number, pair in enumerate(pairs, 1):
            is_last = len(pairs) == 1
            match_id = "LOWER_FINAL" if is_last else f"L{lower_round}M{match_number}"
            winner_node, _ = play_match(
                match_id,
                "L",
                lower_round,
                match_number,
                pair[0],
                pair[1],
                lower_final_need if is_last else normal_need,
                "lower_final" if is_last else "",
            )
            next_lower.append(winner_node)
        lower_survivors = next_lower

    if not lower_survivors:
        raise RuntimeError("Losersブラケット勝者を決定できません")

    grand_winner, grand_loser = play_match(
        "GRAND_FINAL",
        "G",
        1,
        1,
        winners_champion,
        lower_survivors[0],
        grand_final_need,
        "grand_final",
    )

    champion = grand_winner["team"]
    runner_up = grand_loser["team"]
    ranking = build_double_elimination_ranking(
        team_names,
        bracket_matches,
        champion,
        runner_up,
        records,
    )

    data = {
        "mode": "double_elimination",
        "seed_mode": seed_mode,
        "base_seed": base_seed if seed_mode == "fixed" else None,
        "normal_maps_to_win": normal_need,
        "lower_final_maps_to_win": lower_final_need,
        "grand_final_maps_to_win": grand_final_need,
        "teams": team_names,
        "team_controllers": {
            name: (team_controllers or {}).get(
                name,
                "fnatic_v1",
            )
            for name in team_names
        },
        "tournament_seeding": {
            "seed_count": seed_count,
            "seed_method": seed_method,
            "resolved_seeds": resolved_seeds,
            "manual_seeds": (manual_seeds if seed_method == "manual" else []),
            "rating_snapshot": (rating_snapshot if seed_method == "rating" else {}),
        },
        "slots": slots,
        "bracket_size": bracket_size,
        "champion": champion,
        "runner_up": runner_up,
        "records": records,
        "ranking": ranking,
        "bracket_matches": bracket_matches,
        "player_leaderboards": build_player_leaderboards(completed_series),
    }
    path = save_json(f"double_elimination_{champion}", data)
    emit(("competition_done", data, str(path)))
    return data


def run_round_robin(
    team_names: list[str],
    maps_to_win: int,
    seed_mode: str,
    base_seed: int | None,
    render: bool | Callable[[], bool],
    emit: Callable[[tuple[Any, ...]], None],
    team_controllers: dict[str, str] | None = None,
) -> dict[str, Any]:
    if len(team_names) < 2:
        raise ValueError("総当たりリーグには2チーム以上必要です")
    if len(set(team_names)) != len(team_names):
        raise ValueError("同じチームを複数の枠へ設定できません")

    for name in team_names:
        validate_preset(get_preset(name))

    need = validate_maps_to_win(maps_to_win)
    table = {
        name: {
            "series_wins": 0,
            "series_losses": 0,
            "map_wins": 0,
            "map_losses": 0,
            "round_wins": 0,
            "round_losses": 0,
        }
        for name in team_names
    }
    matches: list[dict[str, Any]] = []
    completed_series: list[SeriesResult] = []

    pair_list = list(combinations(team_names, 2))
    for index, (left, right) in enumerate(pair_list, start=1):
        # 初期Attacker偏りを減らすため、奇数試合では枠順、偶数試合では逆順。
        team1, team2 = (left, right) if index % 2 == 1 else (right, left)
        context = f"LEAGUE {index}/{len(pair_list)}"
        series = run_series_core(
            team1_name=team1,
            team2_name=team2,
            maps_to_win=need,
            seed_mode=seed_mode,
            base_seed=base_seed,
            seed_offset=index * 100000,
            render=render,
            emit=emit,
            team_controllers=team_controllers,
            context_label=context,
        )
        completed_series.append(series)
        matches.append(asdict(series))

        table[series.winner]["series_wins"] += 1
        table[series.loser]["series_losses"] += 1
        table[series.team1]["map_wins"] += series.team1_wins
        table[series.team1]["map_losses"] += series.team2_wins
        table[series.team2]["map_wins"] += series.team2_wins
        table[series.team2]["map_losses"] += series.team1_wins

        for map_result in series.maps:
            table[map_result.team1]["round_wins"] += map_result.score1
            table[map_result.team1]["round_losses"] += map_result.score2
            table[map_result.team2]["round_wins"] += map_result.score2
            table[map_result.team2]["round_losses"] += map_result.score1

        emit(("series_done", series, context))
        emit(("standings", table.copy(), "league"))

    ranking = sorted(
        team_names,
        key=lambda name: (
            -table[name]["series_wins"],
            -(table[name]["map_wins"] - table[name]["map_losses"]),
            -(table[name]["round_wins"] - table[name]["round_losses"]),
            name.lower(),
        ),
    )
    data = {
        "mode": "round_robin",
        "seed_mode": seed_mode,
        "base_seed": base_seed if seed_mode == "fixed" else None,
        "maps_to_win": need,
        "teams": team_names,
        "team_controllers": {
            name: (team_controllers or {}).get(
                name,
                "fnatic_v1",
            )
            for name in team_names
        },
        "champion": ranking[0],
        "standings": table,
        "ranking": ranking,
        "matches": matches,
        "player_leaderboards": build_player_leaderboards(completed_series),
    }
    path = save_json(f"round_robin_{ranking[0]}", data)
    emit(("competition_done", data, str(path)))
    return data


class TeamRatingStore:
    """大会をまたいでチームレーティングと推移を保存する。"""

    def __init__(
        self,
        team_names: list[str],
        path: Path = RATING_FILE,
    ) -> None:
        self.path = path
        self.default_rating = float(DEFAULT_TEAM_RATING)
        self.ratings: dict[str, float] = {}
        self.history: list[dict[str, Any]] = []
        self._load()

        changed = False
        for name in team_names:
            if name not in self.ratings:
                self.ratings[name] = self.default_rating
                changed = True
        if changed or not self.path.exists():
            self.save()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.default_rating = float(data.get("default_rating", DEFAULT_TEAM_RATING))
            self.ratings = {
                str(name): float(value)
                for name, value in data.get("ratings", {}).items()
            }
            self.history = list(data.get("history", []))
        except Exception:
            # 壊れたファイルで大会自体が起動不能になるのを避ける。
            self.ratings = {}
            self.history = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "default_rating": self.default_rating,
            "k_factor": RATING_K_FACTOR,
            "ratings": {
                name: round(value, 3) for name, value in sorted(self.ratings.items())
            },
            "history": self.history,
        }
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, team: str) -> float:
        if team not in self.ratings:
            self.ratings[team] = self.default_rating
        return float(self.ratings[team])

    def set_rating(
        self,
        team: str,
        value: float,
        *,
        reason: str = "manual",
    ) -> None:
        before = self.get(team)
        after = max(0.0, float(value))
        self.ratings[team] = after
        self.history.append(
            {
                "event": len(self.history) + 1,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "type": reason,
                "team": team,
                "before": round(before, 3),
                "after": round(after, 3),
                "delta": round(after - before, 3),
            }
        )
        self.save()

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def margin_multiplier(
        wins_a: int,
        wins_b: int,
    ) -> float:
        total = max(1, int(wins_a) + int(wins_b))
        diff = abs(int(wins_a) - int(wins_b))
        # 接戦=1.0、完封に近いほど最大1.6。
        return min(1.6, 1.0 + 0.6 * diff / total)

    def update_series(
        self,
        series: SeriesResult,
        context: str,
        competition_id: str,
    ) -> dict[str, Any]:
        team1 = series.team1
        team2 = series.team2
        before1 = self.get(team1)
        before2 = self.get(team2)

        expected1 = self.expected_score(before1, before2)
        actual1 = 1.0 if series.winner == team1 else 0.0
        multiplier = self.margin_multiplier(
            series.team1_wins,
            series.team2_wins,
        )
        delta1 = RATING_K_FACTOR * multiplier * (actual1 - expected1)
        delta2 = -delta1

        after1 = max(0.0, before1 + delta1)
        after2 = max(0.0, before2 + delta2)
        self.ratings[team1] = after1
        self.ratings[team2] = after2

        record = {
            "event": len(self.history) + 1,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "type": "series",
            "competition_id": competition_id,
            "context": context,
            "team1": team1,
            "team2": team2,
            "score1": int(series.team1_wins),
            "score2": int(series.team2_wins),
            "winner": series.winner,
            "before": {
                team1: round(before1, 3),
                team2: round(before2, 3),
            },
            "after": {
                team1: round(after1, 3),
                team2: round(after2, 3),
            },
            "delta": {
                team1: round(delta1, 3),
                team2: round(delta2, 3),
            },
            "expected": {
                team1: round(expected1, 4),
                team2: round(1.0 - expected1, 4),
            },
            "margin_multiplier": round(multiplier, 4),
        }
        self.history.append(record)
        self.save()
        return record

    def ranking(self) -> list[tuple[str, float]]:
        return sorted(
            self.ratings.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )

    def team_history(self, team: str) -> list[dict[str, Any]]:
        points = [
            {
                "event": 0,
                "rating": self.default_rating,
                "label": "START",
            }
        ]
        current = self.default_rating
        for row in self.history:
            if row.get("type") == "series":
                after = row.get("after", {})
                if team in after:
                    current = float(after[team])
                    points.append(
                        {
                            "event": int(row.get("event", len(points))),
                            "rating": current,
                            "label": str(row.get("context", "SERIES")),
                            "opponent": (
                                row.get("team2")
                                if row.get("team1") == team
                                else row.get("team1")
                            ),
                            "score": (
                                f"{row.get('score1')}-{row.get('score2')}"
                                if row.get("team1") == team
                                else f"{row.get('score2')}-{row.get('score1')}"
                            ),
                        }
                    )
            elif row.get("team") == team and row.get("after") is not None:
                current = float(row["after"])
                points.append(
                    {
                        "event": int(row.get("event", len(points))),
                        "rating": current,
                        "label": str(row.get("type", "manual")),
                    }
                )
        # 既存ファイルが履歴なしで初期値だけ持つ場合。
        if len(points) == 1 and team in self.ratings:
            points[0]["rating"] = float(self.ratings[team])
        return points


class TeamSlotEditor(tk.LabelFrame):
    def __init__(
        self, master: tk.Widget, names: list[str], title: str, initial_count: int = 8
    ):
        super().__init__(master, text=title, padx=8, pady=8)
        self.names = names
        self.slot_vars: list[tk.StringVar] = []
        self.slot_boxes: list[ttk.Combobox] = []
        self.controller_vars: list[tk.StringVar] = []
        self.controller_boxes: list[ttk.Combobox] = []
        self.count_var = tk.IntVar(value=max(2, min(MAX_TEAM_SLOTS, initial_count)))

        header = tk.Frame(self)
        header.pack(fill="x", pady=(0, 6))
        tk.Label(header, text="参加枠数").pack(side="left")
        self.count_spin = tk.Spinbox(
            header,
            from_=2,
            to=MAX_TEAM_SLOTS,
            textvariable=self.count_var,
            width=6,
            command=self.rebuild,
        )
        self.count_spin.pack(side="left", padx=8)
        tk.Button(header, text="枠を更新", command=self.rebuild).pack(side="left")

        self.canvas = tk.Canvas(self, height=250, highlightthickness=0)
        self.scroll = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas)
        self.inner.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")
        self.rebuild()

    def rebuild(self) -> None:
        try:
            count = max(2, min(MAX_TEAM_SLOTS, int(self.count_var.get())))
        except (TypeError, ValueError):
            count = 8
            self.count_var.set(count)

        previous = [var.get() for var in self.slot_vars]
        previous_controllers = [var.get() for var in self.controller_vars]
        for child in self.inner.winfo_children():
            child.destroy()
        self.slot_vars.clear()
        self.slot_boxes.clear()
        self.controller_vars.clear()
        self.controller_boxes.clear()

        options = [UNUSED] + self.names
        controller_options = list(CONTROLLER_OPTIONS.keys())
        for index in range(count):
            default = (
                previous[index]
                if index < len(previous)
                else (self.names[index] if index < len(self.names) else UNUSED)
            )
            var = tk.StringVar(value=default)
            box = ttk.Combobox(
                self.inner, values=options, textvariable=var, state="readonly", width=34
            )
            tk.Label(
                self.inner, text=f"Slot {index + 1:02d}", width=9, anchor="e"
            ).grid(row=index, column=0, padx=(0, 8), pady=3)
            box.grid(row=index, column=1, sticky="ew", pady=3)

            controller_default = (
                previous_controllers[index]
                if index < len(previous_controllers)
                else DEFAULT_CONTROLLER_DISPLAY
            )
            controller_var = tk.StringVar(value=controller_default)
            controller_box = ttk.Combobox(
                self.inner,
                values=controller_options,
                textvariable=controller_var,
                state="readonly",
                width=18,
            )
            tk.Label(
                self.inner,
                text="Controller",
                width=10,
                anchor="e",
            ).grid(
                row=index,
                column=2,
                padx=(12, 5),
                pady=3,
            )
            controller_box.grid(
                row=index,
                column=3,
                sticky="ew",
                pady=3,
            )

            self.slot_vars.append(var)
            self.slot_boxes.append(box)
            self.controller_vars.append(controller_var)
            self.controller_boxes.append(controller_box)

        self.inner.grid_columnconfigure(1, weight=1)
        self.inner.grid_columnconfigure(3, weight=0)

    def selected_teams(self) -> list[str]:
        return [
            var.get() for var in self.slot_vars if var.get() and var.get() != UNUSED
        ]

    def selected_team_controllers(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for team_var, controller_var in zip(
            self.slot_vars,
            self.controller_vars,
        ):
            team = team_var.get()
            if not team or team == UNUSED:
                continue
            display = controller_var.get()
            result[team] = CONTROLLER_OPTIONS.get(
                display,
                "fnatic_v1",
            )
        return result

    def set_enabled(self, enabled: bool) -> None:
        self.count_spin.config(state="normal" if enabled else "disabled")
        for box in self.slot_boxes:
            box.config(state="readonly" if enabled else "disabled")
        for box in self.controller_boxes:
            box.config(state="readonly" if enabled else "disabled")


class TournamentSeedEditor(tk.LabelFrame):
    """ダブルエリミネーション用のシード設定UI。"""

    def __init__(
        self,
        master: tk.Widget,
        names: list[str],
        rating_store: TeamRatingStore,
    ) -> None:
        super().__init__(
            master,
            text="シード設定",
            padx=8,
            pady=7,
        )
        self.names = names
        self.rating_store = rating_store

        self.seed_count_var = tk.StringVar(value="4")
        self.seed_method_var = tk.StringVar(value="rating")
        self.manual_vars = [tk.StringVar(value=UNUSED) for _ in range(8)]
        self.manual_boxes: list[ttk.Combobox] = []

        top = tk.Frame(self)
        top.pack(fill="x")

        tk.Label(top, text="シード数").pack(side="left")
        self.count_box = ttk.Combobox(
            top,
            values=["0", "2", "4", "8"],
            textvariable=self.seed_count_var,
            state="readonly",
            width=5,
        )
        self.count_box.pack(side="left", padx=(6, 16))
        self.count_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._refresh(),
        )

        tk.Label(top, text="決定方法").pack(side="left")
        methods = [
            ("レート順", "rating"),
            ("手動", "manual"),
            ("ランダム", "random"),
        ]
        for label, value in methods:
            tk.Radiobutton(
                top,
                text=label,
                variable=self.seed_method_var,
                value=value,
                command=self._refresh,
            ).pack(side="left", padx=(8, 0))

        self.fill_button = tk.Button(
            top,
            text="レート順を手動欄へ反映",
            command=self.fill_manual_from_rating,
        )
        self.fill_button.pack(side="right")

        self.preview_var = tk.StringVar()
        tk.Label(
            self,
            textvariable=self.preview_var,
            anchor="w",
            justify="left",
            fg="#334155",
        ).pack(fill="x", pady=(7, 3))

        self.manual_frame = tk.Frame(self)
        self.manual_frame.pack(fill="x")
        options = [UNUSED] + self.names
        for index, var in enumerate(self.manual_vars):
            tk.Label(
                self.manual_frame,
                text=f"Seed {index + 1}",
                width=8,
                anchor="e",
            ).grid(
                row=index // 4,
                column=(index % 4) * 2,
                padx=(0, 4),
                pady=2,
            )
            box = ttk.Combobox(
                self.manual_frame,
                values=options,
                textvariable=var,
                state="readonly",
                width=23,
            )
            box.grid(
                row=index // 4,
                column=(index % 4) * 2 + 1,
                padx=(0, 10),
                pady=2,
            )
            self.manual_boxes.append(box)

        self._refresh()

    def _seed_count(self) -> int:
        try:
            return int(self.seed_count_var.get())
        except ValueError:
            return 0

    def _refresh(self) -> None:
        count = self._seed_count()
        method = self.seed_method_var.get()

        rating_names = [name for name, _ in self.rating_store.ranking()][:count]
        if method == "rating":
            preview = "自動シード: " + (
                " / ".join(
                    f"{index + 1}.{name}" for index, name in enumerate(rating_names)
                )
                if rating_names
                else "なし"
            )
        elif method == "manual":
            preview = (
                f"Seed 1～{count}を下の欄で指定してください" if count else "シードなし"
            )
        else:
            preview = (
                f"参加チームから{count}チームを大会開始時に抽選"
                if count
                else "シードなし"
            )
        self.preview_var.set(preview)

        for index, box in enumerate(self.manual_boxes):
            visible = index < count
            if visible:
                box.grid()
            else:
                box.grid_remove()
            box.config(
                state=("readonly" if method == "manual" and visible else "disabled")
            )
        self.fill_button.config(state="normal" if count > 0 else "disabled")

    def fill_manual_from_rating(self) -> None:
        count = self._seed_count()
        names = [name for name, _ in self.rating_store.ranking()][:count]
        for index, var in enumerate(self.manual_vars):
            var.set(names[index] if index < len(names) else UNUSED)
        self.seed_method_var.set("manual")
        self._refresh()

    def get_config(
        self,
        participants: list[str],
    ) -> dict[str, Any]:
        count = self._seed_count()
        if count not in {0, 2, 4, 8}:
            raise ValueError("シード数は0・2・4・8から選んでください")
        if count > len(participants):
            raise ValueError("シード数が参加チーム数を超えています")

        method = self.seed_method_var.get()
        manual = [var.get() for var in self.manual_vars[:count]]
        rating_snapshot = {team: self.rating_store.get(team) for team in participants}

        # 12チームの16枠大会ならBYEは4つ。
        bye_count = _next_power_of_two(len(participants)) - len(participants)
        if count != bye_count and bye_count > 0:
            # 禁止はせず、BYE数との違いを開始ログで分かるよう保存。
            pass

        config = {
            "seed_count": count,
            "seed_method": method,
            "manual_seeds": manual,
            "rating_snapshot": rating_snapshot,
            "bye_count": bye_count,
        }

        # 共通ロジックで事前検証。
        build_seeded_bracket_slots(
            participants,
            count,
            method,
            manual,
            rating_snapshot,
        )
        return config

    def set_enabled(self, enabled: bool) -> None:
        self.count_box.config(state="readonly" if enabled else "disabled")
        for child in self.winfo_children():
            if isinstance(child, tk.Frame):
                for widget in child.winfo_children():
                    try:
                        if isinstance(widget, tk.Radiobutton):
                            widget.config(state="normal" if enabled else "disabled")
                    except tk.TclError:
                        pass
        self.fill_button.config(
            state=("normal" if enabled and self._seed_count() > 0 else "disabled")
        )
        if enabled:
            self._refresh()
        else:
            for box in self.manual_boxes:
                box.config(state="disabled")


# ---------------------------------------------------------------------------
# Team combat-power index (after player combos)
# ---------------------------------------------------------------------------
COMBAT_POWER_IQ_EFFECTIVE_CAP = 200.0


def calculate_combat_power_index(
    hs_rate: float,
    dodge_rate: float,
    iq: float,
    accuracy: float,
    reaction: float,
) -> float:
    """Current combat-power formula with IQ effective cap at 200.

    Formula:
        HS% * 130
        + dodge% * 170
        + min(IQ, 200) / 2
        + (accuracy - 0.2) * 130
        + reaction / 2.2

    IQ itself is not modified. Only its contribution to the index is capped.
    """
    try:
        hs_rate = max(0.0, min(1.0, float(hs_rate)))
    except (TypeError, ValueError):
        hs_rate = 0.0
    try:
        dodge_rate = max(0.0, min(1.0, float(dodge_rate)))
    except (TypeError, ValueError):
        dodge_rate = 0.0
    try:
        accuracy = max(0.0, float(accuracy))
    except (TypeError, ValueError):
        accuracy = 0.0
    try:
        iq = max(0.0, float(iq))
    except (TypeError, ValueError):
        iq = 0.0
    try:
        reaction = max(0.0, float(reaction))
    except (TypeError, ValueError):
        reaction = 0.0

    iq_for_power = min(iq, COMBAT_POWER_IQ_EFFECTIVE_CAP)
    return (
        hs_rate * 130.0
        + dodge_rate * 170.0
        + iq_for_power / 2.0
        + (accuracy - 0.2) * 130.0
        + reaction / 2.2
    )


def _combo_stat_key(stat_key: Any) -> str | None:
    normalized = str(stat_key).strip().lower()
    aliases = {
        'accuracy': 'accuracy', 'aim': 'accuracy', 'hit': 'accuracy',
        'hit%': 'accuracy', 'hit_pct': 'accuracy', 'hit_rate': 'accuracy', '命中率': 'accuracy',
        'hs': 'hs_rate', 'hs%': 'hs_rate', 'hs_pct': 'hs_rate',
        'hs_rate': 'hs_rate', 'headshot_rate': 'hs_rate', 'ヘッドショット率': 'hs_rate',
        'dodge': 'dodge_rate', 'dodge%': 'dodge_rate', 'dodge_pct': 'dodge_rate',
        'dodge_rate': 'dodge_rate', '回避率': 'dodge_rate', '弾除け率': 'dodge_rate',
        'reaction': 'reaction', 'reaction_speed': 'reaction', '反応速度': 'reaction',
        'iq': 'iq', 'intelligence': 'iq', '判断力': 'iq', '知能': 'iq',
    }
    return aliases.get(normalized)


def _apply_combo_bonus_to_stats(stats: dict[str, float], stat_key: Any, value: Any) -> None:
    """Mirror game_core combo stat behavior for the five power-index stats."""
    key = _combo_stat_key(stat_key)
    if key is None:
        return
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return

    if key in {'accuracy', 'hs_rate', 'dodge_rate'}:
        if abs(amount) > 1.0:
            amount /= 100.0
        updated = float(stats[key]) + amount
        if key == 'accuracy':
            stats[key] = max(0.0, updated)  # game_core allows >100% accuracy after combos
        else:
            stats[key] = max(0.0, min(1.0, updated))
    elif key in {'reaction', 'iq'}:
        stats[key] = max(0.0, float(stats[key]) + amount)


def build_team_combo_power_report(team_name: str) -> dict[str, Any]:
    """Calculate a preset's combat power before/after every active player combo.

    This intentionally includes player-combo bonuses only. IGL correction,
    awakenings, mental/condition effects, and in-round temporary effects are not
    included in this static comparison.
    """
    preset = get_preset(team_name)
    validate_preset(preset)
    player_names = [str(name) for name in preset.players]
    team_set = set(player_names)

    base_stats: dict[str, dict[str, float]] = {}
    combo_stats: dict[str, dict[str, float]] = {}
    for name in player_names:
        raw = get_character_combat_stats(name)
        row = {
            'hs_rate': float(raw.get('hs_rate', 0.0)),
            'dodge_rate': float(raw.get('dodge_rate', 0.0)),
            'iq': float(raw.get('iq', 0.0)),
            'accuracy': float(raw.get('accuracy', 0.0)),
            'reaction': float(raw.get('reaction', 0.0)),
        }
        base_stats[name] = dict(row)
        combo_stats[name] = dict(row)

    active_combos: list[str] = []
    for combo in PLAYER_COMBOS:
        if not isinstance(combo, dict):
            continue
        required = tuple(str(x) for x in combo.get('players', ()))
        if not required or not set(required).issubset(team_set):
            continue
        active_combos.append(str(combo.get('name', '名称未設定コンボ')))

        common = combo.get('bonuses', {})
        per_player = combo.get('player_bonuses', {})
        for name in required:
            if name not in combo_stats:
                continue
            if isinstance(common, dict):
                for key, value in common.items():
                    _apply_combo_bonus_to_stats(combo_stats[name], key, value)
            if isinstance(per_player, dict):
                bonuses = per_player.get(name, {})
                if isinstance(bonuses, dict):
                    for key, value in bonuses.items():
                        _apply_combo_bonus_to_stats(combo_stats[name], key, value)

    rows: list[dict[str, Any]] = []
    base_total = 0.0
    combo_total = 0.0
    for name in player_names:
        base = base_stats[name]
        after = combo_stats[name]
        base_power = calculate_combat_power_index(**base)
        combo_power = calculate_combat_power_index(**after)
        base_total += base_power
        combo_total += combo_power
        rows.append({
            'name': name,
            'base': base,
            'after': after,
            'base_power': base_power,
            'combo_power': combo_power,
            'delta': combo_power - base_power,
        })

    return {
        'team': team_name,
        'players': rows,
        'active_combos': active_combos,
        'base_total': base_total,
        'combo_total': combo_total,
        'delta': combo_total - base_total,
        'base_average': base_total / len(rows) if rows else 0.0,
        'combo_average': combo_total / len(rows) if rows else 0.0,
    }


class CompetitionApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("FPS AI Competition Manager")
        self.root.geometry("1120x820")
        self.root.minsize(900, 680)

        self.events: queue.Queue = queue.Queue()
        self.render_requests: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.live_render_enabled = False
        self.names = all_preset_names()
        if len(self.names) < 2:
            raise RuntimeError("party_presets.pyに2チーム以上必要です")

        self.rating_store = TeamRatingStore(self.names)
        self.competition_id = ""
        self.competition_rating_updates: list[dict[str, Any]] = []
        self.rating_window: tk.Toplevel | None = None
        self.rating_team_var: tk.StringVar | None = None
        self.rating_tree: ttk.Treeview | None = None
        self.rating_chart: tk.Canvas | None = None

        self.maps_to_win_var = tk.StringVar(value="2")
        self.lower_final_maps_to_win_var = tk.StringVar(value="3")
        self.grand_final_maps_to_win_var = tk.StringVar(value="3")
        self.seed_mode_var = tk.StringVar(value="random")
        self.seed_var = tk.StringVar(value="42")
        self.render_var = tk.BooleanVar(value=False)
        self.rating_enabled_var = tk.BooleanVar(value=True)
        self.current_rating_enabled = True
        self.status_var = tk.StringVar(value="モードとチームを設定してください")
        self.series_score_var = tk.StringVar(value="-")
        self.power_team_var = tk.StringVar(value=self.names[0])
        self.power_summary_var = tk.StringVar(value="チームを選択して計算してください")
        self.power_combo_var = tk.StringVar(value="発動コンボ: -")
        self.power_tree: ttk.Treeview | None = None

        # 右側の図表示に使う進行状態。
        self.visual_mode = "series"
        self.visual_series_maps: list[MResult] = []
        self.visual_swiss_rounds: list[dict[str, Any]] = []
        self.visual_swiss_records: dict[str, dict[str, int]] = {}
        self.visual_league_table: dict[str, dict[str, int]] = {}
        self.visual_current_pairings: list[tuple[str, str]] = []
        self.visual_bye: str | None = None
        self.visual_bracket_matches: dict[str, dict[str, Any]] = {}

        self._build_common_settings()
        self._build_notebook()
        self._build_results()
        self.root.after(100, self._poll_events)

    def _build_common_settings(self) -> None:
        frame = tk.LabelFrame(
            self.root,
            text="共通設定",
            padx=10,
            pady=8,
        )
        frame.pack(fill="x", padx=12, pady=(12, 6))

        tk.Label(frame, text="通常戦 先取").grid(row=0, column=0)
        self.maps_entry = tk.Entry(
            frame,
            textvariable=self.maps_to_win_var,
            width=5,
        )
        self.maps_entry.grid(row=0, column=1, padx=(5, 10))

        tk.Label(frame, text="Lower Final 先取").grid(row=0, column=2)
        self.lower_final_maps_entry = tk.Entry(
            frame,
            textvariable=self.lower_final_maps_to_win_var,
            width=5,
        )
        self.lower_final_maps_entry.grid(row=0, column=3, padx=(5, 10))

        tk.Label(frame, text="Grand Final 先取").grid(row=0, column=4)
        self.grand_final_maps_entry = tk.Entry(
            frame,
            textvariable=self.grand_final_maps_to_win_var,
            width=5,
        )
        self.grand_final_maps_entry.grid(row=0, column=5, padx=(5, 12))

        seed_box = tk.LabelFrame(frame, text="Seed", padx=8, pady=4)
        seed_box.grid(row=0, column=6, padx=(4, 8))
        self.random_seed_radio = tk.Radiobutton(
            seed_box,
            text="ランダム",
            variable=self.seed_mode_var,
            value="random",
            command=self._update_seed_entry_state,
        )
        self.random_seed_radio.grid(row=0, column=0)
        self.fixed_seed_radio = tk.Radiobutton(
            seed_box,
            text="固定",
            variable=self.seed_mode_var,
            value="fixed",
            command=self._update_seed_entry_state,
        )
        self.fixed_seed_radio.grid(row=0, column=1, padx=(8, 0))
        self.seed_entry = tk.Entry(
            seed_box,
            textvariable=self.seed_var,
            width=10,
        )
        self.seed_entry.grid(row=0, column=2, padx=(8, 0))

        self.render_check = tk.Checkbutton(
            frame,
            text="描画（次マップから反映）",
            variable=self.render_var,
            command=self._on_render_toggle,
        )
        self.render_check.grid(row=0, column=7, padx=(4, 0))
        self.rating_enabled_check = tk.Checkbutton(
            frame,
            text="レートに反映",
            variable=self.rating_enabled_var,
        )
        self.rating_enabled_check.grid(row=0, column=8, padx=(8, 0))
        self.start_button = tk.Button(
            frame,
            text="現在のモードを開始",
            font=("Arial", 11, "bold"),
            command=self.start_current_mode,
        )
        self.start_button.grid(row=0, column=9, padx=(12, 0))

        self.rating_button = tk.Button(
            frame,
            text="レーティング・推移",
            command=self.open_rating_window,
        )
        self.rating_button.grid(row=0, column=10, padx=(8, 0))
        frame.grid_columnconfigure(11, weight=1)
        self._update_seed_entry_state()

    def _on_render_toggle(self) -> None:
        self.live_render_enabled = bool(self.render_var.get())
        state = "ON" if self.live_render_enabled else "OFF"
        if self.worker is not None or self.start_button.cget("state") == "disabled":
            self.status_var.set(
                f"描画を{state}に変更しました。次のマップから反映されます"
            )

    def _update_seed_entry_state(self) -> None:
        if hasattr(self, "seed_entry"):
            self.seed_entry.config(
                state=("normal" if self.seed_mode_var.get() == "fixed" else "disabled")
            )

    def open_rating_window(self) -> None:
        if self.rating_window is not None and self.rating_window.winfo_exists():
            self.rating_window.deiconify()
            self.rating_window.lift()
            self.refresh_rating_window()
            return

        win = tk.Toplevel(self.root)
        self.rating_window = win
        win.title("Team Rating Ranking / History")
        win.geometry("980x680")
        win.minsize(760, 520)

        top = tk.Frame(win)
        top.pack(fill="x", padx=10, pady=8)

        tk.Label(
            top,
            text="チーム",
            font=("Arial", 10, "bold"),
        ).pack(side="left")

        self.rating_team_var = tk.StringVar(value=self.rating_store.ranking()[0][0])
        team_box = ttk.Combobox(
            top,
            values=[name for name, _ in self.rating_store.ranking()],
            textvariable=self.rating_team_var,
            state="readonly",
            width=32,
        )
        team_box.pack(side="left", padx=8)
        team_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.refresh_rating_chart(),
        )

        tk.Label(top, text="設定値").pack(side="left", padx=(18, 4))
        self.rating_edit_var = tk.StringVar()
        tk.Entry(
            top,
            textvariable=self.rating_edit_var,
            width=10,
        ).pack(side="left")
        tk.Button(
            top,
            text="選択チームへ設定",
            command=self.set_selected_team_rating,
        ).pack(side="left", padx=6)
        tk.Button(
            top,
            text="再読込",
            command=self.refresh_rating_window,
        ).pack(side="left", padx=6)

        pane = tk.PanedWindow(
            win,
            orient="horizontal",
            sashwidth=6,
            sashrelief="raised",
        )
        pane.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left = tk.Frame(pane)
        columns = ("rank", "team", "rating")
        self.rating_tree = ttk.Treeview(
            left,
            columns=columns,
            show="headings",
        )
        self.rating_tree.heading("rank", text="#")
        self.rating_tree.heading("team", text="TEAM")
        self.rating_tree.heading("rating", text="RATING")
        self.rating_tree.column("rank", width=45, anchor="center")
        self.rating_tree.column("team", width=220)
        self.rating_tree.column("rating", width=90, anchor="e")
        scroll = tk.Scrollbar(
            left,
            orient="vertical",
            command=self.rating_tree.yview,
        )
        self.rating_tree.configure(yscrollcommand=scroll.set)
        self.rating_tree.pack(
            side="left",
            fill="both",
            expand=True,
        )
        scroll.pack(side="right", fill="y")
        self.rating_tree.bind(
            "<<TreeviewSelect>>",
            self._on_rating_tree_select,
        )
        pane.add(left, minsize=310)

        right = tk.Frame(pane, bg="#111827")
        self.rating_chart = tk.Canvas(
            right,
            bg="#111827",
            highlightthickness=0,
        )
        self.rating_chart.pack(fill="both", expand=True)
        self.rating_chart.bind(
            "<Configure>",
            lambda _event: self.refresh_rating_chart(),
        )
        pane.add(right, minsize=430)

        win.protocol("WM_DELETE_WINDOW", win.withdraw)
        self.refresh_rating_window()

    def _on_rating_tree_select(self, _event=None) -> None:
        if self.rating_tree is None or self.rating_team_var is None:
            return
        selected = self.rating_tree.selection()
        if not selected:
            return
        values = self.rating_tree.item(selected[0], "values")
        if len(values) >= 2:
            self.rating_team_var.set(str(values[1]))
            self.rating_edit_var.set(f"{self.rating_store.get(str(values[1])):.1f}")
            self.refresh_rating_chart()

    def set_selected_team_rating(self) -> None:
        if self.rating_team_var is None:
            return
        team = self.rating_team_var.get()
        try:
            value = float(self.rating_edit_var.get())
        except ValueError:
            messagebox.showerror(
                "入力エラー",
                "レーティングは数値で入力してください",
                parent=self.rating_window,
            )
            return
        self.rating_store.set_rating(
            team,
            value,
            reason="manual",
        )
        self.refresh_rating_window()

    def refresh_rating_window(self) -> None:
        if (
            self.rating_window is None
            or not self.rating_window.winfo_exists()
            or self.rating_tree is None
        ):
            return

        current_team = (
            self.rating_team_var.get() if self.rating_team_var is not None else ""
        )
        for item in self.rating_tree.get_children():
            self.rating_tree.delete(item)

        for rank, (team, rating) in enumerate(
            self.rating_store.ranking(),
            1,
        ):
            self.rating_tree.insert(
                "",
                "end",
                values=(rank, team, f"{rating:.1f}"),
            )

        if self.rating_team_var is not None:
            names = [name for name, _ in self.rating_store.ranking()]
            if current_team not in names and names:
                self.rating_team_var.set(names[0])
            if self.rating_team_var.get():
                self.rating_edit_var.set(
                    f"{self.rating_store.get(self.rating_team_var.get()):.1f}"
                )
        self.refresh_rating_chart()

    def refresh_rating_chart(self) -> None:
        canvas = self.rating_chart
        if canvas is None or self.rating_team_var is None or not canvas.winfo_exists():
            return

        canvas.delete("all")
        width = max(430, canvas.winfo_width())
        height = max(360, canvas.winfo_height())
        team = self.rating_team_var.get()
        points = self.rating_store.team_history(team)

        margin_l = 62
        margin_r = 25
        margin_t = 55
        margin_b = 55
        plot_w = width - margin_l - margin_r
        plot_h = height - margin_t - margin_b

        canvas.create_text(
            20,
            16,
            anchor="nw",
            text=f"{team}  RATING HISTORY",
            fill="#f8fafc",
            font=("Arial", 14, "bold"),
        )
        canvas.create_text(
            20,
            38,
            anchor="nw",
            text=f"CURRENT {self.rating_store.get(team):.1f}",
            fill="#fbbf24",
            font=("Arial", 11, "bold"),
        )

        ratings = [float(point["rating"]) for point in points]
        low = min(ratings + [self.rating_store.default_rating])
        high = max(ratings + [self.rating_store.default_rating])
        padding = max(40.0, (high - low) * 0.18)
        low -= padding
        high += padding
        if high <= low:
            high = low + 1.0

        def x_at(index: int) -> float:
            if len(points) <= 1:
                return margin_l + plot_w / 2
            return margin_l + plot_w * index / (len(points) - 1)

        def y_at(rating: float) -> float:
            return margin_t + plot_h - (rating - low) / (high - low) * plot_h

        # Grid and labels
        for tick in range(6):
            rating = low + (high - low) * tick / 5
            y = y_at(rating)
            canvas.create_line(
                margin_l,
                y,
                width - margin_r,
                y,
                fill="#334155",
            )
            canvas.create_text(
                margin_l - 8,
                y,
                anchor="e",
                text=f"{rating:.0f}",
                fill="#94a3b8",
                font=("Arial", 8),
            )

        canvas.create_line(
            margin_l,
            margin_t,
            margin_l,
            margin_t + plot_h,
            fill="#64748b",
            width=2,
        )
        canvas.create_line(
            margin_l,
            margin_t + plot_h,
            width - margin_r,
            margin_t + plot_h,
            fill="#64748b",
            width=2,
        )

        coords = []
        for index, point in enumerate(points):
            coords.extend([x_at(index), y_at(float(point["rating"]))])
        if len(coords) >= 4:
            canvas.create_line(
                *coords,
                fill="#60a5fa",
                width=3,
                smooth=False,
            )

        for index, point in enumerate(points):
            x = x_at(index)
            y = y_at(float(point["rating"]))
            canvas.create_oval(
                x - 4,
                y - 4,
                x + 4,
                y + 4,
                fill="#f8fafc",
                outline="#60a5fa",
                width=2,
            )
            if index == 0 or index == len(points) - 1 or len(points) <= 12:
                canvas.create_text(
                    x,
                    margin_t + plot_h + 13,
                    anchor="n",
                    text=str(point.get("event", index)),
                    fill="#94a3b8",
                    font=("Arial", 8),
                )

        if points:
            last = points[-1]
            description = str(last.get("label", ""))
            opponent = last.get("opponent")
            score = last.get("score")
            if opponent:
                description += f" / vs {opponent}"
            if score:
                description += f" / {score}"
            canvas.create_text(
                margin_l,
                height - 24,
                anchor="w",
                text=description,
                fill="#cbd5e1",
                font=("Arial", 9),
            )

    def format_rating_ranking(self) -> str:
        lines = [
            "",
            "=" * 62,
            "TEAM RATING RANKING",
            "=" * 62,
        ]
        for rank, (team, rating) in enumerate(
            self.rating_store.ranking(),
            1,
        ):
            lines.append(f"{rank:2d}. {team:<36} {rating:7.1f}")
        lines.append("=" * 62)
        return "\n".join(lines) + "\n"

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=False, padx=12, pady=6)

        self.series_tab = tk.Frame(self.notebook)
        self.swiss_tab = tk.Frame(self.notebook)
        self.league_tab = tk.Frame(self.notebook)
        self.power_tab = tk.Frame(self.notebook)
        self.notebook.add(self.series_tab, text="単独シリーズ")
        self.notebook.add(self.swiss_tab, text="ダブルエリミネーション")
        self.notebook.add(self.league_tab, text="総当たりリーグ")
        self.notebook.add(self.power_tab, text="戦闘力指数")

        self.team1_var = tk.StringVar(value=self.names[0])
        self.team2_var = tk.StringVar(value=self.names[1])
        series_box = tk.LabelFrame(self.series_tab, text="対戦カード", padx=12, pady=12)
        series_box.pack(fill="x", padx=8, pady=8)
        tk.Label(series_box, text="Team 1").grid(row=0, column=0, sticky="w")
        self.team1_box = ttk.Combobox(
            series_box,
            values=self.names,
            textvariable=self.team1_var,
            state="readonly",
            width=34,
        )
        self.team1_box.grid(row=0, column=1, padx=8)
        tk.Label(series_box, text="Team 2").grid(
            row=0, column=2, sticky="w", padx=(20, 0)
        )
        self.team2_box = ttk.Combobox(
            series_box,
            values=self.names,
            textvariable=self.team2_var,
            state="readonly",
            width=34,
        )
        self.team2_box.grid(row=0, column=3, padx=8)

        self.team1_controller_var = tk.StringVar(value=DEFAULT_CONTROLLER_DISPLAY)
        self.team2_controller_var = tk.StringVar(value=DEFAULT_CONTROLLER_DISPLAY)
        tk.Label(
            series_box,
            text="Team 1 Controller",
        ).grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.team1_controller_box = ttk.Combobox(
            series_box,
            values=list(CONTROLLER_OPTIONS.keys()),
            textvariable=self.team1_controller_var,
            state="readonly",
            width=20,
        )
        self.team1_controller_box.grid(
            row=1,
            column=1,
            padx=8,
            pady=(10, 0),
            sticky="w",
        )
        tk.Label(
            series_box,
            text="Team 2 Controller",
        ).grid(
            row=1,
            column=2,
            sticky="w",
            padx=(20, 0),
            pady=(10, 0),
        )
        self.team2_controller_box = ttk.Combobox(
            series_box,
            values=list(CONTROLLER_OPTIONS.keys()),
            textvariable=self.team2_controller_var,
            state="readonly",
            width=20,
        )
        self.team2_controller_box.grid(
            row=1,
            column=3,
            padx=8,
            pady=(10, 0),
            sticky="w",
        )

        swiss_note = tk.Label(
            self.swiss_tab,
            text="Slot順で初戦を固定し、Winners敗者はLosersへ移動します。Lower FinalとGrand Finalは上部で個別設定できます。",
            anchor="w",
            fg="#444",
        )
        swiss_note.pack(fill="x", padx=10, pady=(8, 0))

        self.tournament_seed_editor = TournamentSeedEditor(
            self.swiss_tab,
            self.names,
            self.rating_store,
        )
        self.tournament_seed_editor.pack(
            fill="x",
            padx=8,
            pady=(8, 2),
        )

        self.swiss_slots = TeamSlotEditor(
            self.swiss_tab,
            self.names,
            "参加チーム（Slot順はノンシード配置順として使用）",
            12,
        )
        self.swiss_slots.pack(
            fill="both",
            expand=True,
            padx=8,
            pady=8,
        )

        league_note = tk.Label(
            self.league_tab,
            text="設定した全チームが1回ずつ対戦します。順位はシリーズ勝数 → マップ差 → ラウンド差です。",
            anchor="w",
            fg="#444",
        )
        league_note.pack(fill="x", padx=10, pady=(8, 0))
        self.league_slots = TeamSlotEditor(
            self.league_tab, self.names, "参加チーム（各Slotへチームを設定）", 8
        )
        self.league_slots.pack(fill="both", expand=True, padx=8, pady=8)

        self._build_power_index_tab()

    def _build_power_index_tab(self) -> None:
        top = tk.LabelFrame(self.power_tab, text="コンボ後チーム戦闘力指数", padx=10, pady=8)
        top.pack(fill="x", padx=8, pady=(8, 4))

        tk.Label(top, text="チーム").grid(row=0, column=0, sticky="w")
        self.power_team_box = ttk.Combobox(
            top, values=self.names, textvariable=self.power_team_var,
            state="readonly", width=34,
        )
        self.power_team_box.grid(row=0, column=1, padx=(8, 12), sticky="w")
        tk.Button(top, text="戦闘力指数を計算", command=self._refresh_power_index).grid(
            row=0, column=2, padx=4
        )

        tk.Label(
            top,
            text="IQは実値を保持しつつ、戦闘力指数への寄与だけ200を上限として計算。",
            fg="#555",
            anchor="w",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(7, 0))

        tk.Label(
            self.power_tab, textvariable=self.power_summary_var,
            font=("Arial", 11, "bold"), anchor="w", fg="#1f5f7a"
        ).pack(fill="x", padx=12, pady=(4, 2))
        tk.Label(
            self.power_tab, textvariable=self.power_combo_var,
            anchor="w", justify="left", wraplength=1040, fg="#444"
        ).pack(fill="x", padx=12, pady=(0, 5))

        columns = (
            "player", "base_power", "combo_power", "delta",
            "hs", "dodge", "iq", "iq_used", "accuracy", "reaction"
        )
        self.power_tree = ttk.Treeview(
            self.power_tab, columns=columns, show="headings", height=7
        )
        headings = {
            "player": "Player", "base_power": "素指数", "combo_power": "コンボ後",
            "delta": "増減", "hs": "HS%", "dodge": "回避%", "iq": "IQ",
            "iq_used": "指数IQ", "accuracy": "命中%", "reaction": "反応",
        }
        widths = {
            "player": 130, "base_power": 78, "combo_power": 78, "delta": 70,
            "hs": 62, "dodge": 62, "iq": 58, "iq_used": 64,
            "accuracy": 62, "reaction": 62,
        }
        for key in columns:
            self.power_tree.heading(key, text=headings[key])
            self.power_tree.column(key, width=widths[key], anchor="center")
        self.power_tree.pack(fill="x", padx=10, pady=(2, 8))

        self.power_team_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh_power_index())
        self._refresh_power_index()

    def _refresh_power_index(self) -> None:
        if self.power_tree is None:
            return
        team_name = self.power_team_var.get()
        try:
            report = build_team_combo_power_report(team_name)
        except Exception as exc:
            self.power_summary_var.set(f"計算エラー: {exc}")
            self.power_combo_var.set("発動コンボ: -")
            return

        for item in self.power_tree.get_children():
            self.power_tree.delete(item)

        for row in report["players"]:
            after = row["after"]
            iq = float(after["iq"])
            self.power_tree.insert(
                "", "end", values=(
                    row["name"],
                    f'{row["base_power"]:.1f}',
                    f'{row["combo_power"]:.1f}',
                    f'{row["delta"]:+.1f}',
                    f'{after["hs_rate"] * 100:.1f}',
                    f'{after["dodge_rate"] * 100:.1f}',
                    f'{iq:.0f}',
                    f'{min(iq, COMBAT_POWER_IQ_EFFECTIVE_CAP):.0f}',
                    f'{after["accuracy"] * 100:.1f}',
                    f'{after["reaction"]:.0f}',
                )
            )

        self.power_summary_var.set(
            f'{team_name}  |  チーム合計: {report["base_total"]:.1f} → '
            f'{report["combo_total"]:.1f} ({report["delta"]:+.1f})  |  '
            f'5人平均: {report["base_average"]:.1f} → {report["combo_average"]:.1f}'
        )
        combos = report["active_combos"]
        self.power_combo_var.set(
            "発動コンボ: " + (" / ".join(combos) if combos else "なし")
            + "  ※ IGL補正・覚醒・メンタル/コンディション等は含めません"
        )

    def _build_results(self) -> None:
        status_bar = tk.Frame(self.root)
        status_bar.pack(fill="x", padx=12, pady=(2, 4))
        tk.Label(
            status_bar,
            textvariable=self.status_var,
            anchor="w",
            fg="#22577a",
            font=("Arial", 10, "bold"),
        ).pack(side="left", fill="x", expand=True)
        tk.Label(
            status_bar,
            textvariable=self.series_score_var,
            font=("Arial", 12, "bold"),
        ).pack(side="right")

        frame = tk.LabelFrame(
            self.root,
            text="進行・結果（左：テキスト / 右：図）",
            padx=8,
            pady=8,
        )
        frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        pane = tk.PanedWindow(
            frame,
            orient="horizontal",
            sashrelief="raised",
            sashwidth=6,
        )
        pane.pack(fill="both", expand=True)

        text_frame = tk.Frame(pane)
        self.text = tk.Text(
            text_frame,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
        )
        text_scroll = tk.Scrollbar(
            text_frame,
            command=self.text.yview,
        )
        self.text.configure(yscrollcommand=text_scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        text_scroll.pack(side="right", fill="y")
        pane.add(text_frame, minsize=430, stretch="always")

        visual_frame = tk.Frame(pane, bg="#111827")
        self.visual_canvas = tk.Canvas(
            visual_frame,
            bg="#111827",
            highlightthickness=0,
        )
        visual_scroll_y = tk.Scrollbar(
            visual_frame,
            orient="vertical",
            command=self.visual_canvas.yview,
        )
        visual_scroll_x = tk.Scrollbar(
            visual_frame,
            orient="horizontal",
            command=self.visual_canvas.xview,
        )
        self.visual_canvas.configure(
            yscrollcommand=visual_scroll_y.set,
            xscrollcommand=visual_scroll_x.set,
        )
        self.visual_canvas.grid(row=0, column=0, sticky="nsew")
        visual_scroll_y.grid(row=0, column=1, sticky="ns")
        visual_scroll_x.grid(row=1, column=0, sticky="ew")
        visual_frame.grid_rowconfigure(0, weight=1)
        visual_frame.grid_columnconfigure(0, weight=1)
        pane.add(visual_frame, minsize=430, stretch="always")

        self.visual_canvas.bind(
            "<Configure>",
            lambda _event: self.redraw_visual(),
        )
        self.redraw_visual()

    def _canvas_text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        anchor: str = "nw",
        font: tuple[Any, ...] = ("Arial", 10),
        fill: str = "#e5e7eb",
        width: int | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "anchor": anchor,
            "font": font,
            "fill": fill,
            "text": text,
        }
        if width is not None:
            kwargs["width"] = width
        self.visual_canvas.create_text(x, y, **kwargs)

    def _card(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        subtitle: str = "",
        *,
        outline: str = "#4b5563",
        fill: str = "#1f2937",
    ) -> None:
        self.visual_canvas.create_rectangle(
            x,
            y,
            x + width,
            y + height,
            fill=fill,
            outline=outline,
            width=2,
        )
        self._canvas_text(
            x + 10,
            y + 8,
            title,
            font=("Arial", 10, "bold"),
            width=int(width - 20),
        )
        if subtitle:
            self._canvas_text(
                x + 10,
                y + 31,
                subtitle,
                font=("Arial", 9),
                fill="#cbd5e1",
                width=int(width - 20),
            )

    def _show_bracket_match_details(self, match_id: str) -> None:
        """クリックされたブラケット試合のマップ詳細を別窓で表示する。"""
        match = self.visual_bracket_matches.get(match_id)
        if not match:
            return

        window = tk.Toplevel(self.root)
        window.title(f"Match Details - {match_id}")
        window.geometry("640x480")
        window.minsize(520, 360)

        team1 = match.get("team1") or "BYE / TBD"
        team2 = match.get("team2") or "BYE / TBD"
        winner = match.get("winner") or "未確定"

        header = tk.Label(
            window,
            text=(
                f"{match_id}\n"
                f"{team1} {match.get('team1_wins', 0)} - "
                f"{match.get('team2_wins', 0)} {team2}\n"
                f"WINNER: {winner}"
            ),
            font=("Arial", 12, "bold"),
            justify="center",
            pady=10,
        )
        header.pack(fill="x")

        body_frame = tk.Frame(window)
        body_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        body = tk.Text(
            body_frame,
            wrap="word",
            font=("Consolas", 10),
            state="normal",
        )
        scroll = tk.Scrollbar(body_frame, command=body.yview)
        body.configure(yscrollcommand=scroll.set)
        body.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        maps = match.get("maps", []) or []
        if not maps:
            body.insert("end", "まだマップ結果はありません。\n")
        else:
            for map_data in maps:
                overtime = " [OT]" if map_data.get("overtime") else ""
                mvp1 = map_data.get("mvp1", {})
                mvp2 = map_data.get("mvp2", {})
                body.insert(
                    "end",
                    (
                        f"MAP {map_data.get('number', '?')}{overtime}\n"
                        f"Seed: {map_data.get('seed', '?')}\n"
                        f"{map_data.get('team1')} "
                        f"{map_data.get('score1')} - "
                        f"{map_data.get('score2')} "
                        f"{map_data.get('team2')}\n"
                        f"Winner: {map_data.get('winner')}\n"
                        f"{map_data.get('team1')} MVP: "
                        f"{mvp1.get('name', '?')} "
                        f"{mvp1.get('kills', 0)}K/"
                        f"{mvp1.get('deaths', 0)}D\n"
                        f"{map_data.get('team2')} MVP: "
                        f"{mvp2.get('name', '?')} "
                        f"{mvp2.get('kills', 0)}K/"
                        f"{mvp2.get('deaths', 0)}D\n" + "-" * 70 + "\n"
                    ),
                )
        body.config(state="disabled")

    def redraw_visual(self) -> None:
        if not hasattr(self, "visual_canvas"):
            return
        self.visual_canvas.delete("all")

        if self.visual_mode == "swiss":
            self._draw_double_elimination_visual()
        elif self.visual_mode == "league":
            self._draw_league_visual()
        else:
            self._draw_series_visual()

        bbox = self.visual_canvas.bbox("all")
        if bbox:
            self.visual_canvas.configure(
                scrollregion=(
                    bbox[0] - 20,
                    bbox[1] - 20,
                    bbox[2] + 20,
                    bbox[3] + 20,
                )
            )

    def _draw_series_visual(self) -> None:
        self._canvas_text(
            20,
            18,
            "SERIES MAP FLOW",
            font=("Arial", 15, "bold"),
            fill="#f8fafc",
        )
        if not self.visual_series_maps:
            self._canvas_text(
                20,
                58,
                "シリーズを開始すると、各マップの勝敗がここに並びます。",
                fill="#94a3b8",
                width=380,
            )
            return

        y = 58
        for result in self.visual_series_maps:
            winner_is_1 = result.winner == result.team1
            team1_fill = "#14532d" if winner_is_1 else "#374151"
            team2_fill = "#14532d" if not winner_is_1 else "#374151"

            self._canvas_text(
                20,
                y + 18,
                (
                    f"MAP {result.number}"
                    + (" OT" if result.overtime else "")
                    + f"\nSeed {result.seed}"
                ),
                font=("Arial", 10, "bold"),
                fill="#fbbf24" if result.overtime else "#e5e7eb",
            )
            self._card(
                105,
                y,
                250,
                52,
                result.team1,
                f"{result.score1}  |  MVP {result.mvp1.name} "
                f"{result.mvp1.kills}K/{result.mvp1.deaths}D",
                fill=team1_fill,
                outline="#22c55e" if winner_is_1 else "#64748b",
            )
            self._canvas_text(
                370,
                y + 16,
                "VS",
                font=("Arial", 11, "bold"),
                fill="#94a3b8",
            )
            self._card(
                405,
                y,
                250,
                52,
                result.team2,
                f"{result.score2}  |  MVP {result.mvp2.name} "
                f"{result.mvp2.kills}K/{result.mvp2.deaths}D",
                fill=team2_fill,
                outline="#22c55e" if not winner_is_1 else "#64748b",
            )
            y += 66

    def _draw_double_elimination_visual(self) -> None:
        """VCT風のダブルエリミネーション表を描画する。

        1試合を1枚の大きな箱として扱いつつ、箱の内部はチームごとに
        独立した2行へ分割する。勝者行は緑、敗者行は灰色で表示し、
        前試合の該当チーム行から次試合の該当チーム行へ線を接続する。
        """
        self._canvas_text(
            20,
            14,
            "DOUBLE ELIMINATION BRACKET",
            font=("Arial", 15, "bold"),
            fill="#f8fafc",
        )

        matches = list(self.visual_bracket_matches.values())
        if not matches:
            self._canvas_text(
                20,
                52,
                "開始すると、VCT形式のWinners / Losersブラケットを表示します。",
                fill="#94a3b8",
                width=520,
            )
            return

        winners = sorted(
            [m for m in matches if m.get("bracket") == "W"],
            key=lambda m: (int(m.get("round", 0)), int(m.get("match", 0))),
        )
        losers = sorted(
            [m for m in matches if m.get("bracket") == "L"],
            key=lambda m: (int(m.get("round", 0)), int(m.get("match", 0))),
        )
        grands = sorted(
            [m for m in matches if m.get("bracket") == "G"],
            key=lambda m: (int(m.get("round", 0)), int(m.get("match", 0))),
        )

        card_w = 250
        row_h = 34
        header_h = 22
        card_h = header_h + row_h * 2
        col_gap = 105
        row_gap = 34

        # match_id -> 描画情報。チーム行の中央座標も保存する。
        drawn: dict[str, dict[str, Any]] = {}

        def stage_label(match: dict[str, Any]) -> str:
            special = str(match.get("special", ""))
            if special == "lower_final":
                return "LOWER FINAL"
            if special == "grand_final":
                return "GRAND FINAL"
            side = "UPPER" if match.get("bracket") == "W" else "LOWER"
            return f"{side} ROUND {int(match.get('round', 0))}"

        def row_style(
            match: dict[str, Any],
            team: str | None,
        ) -> tuple[str, str, str]:
            status = str(match.get("status", ""))
            winner = match.get("winner")
            loser = match.get("loser")

            if team is None:
                return "#202938", "#64748b", "#94a3b8"
            if status == "playing":
                return "#172554", "#3b82f6", "#f8fafc"
            if status == "bye":
                return "#1f2937", "#64748b", "#cbd5e1"
            if team == winner:
                return "#14532d", "#22c55e", "#f0fdf4"
            if team == loser:
                return "#343a46", "#6b7280", "#9ca3af"
            return "#1f2937", "#64748b", "#e5e7eb"

        def draw_match(
            match: dict[str, Any],
            x: float,
            y: float,
        ) -> None:
            match_id = str(match.get("id", "?"))
            tag = f"bracket_match_{match_id}"

            # 見出し
            self.visual_canvas.create_rectangle(
                x,
                y,
                x + card_w,
                y + header_h,
                fill="#111827",
                outline="#64748b",
                width=1,
                tags=(tag,),
            )
            self.visual_canvas.create_text(
                x + 7,
                y + header_h / 2,
                anchor="w",
                text=stage_label(match),
                font=("Arial", 8, "bold"),
                fill="#cbd5e1",
                tags=(tag,),
            )

            team_rows: dict[str, tuple[float, float]] = {}
            teams = [match.get("team1"), match.get("team2")]
            scores = [
                int(match.get("team1_wins", 0)),
                int(match.get("team2_wins", 0)),
            ]

            for index, (team, score) in enumerate(zip(teams, scores)):
                row_y = y + header_h + index * row_h
                fill, outline, text_fill = row_style(match, team)
                self.visual_canvas.create_rectangle(
                    x,
                    row_y,
                    x + card_w,
                    row_y + row_h,
                    fill=fill,
                    outline=outline,
                    width=2 if team == match.get("winner") else 1,
                    tags=(tag,),
                )
                display_team = str(team) if team else "BYE / TBD"
                self.visual_canvas.create_text(
                    x + 9,
                    row_y + row_h / 2,
                    anchor="w",
                    text=display_team,
                    font=(
                        "Arial",
                        9,
                        "bold" if team == match.get("winner") else "normal",
                    ),
                    fill=text_fill,
                    width=card_w - 54,
                    tags=(tag,),
                )
                self.visual_canvas.create_text(
                    x + card_w - 13,
                    row_y + row_h / 2,
                    anchor="e",
                    text=str(score),
                    font=("Arial", 11, "bold"),
                    fill=text_fill,
                    tags=(tag,),
                )
                if team:
                    team_rows[str(team)] = (
                        x,
                        row_y + row_h / 2,
                    )

            self.visual_canvas.tag_bind(
                tag,
                "<Button-1>",
                lambda _event, match_id=match_id: (
                    self._show_bracket_match_details(match_id)
                ),
            )
            self.visual_canvas.tag_bind(
                tag,
                "<Enter>",
                lambda _event: self.visual_canvas.config(cursor="hand2"),
            )
            self.visual_canvas.tag_bind(
                tag,
                "<Leave>",
                lambda _event: self.visual_canvas.config(cursor=""),
            )

            drawn[match_id] = {
                "x": x,
                "y": y,
                "right": x + card_w,
                "center_y": y + card_h / 2,
                "team_rows": team_rows,
                "match": match,
            }

        def layout_group(
            title: str,
            group: list[dict[str, Any]],
            top: float,
            accent: str,
        ) -> float:
            self._canvas_text(
                20,
                top,
                title,
                font=("Arial", 12, "bold"),
                fill=accent,
            )
            if not group:
                return top + 48

            rounds = sorted({int(m.get("round", 0)) for m in group})
            by_round = {
                round_no: sorted(
                    [m for m in group if int(m.get("round", 0)) == round_no],
                    key=lambda m: int(m.get("match", 0)),
                )
                for round_no in rounds
            }
            max_matches = max(len(items) for items in by_round.values())
            section_h = max_matches * (card_h + row_gap)

            for col, round_no in enumerate(rounds):
                x = 20 + col * (card_w + col_gap)
                round_matches = by_round[round_no]
                spacing = section_h / max(1, len(round_matches))
                for row, match in enumerate(round_matches):
                    y = top + 35 + row * spacing
                    draw_match(match, x, y)

            return top + 45 + section_h

        bottom = layout_group(
            "UPPER BRACKET",
            winners,
            48,
            "#4ade80",
        )
        bottom = layout_group(
            "LOWER BRACKET",
            losers,
            bottom + 24,
            "#fbbf24",
        )

        # Grand FinalはUpper/Lowerの最終試合より右へ置く。
        if grands:
            grand = grands[-1]
            max_round = max([int(m.get("round", 0)) for m in winners] + [0])
            grand_x = 20 + max_round * (card_w + col_gap)
            grand_y = max(70.0, bottom - card_h - 25)
            draw_match(grand, grand_x, grand_y)

        def source_point(
            source_id: str,
            destination_team: str | None,
        ) -> tuple[float, float] | None:
            source = drawn.get(source_id)
            if source is None:
                return None
            rows = source["team_rows"]
            if destination_team and destination_team in rows:
                _, row_y = rows[destination_team]
                return source["right"], row_y

            source_match = source["match"]
            # BYEや未確定時のフォールバック。
            for candidate in (
                source_match.get("winner"),
                source_match.get("loser"),
                source_match.get("team1"),
                source_match.get("team2"),
            ):
                if candidate and str(candidate) in rows:
                    _, row_y = rows[str(candidate)]
                    return source["right"], row_y
            return source["right"], source["center_y"]

        def target_point(
            match_id: str,
            team: str | None,
        ) -> tuple[float, float] | None:
            target = drawn.get(match_id)
            if target is None:
                return None
            rows = target["team_rows"]
            if team and str(team) in rows:
                row_x, row_y = rows[str(team)]
                return row_x, row_y
            return target["x"], target["center_y"]

        # 試合ブロックを描き終えたあと、該当するチーム行同士を線で結ぶ。
        for match_id, target in drawn.items():
            match = target["match"]
            for source_key, team_key in (
                ("source1", "team1"),
                ("source2", "team2"),
            ):
                source_id = match.get(source_key)
                destination_team = match.get(team_key)
                if not source_id or str(source_id).startswith("SLOT-"):
                    continue
                start = source_point(str(source_id), destination_team)
                end = target_point(match_id, destination_team)
                if start is None or end is None:
                    continue
                sx, sy = start
                ex, ey = end
                middle_x = sx + max(28, (ex - sx) * 0.52)
                line_fill = (
                    "#22c55e" if destination_team == match.get("winner") else "#6b7280"
                )
                self.visual_canvas.create_line(
                    sx,
                    sy,
                    middle_x,
                    sy,
                    middle_x,
                    ey,
                    ex,
                    ey,
                    fill=line_fill,
                    width=2,
                    smooth=False,
                    arrow="last",
                )

        self._canvas_text(
            20,
            bottom + 18,
            "緑＝勝者 / 灰＝敗者　試合ブロックをクリックするとマップ詳細を表示",
            font=("Arial", 9),
            fill="#94a3b8",
        )

    def _draw_swiss_visual(self) -> None:
        self._canvas_text(
            20,
            18,
            "TWO-LOSS SWISS",
            font=("Arial", 15, "bold"),
            fill="#f8fafc",
        )
        records = self.visual_swiss_records
        if not records:
            self._canvas_text(
                20,
                58,
                "0敗＝WINNERS / 1敗＝LOSERS / 2敗＝ELIMINATED",
                fill="#94a3b8",
            )
            return

        groups = [
            ("WINNERS (0敗)", 0, "#166534", "#22c55e"),
            ("LOSERS (1敗)", 1, "#78350f", "#f59e0b"),
            ("ELIMINATED (2敗)", 2, "#7f1d1d", "#ef4444"),
        ]
        x_positions = [20, 255, 490]
        for (title, losses, fill, outline), x in zip(groups, x_positions):
            self._canvas_text(
                x,
                55,
                title,
                font=("Arial", 11, "bold"),
                fill=outline,
            )
            y = 85
            names = [
                name
                for name, rec in records.items()
                if min(2, rec.get("losses", 0)) == losses
            ]
            names.sort(
                key=lambda name: (
                    -records[name].get("wins", 0),
                    name.lower(),
                )
            )
            for name in names:
                rec = records[name]
                self._card(
                    x,
                    y,
                    215,
                    50,
                    name,
                    f"{rec.get('wins', 0)}W-"
                    f"{rec.get('losses', 0)}L  "
                    f"Maps {rec.get('map_wins', 0)}-"
                    f"{rec.get('map_losses', 0)}",
                    fill=fill,
                    outline=outline,
                )
                y += 60

        if self.visual_current_pairings:
            y = max(
                250,
                105
                + max(
                    sum(
                        1
                        for rec in records.values()
                        if min(2, rec.get("losses", 0)) == loss
                    )
                    for loss in (0, 1, 2)
                )
                * 60,
            )
            self._canvas_text(
                20,
                y,
                "CURRENT ROUND PAIRINGS",
                font=("Arial", 11, "bold"),
                fill="#60a5fa",
            )
            y += 30
            for left, right in self.visual_current_pairings:
                self._card(
                    20,
                    y,
                    300,
                    43,
                    left,
                    "vs",
                    outline="#3b82f6",
                )
                self._card(
                    345,
                    y,
                    300,
                    43,
                    right,
                    "",
                    outline="#3b82f6",
                )
                y += 52
            if self.visual_bye:
                self._card(
                    20,
                    y,
                    300,
                    43,
                    f"BYE: {self.visual_bye}",
                    "",
                    outline="#a78bfa",
                    fill="#4c1d95",
                )

    def _draw_league_visual(self) -> None:
        self._canvas_text(
            20,
            18,
            "ROUND ROBIN TABLE",
            font=("Arial", 15, "bold"),
            fill="#f8fafc",
        )
        table = self.visual_league_table
        if not table:
            self._canvas_text(
                20,
                58,
                "試合終了ごとに順位表が更新されます。",
                fill="#94a3b8",
            )
            return

        ordered = sorted(
            table,
            key=lambda name: (
                -table[name].get("series_wins", 0),
                -(table[name].get("map_wins", 0) - table[name].get("map_losses", 0)),
                -(
                    table[name].get("round_wins", 0)
                    - table[name].get("round_losses", 0)
                ),
                name.lower(),
            ),
        )

        headers = ["#", "TEAM", "W-L", "MAP ±", "ROUND ±"]
        xs = [25, 60, 360, 455, 555]
        for x, header in zip(xs, headers):
            self._canvas_text(
                x,
                58,
                header,
                font=("Arial", 10, "bold"),
                fill="#93c5fd",
            )

        y = 86
        for rank, name in enumerate(ordered, 1):
            rec = table[name]
            map_diff = rec.get("map_wins", 0) - rec.get("map_losses", 0)
            round_diff = rec.get("round_wins", 0) - rec.get("round_losses", 0)
            row_fill = "#1e3a5f" if rank == 1 else "#1f2937"
            self.visual_canvas.create_rectangle(
                18,
                y - 7,
                680,
                y + 29,
                fill=row_fill,
                outline="#334155",
            )
            values = [
                str(rank),
                name,
                f"{rec.get('series_wins', 0)}-" f"{rec.get('series_losses', 0)}",
                f"{map_diff:+d}",
                f"{round_diff:+d}",
            ]
            for x, value in zip(xs, values):
                self._canvas_text(
                    x,
                    y,
                    value,
                    font=("Arial", 10, "bold" if rank == 1 else "normal"),
                    fill="#f8fafc",
                )
            y += 42

    def emit(self, event: tuple[Any, ...]) -> None:
        self.events.put(event)

    def append(self, value: str) -> None:
        self.text.config(state="normal")
        self.text.insert("end", value)
        self.text.see("end")
        self.text.config(state="disabled")
        self.root.update_idletasks()

    def clear_output(self) -> None:
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.config(state="disabled")

    def read_common(
        self,
    ) -> tuple[int, int, int, str, int | None, bool]:
        normal_need = validate_maps_to_win(self.maps_to_win_var.get())
        lower_final_need = validate_maps_to_win(self.lower_final_maps_to_win_var.get())
        grand_final_need = validate_maps_to_win(self.grand_final_maps_to_win_var.get())
        seed_mode = validate_seed_mode(self.seed_mode_var.get())
        base_seed = (
            validate_base_seed(self.seed_var.get()) if seed_mode == "fixed" else None
        )
        return (
            normal_need,
            lower_final_need,
            grand_final_need,
            seed_mode,
            base_seed,
            bool(self.render_var.get()),
        )

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.maps_entry.config(state=state)
        self.lower_final_maps_entry.config(state=state)
        self.grand_final_maps_entry.config(state=state)
        self.random_seed_radio.config(state=state)
        self.fixed_seed_radio.config(state=state)
        if enabled:
            self._update_seed_entry_state()
        else:
            self.seed_entry.config(state="disabled")
        # 描画ON/OFFは大会途中でも変更可能。
        self.render_check.config(state="normal")
        self.rating_enabled_check.config(state=state)
        self.start_button.config(state=state)
        self.rating_button.config(state="normal")
        self.team1_box.config(state="readonly" if enabled else "disabled")
        self.team2_box.config(state="readonly" if enabled else "disabled")
        self.team1_controller_box.config(state="readonly" if enabled else "disabled")
        self.team2_controller_box.config(state="readonly" if enabled else "disabled")
        self.swiss_slots.set_enabled(enabled)
        self.tournament_seed_editor.set_enabled(enabled)
        self.league_slots.set_enabled(enabled)

    def start_current_mode(self) -> None:
        try:
            team_controllers: dict[str, str] = {}
            tournament_seed_config: dict[str, Any] = {
                "seed_count": 0,
                "seed_method": "rating",
                "manual_seeds": [],
                "rating_snapshot": {},
            }
            need, lower_final_need, grand_final_need, seed_mode, base_seed, render = (
                self.read_common()
            )
            tab = self.notebook.index(self.notebook.select())
            if tab == 0:
                payload = (
                    "series",
                    [self.team1_var.get(), self.team2_var.get()],
                )
                if payload[1][0] == payload[1][1]:
                    raise ValueError("異なる2チームを選んでください")
                team_controllers = {
                    payload[1][0]: CONTROLLER_OPTIONS.get(
                        self.team1_controller_var.get(),
                        "fnatic_v1",
                    ),
                    payload[1][1]: CONTROLLER_OPTIONS.get(
                        self.team2_controller_var.get(),
                        "fnatic_v1",
                    ),
                }
            elif tab == 1:
                teams = self.swiss_slots.selected_teams()
                if len(teams) < 4:
                    raise ValueError(
                        "ダブルエリミネーションへ4チーム以上設定してください"
                    )
                if len(teams) != len(set(teams)):
                    raise ValueError("同じチームを複数のSlotへ設定できません")
                tournament_seed_config = self.tournament_seed_editor.get_config(teams)
                team_controllers = self.swiss_slots.selected_team_controllers()
                payload = ("swiss", teams)
            else:
                teams = self.league_slots.selected_teams()
                if len(teams) < 2:
                    raise ValueError("総当たりリーグへ2チーム以上設定してください")
                if len(teams) != len(set(teams)):
                    raise ValueError("同じチームを複数のSlotへ設定できません")
                team_controllers = self.league_slots.selected_team_controllers()
                payload = ("league", teams)

            user_teams = [
                team for team, key in team_controllers.items() if key == "user"
            ]
            if len(user_teams) > 1:
                raise ValueError("ユーザー操作は1大会につき1チームまでです")
        except Exception as exc:
            messagebox.showerror("設定エラー", str(exc))
            return

        self.clear_output()
        self.series_score_var.set("-")
        self.append(
            "描画設定は各マップ開始直前に読み込みます。"
            "大会途中のON/OFF変更は次のマップから反映されます。\n"
        )
        self.append("TEAM CONTROLLERS\n")
        for team in payload[1]:
            key = team_controllers.get(team, "fnatic_v1")
            self.append(f"  {team}: " f"{CONTROLLER_KEY_TO_DISPLAY.get(key, key)}\n")
        self.current_rating_enabled = bool(self.rating_enabled_var.get())
        self.append(f"RATING: {'ON' if self.current_rating_enabled else 'OFF'}\n")
        self.append("-" * 78 + "\n")
        self.competition_id = f"{payload[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.competition_rating_updates = []

        self.visual_mode = payload[0]
        self.visual_series_maps = []
        self.visual_swiss_rounds = []
        self.visual_swiss_records = {}
        self.visual_league_table = {}
        self.visual_current_pairings = []
        self.visual_bye = None
        self.visual_bracket_matches = {}
        self.redraw_visual()

        self.set_enabled(False)

        # 大会計算はワーカースレッドで実行する。
        # 描画が必要なマップだけrender_requests経由で
        # Tkメインスレッドへ依頼する。
        self.live_render_enabled = bool(self.render_var.get())
        self.status_var.set("大会を開始します…")
        self.worker = threading.Thread(
            target=self._execute_job,
            args=(
                payload,
                need,
                lower_final_need,
                grand_final_need,
                tournament_seed_config,
                team_controllers,
                seed_mode,
                base_seed,
                self._render_controller,
            ),
            daemon=True,
        )
        self.worker.start()

    class _RenderController:
        def __init__(self, app: "CompetitionApp") -> None:
            self.app = app

        def __call__(self) -> bool:
            return bool(self.app.live_render_enabled)

        def play_map(
            self,
            team1: Any,
            team2: Any,
            map_number: int,
            seed: int,
            team1_controller_key: str,
            team2_controller_key: str,
            team1_series_wins: int,
            team2_series_wins: int,
            series_maps_to_win: int,
        ) -> MResult:
            user_match = "user" in {
                team1_controller_key,
                team2_controller_key,
            }
            if not self.app.live_render_enabled and not user_match:
                return play_map(
                    team1,
                    team2,
                    map_number,
                    seed,
                    False,
                    team1_controller_key,
                    team2_controller_key,
                    team1_series_wins,
                    team2_series_wins,
                    series_maps_to_win,
                )

            request = {
                "team1": team1,
                "team2": team2,
                "map_number": map_number,
                "seed": seed,
                "team1_controller_key": team1_controller_key,
                "team2_controller_key": team2_controller_key,
                "team1_series_wins": team1_series_wins,
                "team2_series_wins": team2_series_wins,
                "series_maps_to_win": series_maps_to_win,
                "done": threading.Event(),
                "result": None,
                "error": None,
            }
            self.app.render_requests.put(request)
            request["done"].wait()

            if request["error"] is not None:
                raise request["error"]
            return request["result"]

    @property
    def _render_controller(self) -> "_RenderController":
        controller = getattr(self, "__render_controller", None)
        if controller is None:
            controller = self._RenderController(self)
            self.__render_controller = controller
        return controller

    def _process_render_requests(self) -> None:
        """描画マップをTkメインスレッドで1件ずつ実行する。"""
        try:
            request = self.render_requests.get_nowait()
        except queue.Empty:
            return

        try:
            request["result"] = play_map(
                request["team1"],
                request["team2"],
                request["map_number"],
                request["seed"],
                True,
                request["team1_controller_key"],
                request["team2_controller_key"],
                request["team1_series_wins"],
                request["team2_series_wins"],
                request["series_maps_to_win"],
            )
        except BaseException as exc:
            request["error"] = exc
        finally:
            request["done"].set()

    def _execute_job(
        self,
        payload: tuple[str, list[str]],
        need: int,
        lower_final_need: int,
        grand_final_need: int,
        tournament_seed_config: dict[str, Any],
        team_controllers: dict[str, str],
        seed_mode: str,
        base_seed: int | None,
        render: bool | Callable[[], bool],
    ) -> None:
        try:
            mode, teams = payload
            if mode == "series":
                series = run_series_core(
                    team1_name=teams[0],
                    team2_name=teams[1],
                    maps_to_win=need,
                    seed_mode=seed_mode,
                    base_seed=base_seed,
                    seed_offset=0,
                    render=render,
                    emit=self.emit,
                    team_controllers=team_controllers,
                    context_label="SERIES",
                )
                data = {
                    "mode": "series",
                    "seed_mode": seed_mode,
                    "base_seed": (base_seed if seed_mode == "fixed" else None),
                    "team_controllers": dict(team_controllers),
                    "rating_enabled": bool(self.current_rating_enabled),
                    **asdict(series),
                    "player_leaderboards": build_player_leaderboards([series]),
                }
                path = save_json(
                    f"series_{series.team1}_vs_{series.team2}_{series.team1_wins}-{series.team2_wins}",
                    data,
                )
                self.emit(("series_done", series, "SERIES"))
                self.emit(("competition_done", data, str(path)))
            elif mode == "swiss":
                config = dict(tournament_seed_config)
                config["rating_snapshot"] = {
                    team: self.rating_store.get(team) for team in teams
                }
                self.emit(
                    (
                        "log",
                        "\nTOURNAMENT SEEDING\n"
                        f"  Method: {config.get('seed_method')}\n"
                        f"  Seed count: {config.get('seed_count')}\n"
                        f"  BYE count: "
                        f"{_next_power_of_two(len(teams)) - len(teams)}\n"
                        + "-" * 78
                        + "\n",
                    )
                )
                run_double_elimination(
                    teams,
                    need,
                    lower_final_need,
                    grand_final_need,
                    config,
                    seed_mode,
                    base_seed,
                    render,
                    self.emit,
                    team_controllers,
                )
            else:
                run_round_robin(
                    teams,
                    need,
                    seed_mode,
                    base_seed,
                    render,
                    self.emit,
                    team_controllers,
                )
        except Exception:
            self.emit(("error", traceback.format_exc()))
        finally:
            # Tk操作は_poll_events側のメインスレッドだけで行う。
            pass

    def format_standings(self, table: dict[str, dict[str, int]], mode: str) -> str:
        if mode == "swiss":
            ordered = sorted(
                table,
                key=lambda name: (
                    table[name]["losses"],
                    -table[name]["wins"],
                    name.lower(),
                ),
            )
            lines = ["\nSWISS STANDINGS"]
            for index, name in enumerate(ordered, 1):
                rec = table[name]
                status = (
                    "WINNERS"
                    if rec["losses"] == 0
                    else ("LOSERS" if rec["losses"] == 1 else "ELIMINATED")
                )
                lines.append(
                    f"{index:2d}. {name:<32} {rec['wins']}-{rec['losses']}  {status}"
                )
            return "\n".join(lines) + "\n"

        ordered = sorted(
            table,
            key=lambda name: (
                -table[name]["series_wins"],
                -(table[name]["map_wins"] - table[name]["map_losses"]),
                -(table[name]["round_wins"] - table[name]["round_losses"]),
                name.lower(),
            ),
        )
        lines = [
            "\nLEAGUE STANDINGS",
            " # Team                              W-L   MapDiff RoundDiff",
        ]
        for index, name in enumerate(ordered, 1):
            rec = table[name]
            map_diff = rec["map_wins"] - rec["map_losses"]
            round_diff = rec["round_wins"] - rec["round_losses"]
            lines.append(
                f"{index:2d}. {name:<32} {rec['series_wins']}-{rec['series_losses']}   {map_diff:+4d}    {round_diff:+5d}"
            )
        return "\n".join(lines) + "\n"

    def format_player_leaderboards(
        self,
        leaderboards: dict[str, Any],
    ) -> str:
        if not leaderboards:
            return ""

        lines = ["", "=" * 78, "PLAYER LEADERBOARDS", "=" * 78]

        def add(title: str, key: str, value_key: str) -> None:
            lines.append("")
            lines.append(title)
            lines.append(
                " # Player                 Team"
                "                           K    D"
                "    K/D Maps MVP   Value"
            )
            for rank, row in enumerate(leaderboards.get(key, []), 1):
                if value_key == "kd":
                    value = f"{row['kd']:.3f}"
                elif value_key == "kills_per_map":
                    value = f"{row['kills_per_map']:.2f}"
                else:
                    value = str(row[value_key])
                lines.append(
                    f"{rank:2d}. {row['player']:<22}"
                    f"{row['team']:<31}"
                    f"{row['kills']:>4}{row['deaths']:>5}"
                    f"{row['kd']:>7.3f}{row['maps']:>5}"
                    f"{row['mvps']:>4}{value:>8}"
                )

        add("K/D TOP 5", "kd_top5", "kd")
        add("MVP COUNT TOP 5", "mvp_top5", "mvps")
        add("TOTAL KILLS TOP 5", "kills_top5", "kills")
        add(
            "KILLS PER MAP TOP 5",
            "kills_per_map_top5",
            "kills_per_map",
        )
        lines.extend(["", "=" * 78])
        return "\n".join(lines) + "\n"

    def _poll_events(self) -> None:
        # 描画要求は必ずTkメインスレッドで処理する。
        self._process_render_requests()

        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]

                if kind == "status":
                    self.status_var.set(event[1])
                elif kind == "log":
                    self.append(event[1])
                elif kind == "map":
                    result: MResult = event[1]
                    context = event[2]
                    self.visual_series_maps.append(result)
                    self.redraw_visual()
                    ot = " [OT]" if result.overtime else ""
                    header = f"{context} / " if context else ""
                    self.append(
                        f"{header}MAP {result.number}{ot}\n"
                        f"  Seed: {result.seed}\n"
                        f"  {result.team1} {result.score1} - {result.score2} {result.team2}\n"
                        f"  Winner: {result.winner}\n"
                        f"  Initial Attacker: {result.initial_attacker}\n"
                        f"  {result.team1} MVP: {result.mvp1.name} {result.mvp1.kills}K/{result.mvp1.deaths}D\n"
                        f"  {result.team2} MVP: {result.mvp2.name} {result.mvp2.kills}K/{result.mvp2.deaths}D\n"
                        + "-" * 78
                        + "\n"
                    )
                elif kind == "series_score":
                    team1, wins1, wins2, team2, context = event[1:]
                    prefix = f"{context}: " if context else ""
                    self.series_score_var.set(
                        f"{prefix}{team1} {wins1} - {wins2} {team2}"
                    )
                elif kind == "series_done":
                    series: SeriesResult = event[1]
                    context = event[2]
                    self.append(
                        f"\n{context} FINAL: {series.team1} {series.team1_wins} - {series.team2_wins} {series.team2}"
                        f" / WINNER: {series.winner}\n"
                    )
                    if self.current_rating_enabled:
                        rating_update = self.rating_store.update_series(
                            series,
                            context,
                            self.competition_id,
                        )
                        self.competition_rating_updates.append(rating_update)
                        d1 = rating_update["delta"][series.team1]
                        d2 = rating_update["delta"][series.team2]
                        a1 = rating_update["after"][series.team1]
                        a2 = rating_update["after"][series.team2]
                        self.append(
                            f"RATING: {series.team1} {a1:.1f} ({d1:+.1f})"
                            f" / {series.team2} {a2:.1f} ({d2:+.1f})\n\n"
                        )
                        self.refresh_rating_window()
                    else:
                        self.append("RATING: OFF（レート変動なし）\n\n")
                elif kind == "bracket_match":
                    match = event[1]
                    self.visual_bracket_matches[match["id"]] = dict(match)
                    self.redraw_visual()
                elif kind == "stage_round":
                    round_number, pairings, bye, _records = event[1:]
                    self.visual_current_pairings = list(pairings)
                    self.visual_bye = bye
                    self.visual_swiss_records = {
                        name: dict(values) for name, values in _records.items()
                    }
                    self.redraw_visual()
                    self.append(f"\n{'=' * 78}\nSWISS ROUND {round_number}\n")
                    for team1, team2 in pairings:
                        self.append(f"  {team1} vs {team2}\n")
                    if bye:
                        self.append(f"  BYE: {bye}\n")
                    self.append("=" * 78 + "\n")
                elif kind == "standings":
                    table, standings_mode = event[1], event[2]
                    self.append(self.format_standings(table, standings_mode))
                    if standings_mode == "swiss":
                        self.visual_swiss_records = {
                            name: dict(values) for name, values in table.items()
                        }
                    else:
                        self.visual_league_table = {
                            name: dict(values) for name, values in table.items()
                        }
                    self.redraw_visual()
                elif kind == "competition_done":
                    data, path = event[1], event[2]
                    champion = data.get("champion", data.get("winner", "?"))

                    data["rating_enabled"] = bool(self.current_rating_enabled)
                    data["rating_system"] = {
                        "default_rating": DEFAULT_TEAM_RATING,
                        "k_factor": RATING_K_FACTOR,
                        "persistent_file": str(RATING_FILE),
                    }
                    data["rating_updates"] = list(self.competition_rating_updates)
                    data["ratings_after_competition"] = {
                        name: round(value, 3)
                        for name, value in self.rating_store.ranking()
                    }
                    try:
                        Path(path).write_text(
                            json.dumps(
                                data,
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    except Exception as exc:
                        self.append(f"\n[WARN] 結果JSONへのRating追記失敗: {exc}\n")

                    self.status_var.set(f"終了：{champion} WIN")
                    self.append(
                        f"\n{'#' * 78}\nCOMPETITION FINISHED\n" f"WINNER: {champion}\n"
                    )
                    self.append(
                        self.format_player_leaderboards(
                            data.get("player_leaderboards", {})
                        )
                    )
                    self.append(self.format_rating_ranking())
                    self.append(
                        f"Rating保存先: {RATING_FILE}\n"
                        f"大会結果保存先: {path}\n{'#' * 78}\n"
                    )
                    self.refresh_rating_window()
                    self.worker = None
                    self.root.deiconify()
                    self.root.lift()
                    self.set_enabled(True)
                elif kind == "error":
                    self.status_var.set("エラーが発生しました")
                    self.append("\nERROR\n" + event[1] + "\n")
                    self.worker = None
                    self.root.deiconify()
                    self.root.lift()
                    self.set_enabled(True)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    CompetitionApp().run()
