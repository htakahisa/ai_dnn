"""Validated season scrim requests and isolated background match processes."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from uuid import uuid4

from character_stats import CharacterStats
from realtime_season import ROSTER_SIZE, SeasonSaveError, validate_player


def ai_options():
    from roster_select import TEAM_AI_OPTIONS
    return dict(TEAM_AI_OPTIONS)


def build_scrim_request(state, own_team_id, opponent_team_id, *, render=True,
                        own_ai=None, opponent_ai=None, initial_side="A",
                        own_igl=None, opponent_igl=None, own_spike=None,
                        opponent_spike=None, tick_time_ms=100, seed=None):
    from game_core import validate_tick_time_ms
    state.validate()
    state.check_day_action()
    own = state.team(own_team_id)
    opponent = next((team for team in state.opponent_teams if team.id == opponent_team_id), None)
    if own is None or opponent is None:
        raise SeasonSaveError("自分のチームと相手チームを選択してください。")
    if len(opponent.players) < ROSTER_SIZE:
        raise SeasonSaveError("相手チームの所属選手が5人未満です。所属設定で選手を補充してください。")
    if any(not state.can_play(name) for name in own.roster):
        raise SeasonSaveError("契約が終了した選手がいます。再契約またはチームの再編成を行ってください。")
    own_players = state.match_players(state.player(name) for name in own.roster)
    opponent_players = state.match_players(opponent.players[:ROSTER_SIZE], enemy=True)

    def team_data(name, players, ai, igl, spike):
        return {"name": name, "players": [asdict(p) for p in players],
                "ai": ai,
                "igl": igl or max(players, key=lambda p: p.iq).name,
                "spike_holder": spike or players[0].name}

    request = {
        "own": team_data(state.team_name, own_players, own.ai if own_ai is None else own_ai,
                         own.igl if own_igl is None else own_igl,
                         own.carrier if own_spike is None else own_spike),
        "opponent": team_data(opponent.name, opponent_players,
                              opponent.ai if opponent_ai is None else opponent_ai,
                              opponent.effective_igl if opponent_igl is None else opponent_igl,
                              opponent.effective_carrier if opponent_spike is None else opponent_spike),
        "render": render, "initial_side": initial_side, "tick_time_ms": validate_tick_time_ms(tick_time_ms),
        "seed": random.SystemRandom().randrange(2**31) if seed is None else seed,
    }
    validate_scrim_request(request)
    return request


def validate_scrim_request(request):
    from game_core import validate_tick_time_ms
    if not isinstance(request, dict):
        raise SeasonSaveError("スクリム設定の形式が不正です。")
    allowed_ai = set(ai_options().values())
    names = set()
    team_names = set()
    for key in ("own", "opponent"):
        team = request.get(key)
        if not isinstance(team, dict) or not isinstance(team.get("name"), str) or not team["name"].strip() or len(team["name"]) > 40:
            raise SeasonSaveError("スクリムのチーム名が不正です。")
        if team["name"].casefold() in team_names:
            raise SeasonSaveError("同じ名前のチーム同士ではスクリムできません。")
        team_names.add(team["name"].casefold())
        if not isinstance(team.get("players"), list) or len(team["players"]) != ROSTER_SIZE:
            raise SeasonSaveError("スクリムには各チーム5人の編成が必要です。")
        members = []
        for row in team["players"]:
            try:
                player = CharacterStats(**row)
            except TypeError as exc:
                raise SeasonSaveError("スクリム選手のデータが不正です。") from exc
            validate_player(player)
            if player.name in names:
                raise SeasonSaveError(f"{player.name}がスクリムの編成内で重複しています。")
            names.add(player.name)
            members.append(player.name)
        if team.get("igl") not in members or team.get("spike_holder") not in members:
            raise SeasonSaveError("IGLとスパイク所持者は出場する5人から選択してください。")
        if team.get("ai") not in allowed_ai:
            raise SeasonSaveError("選択したAIは使用できません。")
    if request.get("initial_side") not in ("A", "D") or type(request.get("render")) is not bool:
        raise SeasonSaveError("開始サイドまたは描画設定が不正です。")
    if not request["render"] and any(request[key]["ai"] == "user" for key in ("own", "opponent")):
        raise SeasonSaveError("ユーザー操作を選ぶ場合は描画ありにしてください。")
    validate_tick_time_ms(request.get("tick_time_ms"))
    if type(request.get("seed")) is not int or not 0 <= request["seed"] < 2**31:
        raise SeasonSaveError("スクリムの乱数設定が不正です。")


class ScrimJob:
    def __init__(self, request, output_directory):
        validate_scrim_request(request)
        self.directory = Path(output_directory).resolve() / uuid4().hex
        self.directory.mkdir(parents=True)
        self.request_path = self.directory / "request.json"
        self.result_path = self.directory / "result.json"
        self.log_path = self.directory / "match.log"
        self.request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        self.cancelled = False
        project_directory = Path(__file__).resolve().parent.parent
        with self.log_path.open("w", encoding="utf-8") as log:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "season.season_scrim_worker", "--request", str(self.request_path), "--result", str(self.result_path)],
                cwd=project_directory, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

    def poll(self):
        code = self.process.poll()
        if code is None:
            return None
        if self.cancelled:
            return {"status": "cancelled"}
        try:
            return json.loads(self.result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"status": "error", "message": f"試合プロセスが終了しました (code={code})。詳細: {self.log_path}"}

    def cancel(self):
        if self.process.poll() is None:
            self.cancelled = True
            self.process.terminate()
