"""Use the isolated normal match engine for season tournament series."""

from dataclasses import asdict

from realtime_season import SeasonSaveError
from season_competitions import next_match
from season_scrim import validate_scrim_request


def build_series_request(state, event_id, *, render=True, tick_time_ms=100):
    state.validate()
    event, run = state.tournament_definition(event_id), state.tournament(event_id)
    if event is None or run is None or run.completed:
        raise SeasonSaveError("進行中の大会を選択してください。")
    state.check_tournament_match_day(event_id)
    match, _ = next_match(event, run)
    if run.own_team_id not in (match.left, match.right):
        raise SeasonSaveError("他チーム同士の試合はレートの勝率で抽選してください。")
    if run.own_team_id in (match.left, match.right):
        own = next(t for t in run.entrants if t.id == run.own_team_id)
        if any(not state.can_play(p.name) for p in own.players):
            raise SeasonSaveError("大会参加選手の契約が終了しています。契約状況から再契約してください。")
    teams = {t.id: t for t in run.entrants}

    def data(team):
        base = tuple(state.player(p.name) for p in team.players) if team.id == state.club_id else team.players
        players = state.match_players(base, enemy=team.id != state.club_id)
        return {"name": team.name, "players": [asdict(p) for p in players],
                "ai": team.ai,
                "igl": team.igl, "spike_holder": team.carrier}

    request = {
        "own": data(teams[match.left]), "opponent": data(teams[match.right]),
        "render": render and run.own_team_id in (match.left, match.right),
        "initial_side": "A", "tick_time_ms": tick_time_ms,
        "seed": (run.seed + len(run.results) * 1000) % (2**31),
        "maps_to_win": match.maps_to_win, "match_id": match.id,
        "left_id": match.left, "right_id": match.right,
    }
    validate_series_request(request)
    return request


def validate_series_request(request):
    validate_scrim_request(request)
    if type(request.get("maps_to_win")) is not int or not 1 <= request["maps_to_win"] <= 10:
        raise SeasonSaveError("シリーズの先取マップ数が不正です。")
    for key in ("match_id", "left_id", "right_id"):
        if not isinstance(request.get(key), str) or not request[key]:
            raise SeasonSaveError("シリーズの対戦IDが不正です。")
    if request["left_id"] == request["right_id"]:
        raise SeasonSaveError("同じチーム同士ではシリーズを実行できません。")


def play_series(request):
    from season_scrim_worker import play_scrim
    validate_series_request(request)
    need = request["maps_to_win"]
    left, right, maps = 0, 0, []
    while max(left, right) < need:
        index = len(maps)
        single = {**request, "seed": (request["seed"] + index) % (2**31),
                  "initial_side": "A" if index % 2 == 0 else "D"}
        result = play_scrim(single)
        if result["status"] != "completed":
            return {"status": "cancelled", "maps": maps}
        maps.append(result)
        if result["winner"] == request["own"]["name"]:
            left += 1
        else:
            right += 1
    return {"status": "completed", "match_id": request["match_id"],
            "left_id": request["left_id"], "right_id": request["right_id"],
            "left_wins": left, "right_wins": right, "maps": maps}
