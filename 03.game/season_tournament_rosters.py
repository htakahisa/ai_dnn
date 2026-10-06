"""Editable tournament lineups and temporary substitutes for missing players."""

from dataclasses import replace

from season_player_stats import SeasonPlayerStats


def friend_player(number):
    # Match the engine's initial abilities for an undefined character.
    return SeasonPlayerStats("友達" if number == 1 else f"友達{number}",
                             .20, .10, 50, .50, 100, "フラッシュ", 0,
                             monthly_salary=0)


def tournament_team(state, event_id, names=None, *, ai=None, igl=None, carrier=None):
    from realtime_season import ROSTER_SIZE, SeasonSaveError

    run = state.tournament(event_id)
    if run is None or run.own_team_id is None or run.declined:
        raise SeasonSaveError("自チームが参加登録した大会を選択してください。")
    own = next(t for t in run.entrants if t.id == run.own_team_id)
    if run.completed and names is None:
        return own
    if names is None:
        players = [state.player(p.name) for p in own.players if state.can_play(p.name)]
    else:
        names = tuple(names)
        if (len(names) > ROSTER_SIZE or any(not isinstance(n, str) for n in names)
                or len(set(names)) != len(names)):
            raise SeasonSaveError("出場選手は重複のない5人以内で選択してください。")
        if any(not state.can_play(n) for n in names):
            raise SeasonSaveError("出場できる契約中の所持選手を選択してください。")
        players = [state.player(n) for n in names]
    opponents = {p.name for t in run.entrants if t.id != run.own_team_id for p in t.players}
    if any(p.name in opponents for p in players):
        raise SeasonSaveError("この大会で他チームから出場する選手は選択できません。")
    occupied = opponents | {p.name for p in state.owned_players}
    number = 1
    while len(players) < ROSTER_SIZE:
        friend = friend_player(number)
        number += 1
        if friend.name not in occupied:
            players.append(friend)
            occupied.add(friend.name)
    members = {p.name for p in players}
    leader = igl if igl is not None else own.igl
    holder = carrier if carrier is not None else own.carrier
    return replace(own, players=tuple(players), ai=own.ai if ai is None else ai,
                   igl=leader if leader in members else max(players, key=lambda p: p.iq).name,
                   carrier=holder if holder in members else players[0].name)


def replace_tournament_team(state, event_id, team):
    run = state.tournament(event_id)
    updated = replace(run, entrants=tuple(team if t.id == team.id else t for t in run.entrants))
    return replace(state, tournaments=tuple(updated if r.tournament_id == event_id else r
                                             for r in state.tournaments))
