"""Monthly NPC club decisions, independent of wall-clock time and UI."""

from collections import Counter
from dataclasses import dataclass, replace

from character_stats import all_characters
from season_transfers import restore_regular_members


DEFAULT_ROLES = ("タイガー", "シーカー", "フラッシュ", "スモーカー", "エンジニア")


@dataclass(frozen=True)
class MonthlyEvent:
    id: str
    date: str
    kind: str
    team_id: str | None
    team_name: str
    player_name: str | None
    message: str


def player_strength(player):
    return (player.iq + player.influence * .25 + player.hit_pct * 100 + player.hs_pct * 50
            + player.dodge_pct * 50 + player.reaction * .2 + player.mental * 4 - player.form_variance * 2)


def choose_recruit(club, candidates, rating):
    """Prefer missing roles, then players of a plausible level and salary."""
    if not candidates:
        return None
    desired = Counter(club.preferred_roles if len(club.preferred_roles) == 5 else DEFAULT_ROLES)
    present = Counter(p.role for p in club.players[:5])
    population = club.players or candidates
    target = sum(player_strength(p) for p in population) / len(population)
    target *= max(.5, min(1.5, rating / 1500))
    salary = sum(p.monthly_salary for p in population) / len(population)
    return min(candidates, key=lambda p: (
        -max(0, desired[p.role] - present[p.role]),
        abs(player_strength(p) - target), abs(p.monthly_salary - salary), -p.loyalty, p.name))


def process_monthly_events(state):
    from realtime_season import ROSTER_SIZE, initial_contract

    if state.monthly_events_through >= state.game_month or state.starter_selection_pending:
        return state
    events = []
    clubs = list(state.opponent_teams)
    month = state.game_month
    # Never recruit a player in an unfinished event, including a released
    # player whose historical tournament roster still exists.
    reserved = {p.name for run in state.tournaments if not run.completed for t in run.entrants for p in t.players}

    def emit(kind, club, player_name, message):
        events.append(MonthlyEvent(f"{month}:{kind}:{club.id}:{player_name or ''}", state.date.isoformat(),
                                   kind, club.id, club.name, player_name, message))

    # Resolve all departures before any recruitment, giving every club the
    # same current LFT pool. Iterate clubs in saved order for deterministic ties.
    for index, club in enumerate(clubs):
        contracts = {c.player_name: c for c in club.contracts}
        members = []
        for player in club.players:
            contract = contracts.get(player.name) or initial_contract(player, month - 1)
            if contract.active(month - 1) and player.loyalty < 10:
                contract = replace(contract, team_loyalty=round(contract.team_loyalty - (10 - player.loyalty) / 10, 10))
            short_exit = contract.kind == "short" and contract.team_loyalty <= 0
            expired = not contract.active(month)
            # Event snapshots are immutable; defer changes to participants until
            # their tournament is finished, instead of invalidating live matches.
            if player.name not in reserved and (short_exit or expired and contract.team_loyalty <= 0):
                contracts[player.name] = replace(contract, end_reason="left")
                emit("departure", club, player.name, f"{player.name}が契約を終了して退団し、LFTになりました。")
                continue
            if player.name not in reserved and expired:
                renewed = initial_contract(player, month)
                contract = replace(renewed, team_loyalty=contract.team_loyalty)
                emit("renewal", club, player.name,
                     f"{player.name}と{contract.duration_months}か月・月給{contract.monthly_salary:,}円で再契約しました。")
            contracts[player.name] = contract
            members.append(player)
        roster = {p.name for p in members[:ROSTER_SIZE]}
        clubs[index] = replace(club, players=tuple(members), contracts=tuple(contracts.values()),
                               igl=club.igl if club.igl in roster else None,
                               carrier=club.carrier if club.carrier in roster else None)

    candidate = restore_regular_members(replace(state, opponent_teams=tuple(clubs)), reserved, emit)
    candidate = candidate.with_resolved_transfer_offers()
    clubs = list(candidate.opponent_teams)
    affiliated = {p.name for p in candidate.owned_players} | {p.name for c in clubs for p in c.players} | reserved
    for index, club in enumerate(clubs):
        rejected = {c.player_name for c in club.contracts if c.team_loyalty <= 0}
        while len(club.players) < ROSTER_SIZE:
            pool = tuple(candidate.salary_player(p) for p in all_characters() if p.name not in affiliated and p.name not in rejected)
            player = choose_recruit(club, pool, state.rating(club.id))
            if player is None:
                emit("recruitment_unfilled", club, None, f"契約できるLFT選手がいないため、{len(club.players)}人で補充を見送りました。")
                break
            contract = initial_contract(player, month)
            club = replace(club, players=(*club.players, player),
                           contracts=tuple(c for c in club.contracts if c.player_name != player.name) + (contract,))
            affiliated.add(player.name)
            emit("recruitment", club, player.name,
                 f"LFTの{player.name}（{player.role}）と{contract.duration_months}か月・月給{contract.monthly_salary:,}円で契約しました。")
        clubs[index] = club

    count = len(events)
    events.append(MonthlyEvent(f"{month}:month_completed", state.date.isoformat(), "month_completed", None, "シーズン", None,
                               f"月次処理が完了しました。他チームの出来事: {count}件。"))
    candidate = replace(candidate, opponent_teams=tuple(clubs), monthly_events=(*state.monthly_events, *events),
                        monthly_events_through=month)
    candidate.validate()
    return candidate
