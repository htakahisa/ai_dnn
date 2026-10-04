"""Monthly NPC club decisions, independent of wall-clock time and UI."""

from collections import Counter
from dataclasses import dataclass, replace
from datetime import timedelta

from character_stats import all_characters
from season_loyalty import remember_loyalties, CONTRACT_LOYALTY
from season_transfers import restore_regular_members
from season_rival_economy import can_sign, opportunistic_offers, signing_terms, signed_contract


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


def settle_deferred_contracts(state, deferred, event_id):
    """End the extension immediately after the last registered event finishes."""
    events, processed = [], []
    candidate = state
    for contract in state.contracts:
        if ((state.club_id, contract.player_name) not in deferred
                or candidate.tournament_reserves_contract(contract)
                or candidate.player(contract.player_name) is None):
            continue
        if contract.team_loyalty < 0:
            candidate = candidate._with_departed_player(contract.player_name, record=False)
        else:
            candidate = replace(candidate, contracts=tuple(
                replace(c, expired_on=candidate.date.isoformat()) if c.player_name == contract.player_name
                and not candidate.contract_active(c) and c.expired_on is None else c for c in candidate.contracts))
        # Normal expiry keeps the player available for the usual renewal UI.
        processed.append(contract.player_name)
    clubs = []
    for club in candidate.opponent_teams:
        contracts = {c.player_name: c for c in club.contracts}
        players = []
        for player in club.players:
            contract = contracts.get(player.name)
            if (contract is None or (club.id, player.name) not in deferred
                    or candidate.tournament_reserves_contract(contract, club.id)):
                players.append(player)
                continue
            short_exit = contract.kind == "short" and contract.team_loyalty < 0
            expired = not candidate.contract_active(contract)
            if not short_exit and not expired:
                players.append(player)
                continue
            terms = signing_terms(candidate, player)
            if short_exit or expired and contract.team_loyalty < 0:
                contracts[player.name] = replace(contract, end_reason="left")
                kind, message = "departure", f"大会終了に伴い、{player.name}が契約を終了して退団し、LFTになりました。"
            else:
                club = replace(club, money=club.money - terms.signing_bonus)
                if not can_sign(candidate, club, terms):
                    contracts[player.name] = replace(contract, expired_on=contract.expired_on or candidate.date.isoformat())
                    players.append(player)
                    processed.append(player.name)
                    continue
                contracts[player.name] = signed_contract(player, terms, candidate.game_month, state=candidate, team_id=club.id)
                players.append(player)
                kind, message = "renewal", f"大会終了に伴い、{player.name}と{terms.days}日間の再契約を結びました。"
            events.append(MonthlyEvent(
                f"{candidate.game_month}:{kind}:{club.id}:{player.name}:after:{event_id}",
                candidate.date.isoformat(), kind, club.id, club.name, player.name, message))
            processed.append(player.name)
        members = {p.name for p in players}
        roster = {p.name for p in players[:5]}
        clubs.append(replace(club, players=tuple(players), contracts=tuple(contracts.values()),
                             acquired_members=tuple(n for n in club.acquired_members if n in members),
                             igl=club.igl if club.igl in roster else None,
                             carrier=club.carrier if club.carrier in roster else None))
    candidate = replace(candidate, opponent_teams=tuple(clubs), monthly_events=(*candidate.monthly_events, *events))
    from season_contract_endings import settle_contract_endings
    candidate = settle_contract_endings(candidate, deferred)
    candidate.validate()
    return candidate._record_history("大会後の契約終了処理") if processed else candidate


def process_monthly_events(state):
    from realtime_season import ROSTER_SIZE, initial_contract

    if state.monthly_events_through >= state.game_month or state.starter_selection_pending:
        return state
    state = remember_loyalties(state)
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
            contract = contracts.get(player.name) or replace(
                initial_contract(player, month - 1), team_loyalty=state.team_loyalty(player.name, club.id))
            if (contract.active(state.date - timedelta(days=1), state.start_date)
                    or state.contract_end_deferred(contract, club.id)) and player.loyalty < 10:
                contract = replace(contract, team_loyalty=round(contract.team_loyalty - (10 - player.loyalty) / 10, 10))
            short_exit = contract.kind == "short" and contract.team_loyalty < 0
            expired = not state.contract_active(contract)
            # Event snapshots are immutable; defer changes to participants until
            # their tournament is finished, instead of invalidating live matches.
            if player.name not in reserved and (short_exit or expired and contract.team_loyalty < 0):
                contracts[player.name] = replace(contract, end_reason="left")
                emit("departure", club, player.name, f"{player.name}が契約を終了して退団し、LFTになりました。")
                continue
            if player.name not in reserved and expired:
                terms = signing_terms(state, player)
                if not can_sign(state, club, terms):
                    contracts[player.name] = contract
                    members.append(player)
                    emit("renewal_waiting", club, player.name, f"{player.name}との再契約に必要な資金が足りず、猶予期間中です。")
                    continue
                club = replace(club, money=club.money - terms.signing_bonus)
                loyalty = contract.team_loyalty + CONTRACT_LOYALTY[terms.kind][1]
                contract = signed_contract(player, terms, month, state=state, team_id=club.id)
                contract = replace(contract, team_loyalty=loyalty)
                emit("renewal", club, player.name,
                     f"{player.name}と{contract.duration_days}日間・月給{contract.monthly_salary:,}円で再契約しました。")
            contracts[player.name] = contract
            members.append(player)
        roster = {p.name for p in members[:ROSTER_SIZE]}
        clubs[index] = replace(club, players=tuple(members), contracts=tuple(contracts.values()),
                               acquired_members=tuple(n for n in club.acquired_members if n in {p.name for p in members}),
                               igl=club.igl if club.igl in roster else None,
                               carrier=club.carrier if club.carrier in roster else None)

    candidate = restore_regular_members(replace(state, opponent_teams=tuple(clubs)), reserved, emit)
    candidate = candidate.with_resolved_transfer_offers()
    clubs = list(candidate.opponent_teams)
    affiliated = {p.name for p in candidate.owned_players} | {p.name for c in clubs for p in c.players} | reserved
    for index, club in enumerate(clubs):
        rejected = {c.player_name for c in club.contracts if candidate.contract_refused(c.player_name, club.id)}
        while len(club.players) < ROSTER_SIZE:
            if candidate.scout_blocked(club.id):
                emit("recruitment_unfilled", club, None, "インシーズンのスカウト上限に達したため、補充を見送りました。")
                break
            pool = tuple(candidate.salary_player(p) for p in all_characters() if p.name not in affiliated and p.name not in rejected)
            pool = tuple(p for p in pool if not candidate.contract_refused(p.name, club.id)
                         and can_sign(candidate, club, signing_terms(candidate, p)))
            player = choose_recruit(club, pool, state.rating(club.id))
            if player is None:
                emit("recruitment_unfilled", club, None, f"資金条件を満たすLFT選手がいないため、{len(club.players)}人で補充を見送りました。")
                break
            terms = signing_terms(candidate, player)
            contract = signed_contract(player, terms, month, state=candidate, team_id=club.id)
            candidate = candidate._with_scout_use(club.id)
            club = replace(club, players=(*club.players, player), money=club.money - terms.signing_bonus,
                           contracts=tuple(c for c in club.contracts if c.player_name != player.name) + (contract,))
            affiliated.add(player.name)
            emit("recruitment", club, player.name,
                 f"LFTの{player.name}（{player.role}）と{contract.duration_days}日間・月給{contract.monthly_salary:,}円で契約しました。")
        clubs[index] = club

    candidate = replace(candidate, opponent_teams=tuple(clubs))
    candidate = opportunistic_offers(candidate, emit).with_resolved_transfer_offers()
    count = len(events)
    events.append(MonthlyEvent(f"{month}:month_completed", state.date.isoformat(), "month_completed", None, "シーズン", None,
                               f"月次処理が完了しました。他チームの出来事: {count}件。"))
    candidate = replace(candidate, monthly_events=(*state.monthly_events, *events),
                        monthly_events_through=month)
    candidate.validate()
    return remember_loyalties(candidate)
