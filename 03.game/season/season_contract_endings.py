"""Calendar-month renewal grace and permanent club-specific refusal."""

from dataclasses import replace

from season.season_competitions import add_months, parse_date
from season.season_loyalty import remember_loyalties, UNRENEWED_LOYALTY_LOSS


def settle_contract_endings(state, deferred=()):
    """Settle once per contract; active tournament contracts remain untouched."""
    candidate = state
    events = []

    def ending(contract, team_id):
        if (contract.end_reason is not None or candidate.contract_active(contract)
                or candidate.tournament_reserves_contract(contract, team_id)):
            return contract, None
        if contract.expired_on is None:
            date = (candidate.date if (team_id, contract.player_name) in deferred
                    else contract.ends_on(candidate.start_date))
            contract = replace(contract, expired_on=date.isoformat())
        if contract.team_loyalty < 0:
            return contract, "忠誠が負になったため、このチームとの再契約を永久に拒否してLFTになりました。"
        if candidate.date >= add_months(parse_date(contract.expired_on), 1):
            return replace(contract, team_loyalty=contract.team_loyalty - UNRENEWED_LOYALTY_LOSS), (
                "契約終了から1か月間再契約しなかったため、忠誠が5下がりLFTになりました。")
        return contract, None

    for original in state.contracts:
        if candidate.player(original.player_name) is None:
            continue
        contract, message = ending(original, state.club_id)
        candidate = replace(candidate, contracts=tuple(
            contract if c.player_name == contract.player_name else c for c in candidate.contracts))
        if message:
            from season.season_monthly_events import MonthlyEvent
            events.append(MonthlyEvent(
                f"contract-end:{state.club_id}:{contract.player_name}:{contract.start_month}:{contract.signing_number}",
                candidate.date.isoformat(), "departure", state.club_id, state.team_name,
                contract.player_name, f"{contract.player_name}: {message}"))
            candidate = candidate._with_departed_player(contract.player_name)

    clubs = []
    for club in candidate.opponent_teams:
        members, contracts = [], {c.player_name: c for c in club.contracts}
        for player in club.players:
            contract, message = ending(contracts[player.name], club.id)
            contracts[player.name] = replace(contract, end_reason="left") if message else contract
            if message:
                from season.season_monthly_events import MonthlyEvent
                events.append(MonthlyEvent(
                    f"contract-end:{club.id}:{player.name}:{contract.start_month}:{contract.signing_number}",
                    candidate.date.isoformat(), "departure", club.id, club.name, player.name,
                    f"{player.name}: {message}"))
            else:
                members.append(player)
        names = {p.name for p in members}
        starters = {p.name for p in members[:5]}
        clubs.append(replace(club, players=tuple(members), contracts=tuple(contracts.values()),
                             acquired_members=tuple(n for n in club.acquired_members if n in names),
                             igl=club.igl if club.igl in starters else None,
                             carrier=club.carrier if club.carrier in starters else None))
    candidate = remember_loyalties(replace(candidate, opponent_teams=tuple(clubs),
                                         monthly_events=(*candidate.monthly_events, *events)))
    return candidate
