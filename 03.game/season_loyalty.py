"""Persistent relationships between each season player and each club ID."""

from dataclasses import replace
import math

from character_stats import all_characters


INITIAL_TEAM_LOYALTY = 50.0
CONTRACT_LOYALTY = {
    "short": (20.0, 5.0),
    "year1": (30.0, 10.0),
    "year2": (40.0, 15.0),
    "year3": (50.0, 20.0),
}
UNRENEWED_LOYALTY_LOSS = 5.0
BENCHED_LOYALTY_LOSS = 3.0


def signing_count(state, player_name, team_id):
    count = state.contract_signings.get(player_name, {}).get(team_id, 0)
    for identifier, contracts in club_contracts(state):
        if identifier == team_id:
            count = max(count, *(c.signing_number for c in contracts if c.player_name == player_name), 0)
    return count


def signing_loyalty(state, player_name, team_id, kind):
    initial, recovery = CONTRACT_LOYALTY[kind]
    count = signing_count(state, player_name, team_id)
    return (loyalty_for(state, player_name, team_id) + recovery if count else initial), count + 1


def contract_refused(state, player_name, team_id):
    return ((player_name, team_id) in state.contract_bans
            or loyalty_for(state, player_name, team_id) < 0)


def club_contracts(state):
    yield state.club_id, state.contracts
    for club in state.opponent_teams:
        yield club.id, club.contracts


def loyalty_for(state, player_name, team_id):
    # Contracts contain the latest value, including retained ended contracts.
    for identifier, contracts in club_contracts(state):
        if identifier == team_id:
            contract = next((c for c in contracts if c.player_name == player_name), None)
            if contract is not None:
                return contract.team_loyalty
    return state.team_loyalties.get(player_name, {}).get(team_id, INITIAL_TEAM_LOYALTY)


def remember_loyalties(state):
    values = {name: dict(teams) for name, teams in state.team_loyalties.items()}
    counts = {name: dict(teams) for name, teams in state.contract_signings.items()}
    bans = set(state.contract_bans)
    names = {p.name for p in all_characters()} | set(values)
    names.update(p.name for p in (*state.owned_players, *state.developed_players))
    ids = {state.club_id, *(c.id for c in state.opponent_teams)}
    ids.update(identifier for teams in values.values() for identifier in teams)
    for identifier, contracts in club_contracts(state):
        for contract in contracts:
            names.add(contract.player_name)
            values.setdefault(contract.player_name, {})[identifier] = contract.team_loyalty
            counts.setdefault(contract.player_name, {})[identifier] = max(
                counts.get(contract.player_name, {}).get(identifier, 0), contract.signing_number)
            if contract.end_reason is not None and contract.team_loyalty < 0:
                bans.add((contract.player_name, identifier))
    for name in names:
        teams = values.setdefault(name, {})
        for identifier in ids:
            teams.setdefault(identifier, INITIAL_TEAM_LOYALTY)
    bans = tuple(sorted(bans))
    if values == state.team_loyalties and counts == state.contract_signings and bans == state.contract_bans:
        return state
    return replace(state, team_loyalties=values, contract_signings=counts, contract_bans=bans)


def validate_contract_memory(state):
    if not isinstance(state.contract_signings, dict):
        raise ValueError("契約回数の形式が不正です。")
    for name, teams in state.contract_signings.items():
        if not isinstance(name, str) or not name or not isinstance(teams, dict):
            raise ValueError("契約回数の選手データが不正です。")
        for identifier, count in teams.items():
            if not isinstance(identifier, str) or not identifier or type(count) is not int or count < 1:
                raise ValueError("契約回数が不正です。")
    if not isinstance(state.contract_bans, tuple) or len(set(state.contract_bans)) != len(state.contract_bans):
        raise ValueError("再契約拒否の記録が不正です。")
    for pair in state.contract_bans:
        if not isinstance(pair, tuple) or len(pair) != 2 or any(not isinstance(v, str) or not v for v in pair):
            raise ValueError("再契約拒否の記録が不正です。")
    for _, contracts in club_contracts(state):
        for contract in contracts:
            if type(contract.signing_number) is not int or contract.signing_number < 1:
                raise ValueError("契約回数が不正です。")
            if contract.signed_on is not None:
                from season_competitions import parse_date, month_index
                signed = parse_date(contract.signed_on)
                if (not parse_date(state.start_date) <= signed <= state.date
                        or month_index(parse_date(state.start_date), signed) != contract.start_month):
                    raise ValueError("契約開始日が不正です。")
            if contract.expired_on is not None:
                from season_competitions import parse_date
                parse_date(contract.expired_on)


def validate_loyalties(values):
    if not isinstance(values, dict):
        raise ValueError("チーム別忠誠の形式が不正です。")
    for name, teams in values.items():
        if not isinstance(name, str) or not name or not isinstance(teams, dict):
            raise ValueError("チーム別忠誠の選手データが不正です。")
        for identifier, value in teams.items():
            if (not isinstance(identifier, str) or not identifier
                    or type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("チーム別忠誠の値が不正です。")
