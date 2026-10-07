"""Initial NPC roster mixing and the return of each club's regular members."""

from dataclasses import dataclass, replace
from random import Random

from character_stats import all_characters
from season.season_loyalty import remember_loyalties, signing_count


@dataclass(frozen=True)
class TransferOffer:
    id: str
    team_id: str
    player_name: str
    created_month: int
    fee: int
    status: str = "pending"
    contract_kind: str | None = None
    contract_months: int | None = None
    monthly_salary: int | None = None


def resolve_club_memberships(clubs, priority_names=(), *, existing_teams=(), rng=None):
    """Allocate overlapping configured players once, retaining saved owners."""
    priority = set(priority_names)
    claims = {}
    for club in clubs:
        for player in club.players:
            claims.setdefault(player.name, []).append(club.id)
    saved_owners = {p.name: c.id for c in existing_teams for p in c.players}
    owners = {}
    shared = {name for name, ids in claims.items() if len(ids) > 1}
    for name, ids in claims.items():
        if name in priority:
            owners[name] = None
        elif saved_owners.get(name) in ids:
            owners[name] = saved_owners[name]
        elif len(ids) == 1:
            owners[name] = ids[0]
        else:
            if rng is None:
                rng = Random()
            owners[name] = rng.choice(ids)
    result = []
    for club in clubs:
        players = tuple(p for p in club.players if owners[p.name] == club.id)
        members = {p.name for p in players}
        removed = set(club.members) - members
        regular = tuple(n for n in club.regular_members if n not in removed)
        starters = {p.name for p in players[:5]}
        result.append(replace(
            club, players=players,
            contracts=tuple(replace(c, end_reason=c.end_reason or "released")
                            if c.player_name in removed else c for c in club.contracts),
            acquired_members=tuple(n for n in club.acquired_members if n in members),
            regular_members=regular,
            igl=club.igl if club.igl in starters else None,
            carrier=club.carrier if club.carrier in starters else None,
            regular_igl=club.regular_igl if club.regular_igl in regular[:5] else None,
            regular_carrier=club.regular_carrier if club.regular_carrier in regular[:5] else None,
            initial_shared_members=tuple(dict.fromkeys(
                [n for n in club.initial_shared_members if n not in removed]
                + [p.name for p in players if p.name in shared])),
        ))
    return tuple(result)


def with_randomized_clubs(state, rng=None):
    """Replace 1–3 starters per NPC club, preserving unique ownership."""
    from realtime_season import initial_contract

    state = remember_loyalties(state)
    rng = rng or Random()
    retained = []
    slots = []
    removed = []
    for club in state.opponent_teams:
        eligible = [i for i, p in enumerate(club.players[:5]) if p.name not in club.initial_shared_members]
        indices = rng.sample(eligible, min(rng.choice((1, 2, 3)), len(eligible)))
        players = list(club.players)
        for index in indices:
            removed.append((club.id, players[index]))
            players[index] = None
            slots.append((len(retained), index))
        retained.append((club, players))
    occupied = {p.name for p in state.owned_players}
    occupied.update(p.name for _, players in retained for p in players if p is not None)
    originals = {p.name for _, p in removed}
    pool = [state.salary_player(p) for p in all_characters() if state.player_available(p) and p.name not in occupied | originals]
    rng.shuffle(slots)
    for club_index, slot in slots:
        club, players = retained[club_index]
        # Some originals move to another team; the rest remain in LFT.
        swaps = [p for owner, p in removed if owner != club.id and p.name not in occupied]
        candidates = swaps if swaps and rng.random() < .5 else [p for p in pool if p.name not in occupied]
        if not candidates:
            candidates = swaps
        if not candidates:
            # Small custom catalogs may not have enough replacement players.
            candidates = [p for _, p in removed if p.name not in occupied]
        player = rng.choice(candidates)
        players[slot] = player
        occupied.add(player.name)
    clubs = []
    for club, players in retained:
        roster = {p.name for p in players[:5]}
        clubs.append(replace(club, players=tuple(players), regular_members=club.members,
            regular_igl=club.igl, regular_carrier=club.carrier,
            igl=club.igl if club.igl in roster else None,
            carrier=club.carrier if club.carrier in roster else None,
            contracts=tuple(replace(initial_contract(p, state.game_month, signed_on=state.date.isoformat()),
                                    team_loyalty=state.team_loyalty(p.name, club.id)
                                    if signing_count(state, p.name, club.id) else initial_contract(p).team_loyalty,
                                    signing_number=max(1, signing_count(state, p.name, club.id))) for p in players)))
    return remember_loyalties(replace(state, opponent_teams=tuple(clubs)))


def restore_regular_members(state, reserved, emit):
    """Return available originals in a batch so cross-team swaps cannot duplicate players."""
    from realtime_season import initial_contract
    from season.season_monthly_events import player_strength
    from season.season_rival_economy import can_sign, make_offer, signing_terms, signed_contract, transfer_cost

    state = remember_loyalties(state)
    clubs = list(state.opponent_teams)
    snapshots = {p.name: p for p in state.scout_players}
    destinations = {}
    incoming_contracts = {}
    for original in clubs:
        club = next(c for c in state.opponent_teams if c.id == original.id)
        for name in club.regular_members:
            if state.scout_blocked(club.id):
                break
            if name in club.members:
                continue
            if state.contract_refused(name, club.id):
                continue
            if state.player(name) is not None:
                state = make_offer(state, club, state.player(name), emit)
                continue
            # Preserve active tournament lineups before charging any money.
            if name in reserved or name not in snapshots or any(p.name in reserved for p in club.players[:5]):
                continue
            player = snapshots[name]
            terms = signing_terms(state, player)
            owner = state.opponent_owner(name)
            fee = transfer_cost(player, owner.transfer_multiplier) if owner else 0
            if not can_sign(state, club, terms, fee):
                continue
            destinations[name] = club.id
            incoming_contracts[name] = signed_contract(player, terms, state.game_month, state=state, team_id=club.id)
            state = state._with_scout_use(club.id)
            state = replace(state, opponent_teams=tuple(
                replace(c, money=c.money - fee - terms.signing_bonus) if c.id == club.id else
                replace(c, money=c.money + fee) if owner and c.id == owner.id else c
                for c in state.opponent_teams))
            club = next(c for c in state.opponent_teams if c.id == club.id)
    clubs = list(state.opponent_teams)
    for index, club in enumerate(clubs):
        players = [p for p in club.players if p.name not in destinations or destinations[p.name] == club.id]
        incoming = [snapshots[name] for name, target in destinations.items() if target == club.id]
        players.extend(p for p in incoming if p.name not in {q.name for q in players})
        regular = {name: i for i, name in enumerate(club.regular_members)}
        if (club.regular_members or club.acquired_members) and not any(p.name in reserved for p in club.players[:5]):
            players.sort(key=(lambda p: -player_strength(p)) if club.acquired_members
                         else (lambda p: regular.get(p.name, len(regular))))
            # Keep substitutes only until their original slot is restored.
            limit = max(5, len(club.regular_members))
            players = [p for i, p in enumerate(players) if i < limit or p.name in reserved
                       or p.name in club.acquired_members or p.name in incoming_contracts
                       or club.acquired_members and p.name in regular]
        members = {p.name for p in players}
        contracts = {c.player_name: replace(c, end_reason="released") if c.player_name not in members and c.end_reason is None else c
                     for c in club.contracts}
        for player in players:
            old = contracts.get(player.name)
            if player.name in incoming_contracts:
                contracts[player.name] = incoming_contracts[player.name]
            elif old is None or old.end_reason is not None:
                contracts[player.name] = signed_contract(player, signing_terms(state, player), state.game_month,
                                                        state=state, team_id=club.id)
        roster = {p.name for p in players[:5]}
        clubs[index] = replace(club, players=tuple(players), contracts=tuple(contracts.values()),
            acquired_members=tuple(n for n in club.acquired_members if n in members),
            igl=club.regular_igl if club.regular_igl in roster else club.igl if club.igl in roster else None,
            carrier=club.regular_carrier if club.regular_carrier in roster else club.carrier if club.carrier in roster else None)
        for player in incoming:
            emit("roster_return", club, player.name, f"正規メンバーの{player.name}が復帰しました。")
    return remember_loyalties(replace(state, opponent_teams=tuple(clubs)))
