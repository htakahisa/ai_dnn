"""Initial NPC roster mixing and the return of each club's regular members."""

from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_CEILING
from random import Random

from character_stats import all_characters


@dataclass(frozen=True)
class TransferOffer:
    id: str
    team_id: str
    player_name: str
    created_month: int
    fee: int
    status: str = "pending"


def with_randomized_clubs(state, rng=None):
    """Replace 1–3 starters per NPC club, preserving unique ownership."""
    from realtime_season import initial_contract

    rng = rng or Random()
    retained = []
    slots = []
    removed = []
    for club in state.opponent_teams:
        indices = rng.sample(range(min(5, len(club.players))), min(rng.choice((1, 2, 3)), len(club.players)))
        players = list(club.players)
        for index in indices:
            removed.append((club.id, players[index]))
            players[index] = None
            slots.append((len(retained), index))
        retained.append((club, players))
    occupied = {p.name for p in state.owned_players}
    occupied.update(p.name for _, players in retained for p in players if p is not None)
    originals = {p.name for _, p in removed}
    pool = [state.salary_player(p) for p in all_characters() if p.name not in occupied | originals]
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
            contracts=tuple(initial_contract(p, state.game_month) for p in players)))
    return replace(state, opponent_teams=tuple(clubs))


def restore_regular_members(state, reserved, emit):
    """Return available originals in a batch so cross-team swaps cannot duplicate players."""
    from realtime_season import initial_contract

    clubs = list(state.opponent_teams)
    snapshots = {p.name: p for p in state.scout_players}
    destinations = {}
    offers = list(state.transfer_offers)
    for club in clubs:
        for name in club.regular_members:
            if name in club.members:
                continue
            if any(c.player_name == name and c.end_reason == "left" and c.team_loyalty <= 0 for c in club.contracts):
                continue
            if state.player(name) is not None:
                if not any(o.status == "pending" and o.player_name == name for o in offers):
                    fee = int((Decimal(state.player(name).monthly_salary) * Decimal(str(club.transfer_multiplier)))
                              .to_integral_value(rounding=ROUND_CEILING))
                    offer = TransferOffer(f"{state.game_month}:{club.id}:{name}", club.id, name, state.game_month, fee)
                    offers.append(offer)
                    emit("transfer_offer", club, name, f"{name}に移籍金{fee:,}円のオファーを送りました。忠誠が30未満になると強制成立します。")
                continue
            if name not in reserved and name in snapshots:
                destinations[name] = club.id
    # Preserve the active tournament rosters, including their current outsiders.
    destinations = {name: target for name, target in destinations.items()
                    if not any(p.name in reserved for c in clubs if c.id == target for p in c.players[:5])}
    for index, club in enumerate(clubs):
        players = [p for p in club.players if p.name not in destinations or destinations[p.name] == club.id]
        incoming = [snapshots[name] for name, target in destinations.items() if target == club.id]
        players.extend(p for p in incoming if p.name not in {q.name for q in players})
        regular = {name: i for i, name in enumerate(club.regular_members)}
        if club.regular_members and not any(p.name in reserved for p in club.players[:5]):
            players.sort(key=lambda p: regular.get(p.name, len(regular)))
            # Keep substitutes only until their original slot is restored.
            limit = max(5, len(club.regular_members))
            players = [p for i, p in enumerate(players) if i < limit or p.name in reserved]
        members = {p.name for p in players}
        contracts = {c.player_name: replace(c, end_reason="released") if c.player_name not in members and c.end_reason is None else c
                     for c in club.contracts}
        for player in players:
            old = contracts.get(player.name)
            if old is None or old.end_reason is not None:
                contracts[player.name] = initial_contract(player, state.game_month)
        roster = {p.name for p in players[:5]}
        clubs[index] = replace(club, players=tuple(players), contracts=tuple(contracts.values()),
            igl=club.regular_igl if club.regular_igl in roster else club.igl if club.igl in roster else None,
            carrier=club.regular_carrier if club.regular_carrier in roster else club.carrier if club.carrier in roster else None)
        for player in incoming:
            emit("roster_return", club, player.name, f"正規メンバーの{player.name}が復帰しました。")
    return replace(state, opponent_teams=tuple(clubs), transfer_offers=tuple(offers))
