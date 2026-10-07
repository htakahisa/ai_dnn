"""Shared contract costs and deterministic, budget-aware rival decisions."""

from dataclasses import replace
from decimal import Decimal, ROUND_CEILING
from random import Random

import realtime_season_rival_economy as settings


def transfer_cost(player, multiplier):
    return int((Decimal(player.monthly_salary) * Decimal(str(multiplier)))
               .to_integral_value(rounding=ROUND_CEILING))


def signing_terms(state, player):
    return state.contract_terms(player, "short" if player.loyalty == 0 else "year1", 6)


def offer_terms(state, offer):
    from realtime_season import ContractTerms

    if offer.contract_kind is not None:
        return ContractTerms(offer.contract_kind, offer.contract_months, offer.monthly_salary)
    return signing_terms(state, state.player(offer.player_name))


def reserved_funds(state, club, *, exclude=None):
    # A rejected offer can still be forced by loyalty, so keep its budget too.
    latest = {o.player_name: o for o in state.transfer_offers}
    return sum(o.fee + offer_terms(state, o).total_required_funds for o in latest.values()
               if o.team_id == club.id and o.id != exclude and o.status in ("pending", "rejected")
               and state.player(o.player_name) is not None)


def can_sign(state, club, terms, fee=0, *, exclude=None, surplus=0):
    return club.money - reserved_funds(state, club, exclude=exclude) >= fee + terms.total_required_funds + surplus


def signed_contract(player, terms, month, loyalty=None, *, state=None, team_id=None):
    from realtime_season import PlayerContract
    from season.season_loyalty import CONTRACT_LOYALTY, signing_loyalty

    count = 1
    if state is not None:
        loyalty, count = signing_loyalty(state, player.name, team_id, terms.kind)
    elif loyalty is None:
        loyalty = CONTRACT_LOYALTY[terms.kind][0]

    return PlayerContract(player.name, terms.kind, terms.monthly_salary, month, terms.months, loyalty,
                          signing_number=count, signed_on=state.date.isoformat() if state is not None else None)


def make_offer(state, club, player, emit, *, surplus=0):
    from season.season_transfers import TransferOffer

    if state.contract_refused(player.name, club.id):
        return state
    remaining = state.scout_remaining(club.id)
    reserved = sum(o.team_id == club.id and o.status in ("pending", "rejected")
                   and state.player(o.player_name) is not None for o in state.transfer_offers)
    if remaining is not None and remaining <= reserved:
        return state

    if any(o.status == "pending" and o.player_name == player.name for o in state.transfer_offers):
        return state
    if any(o.team_id == club.id and o.player_name == player.name and o.created_month == state.game_month
           for o in state.transfer_offers):
        return state
    terms = signing_terms(state, player)
    fee = transfer_cost(player, state.transfer_multiplier)
    # Replace this player's old rejected proposal, releasing that reservation.
    offers = tuple(replace(o, status="cancelled") if o.player_name == player.name and o.status == "rejected" else o
                   for o in state.transfer_offers)
    candidate = replace(state, transfer_offers=offers)
    if not can_sign(candidate, club, terms, fee, surplus=surplus):
        return state
    offer = TransferOffer(f"{state.game_month}:{club.id}:{player.name}", club.id, player.name,
                          state.game_month, fee, contract_kind=terms.kind,
                          contract_months=terms.months, monthly_salary=terms.monthly_salary)
    emit("transfer_offer", club, player.name,
         f"{player.name}に移籍金{fee:,}円のオファーを送りました。忠誠が30未満になると強制成立します。")
    return replace(candidate, transfer_offers=(*offers, offer))


def opportunistic_offers(state, emit):
    from season.season_monthly_events import player_strength

    for club in state.opponent_teams:
        # Regular-member returns have priority; one extra proposal per month.
        if any(o.team_id == club.id and (o.status == "pending" or o.created_month == state.game_month)
               for o in state.transfer_offers):
            continue
        if Random(f"rival-offer:{club.id}:{state.game_month}").random() >= settings.NON_REGULAR_OFFER_CHANCE:
            continue
        weakest = min((player_strength(p) for p in club.players[:5]), default=0)
        candidates = [p for p in state.owned_players if p.name not in club.regular_members
                      and state.can_play(p.name) and player_strength(p) > weakest
                      and not any(o.status == "pending" and o.player_name == p.name for o in state.transfer_offers)]
        surplus = sum(c.monthly_salary for c in club.contracts if state.contract_usable(c, club.id)) * settings.OFFER_SURPLUS_MONTHS
        for player in sorted(candidates, key=lambda p: (-player_strength(p), p.name)):
            candidate = make_offer(state, club, player, emit, surplus=surplus)
            if candidate is not state:
                state = candidate
                break
    return state


def monthly_settlement(state, *, pay_salaries=True):
    from season.season_world_levels import world_level_for_rank

    ranking = state.rating_ranking
    ranks = {r.team_id: i for i, r in enumerate(ranking, 1)}
    clubs = []
    for club in state.opponent_teams:
        level = club.world_level_lock if club.world_level_lock is not None and state.active_tournaments else world_level_for_rank(ranks[club.id], len(ranking))
        income = level.sponsor_funds if club.sponsor_active else 0
        payroll = sum(c.monthly_salary for c in club.contracts if state.contract_usable(c, club.id)) if pay_salaries else 0
        clubs.append(replace(club, money=club.money + income - payroll))
    return replace(state, opponent_teams=tuple(clubs))
