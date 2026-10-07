"""Save-local Elo rules matching competition_manager's 1500-point scale."""

from dataclasses import dataclass

DEFAULT_TEAM_RATING = 1500.0
RATING_K_FACTOR = 40.0


@dataclass(frozen=True)
class SeasonRating:
    team_id: str
    team_name: str
    value: float = DEFAULT_TEAM_RATING


def expected_score(rating1, rating2):
    """Elo win probability, shared by series updates and NPC draws."""
    exponent = (rating2 - rating1) / 400.0
    return 0.0 if exponent > 308 else 1.0 / (1.0 + 10.0 ** exponent)


def series_ratings(before1, before2, wins1, wins2):
    # Identical expectation, margin multiplier, K, and zero floor to
    # run_competition_manager.TeamRatingStore.update_series.
    expected1 = expected_score(before1, before2)
    multiplier = min(1.6, 1.0 + 0.6 * abs(wins1 - wins2) / max(1, wins1 + wins2))
    actual1 = 1.0 if wins1 > wins2 else 0.0
    delta = RATING_K_FACTOR * multiplier * (actual1 - expected1)
    return max(0.0, before1 + delta), max(0.0, before2 - delta)
