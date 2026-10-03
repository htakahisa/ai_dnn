"""Save-local Elo rules matching competition_manager's 1500-point scale."""

from dataclasses import dataclass

DEFAULT_TEAM_RATING = 1500.0
RATING_K_FACTOR = 40.0


@dataclass(frozen=True)
class SeasonRating:
    team_id: str
    team_name: str
    value: float = DEFAULT_TEAM_RATING


def series_ratings(before1, before2, wins1, wins2):
    # Identical expectation, margin multiplier, K, and zero floor to
    # run_competition_manager.TeamRatingStore.update_series.
    exponent = (before2 - before1) / 400.0
    expected1 = 0.0 if exponent > 308 else 1.0 / (1.0 + 10.0 ** exponent)
    multiplier = min(1.6, 1.0 + 0.6 * abs(wins1 - wins2) / max(1, wins1 + wins2))
    actual1 = 1.0 if wins1 > wins2 else 0.0
    delta = RATING_K_FACTOR * multiplier * (actual1 - expected1)
    return max(0.0, before1 + delta), max(0.0, before2 - delta)
