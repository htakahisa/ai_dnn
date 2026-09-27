"""Validate map scores before exporting a series to the analysis dataset."""

import math


def series_export_skip_reason(series_data):
    """Return None when every recorded map has a team with at least 13 wins."""
    if not isinstance(series_data, dict):
        return "series data is not an object"
    maps = series_data.get("maps")
    if not isinstance(maps, list) or not maps:
        return "no recorded maps"
    reasons = []
    for index, data in enumerate(maps, 1):
        if not isinstance(data, dict):
            reasons.append(f"Map {index}: invalid map data")
            continue
        number = data.get("number", index)
        score1, score2 = data.get("score1"), data.get("score2")
        if not all(
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(score)
            and score >= 0
            and score == int(score)
            for score in (score1, score2)
        ):
            reasons.append(f"Map {number}: missing or invalid scores ({score1}-{score2})")
        elif score1 < 13 and score2 < 13:
            reasons.append(f"Map {number}: incomplete ({score1}-{score2}; neither team reached 13)")
    return "; ".join(reasons) if reasons else None
