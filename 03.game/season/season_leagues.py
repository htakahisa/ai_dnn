"""Chapter configuration and save locations for real-time seasons only."""

from pathlib import Path


def configured_leagues():
    from realtime_season_leagues import LEAGUE_NAMES

    if (not isinstance(LEAGUE_NAMES, dict) or not {1, 2, 3}.issubset(LEAGUE_NAMES)
            or any(type(k) is not int or k < 1 for k in LEAGUE_NAMES)
            or any(not isinstance(name, str) or not name.strip() for name in LEAGUE_NAMES.values())):
        raise ValueError("LEAGUE_NAMES に1・2・3章のリーグ名を設定してください。追加章は1以上の整数で指定できます。")
    return {chapter: name.strip() for chapter, name in sorted(LEAGUE_NAMES.items())}


def validate_chapter(chapter):
    if type(chapter) is not int or chapter not in configured_leagues():
        raise ValueError("章はLEAGUE_NAMESに設定した章番号の整数にしてください。")


def validate_debut_chapter(chapter):
    if type(chapter) is not int or chapter < 1:
        raise ValueError("登場章は1以上の整数にしてください。")


def chapter_save_path(path, chapter):
    validate_chapter(chapter)
    path = Path(path)
    return path if chapter == 1 else path.with_name(f"{path.stem}_chapter{chapter}{path.suffix}")
