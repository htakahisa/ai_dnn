"""Power indices calculated from the season's saved player abilities."""

from dataclasses import asdict, dataclass

from character_stats import CharacterStats


@dataclass(frozen=True)
class SeasonPlayerStats(CharacterStats):
    """Season progress kept separately from the initial character catalog."""

    research_level: int = 0
    aim_lab_level: int = 0


def season_player(player):
    if isinstance(player, SeasonPlayerStats):
        return player
    return SeasonPlayerStats(**asdict(player))


def player_combat_power(player):
    from game_core import calculate_combat_power
    # CharacterStats rates are already fractions (1.0 = 100%).
    return calculate_combat_power(player.hs_pct, player.dodge_pct, player.iq,
                                  player.hit_pct, player.reaction)


def player_duel_power(player):
    from game_core import calculate_combat_power
    return calculate_combat_power(player.hs_pct, player.dodge_pct, 0,
                                  player.hit_pct, player.reaction)
