"""Shared site-specific ability limits for base and battle training."""

DEFAULT_FLASH_DISTANCE_L = 6
DEFAULT_RECON_DISTANCE_L = 11
DEFAULT_SMOKE_DISTANCE_L = 16

DEFAULT_FLASH_DISTANCE_R = 7
DEFAULT_RECON_DISTANCE_R = 6
DEFAULT_SMOKE_DISTANCE_R = 11

DEFAULT_ABILITY_DISTANCES = {
    "L": dict(FLASH=DEFAULT_FLASH_DISTANCE_L, RECON=DEFAULT_RECON_DISTANCE_L, SMOKE=DEFAULT_SMOKE_DISTANCE_L),
    "R": dict(FLASH=DEFAULT_FLASH_DISTANCE_R, RECON=DEFAULT_RECON_DISTANCE_R, SMOKE=DEFAULT_SMOKE_DISTANCE_R),
}


def add_ability_arguments(parser):
    parser.add_argument("--ability-distance", type=int, help="legacy shared BFS limit; per-ability options override it")
    for name in ("flash", "recon", "smoke"):
        parser.add_argument(f"--{name}-distance", type=int, help=f"shared {name.upper()} BFS limit for both sites")
        for site, label in (("L", "left"), ("R", "right")):
            parser.add_argument(f"--{name}-distance-{site.lower()}", f"--{label}-{name}-distance", type=int,
                                dest=f"{name}_distance_{site.lower()}",
                                help=f"{label} site {name.upper()} BFS limit (default: {DEFAULT_ABILITY_DISTANCES[site][name.upper()]})")


def ability_distances_from_args(args):
    distances = {}
    for site in ("L", "R"):
        distances[site] = {}
        for name in ("FLASH", "RECON", "SMOKE"):
            candidates = (getattr(args, f"{name.lower()}_distance_{site.lower()}"),
                          getattr(args, f"{name.lower()}_distance"), args.ability_distance,
                          DEFAULT_ABILITY_DISTANCES[site][name])
            distances[site][name] = next(value for value in candidates if value is not None)
    return distances
