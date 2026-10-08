"""Load and validate tunable attacker parameters without changing v1 globals."""
import copy
import json
import math
import os
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("gc_profiles.json")
STAGES = ("baseline", "hold", "recon", "sites", "entry")
FLAG_STAGE = dict(post_plant_hold="hold", early_recon="recon",
                  site_selection="sites", entry_discipline="entry")


def load_config(path=None):
    path = Path(path or os.environ.get("GC_V2_CONFIG") or DEFAULT_CONFIG)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_config(data)
    return data


def validate_config(data):
    if data.get("schema_version") != 2:
        raise ValueError("Unsupported GC v2 config schema")
    if "profiles" in data:
        raise ValueError("Opponent-specific tactical profiles are no longer supported")
    opening=data["opening"]
    weights=opening["weights"]
    if set(weights)!={"DEFAULT","RUSH","SPLIT"} or any(
            isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0
            for v in weights.values()) or sum(weights.values())<=0:
        raise ValueError("Opening weights must cover DEFAULT/RUSH/SPLIT and have a positive total")
    if (isinstance(opening["utility_distance"],bool) or not isinstance(opening["utility_distance"],(int,float))
            or not math.isfinite(opening["utility_distance"]) or opening["utility_distance"]<=0):
        raise ValueError("Opening utility distance must be finite and positive")
    for point in (opening["split_waypoint"],*opening["split_entries"].values()):
        if len(point)!=2 or any(type(v) is not int or v<0 for v in point):
            raise ValueError("Invalid opening waypoint")
    if set(opening["split_entries"])!={"A","B"}:
        raise ValueError("Split entries must cover A and B")
    for flag in FLAG_STAGE:
        if not isinstance(data["flags"][flag], bool):
            raise ValueError(f"{flag} must be boolean")
    for group in ("hold", "entry", "recon", "support"):
        for name, value in data[group].items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{group}.{name} must be finite and positive")
    hold = data["hold"]
    if not 0 < hold["minimum_radius"] <= hold["radius"] <= hold["leash_radius"]:
        raise ValueError("Hold radii must satisfy minimum <= radius <= leash")
    if not 0 < hold["crossfire_angle"] <= 180:
        raise ValueError("Crossfire angle must be in (0, 180]")
    if not isinstance(hold["responders"], int):
        raise ValueError("Responder count must be an integer")
    for group in ("waypoints", "aims", "secondary_waypoints", "secondary_aims"):
        for axis in ("A", "Mid", "B"):
            point = data["recon"][group][axis]
            if len(point) != 2 or any(not isinstance(v, int) for v in point):
                raise ValueError(f"Invalid recon {group}.{axis}")
    for axis in ("A", "B"):
        point = data["plant_targets"][axis]
        if len(point) != 2 or any(not isinstance(v,int) for v in point):
            raise ValueError(f"Invalid plant target {axis}")
    for flag in ("complete_site_survey","carrier_route_priority"):
        if not isinstance(data["team_tactics"][flag],bool):
            raise ValueError(f"Invalid team_tactics.{flag}")


def active_flags(config, stage):
    stage="sites" if stage=="profiles" else stage  # Historical evaluation-stage alias.
    if stage not in STAGES:
        raise ValueError(f"Unknown GC v2 evaluation stage: {stage}")
    flags = copy.deepcopy(config["flags"])
    for name, introduced in FLAG_STAGE.items():
        flags[name] = (flags[name] and STAGES.index(stage) >= STAGES.index(introduced))
    return flags
