"""Load and validate tunable attacker parameters without changing v1 globals."""
import copy
import json
import math
import os
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("gc_profiles.json")
STAGES = ("baseline", "hold", "recon", "profiles", "entry")
FLAG_STAGE = dict(post_plant_hold="hold", early_recon="recon",
                  opponent_profiles="profiles", entry_discipline="entry")


def load_config(path=None):
    path = Path(path or os.environ.get("GC_V2_CONFIG") or DEFAULT_CONFIG)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_config(data)
    return data


def validate_config(data):
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported GC v2 config schema")
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
    for name,profile in data["profiles"].items():
        for flag in ("complete_site_survey","carrier_route_priority"):
            if flag in profile and not isinstance(profile[flag],bool):
                raise ValueError(f"Invalid {name}.{flag}")
        for group,values in profile.get("recon",{}).items():
            if group not in ("waypoints","aims","secondary_waypoints","secondary_aims") or not isinstance(values,dict):
                raise ValueError(f"Invalid {name}.recon.{group}")
            for axis,point in values.items():
                if axis not in ("A","Mid","B") or len(point)!=2 or any(not isinstance(v,int) for v in point):
                    raise ValueError(f"Invalid {name}.recon.{group}.{axis}")


def active_flags(config, stage, profile):
    if stage not in STAGES:
        raise ValueError(f"Unknown GC v2 evaluation stage: {stage}")
    flags = copy.deepcopy(config["flags"])
    for name, introduced in FLAG_STAGE.items():
        flags[name] = (flags[name] and STAGES.index(stage) >= STAGES.index(introduced))
    # Profiles cannot silently re-enable an ablated/global-disabled feature.
    for name in flags:
        if name in profile:
            flags[name] = flags[name] and bool(profile[name])
    if profile.get("preserve_v1"):
        flags = dict.fromkeys(flags, False)
    return flags
