"""Offline FRC diagnostics. Replay positions are never provided to the live AI."""
import argparse
from collections import Counter
import json
from pathlib import Path


def classify(frames):
    frames=sorted((f for f in frames if not f.get("setup")),key=lambda f:f["tick"])
    if not frames:
        return {"category":"missing_replay"}
    last=frames[-1]
    meta=last.get("gc_attacker_v2",{})
    selected=meta.get("selected_site")
    peaks=meta.get("max_observed_defenders_by_axis",{})
    events=meta.get("events",[])
    casts=[e for e in events if e.get("type")=="recon_cast"]
    allies=[c for c in last["chars"] if c["team"]=="A" and c["alive"]]
    carriers=[c for c in allies if c.get("has_spike")]
    first_selected=next((f["tick"] for f in frames
                         if f.get("gc_attacker_v2",{}).get("selected_site")),None)
    if not carriers:
        category="carrier_lost_or_recovery"
    elif selected is None:
        if len(casts)<4:
            category="unconfirmed_incomplete_recon"
        elif max(peaks.get("A",0),peaks.get("B",0))>=3 and min(peaks.get("A",0),peaks.get("B",0))<2:
            category="unconfirmed_other_site_underobserved"
        elif peaks.get("A",0)>2 and peaks.get("B",0)>2:
            category="unconfirmed_both_sites_crowded"
        else:
            category="unconfirmed_no_current_two"
    else:
        carrier=carriers[0]
        recent=[f for f in frames if f["tick"]>=last["tick"]-15]
        positions={tuple(c["pos"]) for f in recent for c in f["chars"]
                   if c["name"]==carrier["name"] and c["team"]=="A"}
        target=meta.get("target_plant_pos")
        if target and tuple(carrier["pos"])==tuple(target):
            category="selected_waiting_plant_support"
        elif len(positions)<=2:
            category="selected_movement_stalled"
        else:
            category="selected_late_or_fighting"
    return dict(category=category,selected_site=selected,first_selected_tick=first_selected,
                final_tick=last["tick"],observed_peak=peaks,observed_last=meta.get("observed_defenders_by_axis"),
                stale_reason=meta.get("site_reason"),recon_casts=casts,
                surviving_allies=[dict(name=c["name"],pos=c["pos"],ability=c["ability"],
                                      charges=c["ability_charges"],carrier=c.get("has_spike",False)) for c in allies])


def analyze(paths):
    rounds=[]
    for path in paths:
        data=json.loads(Path(path).read_text(encoding="utf-8"))
        for mi,mp in enumerate(data["maps"]):
            for record in mp["round_records"]:
                if (record["round_number"]>24 or record["reason"]!="time_expired" or record["planted"]
                        or not any(p.get("team")=="Ghost Champions" and p.get("side")=="attacker"
                                   for p in record.get("players",{}).values())):
                    continue
                frames=[f for f in mp.get("replay_frames",[]) if f["round"]==record["round_number"]]
                rounds.append(dict(file=Path(path).name,map=mi+1,round=record["round_number"],
                                   actual_initial_setup=record.get("tactic",{}).get("defender_initial_setup"),
                                   **classify(frames)))
    return dict(timeout_count=len(rounds),categories=dict(Counter(r["category"] for r in rounds)),rounds=rounds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(sorted(args.folder.glob("series_FRC_*.json")))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="rounds"},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
