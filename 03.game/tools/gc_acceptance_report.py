"""Report the requested GC v2 gates from saved evaluations, without running games."""
import argparse
import json
from pathlib import Path


TARGETS = {"Touyama Gaming": .80, "Fnatic2023": .50, "Furina Classic": .40,
           "Omoko Gaming": .35, "Gorigons": .40, "SUPES": .40}


def report(results, baseline, complete):
    checks = []
    lines = ["# GC v2 acceptance results", "",
             f"Full evaluation sample complete: **{complete}**. "
             "Rates below are point estimates; uncertainty is shown separately.", "",
             "| Opponent | v1 attack | v2 attack W/n | Wilson 95% | Target | Result |",
             "|---|---:|---:|---|---:|---|"]

    def check(name, value, threshold, enough, minimum=True):
        passed = value is not None and (value >= threshold if minimum else value <= threshold)
        outcome = "PASS" if enough and passed else "FAIL" if enough else "INCOMPLETE"
        checks.append(dict(name=name, value=value, threshold=threshold,
                           minimum=minimum, sufficient_sample=enough, outcome=outcome))
        return outcome

    def rate(w, n):
        return f"{w}/{n} ({w/n:.1%})" if n else f"{w}/{n} (unavailable)"

    for name, target in TARGETS.items():
        r, b = results.get(name, {}), baseline.get(name, {})
        n, w = r.get("n", 0), r.get("wins", 0)
        ci = r.get("attack_wilson95")
        interval = f"{ci[0]:.1%}–{ci[1]:.1%}" if ci else "unavailable"
        outcome = check(f"attack:{name}", w/n if n else None, target, complete and n >= 100)
        lines.append(f"| {name} | {rate(b.get('wins',0),b.get('n',0))} | {rate(w,n)} | {interval} | {target:.0%} | {outcome} |")

    lines += ["", "## T2: postplant", "",
              "| Opponent | Postplant W/n (Wilson 95%) | ≥40% | +10 distance (n) | ≤6 | +20 distance (n) | ≤6 |",
              "|---|---|---|---:|---|---:|---|"]
    for name in ("Omoko Gaming", "Gorigons", "SUPES"):
        r = results.get(name, {})
        ci = r.get("postplant_wilson95")
        interval = f"{ci[0]:.1%}–{ci[1]:.1%}" if ci else "unavailable"
        enough = complete and r.get("n",0) >= 100
        post = check(f"postplant:{name}", r.get("postplant_rate"), .4, enough)
        cells = []
        for tick in (10,20):
            distance, n = r.get(f"distance_{tick}"), r.get(f"distance_{tick}_n",0)
            verdict = check(f"distance+{tick}:{name}", distance, 6, enough and n > 0, False)
            cells.extend([f"{distance:.2f} ({n})" if distance is not None else "unavailable", verdict])
        lines.append(f"| {name} | {rate(r.get('postplant_wins',0),r.get('plants',0))}; {interval} | {post} | " + " | ".join(cells) + " |")

    lines += ["", "## T3: recon", "",
              "| Opponent | All-round used/available | By tick 30 / available |",
              "|---|---:|---:|"]
    used, early, available = 0, 0, 0
    for name in TARGETS:
        r = results.get(name,{})
        u, e, a = r.get("recon_used",0), r.get("recon_used_by_30",0), r.get("recon_available",0)
        used, early, available = used+u, early+e, available+a
        lines.append(f"| {name} | {rate(u,a)} | {rate(e,a)} |")
    verdict = check("recon:all", used/available if available else None, .7, complete)
    early_verdict = check("recon:by30", early/available if available else None, .7, complete)
    lines += [f"| Total | {rate(used,available)}; {verdict} | {rate(early,available)}; {early_verdict} |", "",
              "TYG deliberately keeps the v1 policy, including its recon behavior.", "", "## T4–T5: site and entry", ""]
    frc, fnc, old_fnc = results.get("Furina Classic",{}), results.get("Fnatic2023",{}), baseline.get("Fnatic2023",{})
    two, two_three = frc.get("site_two",0), frc.get("site_two_or_three",0)
    verdict = check("FRC:two_player_site", two/two_three if two_three else None, .7,
                    complete and frc.get("n",0) >= 100 and two_three > 0)
    lines.append(f"- FRC two-player site / two-or-three-player site: {rate(two,two_three)}; target ≥70%: **{verdict}**.")
    lines.append(f"- FRC selected-site rounds / all attacker rounds: {frc.get('site_selected',0)}/{frc.get('n',0)}. Unselected rounds are excluded from the site-choice denominator.")
    wipes, n = frc.get("preplant_wipes",0), frc.get("n",0)
    verdict = check("FRC:preplant_wipes", wipes/n if n else None, .4, complete and n >= 100, False)
    lines.append(f"- FRC preplant wipes: {rate(wipes,n)}; target ≤40%: **{verdict}**.")
    lines.append(f"- FRC preplant timeouts: {rate(frc.get('preplant_timeouts',0),n)}. Reported separately so waiting is not mistaken for a survival improvement.")
    crowded, n = fnc.get("site_four_plus",0), fnc.get("n",0)
    old_rate = old_fnc.get("site_four_plus",0)/old_fnc["n"] if old_fnc.get("n") else None
    limit = old_rate/2 if old_rate is not None else 0
    verdict = check("FNC:four_plus_site", crowded/n if n else None, limit,
                    complete and n >= 100 and old_rate is not None, False)
    lines.append(f"- FNC attacks on four-or-more-player sites: {rate(crowded,n)}; half-baseline limit {limit:.2%}: **{verdict}**.")
    failures = [c["name"] for c in checks if c["outcome"]=="FAIL"]
    lines += ["", "## T6 decision", "",
              "Rule-only targets remain unmet: " + (", ".join(failures) if failures else "none confirmed") + ".",
              "RL training, BC recollection/retraining and PFSP have not been performed. "
              "These measurements are the required report before starting T6.", ""]
    return "\n".join(lines), checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation",type=Path)
    args=parser.parse_args()
    folder=args.evaluation
    read=lambda name: json.loads((folder/name).read_text(encoding="utf-8"))
    text,checks=report(read("summary.json"),read("baseline_summary.json"),read("status.json")["complete"])
    (folder/"acceptance.md").write_text(text,encoding="utf-8")
    (folder/"acceptance.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
    print(text)


if __name__=="__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
