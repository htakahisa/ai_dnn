"""GC(など任意のチーム)のアタッカー側を、対戦相手ごとに診断するレポート。

使い方:
    py -3.10 gc_attack_report.py "series_Ghost_Champions_vs_*.json"
    py -3.10 gc_attack_report.py "series_*.json" --team "Ghost Champions"

前提: series形式のJSON(round_records と replay_frames が入っているもの)。
延長(25ラウンド目以降)はサイド交代の規則が不明なため、集計から除外します。
"""
import argparse
import collections
import glob
import json
import statistics as st


def mean(xs):
    xs = list(xs)
    return round(st.mean(xs), 2) if xs else None


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "-"


def analyze(paths, team):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    skipped = 0
    for path in paths:
        d = json.load(open(path, encoding="utf-8"))
        opp = d["team2"] if d["team1"] == team else d["team1"]
        for mp in d["maps"]:
            first_atk = mp["initial_attacker"]
            other = opp if first_atk == team else team
            frames_by_round = collections.defaultdict(list)
            for fr in mp.get("replay_frames", []):
                frames_by_round[fr["round"]].append(fr)
            for r in mp["round_records"]:
                rn = r["round_number"]
                if rn > 24:
                    skipped += 1
                    continue
                atk = first_atk if rn <= 12 else other
                if atk != team:
                    continue
                t = r["tactic"]
                a, m, b = map(int, t["defender_initial_setup"].split("-"))
                site = t["final_attack_site"]
                row = per[opp]
                row["win"].append(r["winner"] == "attacker")
                row["planted"].append(bool(r["planted"]))
                row["reason"].append(r["reason"])
                row["strategy"].append(t["attacker_strategy"])
                row["site"].append(site)
                row["setup"].append(t["defender_initial_setup"])
                row["def_at_site"].append(a if site == "A" else b)
                row["kills"].append(sum(p["kills"] for p in r["players"].values() if p["team"] == team))
                row["deaths"].append(sum(p["deaths"] for p in r["players"].values() if p["team"] == team))
                for name, p in r["players"].items():
                    if p["team"] == team:
                        row["first_death_by"].append(name) if p["first_deaths"] else None
                frames = [x for x in frames_by_round.get(rn, []) if not x["setup"]]
                if frames:
                    start = {c["name"]: c["ability_charges"] for c in frames[0]["chars"] if c["ability"] == "RECON" and c["team"] == "A"}
                    prev = {c["name"]: c["ability_charges"] for c in frames[0]["chars"]}
                    used = 0
                    for x in frames[1:]:
                        for c in x["chars"]:
                            if c["ability"] == "RECON" and c["team"] == "A":
                                if c["name"] in prev and c["ability_charges"] < prev[c["name"]]:
                                    used += prev[c["name"]] - c["ability_charges"]
                            prev[c["name"]] = c["ability_charges"]
                    row["recon_avail"].append(sum(start.values()))
                    row["recon_used"].append(used)
                    pk = next((x for x in frames if x["planted"]), None)
                    if pk:
                        row["atk_alive_at_plant"].append(sum(1 for c in pk["chars"] if c["team"] == "A" and c["alive"]))
                        row["def_alive_at_plant"].append(sum(1 for c in pk["chars"] if c["team"] == "D" and c["alive"]))
                        row["plant_tick"].append(pk["tick"])
    return per, skipped


def report(per, team):
    for opp, row in sorted(per.items()):
        n = len(row["win"])
        wins = sum(row["win"])
        planted = sum(row["planted"])
        won_after_plant = sum(w for w, p in zip(row["win"], row["planted"]) if p)
        defused = sum(1 for w, rs in zip(row["win"], row["reason"]) if not w and rs == "defused")
        print(f"\n===== vs {opp}  (attack rounds: {n}) =====")
        print(f"attack win      : {pct(wins, n)}")
        print(f"planted         : {pct(planted, n)}   post-plant win: {pct(won_after_plant, planted)}   defused by opp: {defused}")
        print(f"loss reasons    : {dict(collections.Counter(rs for w, rs in zip(row['win'], row['reason']) if not w))}")
        print(f"kills/deaths per round: {mean(row['kills'])} / {mean(row['deaths'])}")
        print(f"strategy        : {dict(collections.Counter(row['strategy']))}")
        print(f"attacked site   : {dict(collections.Counter(row['site']))}")
        print(f"opp def setups  : {collections.Counter(row['setup']).most_common(5)}")
        by_n = collections.defaultdict(lambda: [0, 0])
        for w, k in zip(row["win"], row["def_at_site"]):
            by_n[k][0] += 1
            by_n[k][1] += w
        print("win by # defenders at attacked site: " + ", ".join(f"{k}人 {v[1]}/{v[0]}" for k, v in sorted(by_n.items())))
        if row["recon_avail"]:
            av, us = sum(row["recon_avail"]), sum(row["recon_used"])
            print(f"recon usage     : {us}/{av:.0f} charges ({us / av:.0%})" if av else "recon usage     : no recon charges")
            if row["atk_alive_at_plant"]:
                print(f"alive at plant  : {team} {mean(row['atk_alive_at_plant'])} vs opp {mean(row['def_alive_at_plant'])}  (plant tick avg {mean(row['plant_tick'])})")
        fd = collections.Counter(row["first_death_by"])
        if fd:
            print(f"first deaths    : {dict(fd.most_common())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help='例: "series_Ghost_Champions_vs_*.json"')
    ap.add_argument("--team", default="Ghost Champions")
    args = ap.parse_args()
    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit("ファイルが見つかりません: " + args.pattern)
    per, skipped = analyze(paths, args.team)
    report(per, args.team)
    if skipped:
        print(f"\n(延長ラウンド {skipped} 件は除外)")
