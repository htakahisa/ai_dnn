"""Human-readable public history and intended defensive allocation."""


def format_defender_status(snapshot):
    analysis = snapshot["analysis"]
    tendency = analysis["tendency"]
    attacks, rushes = tendency["planted_attacks"], tendency["observed_rushes"]
    allocation = snapshot["allocation"]
    source = {"round_history": "履歴", "current_contacts": "今ラウンドの目撃",
              "learned_with_contact": "目撃＋解析", "balanced": "均等", "existing_gc": "既存GC"}[snapshot["allocation_source"]]
    probabilities = analysis["probabilities"]
    deployment = "既存GCの配置（観測不足・偏り弱）" if snapshot["allocation_source"] == "existing_gc" else (
        f"配分目標 A{allocation['A']} / Mid{allocation['Mid']} / B{allocation['B']} ({source})")
    if snapshot.get("reinforcement"):
        deployment += f" 増援: {snapshot['reinforcement']}"
    return (f"履歴{tendency['rounds']}R: A設置{attacks['A']} / B設置{attacks['B']}"
            f"・集中攻撃 A{rushes['A']} / B{rushes['B']}\n"
            f"{deployment}"
            f"  解析 A {probabilities['A']:.0%} / B {probabilities['B']:.0%}"
            f"  交戦の向き補正 {snapshot.get('combat_facing_corrections', 0)}回"
            f"  煙内解除の誘導 {snapshot.get('covered_retake_commits', 0)}回")
