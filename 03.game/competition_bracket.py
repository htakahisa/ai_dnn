"""Competition-manager bracket traversal and projection, without simulations."""


def double_elimination_bracket(slots, normal_need, lower_need, grand_need):
    """Yield the actual cards (including byes), accepting each winner via send."""
    def pairs(nodes):
        if len(nodes) % 2:
            raise RuntimeError(f"ブラケットノード数が偶数ではありません: {len(nodes)}")
        return zip(nodes[::2], nodes[1::2])

    def play(match_id, side, round_number, match_number, left, right, need, special=""):
        match = {"id": match_id, "bracket": side, "round": round_number, "match": match_number,
                 "team1": left["team"], "team2": right["team"], "source1": left["source"],
                 "source2": right["source"], "source1_outcome": left.get("outcome"),
                 "source2_outcome": right.get("outcome"), "maps_to_win": need, "special": special}
        winner = yield match
        if winner not in (left["team"], right["team"]) or winner is None and (left["team"] or right["team"]):
            raise ValueError("ブラケットの勝者が対戦チームに含まれていません。")
        loser = right["team"] if winner == left["team"] else left["team"]
        return ({"team": winner, "source": match_id, "outcome": "winner"},
                {"team": loser, "source": match_id, "outcome": "loser"})

    winners = [{"team": team, "source": f"SLOT-{i + 1}"} for i, team in enumerate(slots)]
    lower = []
    wr, lr = 1, 0
    while len(winners) > 1:
        next_winners, incoming = [], []
        for i, (left, right) in enumerate(pairs(winners), 1):
            winner, loser = yield from play(f"W{wr}M{i}", "W", wr, i, left, right, normal_need)
            next_winners.append(winner)
            if loser["team"] is not None:
                incoming.append(loser)
        incoming.reverse()
        if wr == 1:
            lr = 1
            if len(incoming) % 2:
                lower.append(incoming.pop(0))
            for i, (left, right) in enumerate(pairs(incoming), 1):
                winner, _ = yield from play(f"L{lr}M{i}", "L", lr, i, left, right, normal_need)
                lower.append(winner)
        else:
            for group, target in ((lower, incoming), (incoming, lower)):
                while len(group) > len(target):
                    lr += 1
                    count = min(len(group) // 2, len(group) - len(target))
                    if count == 0:
                        raise RuntimeError("Losersブラケットの人数調整に失敗しました")
                    carry = len(group) - count * 2
                    reduced = group[:carry]
                    for i, (left, right) in enumerate(pairs(group[carry:]), 1):
                        winner, _ = yield from play(f"L{lr}M{i}", "L", lr, i, left, right, normal_need)
                        reduced.append(winner)
                    group[:] = reduced
            if len(lower) != len(incoming):
                raise RuntimeError("Losersブラケットの人数調整に失敗しました")
            lr += 1
            merged = []
            for i, (survivor, dropped) in enumerate(zip(lower, incoming), 1):
                final = len(next_winners) == 1 and len(lower) == 1
                winner, _ = yield from play("LOWER_FINAL" if final else f"L{lr}M{i}", "L", lr, i,
                                            survivor, dropped, lower_need if final else normal_need,
                                            "lower_final" if final else "")
                merged.append(winner)
            lower = merged
        winners, wr = next_winners, wr + 1
    while len(lower) > 1:
        lr += 1
        next_lower = []
        paired = list(pairs(lower))
        for i, (left, right) in enumerate(paired, 1):
            final = len(paired) == 1
            winner, _ = yield from play("LOWER_FINAL" if final else f"L{lr}M{i}", "L", lr, i,
                                        left, right, lower_need if final else normal_need,
                                        "lower_final" if final else "")
            next_lower.append(winner)
        lower = next_lower
    if not lower:
        raise RuntimeError("Losersブラケット勝者を決定できません")
    winner, loser = yield from play("GRAND_FINAL", "G", 1, 1, winners[0], lower[0], grand_need, "grand_final")
    return winner["team"], loser["team"]


def project_bracket(setup, results):
    """Hide provisional future participants and show their feeder instead."""
    generator = double_elimination_bracket(setup["slots"], setup["normal_maps_to_win"],
                                          setup["lower_final_maps_to_win"], setup["grand_final_maps_to_win"])
    cards = []
    first_pending = True
    try:
        match = next(generator)
        while True:
            result = results.get(match["id"])
            finished = result is not None and result.get("status") in ("finished", "bye")
            winner = result.get("winner") if finished else match["team1"] or match["team2"]
            card = {**match, "winner": None, "loser": None, "status": "pending"}
            for row in (1, 2):
                source = match[f"source{row}"]
                card[f"empty{row}"] = match[f"team{row}"] is None
                if not source.startswith("SLOT-") and source not in results:
                    card[f"team{row}"] = None
                elif source in results and results[source].get("status") not in ("finished", "bye"):
                    card[f"team{row}"] = None
            if result is not None:
                card.update(result)
            elif first_pending:
                card["status"] = "ready"
            if not finished:
                first_pending = False
            cards.append(card)
            match = generator.send(winner)
    except StopIteration:
        return tuple(cards)
