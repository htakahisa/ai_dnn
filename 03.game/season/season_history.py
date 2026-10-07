"""Persisted season snapshots and a rebuildable Japanese JSON history export."""

from dataclasses import replace
from datetime import date
import json
import math
import os
from pathlib import Path
import tempfile


def cash_item(category, amount, player=None):
    item = {"内訳": category, "金額": amount}
    if player is not None:
        item["選手"] = player
    return item


def snapshot(state):
    standings = []
    for run in state.tournaments:
        event = state.tournament_definition(run.tournament_id)
        names = {team.id: team.name for team in run.entrants}
        rank = run.ranking.index(run.own_team_id) + 1 if run.own_team_id in run.ranking else None
        standings.append({"大会ID": run.tournament_id, "大会名": event.display_name,
                          "状態": "不参加" if run.declined else "終了" if run.completed else "進行中",
                          "順位": rank, "順位表": [names[team_id] for team_id in run.ranking],
                          "受取賞金": run.prize_paid})
    clubs = {club.id: club.name for club in state.opponent_teams}
    affiliations = {p.name: club.name for club in state.opponent_teams for p in club.players}
    affiliations.update((p.name, state.team_name) for p in state.owned_players)
    affiliation_ids = {p.name: club.id for club in state.opponent_teams for p in club.players}
    affiliation_ids.update((p.name, state.club_id) for p in state.owned_players)
    rank, count = state.world_rank
    return {"ゲーム内日付": state.date.isoformat(), "チーム名": state.team_name,
            "資金": state.money, "月給総額": state.monthly_payroll,
            "レート": state.rating(state.club_id), "世界レベル": state.world_level,
            "世界順位": rank, "チーム数": count, "大会中の世界レベル固定": bool(state.active_tournaments),
            "月額スポンサー収入": state.monthly_sponsor_income, "大会順位": standings,
            "所属": affiliations, "所属チームID": affiliation_ids,
            "オファー": {offer.id: {"選手": offer.player_name, "相手チーム": clubs.get(offer.team_id, offer.team_id),
                                  "状態": offer.status, "移籍金": offer.fee} for offer in state.transfer_offers}}


def record_state(state, kind, *, income=(), expenses=()):
    current = snapshot(state)
    previous = state.history[-1] if state.history else None
    incoming, outgoing = list(income), list(expenses)
    # Explicit administrative changes are distinguishable from game transactions.
    if previous is not None:
        difference = current["資金"] - previous["資金"] - sum(i["金額"] for i in incoming) + sum(i["金額"] for i in outgoing)
        if difference:
            (incoming if difference > 0 else outgoing).append(cash_item("資金調整（詳細不明）", abs(difference)))
    transfers = []
    if previous is not None:
        before, after = previous["所属"], current["所属"]
        for name in sorted(before.keys() | after.keys()):
            origin, destination = before.get(name, "LFT"), after.get(name, "LFT")
            if previous["所属チームID"].get(name) != current["所属チームID"].get(name):
                transfers.append({"選手": name, "移籍元": origin, "移籍先": destination,
                                  "種別": "加入" if origin == "LFT" else "退団" if destination == "LFT" else "移籍"})
        statuses = {"pending": "オファー到着", "accepted": "オファー承認", "rejected": "オファー拒否",
                    "forced": "強制移籍成立", "cancelled": "オファー取消"}
        for offer_id, offer in current["オファー"].items():
            if offer != previous["オファー"].get(offer_id):
                transfers.append({"種別": statuses[offer["状態"]], "オファーID": offer_id, **offer})
    entry = {"番号": len(state.history) + 1, "種別": kind, **current,
             "収入": incoming, "支出": outgoing, "移籍ログ": transfers}
    if kind in ("月次決算", "大会参加登録", "大会自動開催（自チーム不参加）"):
        from season.season_pair_familiarity import history_metrics
        run = state.tournaments[-1] if kind != "月次決算" else None
        entry["ペア練度"] = history_metrics(state, run)
    return replace(state, history=(*state.history, entry))


def validate_history(history):
    if not isinstance(history, tuple):
        raise ValueError("シーズン履歴の形式が不正です。")
    previous_date = None
    required = {"番号", "種別", "ゲーム内日付", "チーム名", "資金", "月給総額", "レート", "世界レベル",
                "世界順位", "チーム数", "大会中の世界レベル固定", "月額スポンサー収入", "大会順位",
                "所属", "所属チームID", "オファー", "収入", "支出", "移籍ログ"}
    for number, entry in enumerate(history, 1):
        if (not isinstance(entry, dict) or not required.issubset(entry)
                or set(entry) - required - {"ペア練度"}
                or type(entry["番号"]) is not int or entry["番号"] != number):
            raise ValueError("シーズン履歴の番号または項目が不正です。")
        day = date.fromisoformat(entry["ゲーム内日付"])
        if previous_date is not None and day < previous_date:
            raise ValueError("シーズン履歴の日付が前後しています。")
        previous_date = day
        for key in ("資金", "月給総額", "月額スポンサー収入", "世界レベル", "世界順位", "チーム数"):
            if type(entry[key]) is not int or key != "資金" and entry[key] < (1 if key in ("世界レベル", "世界順位", "チーム数") else 0):
                raise ValueError("シーズン履歴の数値が不正です。")
        if type(entry["レート"]) not in (int, float) or not math.isfinite(entry["レート"]) or entry["レート"] < 0:
            raise ValueError("シーズン履歴のレートが不正です。")
        for key in ("種別", "チーム名"):
            if not isinstance(entry[key], str) or not entry[key]:
                raise ValueError("シーズン履歴の名称が不正です。")
        if (type(entry["大会中の世界レベル固定"]) is not bool
                or any(not isinstance(entry[key], list) for key in ("収入", "支出", "移籍ログ", "大会順位"))
                or any(not isinstance(entry[key], dict) for key in ("所属", "所属チームID", "オファー"))):
            raise ValueError("シーズン履歴の内訳が不正です。")
        if any(not isinstance(name, str) or not isinstance(team, str) or not name or not team
               for key in ("所属", "所属チームID") for name, team in entry[key].items()):
            raise ValueError("シーズン履歴の所属が不正です。")
        for offer_id, offer in entry["オファー"].items():
            if (not isinstance(offer_id, str) or not isinstance(offer, dict)
                    or set(offer) != {"選手", "相手チーム", "状態", "移籍金"}
                    or any(not isinstance(offer[key], str) or not offer[key] for key in ("選手", "相手チーム"))
                    or offer["状態"] not in ("pending", "accepted", "rejected", "forced", "cancelled")
                    or type(offer["移籍金"]) is not int or offer["移籍金"] < 0):
                raise ValueError("シーズン履歴のオファーが不正です。")
        if any(not isinstance(item, dict) or not isinstance(item.get("種別"), str) for item in entry["移籍ログ"]):
            raise ValueError("シーズン履歴の移籍ログが不正です。")
        for standing in entry["大会順位"]:
            if (not isinstance(standing, dict) or set(standing) != {"大会ID", "大会名", "状態", "順位", "順位表", "受取賞金"}
                    or standing["状態"] not in ("不参加", "終了", "進行中")
                    or standing["順位"] is not None and (type(standing["順位"]) is not int or standing["順位"] < 1)
                    or not isinstance(standing["順位表"], list)
                    or any(not isinstance(name, str) for name in standing["順位表"])
                    or type(standing["受取賞金"]) is not int or standing["受取賞金"] < 0):
                raise ValueError("シーズン履歴の大会順位が不正です。")
        for item in (*entry["収入"], *entry["支出"]):
            if (not isinstance(item, dict) or not isinstance(item.get("内訳"), str)
                    or type(item.get("金額")) is not int or item["金額"] < 0):
                raise ValueError("シーズン履歴の収支が不正です。")
        if "ペア練度" in entry:
            rows = entry["ペア練度"]
            if not isinstance(rows, list):
                raise ValueError("履歴のペア練度一覧が不正です。")
            for row in rows:
                if (not isinstance(row, dict) or set(row) != {"チームID", "チーム名", "スタメン", "C", "M", "適用倍率"}
                        or any(not isinstance(row[k], str) or not row[k] for k in ("チームID", "チーム名"))
                        or not isinstance(row["スタメン"], list)
                        or any(not isinstance(n, str) for n in row["スタメン"])
                        or row["C"] is not None and (type(row["C"]) not in (int, float) or not 0 <= row["C"] <= 1)
                        or any(type(row[k]) not in (int, float) or not math.isfinite(row[k]) or row[k] <= 0
                               for k in ("M", "適用倍率"))):
                    raise ValueError("履歴のチーム練度・倍率が不正です。")
        json.dumps(entry, ensure_ascii=False, allow_nan=False)


def export_history(path, state):
    income, expenses, transfers = {}, {}, []
    entries = state.history or record_state(state, "記録開始").history
    for entry in entries:
        for key, totals in (("収入", income), ("支出", expenses)):
            for item in entry[key]:
                totals[item["内訳"]] = totals.get(item["内訳"], 0) + item["金額"]
        transfers.extend({"番号": entry["番号"], "ゲーム内日付": entry["ゲーム内日付"], **item}
                         for item in entry["移籍ログ"])
    payload = {"説明": "記録開始以降の推移。月給総額は自チームの有効な契約の月給合計。進行中の大会順位は未確定。旧セーブの過去の収支は復元しません。",
               "収入累計": sum(income.values()), "支出累計": sum(expenses.values()),
               "収入の内訳": income, "支出の内訳": expenses, "移籍ログ": transfers,
               "履歴": list(entries)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
