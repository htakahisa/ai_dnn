"""Shared tournament card drawing for season mode and competition manager."""

from tkinter import font as tkfont


class BracketRenderer:
    def __init__(self, canvas):
        self.canvas = canvas
        self.team_font = tkfont.Font(root=canvas, family="Yu Gothic UI", size=9)

    def draw(self, matches, on_select, *, own_team_id=None, team_labels=None,
             legend="緑: 勝者  /  灰: 敗者  /  青: 次の試合・試合中  /  カードをクリックで詳細",
             empty_message="大会を開始するとトーナメント表を表示します。"):
        canvas = self.canvas
        canvas.delete("all")
        canvas.configure(background="#0f172a", cursor="")
        if not matches:
            canvas.create_text(24, 24, text=empty_message, anchor="nw", fill="#cbd5e1")
            canvas.configure(scrollregion=(0, 0, 650, 180))
            return
        canvas.create_text(24, 14, anchor="nw", fill="#cbd5e1", font=("Yu Gothic UI", 10), text=legend)
        upper, lower, finals = {}, {}, []
        for match in matches:
            if match["bracket"] == "G":
                finals.append(match)
            else:
                groups = upper if match["bracket"] == "W" else lower
                groups.setdefault(match["round"], []).append(match)
        card_w, header_h, row_h, gap = 240, 24, 32, 18
        card_h = header_h + row_h * 2
        col_step = card_w + 100
        upper_start = 72
        upper_h = max((len(cards) for cards in upper.values()), default=1) * (card_h + gap) - gap
        lower_start = upper_start + upper_h + 74
        positions = {}
        for groups, start, name in ((upper, upper_start, "UPPER"), (lower, lower_start, "LOWER")):
            if not groups:
                continue
            canvas.create_text(24, start - 32, anchor="nw", text=name, font=("Arial", 12, "bold"), fill="#93c5fd")
            for round_number, cards in sorted(groups.items()):
                for index, match in enumerate(sorted(cards, key=lambda m: m.get("match", 0))):
                    positions[match["id"]] = (24 + (round_number - 1) * col_step, start + index * (card_h + gap))
        final_col = max(max(upper, default=0), max(lower, default=0))
        final_y = upper_start + upper_h // 2 if lower else upper_start + max(0, (upper_h - card_h) // 2)
        for index, match in enumerate(finals):
            positions[match["id"]] = (24 + final_col * col_step, final_y + index * (card_h + gap))
        by_id = {match["id"]: match for match in matches}
        # Draw connectors first so they cannot cover a card's text or click target.
        for match in matches:
            x, y = positions[match["id"]]
            for row in range(2):
                source_id = match.get(f"source{row + 1}")
                if source_id not in by_id:
                    continue
                source = by_id[source_id]
                outcome = match.get(f"source{row + 1}_outcome")
                if outcome is None:
                    team = match.get(f"team{row + 1}")
                    outcome = "loser" if team is not None and team == source.get("loser") else "winner"
                sx, sy = positions[source_id]
                source_y = sy + card_h / 2
                if source.get("winner") is not None:
                    winner_row = 0 if source["winner"] == source.get("team1") else 1
                    source_row = winner_row if outcome == "winner" else 1 - winner_row
                    source_y = sy + header_h + (source_row + .5) * row_h
                target_y = y + header_h + (row + .5) * row_h
                elbow = (sx + card_w + x) / 2 if x > sx + card_w else sx + card_w + 24
                canvas.create_line(sx + card_w, source_y, elbow, source_y, elbow, target_y, x - 3, target_y,
                                   fill="#64748b" if outcome == "winner" else "#a78bfa", width=2,
                                   arrow="last", dash=() if outcome == "winner" else (4, 3),
                                   tags=("connector", f"feed:{source_id}:{match['id']}:{row}"))
        labels = team_labels or {}
        for match in matches:
            x, y = positions[match["id"]]
            tag = f"match:{match['id']}"
            tags = (tag, f"bracket_match_{match['id']}")
            status = match.get("status", "pending")
            active = status in ("playing", "ready")
            title = {"lower_final": "LOWER FINAL", "grand_final": "GRAND FINAL"}.get(match.get("special"), match["id"])
            status_text = {"finished": "結果", "rating": "レート判定", "playing": "試合中",
                           "ready": "次の試合", "bye": "BYE", "pending": "未確定"}.get(status, "未確定")
            canvas.create_rectangle(x, y, x + card_w, y + header_h,
                                    fill="#1e3a8a" if active else "#111827", outline="#64748b", tags=tags)
            canvas.create_text(x + 8, y + header_h / 2, anchor="w", fill="#e2e8f0", text=f"{title} · {status_text}",
                               font=("Yu Gothic UI", 9, "bold"), tags=tags)
            winner = match.get("winner")
            finished = status in ("finished", "rating")
            for row in range(2):
                team = match.get(f"team{row + 1}")
                row_y = y + header_h + row * row_h
                owned = team is not None and team == own_team_id
                won = team is not None and team == winner
                fill = "#14532d" if won else "#343a46" if finished else "#172554" if active else "#1f2937"
                if team is not None:
                    name = labels.get(team, team)
                elif match.get(f"empty{row + 1}", False):
                    name = "BYE"
                elif match.get(f"source{row + 1}") in by_id:
                    outcome = match.get(f"source{row + 1}_outcome", "winner")
                    name = f"{match[f'source{row + 1}']} {'敗者' if outcome == 'loser' else '勝者'}"
                else:
                    name = "未確定"
                text = ("★ " if owned else "") + str(name)
                while len(text) > 1 and self.team_font.measure(text) > card_w - 54:
                    text = text[:-2] + "…"
                row_tags = (*tags, f"team:{team}") if team is not None else (*tags, "unresolved")
                canvas.create_rectangle(x, row_y, x + card_w, row_y + row_h, fill=fill,
                                        outline="#fbbf24" if owned else "#22c55e" if won else "#64748b",
                                        width=2 if owned else 1, tags=row_tags)
                canvas.create_text(x + 8, row_y + row_h / 2, anchor="w", text=text,
                                   font=self.team_font, fill="#f1f5f9" if not finished or won else "#9ca3af", tags=row_tags)
                value = str(match.get(f"team{row + 1}_wins", 0)) if finished or status == "playing" else "—"
                canvas.create_text(x + card_w - 12, row_y + row_h / 2, anchor="e", text=value,
                                   font=("Arial", 12, "bold"), fill="#f1f5f9", tags=tags)
            canvas.tag_bind(tag, "<Button-1>", lambda _event, match_id=match["id"]: on_select(match_id))
            canvas.tag_bind(tag, "<Enter>", lambda _event: canvas.configure(cursor="hand2"))
            canvas.tag_bind(tag, "<Leave>", lambda _event: canvas.configure(cursor=""))
        right = max(x for x, y in positions.values()) + card_w + 30
        bottom = max(y for x, y in positions.values()) + card_h + 30
        canvas.configure(scrollregion=(0, 0, right, bottom))


def bind_bracket_scroll(canvas):
    canvas.bind("<MouseWheel>", lambda event: canvas.yview_scroll(-int(event.delta / 120), "units"))
    canvas.bind("<Shift-MouseWheel>", lambda event: canvas.xview_scroll(-int(event.delta / 120), "units"))
