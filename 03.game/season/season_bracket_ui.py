"""Season-mode adapter for the shared scrollable tournament cards."""

import tkinter as tk
from tkinter import ttk

from season.season_competitions import bracket_view
from tournament_bracket_ui import BracketRenderer, bind_bracket_scroll


class SeasonBracketPanel(ttk.Frame):
    def __init__(self, parent, on_select):
        super().__init__(parent)
        self.on_select = on_select
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, height=180, background="#0f172a", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(self, command=self.canvas.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        bind_bracket_scroll(self.canvas)
        self.renderer = BracketRenderer(self.canvas)
        self.cards = ()
        self.show(None, None)

    def show(self, event, run):
        self.cards = bracket_view(event, run) if event is not None and run is not None else ()
        teams = {t.id: t.name for t in run.entrants} if run else {}
        matches, details = [], {}
        lower_round = 0

        def slot_name(slot):
            if slot.team_id is not None:
                return teams[slot.team_id]
            return f"{slot.source_match} {'勝者' if slot.source_outcome == 'winner' else '敗者'}"

        for card in self.cards:
            match, score = card.match, card.score
            if match.stage == "grand_final":
                side, round_number, index = "G", 1, 1
            elif match.stage == "lower_final":
                side, round_number, index = "L", lower_round + 1, 1
            else:
                side = "W" if match.stage == "upper" else "L"
                round_part, index_part = match.id[1:].split("M")
                round_number, index = int(round_part), int(index_part)
                if side == "L":
                    lower_round = max(lower_round, round_number)
            winner = (score.left_id if score.left_wins > score.right_wins else score.right_id) if score else None
            status = "rating" if score and score.decided_by_rating else "finished" if score else "ready" if card.pending else "pending"
            matches.append({"id": match.id, "bracket": side, "round": round_number, "match": index,
                "special": match.stage, "team1": card.left.team_id, "team2": card.right.team_id,
                "source1": card.left.source_match, "source2": card.right.source_match,
                "source1_outcome": card.left.source_outcome, "source2_outcome": card.right.source_outcome,
                "team1_wins": score.left_wins if score else 0, "team2_wins": score.right_wins if score else 0,
                "winner": winner, "status": status})
            status_text = {"rating": "レート判定", "finished": "結果", "ready": "次の試合", "pending": "未確定"}[status]
            details[match.id] = f"{match.id}: {slot_name(card.left)} vs {slot_name(card.right)} — {status_text} / {match.maps_to_win}マップ先取"
        message = "この大会は不参加です。" if run and run.declined else "参加登録後にトーナメント表を表示します。"
        self.renderer.draw(matches, lambda match_id: self.on_select(details[match_id]),
                           own_team_id=run.own_team_id if run else None, team_labels=teams,
                           legend="緑: 勝者  /  灰: 敗者  /  青: 次の試合  /  ★: 自チーム  /  レート判定: シミュレーション省略",
                           empty_message=message)
