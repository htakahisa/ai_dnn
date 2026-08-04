from __future__ import annotations
import contextlib, io, json, queue, random, threading, traceback
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from map_data import NEW_MAZE_STR
from party_presets import all_preset_names, get_preset
from run_game import VisualFPSBattle, _build_team_ai

AI_KEY = "fnatic_v1"
RESULT_DIR = Path("series_results")


@dataclass
class PStat:
    name: str
    kills: int
    deaths: int


@dataclass
class MResult:
    number: int
    team1: str
    team2: str
    score1: int
    score2: int
    winner: str
    mvp1: PStat
    mvp2: PStat
    initial_attacker: str
    overtime: bool


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def validate(p):
    if p is None or len(p.players) != 5:
        raise ValueError("無効なチームプリセットです")
    if p.igl not in p.players or p.spike_holder not in p.players:
        raise ValueError(f"{p.name}: IGLまたはスパイク担当が不正です")


def mvp_from_team_chars(team_chars):
    """試合開始時に確保したチーム固有のCharacter参照からMVPを決める。"""
    if not team_chars:
        raise RuntimeError("MVP対象のキャラクターが見つかりません")

    values = [
        PStat(
            str(getattr(c, "name", "?")),
            int(getattr(c, "kills", 0)),
            int(getattr(c, "deaths", 0)),
        )
        for c in team_chars
    ]
    return max(values, key=lambda x: (x.kills, -x.deaths))


def scores(game, t1, t2):
    a, d = set(game.attacker_roster or []), set(game.defender_roster or [])
    s1, s2 = set(t1), set(t2)
    if a == s1 and d == s2:
        return int(game.attacker_wins), int(game.defender_wins)
    if a == s2 and d == s1:
        return int(game.defender_wins), int(game.attacker_wins)
    raise RuntimeError("試合終了後のチームとスコアを対応付けられません")


def play_map(t1, t2, no, seed):
    seed_all(seed)
    ai1 = _build_team_ai(AI_KEY)
    ai2 = _build_team_ai(AI_KEY)
    if no % 2:
        a, d, aai, dai = t1, t2, ai1, ai2
    else:
        a, d, aai, dai = t2, t1, ai2, ai1
    with contextlib.redirect_stdout(io.StringIO()):
        g = VisualFPSBattle(
            NEW_MAZE_STR,
            aai,
            dai,
            headless=True,
            attacker_roster=list(a.players),
            defender_roster=list(d.players),
            spike_holder_name=a.spike_holder,
            defender_spike_holder_name=d.spike_holder,
            attacker_igl_name=a.igl,
            defender_igl_name=d.igl,
            disable_side_swap=False,
        )

        # サイドスワップ前に各チーム固有のCharacterオブジェクトを確保する。
        initial_attackers = [c for c in g.chars if c.team == "A"]
        initial_defenders = [c for c in g.chars if c.team == "D"]

        if a is t1:
            team1_chars = list(initial_attackers)
            team2_chars = list(initial_defenders)
        else:
            team1_chars = list(initial_defenders)
            team2_chars = list(initial_attackers)

        if len(team1_chars) != 5 or len(team2_chars) != 5:
            raise RuntimeError(
                "試合開始時のチームCharacter取得に失敗しました: "
                f"team1={len(team1_chars)}, team2={len(team2_chars)}"
            )

        g.run()
    s1, s2 = scores(g, t1.players, t2.players)
    return MResult(
        no,
        t1.name,
        t2.name,
        s1,
        s2,
        t1.name if s1 > s2 else t2.name,
        mvp_from_team_chars(team1_chars),
        mvp_from_team_chars(team2_chars),
        a.name,
        bool(getattr(g, "overtime", False)),
    )


def run_series(n1, n2, need, seed, events):
    if n1 == n2:
        raise ValueError("異なる2チームを選んでください")
    if not 1 <= need <= 7:
        raise ValueError("勝利マップ数は1～7です")
    t1, t2 = get_preset(n1), get_preset(n2)
    validate(t1)
    validate(t2)
    w1 = w2 = 0
    results = []
    while w1 < need and w2 < need:
        no = len(results) + 1
        events.put(("status", f"MAP {no} 実行中…"))
        r = play_map(t1, t2, no, seed + no - 1)
        results.append(r)
        if r.winner == t1.name:
            w1 += 1
        else:
            w2 += 1
        events.put(("map", r))
        events.put(("score", w1, w2))
    winner = t1.name if w1 > w2 else t2.name
    data = {
        "team1": t1.name,
        "team2": t2.name,
        "maps_to_win": need,
        "team1_wins": w1,
        "team2_wins": w2,
        "winner": winner,
        "maps": [asdict(r) for r in results],
    }
    RESULT_DIR.mkdir(exist_ok=True)
    safe = lambda s: "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
    path = RESULT_DIR / f"{safe(t1.name)}_vs_{safe(t2.name)}_{w1}-{w2}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    events.put(("done", data, str(path)))


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Headless Team Series")
        self.root.geometry("900x650")
        self.q = queue.Queue()
        names = all_preset_names()
        if len(names) < 2:
            raise RuntimeError("party_presets.pyに2チーム以上必要です")
        self.t1 = tk.StringVar(value=names[0])
        self.t2 = tk.StringVar(value=names[1])
        self.need = tk.IntVar(value=2)
        self.seed = tk.IntVar(value=42)
        self.status = tk.StringVar(value="設定を選択してください")
        self.ss = tk.StringVar(value="-")
        f = tk.LabelFrame(self.root, text="シリーズ設定", padx=10, pady=10)
        f.pack(fill="x", padx=12, pady=12)
        tk.Label(f, text="チーム1").grid(row=0, column=0)
        self.b1 = ttk.Combobox(
            f, values=names, textvariable=self.t1, state="readonly", width=28
        )
        self.b1.grid(row=0, column=1, padx=8)
        tk.Label(f, text="チーム2").grid(row=0, column=2)
        self.b2 = ttk.Combobox(
            f, values=names, textvariable=self.t2, state="readonly", width=28
        )
        self.b2.grid(row=0, column=3, padx=8)
        tk.Label(f, text="勝利に必要なマップ数").grid(row=1, column=0, pady=10)
        self.sp = tk.Spinbox(
            f, from_=1, to=7, textvariable=self.need, state="readonly", width=8
        )
        self.sp.grid(row=1, column=1, sticky="w", padx=8)
        tk.Label(f, text="Seed").grid(row=1, column=2)
        self.se = tk.Entry(f, textvariable=self.seed, width=10)
        self.se.grid(row=1, column=3, sticky="w", padx=8)
        bar = tk.Frame(self.root)
        bar.pack(fill="x", padx=12)
        self.start = tk.Button(
            bar, text="シリーズ開始", font=("Arial", 11, "bold"), command=self.go
        )
        self.start.pack(side="left")
        tk.Label(bar, text=f"使用AI: {AI_KEY}", fg="#555").pack(side="left", padx=16)
        tk.Label(bar, textvariable=self.ss, font=("Arial", 14, "bold")).pack(
            side="right"
        )
        tk.Label(self.root, textvariable=self.status, anchor="w", fg="#22577a").pack(
            fill="x", padx=12, pady=8
        )
        rf = tk.LabelFrame(self.root, text="結果", padx=8, pady=8)
        rf.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.text = tk.Text(rf, font=("Consolas", 10), state="disabled")
        sc = tk.Scrollbar(rf, command=self.text.yview)
        self.text.config(yscrollcommand=sc.set)
        self.text.pack(side="left", fill="both", expand=True)
        sc.pack(side="right", fill="y")
        self.root.after(100, self.poll)

    def append(self, s):
        self.text.config(state="normal")
        self.text.insert("end", s)
        self.text.see("end")
        self.text.config(state="disabled")

    def enable(self, v):
        st = "readonly" if v else "disabled"
        self.b1.config(state=st)
        self.b2.config(state=st)
        self.sp.config(state=st)
        self.se.config(state="normal" if v else "disabled")
        self.start.config(state="normal" if v else "disabled")

    def go(self):
        if self.t1.get() == self.t2.get():
            messagebox.showerror("エラー", "異なる2チームを選んでください")
            return
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.config(state="disabled")
        self.ss.set(f"{self.t1.get()} 0 - 0 {self.t2.get()}")
        self.enable(False)
        threading.Thread(target=self.worker, daemon=True).start()

    def worker(self):
        try:
            run_series(
                self.t1.get(),
                self.t2.get(),
                int(self.need.get()),
                int(self.seed.get()),
                self.q,
            )
        except Exception:
            self.q.put(("error", traceback.format_exc()))

    def poll(self):
        try:
            while True:
                e = self.q.get_nowait()
                if e[0] == "status":
                    self.status.set(e[1])
                elif e[0] == "score":
                    self.ss.set(f"{self.t1.get()} {e[1]} - {e[2]} {self.t2.get()}")
                elif e[0] == "map":
                    r = e[1]
                    ot = " [OT]" if r.overtime else ""
                    self.append(
                        f"MAP {r.number}{ot}\n  {r.team1} {r.score1} - {r.score2} {r.team2}\n  Winner: {r.winner}\n  Initial Attacker: {r.initial_attacker}\n  {r.team1} MVP: {r.mvp1.name} {r.mvp1.kills}K/{r.mvp1.deaths}D\n  {r.team2} MVP: {r.mvp2.name} {r.mvp2.kills}K/{r.mvp2.deaths}D\n"
                        + ("-" * 70)
                        + "\n"
                    )
                elif e[0] == "done":
                    d, p = e[1], e[2]
                    self.status.set(f"シリーズ終了: {d['winner']} WIN")
                    self.append(
                        f"\nSERIES FINAL\n{d['team1']} {d['team1_wins']} - {d['team2_wins']} {d['team2']}\nWINNER: {d['winner']}\n保存先: {p}\n"
                    )
                    self.enable(True)
                elif e[0] == "error":
                    self.status.set("エラーが発生しました")
                    self.append("\nERROR\n" + e[1])
                    self.enable(True)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
