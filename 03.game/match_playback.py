"""Responsive playback controls for rendered matches using real battle ticks."""

import time
import tkinter as tk
from tkinter import ttk

from game_core import validate_tick_time_ms


class MatchPlaybackMixin:
    def _init_playback(self):
        self.paused = False
        self.fast_forward_mode = None
        self._match_after_id = None
        self._pending_match_callback = None
        self._side_swap_count = 0

    def _build_playback_controls(self):
        bar = ttk.Frame(self.root, padding=5)
        bar.pack(fill="x", before=self.canvas)
        self.pause_button = ttk.Button(bar, text="一時停止", command=self.toggle_pause)
        self.pause_button.pack(side="left")
        self.skip_buttons = []
        for label, scope in (("ラウンドスキップ", "round"), ("サイドスキップ", "side"), ("マップスキップ", "map")):
            button = ttk.Button(bar, text=label, command=lambda mode=scope: self.start_fast_forward(mode))
            button.pack(side="left", padx=(5, 0))
            self.skip_buttons.append(button)
        ttk.Label(bar, text="1tick (ms)").pack(side="left", padx=(14, 5))
        self.playback_tick_var = tk.StringVar(self.root, value=str(self._tick_delay_ms()))
        entry = ttk.Entry(bar, textvariable=self.playback_tick_var, width=7)
        entry.pack(side="left")
        entry.bind("<Return>", lambda _: self.apply_tick_time())
        ttk.Button(bar, text="変更", command=self.apply_tick_time).pack(side="left", padx=5)
        self.playback_status = tk.StringVar(self.root, value="再生中")
        ttk.Label(bar, textvariable=self.playback_status).pack(side="left", padx=8)

    def _cancel_match_callback(self):
        handle = getattr(self, "_match_after_id", None)
        if handle is not None:
            self.root.after_cancel(handle)
        self._match_after_id = None

    def _schedule_match_callback(self, callback, delay=None):
        self._cancel_match_callback()
        self._pending_match_callback = callback
        self._match_after_id = self.root.after(self._tick_delay_ms() if delay is None else delay, callback)

    def _begin_match_callback(self):
        self._match_after_id = None

    def _update_playback_controls(self):
        if not hasattr(self, "playback_status"):
            return
        finished = self.match_over
        self.pause_button.configure(text="再開" if self.paused else "一時停止", state="disabled" if finished else "normal")
        for button in self.skip_buttons:
            button.configure(state="disabled" if finished else "normal")
        if finished:
            text = "試合終了"
        elif self.paused:
            text = "一時停止中"
        elif self.fast_forward_mode:
            text = {"round": "ラウンドを高速計算中", "side": "サイド交代まで高速計算中", "map": "マップを高速計算中"}[self.fast_forward_mode]
        else:
            text = "再生中"
        self.playback_status.set(text)

    def toggle_pause(self):
        if self.headless or self.match_over:
            return
        self.paused = not getattr(self, "paused", False)
        if self.paused:
            self._cancel_match_callback()
        else:
            callback = self._run_fast_forward_chunk if self.fast_forward_mode else (
                getattr(self, "_pending_match_callback", None) or self.loop
            )
            self._schedule_match_callback(callback, 1 if self.fast_forward_mode else None)
        self._update_playback_controls()

    def set_tick_time_ms(self, value):
        self.tick_time_ms = validate_tick_time_ms(value)
        # Replace the already pending timer, including a round transition timer.
        callback = getattr(self, "_pending_match_callback", None)
        if callback and not getattr(self, "paused", False) and not getattr(self, "fast_forward_mode", None) and not self.match_over:
            self._schedule_match_callback(callback)
        return self.tick_time_ms

    def apply_tick_time(self):
        try:
            value = self.set_tick_time_ms(self.playback_tick_var.get())
        except ValueError:
            self.playback_status.set("msは1以上の整数で入力してください")
            return
        self.playback_tick_var.set(str(value))
        self._update_playback_controls()

    def start_fast_forward(self, scope):
        if scope not in {"round", "side", "map"}:
            raise ValueError("Unknown skip scope")
        if self.headless or self.match_over:
            return
        self._cancel_match_callback()
        self.paused = False
        self.fast_forward_mode = scope
        self._skip_start_round = self.current_round - (1 if self.round_over else 0)
        self._skip_start_side = getattr(self, "_side_swap_count", 0)
        self._update_playback_controls()
        self._schedule_match_callback(self._run_fast_forward_chunk, 1)

    def _fast_forward_finished(self):
        return self.match_over or (
            self.fast_forward_mode == "round" and self.current_round > self._skip_start_round
        ) or (
            self.fast_forward_mode == "side" and getattr(self, "_side_swap_count", 0) != self._skip_start_side
        )

    def _run_fast_forward_chunk(self):
        self._begin_match_callback()
        if getattr(self, "paused", False):
            return
        deadline = time.perf_counter() + 0.02
        self.headless = True
        try:
            # During a rendered transition current_round already refers to the
            # next round, but its characters and side swap are not initialized.
            if self.round_over and not self.match_over:
                self.special_round_banner = self.explosion_effect = None
                self.init_round()
            for _ in range(200):
                if self._fast_forward_finished():
                    break
                self._simulate_tick()
                if time.perf_counter() >= deadline:
                    break
        finally:
            self.headless = False
        if not self._fast_forward_finished():
            self._schedule_match_callback(self._run_fast_forward_chunk, 1)
            return
        self.fast_forward_mode = None
        if self.match_over:
            self.label.config(text=f"MATCH OVER: {self.attacker_team_name} {self.attacker_wins} - {self.defender_wins} {self.defender_team_name}")
        self.draw()
        self._update_playback_controls()
        if not self.match_over:
            self._schedule_match_callback(self.loop)

    def stop_playback(self):
        self._cancel_match_callback()
        self._pending_match_callback = None
