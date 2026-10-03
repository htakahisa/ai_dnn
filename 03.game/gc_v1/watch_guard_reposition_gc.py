"""Watch the evaluated GC Guard candidate in the real, rendered match engine."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import tkinter as tk

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gc_v1.training_opponent_pool_gc import OpponentRotation, rotation_opponents
from game_core import COMBO_BANNER_HEIGHT
from run_game import VisualFPSBattle, _build_team_ai

DEFAULT_RUN = ROOT / "gc_v1/data/guard_reposition_20261002_combo_update"


class GuardWatchBattle(VisualFPSBattle):
    def __init__(self, *args, opponent_index=None, model_label="candidate", **kwargs):
        self.rotation = OpponentRotation()
        self.opponent_index = opponent_index
        self.model_label = model_label
        self.paused = False
        super().__init__(*args, **kwargs)
        if not self.headless:
            controls = tk.Frame(self.root)
            controls.pack(fill="x")
            self.pause_button = tk.Button(controls, text="一時停止 / 再開 (Space)",
                                          command=self.toggle_pause)
            self.pause_button.pack(side="left")
            tk.Label(controls, text="水色の枠：各プレイヤーのGuard担当位置").pack(side="left")
            self.root.bind("<space>", lambda _event: self.toggle_pause())
            # A hidden console launch can suppress Windows' first ShowWindow.
            # Remap the requested Tk window after the initial map completes.
            self.root.update()
            self.root.withdraw()
            self.root.update_idletasks()
            self.root.deiconify()
            self.root.lift()
            self.root.update()

    def init_round(self):
        index = self.current_round - 1 if self.opponent_index is None else self.opponent_index
        opponent = self.rotation.bind(self, index)
        super().init_round()
        print(f"[WATCH] round={self.current_round} opponent={opponent.name} "
              f"ai={opponent.ai_key} guard={self.model_label}", flush=True)
        if not self.headless:
            self.root.title(f"GC Guard {self.model_label} vs {opponent.name} | Round {self.current_round}")

    def toggle_pause(self):
        self.paused = not self.paused

    def loop(self):
        if self.paused and not self.headless:
            self.root.after(self._tick_delay_ms(), self.loop)
            return
        super().loop()

    def draw(self):
        super().draw()
        if self.headless or not self.is_planted:
            return
        controller = getattr(self.attacker_controller, "inner_controller", self.attacker_controller)
        guard = getattr(controller, "guard", None)
        alive = {c.name for c in self.chars if c.team == "A" and c.is_alive}
        for name, (row, col) in getattr(guard, "_assigned_guard_positions", {}).items():
            if name not in alive:
                continue
            # The base renderer moves its map below the announcement banner.
            # This overlay is appended afterwards, so apply that same offset.
            x = self._map_x(col * self.cell_size)
            y = COMBO_BANNER_HEIGHT + row * self.cell_size
            self.canvas.create_rectangle(x + 1, y + 1, x + self.cell_size - 1,
                                         y + self.cell_size - 1, outline="#00dfff", width=2)
            self.canvas.create_text(x + self.cell_size / 2, y + self.cell_size / 2,
                                    text=str(name), fill="#00dfff", font=("Arial", 7))


def main():
    opponents = rotation_opponents()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--guard-model", type=Path, help="defaults to the evaluated candidate")
    parser.add_argument("--baseline", action="store_true", help="watch the original Guard instead")
    parser.add_argument("--opponent", choices=["rotation"] + [o.name for o in opponents], default="rotation")
    parser.add_argument("--seed", type=int, default=9226100200)
    parser.add_argument("--tick-ms", type=int, default=180)
    parser.add_argument("--verify-only", action="store_true", help="load and initialize all requested opponents without a window")
    args = parser.parse_args()
    if args.baseline and args.guard_model:
        parser.error("--baseline and --guard-model cannot be combined")
    if args.tick_ms <= 0:
        parser.error("--tick-ms must be positive")

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    torch.manual_seed(args.seed)
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    sources = {phase: Path(info["path"]) for phase, info in manifest["sources"].items()}
    for phase, path in sources.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sources"][phase]["sha256"]:
            raise RuntimeError(f"The evaluated source checkpoint changed: {path}")
    guard_path = (sources["guard"] if args.baseline else
                  args.guard_model or args.run_dir / "dqn_attacker_guard_gc_best_by_eval.pt")
    from gc_v1.train_guard_reposition_gc import load_guard_checkpoint, load_policies
    checkpoint = load_guard_checkpoint(guard_path)
    load_policies({**sources, "guard": guard_path})  # Strict architecture validation before the legacy runtime loader.

    import ghost_champions_v1 as gc
    # These overrides are local to this viewing process; no checkpoint is copied.
    gc.CARRY = (sources["carry"],)
    gc.ESCORT = (sources["escort"],)
    gc.GUARD = (guard_path.resolve(),)
    from map_data import NEW_MAZE_STR
    from party_presets import get_preset

    index = None if args.opponent == "rotation" else next(i for i, o in enumerate(opponents) if o.name == args.opponent)
    first = opponents[0 if index is None else index]
    own = get_preset("Ghost Champions")
    label = f"{'baseline' if args.baseline else 'candidate'} EP{checkpoint.get('episode')}"
    print(f"[WATCH] checkpoint={guard_path.resolve()} mode={label}", flush=True)
    game = GuardWatchBattle(
        NEW_MAZE_STR, _build_team_ai("ghost_champions_v1"), first.build_loaded_team_ai("D"),
        headless=args.verify_only, attacker_roster=list(own.players), defender_roster=list(first.players),
        spike_holder_name=own.spike_holder, defender_spike_holder_name=first.spike_holder,
        attacker_igl_name=own.igl, defender_igl_name=first.igl,
        attacker_team_name=own.name, defender_team_name=first.name,
        disable_side_swap=True, tick_time_ms=args.tick_ms,
        opponent_index=index, model_label=label,
    )
    controller = game.attacker_controller.inner_controller
    for phase in ("carry", "escort", "guard"):
        if getattr(controller, phase, None) is None:
            raise RuntimeError(f"GC {phase} failed to load; viewing a fallback is not allowed")
    if args.verify_only:
        if index is None:
            for round_number in range(2, len(opponents) + 1):
                game.current_round = round_number
                game.init_round()
        print("[WATCH] VERIFIED", flush=True)
        return
    print("[WATCH] WINDOW_READY", flush=True)
    game.run()


if __name__ == "__main__":
    main()
