"""Browse actual post-plant scenes in one window. Run from ghost_champions_v3/."""
from collections import deque
from pathlib import Path
from functools import partial
import argparse
import gc
import random
import sys

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT.parent))

# Normal settings: edit these constants, then run without options.
CHECKPOINT = AI_ROOT / "data/attacker_postplant_fixed_v1/best.pt"
OPPONENT = "OMG"  # TYG, OMG, FRC, FNC, GG, SPS
SEED = None  # None draws a new scene; an integer repeats the same seed.
TICK_MS = 150
START_PAUSED = True
ENTRY_ATTACKER_CODES = ("TYG", "OMG", "FRC", "FNC", "SPS")
ENTRY_MAX_ATTEMPTS = 24
SITE_MODE = "alternate"  # alternate, A, B
SPREAD_MODE = "random"  # random, attacker, defender, both
SPREAD_MIN_CELLS = 8  # Minimum maximum Manhattan separation among living teammates.
SCENE_SEARCH_MAX_ATTEMPTS = 96
RECENT_SCENES = 12
ENTRY_TICKS_PER_CALLBACK = 6

SITE_LABELS = {"A/B交互": "alternate", "A（左）": "A", "B（右）": "B"}
SPREAD_LABELS = {"ランダム": "random", "攻撃側が分散": "attacker",
                 "防衛側が分散": "defender", "双方が分散": "both"}

from ghost_champions_v3.gc3_rollout import (
    OPPONENTS, POSTPLANT_ENTRY_SEED_STRIDE, POSTPLANT_ENTRY_MAX_TICKS, NoPostplantEntry,
    engine_directory, freeze_modules, phase_start_snapshot,
)
with engine_directory():
    from run_game import VisualFPSBattle, _build_team_ai


class PostplantBattle(VisualFPSBattle):
    def attach_window(self, root=None):
        """Give an already prepared headless game the shared rendered controls."""
        import tkinter as tk
        from game_core import SIDE_PANEL_WIDTH, COMBO_BANNER_HEIGHT
        self.root = root if root is not None else tk.Tk()
        self.headless = False
        self.canvas = tk.Canvas(
            self.root, width=self.map_pixel_width + SIDE_PANEL_WIDTH * 2,
            height=self.map_pixel_height + self.ability_area_height + COMBO_BANNER_HEIGHT,
            bg="#10141c", highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.label = tk.Label(self.root, font=("Arial", 10))
        self.label.pack()
        self._build_playback_controls()
        self.root.minsize(self.map_pixel_width + SIDE_PANEL_WIDTH * 2,
                          self.map_pixel_height + self.ability_area_height + COMBO_BANNER_HEIGHT + 110)
        self.root.protocol("WM_DELETE_WINDOW", self._close_match_window)

    def _update_playback_controls(self):
        super()._update_playback_controls()
        # Shared skip controls initialize later rounds, outside this viewer's phase.
        for button in getattr(self, "skip_buttons", ()):
            button.configure(state="disabled")
        if self.round_over and hasattr(self, "playback_status"):
            self.pause_button.configure(state="disabled")
            self.playback_status.set("ポストプラント終了")


def scene_layout(game):
    def separation(team):
        positions = [c.pos for c in game.chars if c.team == team and c.is_alive]
        return max((abs(a[0]-b[0]) + abs(a[1]-b[1])
                    for a in positions for b in positions), default=0)
    return dict(site="A" if game.planted_pos[1] < game.width // 2 else "B",
                attacker_span=separation("A"), defender_span=separation("D"))


def scene_signature(game):
    return (tuple(game.planted_pos), tuple((str(c.name), c.team, tuple(c.pos), bool(c.is_alive))
                                           for c in game.chars))


def layout_matches(layout, site=None, spread="random"):
    if site is not None and layout["site"] != site:
        return False
    if spread in ("attacker", "both") and layout["attacker_span"] < SPREAD_MIN_CELLS:
        return False
    if spread in ("defender", "both") and layout["defender_span"] < SPREAD_MIN_CELLS:
        return False
    return True


def scene_steps(policy, opponent, seed, *, tick_ms=TICK_MS, site=None, spread="random",
                exclude=(), max_attempts=ENTRY_MAX_ATTEMPTS):
    """Yield progress between real entry ticks, then a matching unchanged plant state."""
    from map_data import NEW_MAZE_STR
    from party_presets import get_preset
    from run_competition_manager import seed_all
    from team_ai import DualRoleTeamAI
    from ghost_champions_v3.gc3_learn_attacker_guard import GhostChampionsV3AttackerController
    from ghost_champions_v3.gc3_learn_defender_search import GhostChampionsV3SearchController

    own, enemy = get_preset("Ghost Champions"), get_preset(OPPONENTS[opponent][0])
    offset = random.Random(seed).randrange(len(ENTRY_ATTACKER_CODES))
    if site not in (None, "A", "B") or spread not in SPREAD_LABELS.values():
        raise ValueError("Unknown scene conditions")
    for attempt in range(max_attempts):
        entry_seed = seed + attempt * POSTPLANT_ENTRY_SEED_STRIDE
        source = ENTRY_ATTACKER_CODES[(offset + attempt) % len(ENTRY_ATTACKER_CODES)]
        seed_all(entry_seed)
        warmup_ai = _build_team_ai(OPPONENTS[source][1])
        opponent_ai = _build_team_ai(OPPONENTS[opponent][1])
        yield None, dict(attempt=attempt+1, source=source, ticks=0)
        game = PostplantBattle(
            NEW_MAZE_STR, warmup_ai, opponent_ai, headless=True,
            disable_side_swap=True, tick_time_ms=tick_ms,
            attacker_roster=list(own.players), defender_roster=list(enemy.players),
            spike_holder_name=own.spike_holder, defender_spike_holder_name=enemy.spike_holder,
            attacker_igl_name=own.igl, defender_igl_name=enemy.igl,
        )
        game.stop_after_round = True
        freeze_modules(game.attacker_controller, excluded=(game, policy))
        freeze_modules(game.defender_controller, excluded=(game, policy))
        # Same plant boundary as training; yield often enough to keep Tk responsive.
        for tick in range(POSTPLANT_ENTRY_MAX_TICKS):
            if game.round_over or game.match_over or game.is_planted:
                break
            game._simulate_tick()
            if (tick+1) % ENTRY_TICKS_PER_CALLBACK == 0:
                yield None, dict(attempt=attempt+1, source=source, ticks=tick+1)
        if not game.is_planted or game.round_over or game.match_over:
            del game, warmup_ai, opponent_ai
            gc.collect()
            continue
        layout = scene_layout(game)
        signature = scene_signature(game)
        if not layout_matches(layout, site, spread) or signature in exclude:
            del game, warmup_ai, opponent_ai
            gc.collect()
            continue
        plant = tuple(game.planted_pos)
        state = phase_start_snapshot(game)
        own_ai = DualRoleTeamAI(
            "GC v3 scratch",
            partial(GhostChampionsV3AttackerController, policy=policy, record_phase="postplant"),
            GhostChampionsV3SearchController,
        )
        own_ai.perception_engine = warmup_ai.perception_engine
        game.current_attacker_team_ai = game.initial_attacker_team_ai = own_ai
        own_ai.bind_game(game)
        game.attacker_controller = own_ai.get_attacker_controller()
        inner = getattr(game.defender_controller, "inner", game.defender_controller)
        actor = getattr(inner, "actor", None)
        if actor is not None and hasattr(actor, "collect_statistics"):
            actor.collect_statistics = False
        if phase_start_snapshot(game) != state:
            raise RuntimeError("Post-plant handoff changed the actual game state")
        # Replay begins at the plant, with positions, HP, charges and effects preserved.
        game.replay_frames.clear()
        game._record_replay_frame()
        yield game, dict(attacker=source, seed=entry_seed, attempts=attempt + 1,
                         plant=plant, state=state, signature=signature, **layout)
        return
    raise NoPostplantEntry(f"{opponent}, seed {seed}: no matching live plant in {max_attempts} rounds")


def prepare_scene(policy, opponent, seed, *, headless=False, tick_ms=TICK_MS,
                  site=None, spread="random", max_attempts=ENTRY_MAX_ATTEMPTS):
    steps = scene_steps(policy, opponent, seed, tick_ms=tick_ms, site=site,
                        spread=spread, max_attempts=max_attempts)
    try:
        for game, entry in steps:
            if game is not None:
                if not headless:
                    game.attach_window()
                    game.root.withdraw()
                return game, entry
    finally:
        steps.close()


class SceneViewer:
    def __init__(self, policy, metadata, *, opponent=OPPONENT, seed=None, tick_ms=TICK_MS,
                 root=None):
        import tkinter as tk
        from tkinter import ttk
        self.root = root if root is not None else tk.Tk()
        self.policy, self.metadata = policy, metadata
        self.base_seed = seed if seed is not None else random.SystemRandom().randrange(2**32)
        self.seeds = random.Random(self.base_seed)
        self.tick_ms, self.scene_count, self.request_count = tick_ms, 0, 0
        self.game, self.entry, self.steps, self.after_id = None, None, None, None
        self.recent = deque(maxlen=RECENT_SCENES)
        self.root.title("GC v3 ポストプラント")
        self.bar = ttk.Frame(self.root, padding=5)
        self.bar.pack(fill="x")
        self.opponent_var = tk.StringVar(self.root, value=opponent)
        self.site_var = tk.StringVar(self.root, value=next(k for k, v in SITE_LABELS.items() if v == SITE_MODE))
        self.spread_var = tk.StringVar(self.root, value=next(k for k, v in SPREAD_LABELS.items() if v == SPREAD_MODE))
        for label, variable, values, width in (
            ("相手", self.opponent_var, tuple(OPPONENTS), 6),
            ("サイト", self.site_var, tuple(SITE_LABELS), 9),
            ("配置", self.spread_var, tuple(SPREAD_LABELS), 13),
        ):
            ttk.Label(self.bar, text=label).pack(side="left", padx=(6, 3))
            ttk.Combobox(self.bar, textvariable=variable, values=values,
                         state="readonly", width=width).pack(side="left")
        self.next_button = ttk.Button(self.bar, text="次の場面", command=self.next_scene)
        self.next_button.pack(side="left", padx=10)
        self.status = tk.StringVar(self.root, value="次の場面で条件を適用します。A＝左、B＝右。")
        self.status_label = ttk.Label(self.root, textvariable=self.status, padding=5)
        self.status_label.pack(fill="x")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def next_scene(self):
        if self.steps is not None:
            return
        import numpy as np
        import torch
        self.saved_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        if self.game is not None:
            self.tick_ms = self.game.tick_time_ms
            self.game.stop_playback()
            self.game.paused = True
            self.game._update_playback_controls()
            self.game.pause_button.configure(state="disabled")
        site = SITE_LABELS[self.site_var.get()]
        if site == "alternate":
            site = ("B" if self.entry["site"] == "A" else "A") if self.entry else None
        self.request_seed = self.base_seed if not self.request_count else self.seeds.getrandbits(32)
        self.request_count += 1
        self.request_opponent = self.opponent_var.get()
        self.steps = scene_steps(
            self.policy, self.request_opponent, self.request_seed, tick_ms=self.tick_ms,
            site=site, spread=SPREAD_LABELS[self.spread_var.get()], exclude=self.recent,
            max_attempts=SCENE_SEARCH_MAX_ATTEMPTS,
        )
        self.next_button.configure(state="disabled")
        self.after_id = self.root.after(1, self._prepare_next)

    def _prepare_next(self):
        import torch
        self.after_id = None
        try:
            with torch.no_grad():
                game, entry = next(self.steps)
            if game is None:
                self.status.set(f"次の場面を準備中… 候補 {entry['attempt']}/{SCENE_SEARCH_MAX_ATTEMPTS}  "
                                f"設置前 {entry['ticks']} tick")
                self.after_id = self.root.after(1, self._prepare_next)
                return
        except Exception as error:
            import numpy as np
            import traceback
            self.steps.close()
            self.steps = None
            random.setstate(self.saved_rng[0])
            np.random.set_state(self.saved_rng[1])
            torch.set_rng_state(self.saved_rng[2])
            if isinstance(error, NoPostplantEntry):
                self.status.set("条件に合う場面が見つかりませんでした。条件を変えるか、もう一度『次の場面』を押してください。")
            else:
                self.status.set(f"場面の準備に失敗しました: {error}")
                traceback.print_exc()
            if self.game is not None:
                self.game._update_playback_controls()
            self.next_button.configure(state="normal")
            return
        self.steps.close()
        self.steps = None
        self._mount(game, entry)

    def _mount(self, game, entry):
        # Replace only the scene widgets. The root and scene selection bar stay alive.
        if self.game is not None:
            self.game.stop_playback()
        for widget in self.root.winfo_children():
            if widget not in (self.bar, self.status_label):
                widget.destroy()
        self.game, self.entry = game, entry
        self.scene_count += 1
        self.recent.append(entry["signature"])
        game.attach_window(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        game.label.configure(text="設置直後から開始。『再開』で再生。途中でも『次の場面』で切り替え。")
        game.paused = START_PAUSED
        game._update_playback_controls()
        game.draw()
        self.root.title(f"GC v3 ポストプラント #{self.scene_count} vs {self.request_opponent} | "
                        f"{entry['site']}サイト | step {self.metadata.get('step')}")
        self.status.set(f"場面 {self.scene_count}：{entry['site']}サイト  "
                        f"位置間隔 攻撃 {entry['attacker_span']}／防衛 {entry['defender_span']} マス  "
                        f"seed={self.request_seed}")
        self.next_button.configure(state="normal")
        gc.collect()
        print(f"[WATCH] scene={self.scene_count}; opponent={self.request_opponent}; "
              f"seed={self.request_seed}; plant={entry['plant']}; "
              f"entry={entry['attacker']}; entry_seed={entry['seed']}; attempts={entry['attempts']}", flush=True)
        if not game.paused:
            game._schedule_match_callback(game.loop)

    def run(self):
        self.next_scene()
        self.root.mainloop()

    def close(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.steps is not None:
            self.steps.close()
            self.steps = None
        if self.game is not None:
            self.game.stop_playback()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--opponent", choices=OPPONENTS, default=OPPONENT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tick-ms", type=int, default=TICK_MS)
    parser.add_argument("--verify-only", action="store_true", help="Simulate one round without a window")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.tick_ms < 1:
        parser.error("tick-ms must be positive")
    checkpoint = args.checkpoint.resolve()
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    from ghost_champions_v3.gc3_policy import FreshPolicy
    from simulation_runtime import cpu_inference
    import torch
    with engine_directory(), cpu_inference(), torch.no_grad():
        policy, metadata = FreshPolicy.load(checkpoint, role="attacker")
        if not metadata.get("trained"):
            raise ValueError("Select a trained post-plant checkpoint")
        policy.requires_grad_(False)
        print(f"[WATCH] GC v3 vs {args.opponent}; seed={seed}; selection=sampled", flush=True)
        print(f"[WATCH] checkpoint={checkpoint}; step={metadata.get('step')}", flush=True)
        if args.verify_only:
            game, entry = prepare_scene(policy, args.opponent, seed, headless=True, tick_ms=args.tick_ms)
            game.run()
            result = game.analytics_tracker.round_records[-1]
            print(f"[WATCH] VERIFIED winner={'A' if game.attacker_wins else 'D'}; "
                  f"reason={result.get('reason')}", flush=True)
        else:
            SceneViewer(policy, metadata, opponent=args.opponent, seed=seed, tick_ms=args.tick_ms).run()


if __name__ == "__main__":
    main()
