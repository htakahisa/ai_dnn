"""Watch micro policies against fixed logic, or explicitly select a macro run."""
from pathlib import Path
from functools import partial
import argparse
import random
import sys

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT.parent))

# Normal settings: edit these constants and run without options.
PREPLANT_RL_RUN = "attacker_preplant_rl_history_v2"
PREPLANT_BC_RUN = "attacker_preplant_bc_history_v2"
POSTPLANT_CHECKPOINT = AI_ROOT/"data/attacker_postplant_fixed_v1/best.pt"
OMOKO_BC_RUN = "attacker_preplant_omoko_bc_site_v1"
OMOKO_RL_RUN = "attacker_preplant_omoko_rl_site_v1"
RESOURCE_RL_RUN = "attacker_preplant_omoko_rl_resources_v2"
HANDOFF_RL_RUN = "attacker_preplant_omoko_rl_handoff_v3"
OBJECTIVE_RL_RUN = "attacker_preplant_objectives_v1"
RETRIEVAL_RL_RUN = "attacker_preplant_retrieval_v2"
RETRIEVAL_BC_RUN = "attacker_preplant_retrieval_bc_v2"
MACRO_RUN = "attacker_macro_v1"
MAJOR_MACRO_RUN = "attacker_macro_major_v1"
MAJOR_MACRO_MODEL_PATHS = {"主要AIマクロbest": AI_ROOT/"macro_v1/data"/MAJOR_MACRO_RUN/"best.pt",
                          "主要AIマクロ最新": AI_ROOT/"macro_v1/data"/MAJOR_MACRO_RUN/"latest.pt"}
MACRO_MODEL_PATHS = {"マクロ best": AI_ROOT/"macro_v1/data"/MACRO_RUN/"best.pt",
                     "マクロ最新": AI_ROOT/"macro_v1/data"/MACRO_RUN/"latest.pt"}
MACRO_MODEL_PATHS.update(MAJOR_MACRO_MODEL_PATHS)
DEFAULT_MODEL = ("回収修正best" if (AI_ROOT/"data"/RETRIEVAL_RL_RUN/"best.pt").is_file()
                 else "回収修正最新" if (AI_ROOT/"data"/RETRIEVAL_RL_RUN/"latest.pt").is_file()
                 else "回収修正BC" if (AI_ROOT/"data"/RETRIEVAL_BC_RUN/"best.pt").is_file()
                 else "回収修正BC最新" if (AI_ROOT/"data"/RETRIEVAL_BC_RUN/"latest.pt").is_file()
                 else "未学習ミクロ")
OPPONENT = "FNC"
FIXED_OPPONENT_AI = "default"
SEED = None
TICK_MS = 150
START_PAUSED = True
MODEL_PATHS = {"RL最新": AI_ROOT/"data"/PREPLANT_RL_RUN/"latest.pt",
               "未学習ミクロ": None,
               "回収修正best": AI_ROOT/"data"/RETRIEVAL_RL_RUN/"best.pt",
               "回収修正最新": AI_ROOT/"data"/RETRIEVAL_RL_RUN/"latest.pt",
               "回収修正BC": AI_ROOT/"data"/RETRIEVAL_BC_RUN/"best.pt",
               "回収修正BC最新": AI_ROOT/"data"/RETRIEVAL_BC_RUN/"latest.pt",
               "ミクロ再学習best": AI_ROOT/"data"/OBJECTIVE_RL_RUN/"best.pt",
               "ミクロ再学習最新": AI_ROOT/"data"/OBJECTIVE_RL_RUN/"latest.pt",
               "主要AIマクロbest": AI_ROOT/"data"/HANDOFF_RL_RUN/"best.pt",
               "主要AIマクロ最新": AI_ROOT/"data"/HANDOFF_RL_RUN/"best.pt",
               "マクロ best": AI_ROOT/"data"/HANDOFF_RL_RUN/"best.pt",
               "マクロ最新": AI_ROOT/"data"/HANDOFF_RL_RUN/"best.pt",
               "勝敗RL best": AI_ROOT/"data"/HANDOFF_RL_RUN/"best.pt",
               "勝敗RL最新": AI_ROOT/"data"/HANDOFF_RL_RUN/"latest.pt",
               "Omoko BC": AI_ROOT/"data"/OMOKO_BC_RUN/"best.pt",
               "Omoko BC最新": AI_ROOT/"data"/OMOKO_BC_RUN/"latest.pt",
               "Omoko RL": AI_ROOT/"data"/OMOKO_RL_RUN/"best.pt",
               "Omoko RL修正版": AI_ROOT/"data"/RESOURCE_RL_RUN/"best.pt",
               "RL best": AI_ROOT/"data"/PREPLANT_RL_RUN/"best.pt",
               "BC best": AI_ROOT/"data"/PREPLANT_BC_RUN/"best.pt"}

from ghost_champions_v3.gc3_rollout import engine_directory, freeze_modules
from ghost_champions_v3.gc3_policy import FreshPolicy
from ghost_champions_v3.preplant.gc3_policy_attacker_history import load_preplant_policy
from ghost_champions_v3.preplant.gc3_learn_attacker_preplant import TwoPhaseAttackerController
from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import MAJOR_TEAMS, distinguish_same_roster
from ghost_champions_v3.tools.gc3_watch_attacker_postplant import PostplantBattle, VisualFPSBattle


class AttackerBattle(PostplantBattle):
    def _update_playback_controls(self):
        VisualFPSBattle._update_playback_controls(self)
        for button in getattr(self, "skip_buttons", ()):
            button.configure(state="disabled")
        if self.round_over and hasattr(self, "playback_status"):
            self.pause_button.configure(state="disabled")
            self.playback_status.set("ラウンド終了：次のラウンドで新しい場面を再生")

    def draw(self):
        super().draw()
        if hasattr(self, "viewer_status"):
            phase = "防衛初期配置" if self.in_defender_setup_phase else "設置後" if self.is_planted else "設置前"
            result = ""
            if self.round_over:
                reason = self.analytics_tracker.round_records[-1].get("reason", "unknown")
                result = f" | 終了：{'攻撃勝利' if self.attacker_wins else '防衛勝利'} ({reason})"
            controller = getattr(self, "attacker_controller", None)
            ctrl = getattr(controller, "inner_controller", controller)
            pre = getattr(ctrl, "preplant", None)
            if hasattr(ctrl, "lower"):
                pre = ctrl.lower.preplant
                phase += f" マクロ={ctrl.last_action} 切替={ctrl.switches}"
            goal = pre.history.goal_sites.get(pre.history.round_id) if pre is not None else None
            if goal is not None:
                phase += f" 目標サイト={('A','B')[goal]}"
            self.viewer_status.set(f"{self.viewer_description} | {phase} tick={self.battle_tick}{result}")


def prepare_round(model, opponent, seed, *, headless=True, tick_ms=TICK_MS):
    if model not in MODEL_PATHS or opponent not in MAJOR_TEAMS or FIXED_OPPONENT_AI != "default":
        raise ValueError("Unknown model/roster or fixed opponent logic")
    from run_competition_manager import seed_all
    from run_game import _build_team_ai
    from party_presets import get_preset
    from map_data import NEW_MAZE_STR
    from team_ai import DualRoleTeamAI
    from controllers import DefaultDefenderController
    with engine_directory():
        if MODEL_PATHS[model] is None:
            from ghost_champions_v3.preplant.gc3_checkpoint_attacker_micro import fresh_micro_policy
            preplant=fresh_micro_policy()
            metadata=dict(stage="fresh",step=0,trained=False)
        else:
            preplant, metadata = load_preplant_policy(MODEL_PATHS[model])
        if metadata.get("stage") not in ("bc", "rl", "fresh"):
            raise ValueError("Select a pre-plant BC/RL checkpoint")
        postplant, guard_metadata = FreshPolicy.load(POSTPLANT_CHECKPOINT, role="attacker")
        preplant.requires_grad_(False)
        postplant.requires_grad_(False)
        seed_all(seed)
        own, enemy = get_preset("Ghost Champions"), get_preset(MAJOR_TEAMS[opponent])
        opponent_key = FIXED_OPPONENT_AI
        opponent_team = _build_team_ai(opponent_key)
        if model in MAJOR_MACRO_MODEL_PATHS:
            from ghost_champions_v3.macro_v1.gc3_opponents_attacker_macro import MAIN_OPPONENT_AIS, build_opponent
            opponent_key = MAIN_OPPONENT_AIS[opponent]
            opponent_team = build_opponent(opponent_key)
        factory = partial(TwoPhaseAttackerController, preplant_policy=preplant, guard_policy=postplant)
        if model in MACRO_MODEL_PATHS:
            from ghost_champions_v3.macro_v1.gc3_learn_attacker_macro import MacroAttackerController
            factory = partial(MacroAttackerController, macro_checkpoint=MACRO_MODEL_PATHS[model],
                preplant_checkpoint=MODEL_PATHS[model], guard_checkpoint=POSTPLANT_CHECKPOINT,
                preplant_policy=preplant, guard_policy=postplant)
        own_ai = DualRoleTeamAI("GC v3 pre/post-plant", factory, DefaultDefenderController)
        game = AttackerBattle(NEW_MAZE_STR, own_ai, opponent_team,
            headless=True, disable_side_swap=True, tick_time_ms=tick_ms,
            attacker_roster=list(own.players), defender_roster=list(enemy.players),
            spike_holder_name=own.spike_holder, defender_spike_holder_name=enemy.spike_holder,
            attacker_igl_name=own.igl, defender_igl_name=enemy.igl)
        game.stop_after_round = True
        distinguish_same_roster(game)
        freeze_modules(game.attacker_controller, excluded=(game,))
        freeze_modules(game.defender_controller, excluded=(game,))
        # Mirror-roster aliases must also appear consistently in replay frames.
        game.replay_frames.clear()
        game._record_replay_frame()
        if not headless:
            game.attach_window()
        return game, dict(model=model, macro_step=getattr(game.attacker_controller, "macro_step", None),
            preplant_checkpoint=str(MODEL_PATHS[model]),
            preplant_step=metadata.get("step", 0), preplant_epoch=metadata.get("epoch"),
            preplant_stage=metadata.get("stage"),
            guard_checkpoint=str(POSTPLANT_CHECKPOINT), guard_step=guard_metadata.get("step"),
            opponent=opponent, opponent_ai=opponent_key, seed=seed)


class AttackerViewer:
    def __init__(self, *, root=None, seed=SEED):
        import tkinter as tk
        from tkinter import ttk
        self.root = root if root is not None else tk.Tk()
        self.root.title("GC v3 設置前からの実戦")
        self.game, self.seed, self.round_count = None, seed, 0
        self.seeds = random.Random(seed if seed is not None else random.SystemRandom().getrandbits(32))
        self.bar = ttk.Frame(self.root, padding=6)
        self.bar.pack(fill="x")
        self.model = tk.StringVar(self.root, value=DEFAULT_MODEL)
        self.opponent = tk.StringVar(self.root, value=OPPONENT)
        for label, variable, values in (("設置前モデル", self.model, tuple(MODEL_PATHS)),
                                        ("相手ロスター", self.opponent, tuple(MAJOR_TEAMS))):
            ttk.Label(self.bar, text=label).pack(side="left", padx=4)
            ttk.Combobox(self.bar, textvariable=variable, values=values,
                         state="readonly", width=12).pack(side="left")
        ttk.Button(self.bar, text="次のラウンド", command=self.next_round).pack(side="left", padx=8)
        ttk.Button(self.bar, text="同じseedで比較", command=self.repeat_round).pack(side="left")
        self.status = tk.StringVar(self.root, value="モデルを読み込みます")
        self.status_label = ttk.Label(self.root, textvariable=self.status, padding=6)
        self.status_label.pack(fill="x")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def next_round(self):
        seed = self.seed if self.round_count == 0 and self.seed is not None else self.seeds.getrandbits(32)
        self._mount(seed)

    def repeat_round(self):
        self._mount(self.seed if self.seed is not None else self.seeds.getrandbits(32))

    def _mount(self, seed):
        import torch
        if self.game is not None:
            self.game.stop_playback()
            self.game.paused = True
            self.game._update_playback_controls()
        try:
            with torch.no_grad():
                game, info = prepare_round(self.model.get(), self.opponent.get(), seed,
                                          tick_ms=self.game.tick_time_ms if self.game else TICK_MS)
        except (OSError, ValueError, RuntimeError) as error:
            self.status.set(f"モデルの読み込みに失敗しました：{error}")
            return
        for widget in self.root.winfo_children():
            if widget not in (self.bar, self.status_label):
                widget.destroy()
        self.game, self.seed = game, seed
        self.round_count += 1
        game.attach_window(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        game.label.configure(text="『再開』で再生。モデルと相手を選び、次のラウンド／同じseedで比較できます。")
        game.viewer_status = self.status
        game.viewer_description = (f"{info['model']} step={info['preplant_step']:,} "
                                   f"vs {info['opponent']} ({info['opponent_ai']}) seed={seed}")
        if info.get("macro_step") is not None:
            game.viewer_description = (f"{info['model']} マクロST={info['macro_step']:,} 下位ST={info['preplant_step']:,} "
                                       f"vs {info['opponent']} ({info['opponent_ai']}) seed={seed}")
        game.paused = START_PAUSED
        game._update_playback_controls()
        game.draw()
        self.root.title(f"GC v3 実戦 #{self.round_count} | {info['model']} | vs {info['opponent']}")
        print(f"[WATCH] {info}", flush=True)
        if not game.paused:
            game._schedule_match_callback(game.loop)

    def close(self):
        if self.game is not None:
            self.game.stop_playback()
        self.root.destroy()

    def run(self):
        self.next_round()
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    from simulation_runtime import cpu_inference
    import torch
    with engine_directory(), cpu_inference(), torch.no_grad():
        if args.verify_only:
            game, info = prepare_round(DEFAULT_MODEL, OPPONENT, SEED if SEED is not None else 20261014)
            game.run_headless_loop()
            print(dict(info, result=game.analytics_tracker.round_records[-1].get("reason")))
        else:
            AttackerViewer().run()


if __name__ == "__main__":
    main()
