from __future__ import annotations
import argparse, random
from pathlib import Path
import torch

from controllers import DefaultAttackerController
from map_data import NEW_MAZE_STR
from party_presets import get_preset
from run_game import VisualFPSBattle, _build_team_ai
from team_ai import DualRoleTeamAI

from train_defender_opening_macro_gc_v2 import (
    ATTACKER_AI, BEST, FINAL, LATEST, GC_PRESET, GC_ROSTER, OPENING_TICKS,
    SelectionNet, DefenderOpeningController, LearningDefenderSetupGCRuntime,
    SETUP_VARIATIONS, VARIATION_DIM,
    ability_charge, dynamic_nearest_origin, bfs_dist, bfs_step,
    R_ORIGIN, R_EXECUTE,
)


def choose_checkpoint(requested):
    if requested:
        p=Path(requested)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for p in (FINAL, BEST, LATEST):
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(f"No checkpoint found: {FINAL}, {BEST}, {LATEST}")


def load_model(path, device):
    ckpt=torch.load(path,map_location=device,weights_only=False)
    net=SelectionNet().to(device)
    net.load_state_dict(ckpt["selection_state_dict"])
    net.eval()
    print("="*72)
    print("GC DEFENDER OPENING MACRO v2 - RANDOMIZED VISUAL WATCH")
    print(f"checkpoint={path} | episode={ckpt.get('episode','?')} | device={device}")
    print(f"best_opening_reward={ckpt.get('best_opening_reward','?')}")
    print("="*72)
    return net


class RandomizedWatchOpeningController(DefenderOpeningController):
    """
    Stage 2A selection remains learned/greedy.
    Setup variation and execution timing are intentionally randomized.

    Timing policy:
      SMOKE : ready + 0..2 ticks
      RECON : ready + 4..10 ticks
      FLASH : ready + 7..14 ticks

    These are not learned decisions. They are set-play variation.
    """

    def __init__(self, setup, sel, device, rng, game_ref=None, variation_mode="random"):
        super().__init__(setup,sel,device,0.0,rng,False)
        self._printed_plan=False
        self._printed_exec=set()
        self.execute_after={}
        self.game_ref=game_ref
        self.variation_mode=variation_mode
        self._round_index=0

    def _choose_variation_index(self):
        if self.variation_mode == "random":
            return self.rng.randrange(VARIATION_DIM)

        # Accept numeric index or exact variation name.
        try:
            idx=int(self.variation_mode)
            if 0 <= idx < VARIATION_DIM:
                return idx
        except Exception:
            pass

        wanted=str(self.variation_mode).strip().upper()
        for i,name in enumerate(SETUP_VARIATIONS):
            if str(name).upper() == wanted:
                return i

        raise ValueError(
            f"Unknown variation={self.variation_mode!r}; "
            f"use random, 0..{VARIATION_DIM-1}, or one of {list(SETUP_VARIATIONS)}"
        )

    def _apply_round_variation(self):
        idx=self._choose_variation_index()
        name=SETUP_VARIATIONS[idx]

        if hasattr(self.setup,"set_context"):
            opponent_name=getattr(self.game,"attacker_team_name",None) if self.game is not None else None
            self.setup.set_context(opponent_name=opponent_name, variation=idx)

        if self.game is not None:
            self.game.gc_setup_variation_index=idx
            self.game.gc_setup_variation=name

        print(f"\n[SETUP VARIATION] round={self._round_index} {name} ({idx})")

    def reset_round(self):
        super().reset_round()
        self._printed_plan=False
        self._printed_exec=set()
        self.execute_after={}
        self._round_index += 1
        self._apply_round_variation()

    def set_game(self, game):
        super().set_game(game)
        self._apply_round_variation()

    def _live_tick(self):
        # In trainer run_episode, ctrl.tick is assigned manually.
        # In the GUI game it is not, so use VisualFPSBattle.battle_tick.
        return int(getattr(self.game, "battle_tick", self.tick) or 0)

    def _random_delay(self, ability):
        ability=str(ability).upper()
        if ability=="SMOKE":
            return self.rng.randint(0,2)
        if ability=="RECON":
            return self.rng.randint(4,10)
        if ability=="FLASH":
            return self.rng.randint(7,14)
        return 0

    def select_plans(self):
        old=self.selected
        super().select_plans()
        if old or self._printed_plan:
            return

        self._printed_plan=True
        print("\n[OPENING] selected set play")
        for name in GC_ROSTER:
            p=self.plans.get(name)
            if p is None:
                print(f"  {name:10s} NONE")
                continue

            delay=self._random_delay(p.ability)
            self.execute_after[name]=delay

            origin="-" if p.origin is None else tuple(map(int,p.origin))
            print(
                f"  {name:10s} {p.ability:5s} #{p.pid} "
                f"origin={origin} target={tuple(map(int,p.target))} "
                f"originDist={p.origin_bfs_distance} "
                f"waitAfterReady={delay}"
            )

    def decide_move(self,char,game_state):
        # During Setup, keep the learned setup controller untouched.
        if self.setup_active():
            chars=list(self.game.chars)
            if not getattr(self.setup,"round_initialized",False):
                self.setup.initialize_round(chars)
            return self.setup.decide_setup_move(char,chars)

        base=list(map(int,char.pos))
        name=str(getattr(char,"name",""))

        if name not in GC_ROSTER:
            return base

        self.select_plans()
        p=self.plans.get(name)

        if p is None or p.state in {"EXECUTED","CANCELLED"}:
            return base

        if not getattr(char,"is_alive",True):
            p.state="CANCELLED"
            p.cancel_reason="DEAD"
            return base

        if ability_charge(char,p.ability)<=0:
            p.state="CANCELLED"
            p.cancel_reason="NO_CHARGE"
            return base

        # Flash/Recon still move to a legal origin exactly like Stage 2A.
        if p.ability!="SMOKE":
            new_origin=dynamic_nearest_origin(char,p,self.game)
            if new_origin is not None:
                p.origin=new_origin

            dm=bfs_dist(self.game.grid,[p.origin])
            r,c=map(int,char.pos)
            d=int(dm[r,c])

            if d<0:
                p.state="CANCELLED"
                p.cancel_reason="UNREACHABLE"
                return base

            if d>0:
                p.state="MOVING"
                return bfs_step(self.game.grid,char,self.game.chars,p.origin)

            if not p.origin_reached:
                p.origin_reached=True
                self.local_reward += R_ORIGIN

        p.state="READY"

        # Random variation in timing instead of "fire instantly".
        live_tick=self._live_tick()
        ready_tick=getattr(p,"ready_tick",None)
        if ready_tick is None:
            p.ready_tick=live_tick
            ready_tick=live_tick

        delay=int(self.execute_after.get(name,0))
        if live_tick < ready_tick + delay:
            return base

        p.state="EXECUTED"
        p.executed_tick=live_tick
        self.local_reward += R_EXECUTE

        if name not in self._printed_exec:
            self._printed_exec.add(name)
            print(
                f"[OPENING EXEC] tick={live_tick:02d} "
                f"{name} {p.ability} #{p.pid} -> {tuple(map(int,p.target))}"
            )

        return (base,{"ability":p.ability,"target":tuple(map(int,p.target))})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",default=None)
    ap.add_argument("--opponent",required=True,help="Exact party preset name")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--device",default="auto")
    ap.add_argument(
        "--variation",
        default="random",
        help="random / 0..N / exact variation name",
    )
    args=ap.parse_args()

    device=torch.device(
        "cuda" if args.device=="auto" and torch.cuda.is_available()
        else "cpu" if args.device=="auto"
        else args.device
    )
    rng=random.Random(args.seed)
    torch.manual_seed(args.seed)
    sel=load_model(choose_checkpoint(args.checkpoint),device)

    preset=get_preset(args.opponent)
    if preset is None:
        raise ValueError(f"Unknown party preset: {args.opponent!r}")

    setup=LearningDefenderSetupGCRuntime(device=str(device),verbose=False)
    ctrl=RandomizedWatchOpeningController(
        setup,sel,device,rng,variation_mode=args.variation
    )

    defender_ai=DualRoleTeamAI(
        "GC Opening v2 Random Watch",
        attacker_factory=lambda:DefaultAttackerController(),
        defender_factory=lambda:ctrl,
        use_iq_perception=False,
    )
    attacker_ai=_build_team_ai(ATTACKER_AI)

    game=VisualFPSBattle(
        NEW_MAZE_STR,attacker_ai,defender_ai,
        headless=False,
        attacker_roster=list(preset.players),
        defender_roster=list(GC_ROSTER),
        spike_holder_name=preset.spike_holder,
        attacker_igl_name=preset.igl,
        attacker_team_name=preset.name,
        defender_team_name=GC_PRESET,
        disable_side_swap=True,
    )

    print(f"opponent={preset.name} | Opening={OPENING_TICKS} ticks")
    print("variation=randomized unless --variation is specified")
    print("timing=randomized by ability")
    game.run()


if __name__=="__main__":
    main()
