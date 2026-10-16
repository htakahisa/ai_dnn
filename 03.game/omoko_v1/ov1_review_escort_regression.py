"""Replay withdrawn escort gates in memory; never train or replace policies."""

from collections import Counter
from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
from unittest.mock import patch

AI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_DIR.parent))
sys.path.insert(0, str(AI_DIR))

import torch
from omoko_v1.ov1_attacker_controller import Ov1AttackerController
import ov1_learning_attacker_carry as carry
import ov1_learning_attacker_escort as escort
from ov1_inference_team_tactics import known_enemies, shot_clear, tactical_state
from omoko_v1.ability_advisor_v1.ov1_collect_attacker_ability_advisor import play_match
from omoko_v1.ability_advisor_v1.ov1_learn_attacker_ability_advisor import FixedAbilityAdvisor

SEEDS = (100000, 100001, 100002, 100003, 100004)
ROUNDS_PER_MATCH = 6
VARIANTS = ("original", "withdrawn")
ABLATION_VARIANTS = ("blind_only", "shot_only", "ignore_smoke", "ignore_bodies")
OPPONENT = "frc_v1"
DISABLE_LATER_CARRY_ESCAPE = True
OUTPUT = AI_DIR / "logs" / "ov1_escort_regression.json"

ORIGINAL_GATE = '''        # Keep inference deterministic with the combat action mask used during
        # training, including pre-existing checkpoints.
        visible_enemy, _ = self._nearest_visible_enemy(grid, chars, char.team, (r, c))
        if visible_enemy is not None and action != ACTION_ABILITY:
            st["last_delta"] = (0.0, 0.0)
            st["stuck"] += 1
            return [r, c], {"facing": _facing_towards((r, c), tuple(visible_enemy.pos))}
'''


class Audit:
    def __init__(self, variant):
        self.variant = variant
        self.events, self.rounds = [], []
        self.stats = Counter()
        self.game = None
        self.before = None
        self.pending = []
        self.plant_snapshot = None

    def gate(self, controller, char, state, action, facing):
        nearest, _ = controller._nearest_visible_enemy(
            state["grid"], state["chars"], char.team, tuple(char.pos))
        shared = tactical_state(state)
        shootable = [e for e in known_enemies(char, shared) if shot_clear(char, e, shared)]
        threats = [e for e in shootable if getattr(e, "blind_remaining", 0) <= 0]
        original = nearest is not None
        if self.variant == "original":
            stop, targets = original, [nearest] if nearest is not None else []
        elif self.variant == "withdrawn":
            stop, targets = bool(threats), threats
        elif self.variant == "blind_only":
            stop = original and getattr(nearest, "blind_remaining", 0) <= 0
            targets = [nearest] if stop else []
        elif self.variant == "shot_only":
            stop, targets = bool(shootable), shootable
        elif self.variant == "ignore_smoke":
            targets = [e for e in known_enemies(char, shared)
                       if getattr(e, "blind_remaining", 0) <= 0
                       and shot_clear(char, e, {**shared, "smoke_cells": ()})]
            stop = bool(targets)
        elif self.variant == "ignore_bodies":
            targets = [e for e in known_enemies(char, shared)
                       if getattr(e, "blind_remaining", 0) <= 0
                       and shot_clear(char, e, {**shared, "chars": [char, e]})]
            stop = bool(targets)
        else:
            raise ValueError(self.variant)
        stop = stop and action != escort.ACTION_ABILITY
        target = min(targets, key=lambda e: max(abs(e.pos[0]-char.pos[0]), abs(e.pos[1]-char.pos[1]))) if targets else None
        if original and action in (escort.ACTION_UP, escort.ACTION_DOWN, escort.ACTION_LEFT, escort.ACTION_RIGHT):
            dr, dc = escort._MOVE_DELTA[action]
            origin = tuple(char.pos)
            proposed = origin[0]+dr, origin[1]+dc
            hypothetical = (proposed if not stop else origin)
            row = dict(round=self.game.current_round, tick=self.game.battle_tick,
                       name=str(char.name), origin=origin, action=action, model_facing=facing,
                       proposed=proposed, stopped=bool(stop), hp=float(char.hp),
                       nearest=str(nearest.name), enemy_pos=tuple(nearest.pos),
                       enemy_blind=int(getattr(nearest, "blind_remaining", 0)),
                       shootable=[str(e.name) for e in shootable],
                       nonblind=[str(e.name) for e in threats],
                       before_shot=bool(shot_clear(char, nearest, shared)),
                       after_proposed_shot=bool(shot_clear(char, nearest, shared, position=hypothetical)),
                       smoke=[tuple(p) for p in shared.get("smoke_cells", ())],
                       allies=[(str(c.name),tuple(c.pos)) for c in shared["chars"] if c.team==char.team and c.is_alive])
            self.events.append(row)
            self.pending.append(row)
        return stop, target

    def snapshot(self):
        return {str(c.name):dict(pos=tuple(c.pos),hp=float(c.hp),alive=bool(c.is_alive),
                               team=c.team, blind=int(c.blind_remaining)) for c in self.game.chars}

    def observer(self, game, phase):
        self.game = game
        if phase == "before":
            self.pending = []
            self.before = self.snapshot()
            return
        after = self.snapshot()
        shots = [dict(shooter=str(s["shooter"].name),target=str(s["target"].name),
                      moved=bool(s["shooter"].moved_this_tick),hit_chance=s["hit_chance"],
                      damage=s["damage"],headshot=s["headshot"]) for s in game.last_shots]
        if not game.defender_setup_phase.active:
            for shot in shots:
                shooter = next(c for c in game.chars if str(c.name)==shot["shooter"])
                self.stats[f'{shooter.team}_shots'] += 1
                self.stats[f'{shooter.team}_damage'] += shot["damage"]
                if shot["moved"]:
                    self.stats[f'{shooter.team}_moving_shots'] += 1
            for row in self.pending:
                row["actual"] = after[row["name"]]
                row["shots"] = shots
        if game.is_planted and self.plant_snapshot is None:
            self.plant_snapshot = dict(tick=game.battle_tick,chars=after)
        if game.round_over:
            self.rounds.append(dict(round=game.current_round, score=[game.attacker_wins,game.defender_wins],
                                    planted=self.plant_snapshot, final=after, stats=dict(self.stats)))
            self.plant_snapshot = None


def run_variant(variant, seed):
    audit = Audit(variant)
    source = inspect.getsource(escort.Ov1LearningAttackerEscortController.decide_move)
    if source.count(ORIGINAL_GATE) != 1:
        raise RuntimeError("Escort source differs from the audited baseline")
    gate = '''        stop, combat_target = _audit.gate(self, char, game_state, action, facing)
        if stop:
            st["last_delta"] = (0.0, 0.0)
            st["stuck"] += 1
            return [r, c], {"facing": _facing_towards((r, c), tuple(combat_target.pos))}
'''
    namespace = {**escort.__dict__, "_audit":audit}
    exec(compile(textwrap.dedent(source.replace(ORIGINAL_GATE, gate)), "<audited_escort>", "exec"), namespace)
    with ExitStack() as stack:
        stack.enter_context(patch.object(escort.Ov1LearningAttackerEscortController, "decide_move", namespace["decide_move"]))
        if DISABLE_LATER_CARRY_ESCAPE:
            stack.enter_context(patch.object(carry, "awakened_carry_escape", return_value=None))
        result = play_match(OPPONENT, seed, ROUNDS_PER_MATCH, FixedAbilityAdvisor(None, enabled=False), tick_observer=audit.observer)
    return dict(variant=variant,seed=seed,result=result,stats=dict(audit.stats),rounds=audit.rounds,events=audit.events)


def main():
    torch.set_num_threads(1)
    models = sorted(AI_DIR.glob("data/attacker_*_data/*.pt"))
    hashes = {str(p.relative_to(AI_DIR)):hashlib.sha256(p.read_bytes()).hexdigest() for p in models}
    report = dict(opponent=OPPONENT,rounds=ROUNDS_PER_MATCH,disable_later_carry_escape=DISABLE_LATER_CARRY_ESCAPE,
                  ability_advisor=False,model_hashes=hashes,runs=[])
    for seed in SEEDS:
        for variant in VARIANTS + (ABLATION_VARIANTS if seed == SEEDS[0] else ()):
            run = run_variant(variant,seed)
            report["runs"].append(run)
            print(json.dumps(dict(variant=variant,seed=seed,wins=run["result"]["attacker_wins"],
                                  plants=run["result"]["plants"],stats=run["stats"])),flush=True)
    if hashes != {str(p.relative_to(AI_DIR)):hashlib.sha256(p.read_bytes()).hexdigest() for p in models}:
        raise RuntimeError("Model files changed during analysis")
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
