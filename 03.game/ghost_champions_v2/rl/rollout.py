"""Real attack rounds against frozen opponents; privileged state is reward-only."""
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
import gc
import hashlib
import os
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OPPONENTS={"TYG":("Touyama Gaming","touyama_gaming_v2"),"OMG":("Omoko Gaming","omoko_gaming_v1"),
           "FRC":("Furina Classic","frc_v1"),"FNC":("Fnatic2023","fnatic_v3"),
           "GG":("Gorigons","concon_v1"),"SPS":("SUPES","toru_ai_v3.1")}


def potential(atk_alive,def_alive,planted,terminal=False):
    return 0. if terminal else .15*(atk_alive-def_alive)+.2*float(planted)


def shaping_coefficient(step,decay_steps=300_000):
    return max(0.,1.-step/decay_steps)


def pfsp_probabilities(rates):
    weights=np.array([(1.-min(1.,max(0.,rates.get(code,.5))))**2 for code in OPPONENTS],dtype=float)
    return weights/weights.sum() if weights.sum()>0 else np.full(len(weights),1/len(weights))


def freeze_modules(controller,excluded=()):
    queue,seen,modules=[controller],{id(item) for item in excluded},[]
    while queue:
        item=queue.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item,torch.nn.Module):
            item.eval()
            for parameter in item.parameters():
                parameter.requires_grad_(False)
            modules.append(item)
        elif isinstance(item,dict):
            queue.extend(item.values())
        elif isinstance(item,(list,tuple)):
            queue.extend(item)
        elif hasattr(item,"__dict__") and not isinstance(item,type):
            queue.extend(value for key,value in vars(item).items()
                         if key not in ("game","_game","_real","real_game","_real_game","encoder","sensor","memory"))
    return modules


def module_digest(modules):
    digest=hashlib.sha256()
    for module in modules:
        for key,value in sorted(module.state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def engine_directory():
    previous=Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


class Recorder:
    def __init__(self):
        self.rows=[]
        self.latest={}
        self.by_agent=defaultdict(list)
        self.observed_names=set()

    def record(self,name,obs,mask,action,logp,value,teacher):
        row=dict(obs=obs,mask=mask,action=action,logp=logp,value=value,teacher=teacher,
                 reward=0.,age=0,discount=1.)
        self.rows.append(row)
        self.by_agent[name].append(row)
        self.latest[name]=row

    def reward(self,reward,gamma,terminal):
        for row in self.latest.values():
            row["reward"]+=gamma**row["age"]*reward
            row["age"]+=1
            row["discount"]=0. if terminal else gamma**row["age"]

    def advantages(self,lam=.95):
        for trajectory in self.by_agent.values():
            next_value,next_adv=0.,0.
            for row in reversed(trajectory):
                delta=row["reward"]+row["discount"]*next_value-row["value"]
                adv=delta+row["discount"]*lam**row["age"]*next_adv
                row["advantage"],row["return"]=adv,adv+row["value"]
                next_value,next_adv=row["value"],adv
        return self.rows


def play_round(code,seed,*,policy=None,mode="teacher",gamma=.99,step=0,decay_steps=300_000,recon_reward=.02,gae_lambda=.95):
    from .controller import LearnedAttackerController
    recorder=Recorder()
    with engine_directory():
        from simulation_runtime import cpu_inference
        from run_competition_manager import seed_all
        from run_game import VisualFPSBattle,_build_team_ai
        from map_data import NEW_MAZE_STR
        from party_presets import get_preset
        from team_ai import DualRoleTeamAI
        from ghost_champions_v1_macro import GhostChampionsV1DefenderController
        seed_all(seed)
        gc_roster=get_preset("Ghost Champions")
        opponent=get_preset(OPPONENTS[code][0])
        factory=partial(LearnedAttackerController,policy=policy,recorder=recorder,mode=mode)
        with cpu_inference():
            game=VisualFPSBattle(NEW_MAZE_STR,DualRoleTeamAI("GC v2 training",factory,GhostChampionsV1DefenderController),
                _build_team_ai(OPPONENTS[code][1]),headless=True,
                attacker_roster=list(gc_roster.players),defender_roster=list(opponent.players),
                spike_holder_name=gc_roster.spike_holder,defender_spike_holder_name=opponent.spike_holder,
                attacker_igl_name=gc_roster.igl,defender_igl_name=opponent.igl,disable_side_swap=True)
            game.stop_after_round = True
            # No opponent optimizer exists. Disable FRC's optional inference statistics too.
            inner=getattr(game.defender_controller,"inner",game.defender_controller)
            actor=getattr(inner,"actor",None)
            if actor is not None and hasattr(actor,"collect_statistics"):
                actor.collect_statistics=False
            frozen_modules=freeze_modules(game.defender_controller,excluded=(game,policy))
            opponent_digest=module_digest(frozen_modules)
            seen_recon=set()
            rewarded_recon=0
            credited_bursts=[]
            # Keep the coefficient constant within an episode so the potential
            # terms telescope. Anneal it between complete training rounds.
            coef=shaping_coefficient(step,decay_steps)
            terminal=False
            for _ in range(1000):
                setup=game.defender_setup_phase.active
                before=potential(sum(c.is_alive for c in game.chars if c.team=="A"),
                                 sum(c.is_alive for c in game.chars if c.team=="D"),game.is_planted)
                game._simulate_tick()
                terminal=game.round_over or game.current_round!=1 or game.match_over
                if not setup:
                    after=potential(sum(c.is_alive for c in game.chars if c.team=="A"),
                                    sum(c.is_alive for c in game.chars if c.team=="D"),game.is_planted,terminal)
                    novel=set()
                    # Only an A-owned RECON explosion can earn novelty credit.
                    for burst in game.recon_bursts:
                        if burst.get("team")!="A" or any(burst is old for old in credited_bursts):
                            continue
                        credited_bursts.append(burst)
                        novel.update(str(c.name) for c in game.chars if c.team=="D" and c.is_alive
                                     and tuple(c.pos) in burst["cells"] and c.reveal_remaining>0)
                    new=len(novel-seen_recon-recorder.observed_names)
                    rewarded_recon+=new
                    seen_recon.update(novel)
                    records=game.analytics_tracker.round_records
                    record=records[-1] if terminal and records else None
                    if terminal and record is None:
                        raise RuntimeError("Terminal training round has no analytics result")
                    env=(1. if record and record["winner"]=="attacker" else -1.) if terminal else 0.
                    recorder.reward(env+coef*(gamma*after-before)+coef*recon_reward*new,gamma,terminal)
                if terminal:
                    break
            if not terminal:
                raise RuntimeError("Real-engine training round exceeded tick limit")
            record=game.analytics_tracker.round_records[-1]
            stats=dict(opponent=code,seed=seed,winner=record["winner"],reason=record["reason"],
                       planted=record["planted"],decisions=len(recorder.rows),recon_new=len(seen_recon),
                       recon_rewarded_new=rewarded_recon,opponent_modules=len(frozen_modules),
                       opponent_model_digest=opponent_digest,round_records=len(game.analytics_tracker.round_records))
            if module_digest(frozen_modules)!=opponent_digest:
                raise RuntimeError("Frozen opponent model changed during a training round")
            if policy is not None and any(not p.requires_grad for p in policy.parameters()):
                raise RuntimeError("Attacker training policy was unexpectedly frozen")
            rows=recorder.advantages(gae_lambda)
            del game
    gc.collect()
    return rows,stats
