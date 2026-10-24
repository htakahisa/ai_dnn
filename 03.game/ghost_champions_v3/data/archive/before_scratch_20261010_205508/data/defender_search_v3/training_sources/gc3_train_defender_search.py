"""Run from ghost_champions_v3/: python gc3_train_defender_search.py."""
from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parent
ROOT = AI_ROOT.parent

# Normal settings: edit here, then execute without command-line options.
RUN_NAME = "defender_search_v3"
SEED = 20261010
TRAINING_STEPS = 20000
BC_ROUNDS_PER_OPPONENT_SITE = 5
BC_EPOCHS = 8
EVALUATE_EVERY_STEPS = 5000
EVALUATION_ROUNDS_PER_OPPONENT_SITE = 4
OPPONENT_CODES = ("TYG", "OMG", "FRC", "FNC", "GG", "SPS")
TRAINING_SITES = ("normal",)
BASELINE_CHECKPOINT = ROOT / "ghost_champions_v2/data/version_snapshots/v2_20261010/data/attacker_rl_unitstatus_20261009_recovered/latest.pt"
DATA_ROOT = AI_ROOT / "data"
LOG_ROOT = AI_ROOT / "logs"
RESUME = False
LEARNING_RATE = .0003
BATCH_SIZE = 256
ROLLOUT_DECISIONS = 1024
PPO_EPOCHS = 3
GAMMA = .99
GAE_LAMBDA = .95
PPO_CLIP = .2
ENTROPY_COEFFICIENT = .02
ABILITY_BC_WEIGHT = 1.5
BC_REGULARIZATION = .3
TORCH_THREADS = 1

sys.path.insert(0, str(ROOT))

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import re
import numpy as np
import torch
from torch.nn import functional as F
from ghost_champions_v3.gc3_learn_defender_search import DefenderSearchPolicy, SCHEMA_HASH, ACTIONS
from ghost_champions_v3.gc3_rollout_defender_search import play_search_round


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def bc_loss(policy, data, indices):
    dist, _ = policy(torch.from_numpy(data["obs"][indices]), torch.from_numpy(data["mask"][indices]))
    labels = torch.from_numpy(data["teacher"][indices])
    weights = torch.where(labels == 2, ABILITY_BC_WEIGHT, 1.)
    return -(weights * dist.log_prob(labels)).mean()


def optimize(policy, optimizer, rows, demos, rng):
    obs = torch.from_numpy(np.stack([r["obs"] for r in rows]))
    mask = torch.from_numpy(np.stack([r["mask"] for r in rows]))
    actions = torch.tensor([r["action"] for r in rows])
    old_logp = torch.tensor([r["logp"] for r in rows])
    returns = torch.tensor([r["return"] for r in rows], dtype=torch.float32)
    advantages = torch.tensor([r["advantage"] for r in rows], dtype=torch.float32)
    advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    training = np.flatnonzero(~demos["validation"])
    losses = []
    policy.train()
    for _ in range(PPO_EPOCHS):
        for batch in np.array_split(rng.permutation(len(rows)), max(1, math.ceil(len(rows)/BATCH_SIZE))):
            dist, value = policy(obs[batch], mask[batch])
            logp = dist.log_prob(actions[batch])
            spatial_logp, spatial_entropy = [], []
            for row_index in batch:
                terms = [policy.spatial.log_prob(record) for record in rows[row_index].get("spatial_choices", ())]
                spatial_logp.append(sum((item[0] for item in terms), logp.new_zeros(())))
                spatial_entropy.extend(item[1] for item in terms)
            logp = logp + torch.stack(spatial_logp)
            ratio = (logp-old_logp[batch]).exp()
            actor = -torch.minimum(ratio*advantages[batch], ratio.clamp(1-PPO_CLIP, 1+PPO_CLIP)*advantages[batch]).mean()
            indices = rng.choice(training, min(BATCH_SIZE, len(training)), replace=False)
            entropy = dist.entropy().mean() + (torch.stack(spatial_entropy).mean() if spatial_entropy else 0.)
            loss = actor + .5*F.mse_loss(value, returns[batch]) - ENTROPY_COEFFICIENT*entropy + BC_REGULARIZATION*bc_loss(policy, demos, indices)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite search training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            losses.append(float(loss.detach()))
    policy.eval()
    return float(np.mean(losses))


def summarize(episodes):
    def group(records):
        return dict(rounds=len(records), win_rate=sum(r["winner"] == "defender" for r in records)/max(1, len(records)),
                    plant_losses=sum(r["planted"] for r in records),
                    abilities_per_round=float(np.mean([r["ability_uses"] for r in records])),
                    reveals_per_round=float(np.mean([r["revealed"] for r in records])),
                    blinds_per_round=float(np.mean([r["blinded"] for r in records])),
                    coverage_ratio=sum(r["coverage_ticks"] for r in records)/max(1, sum(r["alive_ticks"] for r in records)),
                    preaim_ratio=sum(r["aim_ticks"] for r in records)/max(1, sum(r["alive_ticks"] for r in records)))
    return dict(overall=group(episodes), by_site={site: group([r for r in episodes if r["site"] == site]) for site in TRAINING_SITES},
                by_opponent={code: group([r for r in episodes if r["opponent"] == code]) for code in OPPONENT_CODES}, episodes=episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=RUN_NAME)
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--resume", action="store_true", default=RESUME)
    parser.add_argument("--verification", action="store_true", help="Small functional run in a separate directory")
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", "..") or args.steps < 1:
        parser.error("run must be a single directory name and steps must be positive")
    if args.verification and args.run == RUN_NAME:
        args.run = "defender_search_verification"
    config = dict(seed=SEED, steps=min(args.steps, 256) if args.verification else args.steps,
                  bc_rounds=2 if args.verification else BC_ROUNDS_PER_OPPONENT_SITE,
                  bc_epochs=2 if args.verification else BC_EPOCHS,
                  evaluation_rounds=1 if args.verification else EVALUATION_ROUNDS_PER_OPPONENT_SITE,
                  evaluate_every=EVALUATE_EVERY_STEPS, verification=args.verification,
                  opponents=OPPONENT_CODES, sites=TRAINING_SITES,
                  learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, rollout_decisions=ROLLOUT_DECISIONS,
                  ppo_epochs=PPO_EPOCHS, gamma=GAMMA, gae_lambda=GAE_LAMBDA, ppo_clip=PPO_CLIP,
                  entropy_coefficient=ENTROPY_COEFFICIENT, ability_bc_weight=ABILITY_BC_WEIGHT,
                  bc_regularization=BC_REGULARIZATION, baseline=str(BASELINE_CHECKPOINT))
    data_dir, log_dir = DATA_ROOT/args.run, LOG_ROOT/args.run
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir/"manifest.json").exists() and not args.resume:
        parser.error("run already exists; use RESUME=True or choose a new RUN_NAME")
    lock = data_dir/"training.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error(f"Training lock exists: {lock}. Check the recorded process before restarting.")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    status = dict(phase="starting", complete=False, defender_search_only=True, run=args.run, process_id=os.getpid(), step=0)
    def update(**values):
        status.update(values)
        write_json(log_dir/"status.json", status)
    def episode(code, seed, **kwargs):
        with (log_dir/"engine.log").open("a", encoding="utf-8") as engine_log, contextlib.redirect_stdout(engine_log), contextlib.redirect_stderr(engine_log):
            result = play_search_round(code, seed, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
                                      baseline_checkpoint=BASELINE_CHECKPOINT, **kwargs)
        gc.collect()
        return result
    def evaluation(policy, label, baseline=False):
        update(phase="evaluation_"+label, evaluated=0, evaluation_total=len(OPPONENT_CODES)*len(TRAINING_SITES)*config["evaluation_rounds"])
        episodes = []
        for ci, code in enumerate(OPPONENT_CODES):
            for si, site in enumerate(TRAINING_SITES):
                for i in range(config["evaluation_rounds"]):
                    _, stats = episode(code, SEED+100000+ci*1000+si*100+i, site=site, policy=policy, mode="stochastic", baseline=baseline)
                    episodes.append(stats)
                    update(evaluated=len(episodes), opponent=code, site=site)
        result = summarize(episodes)
        write_json(log_dir/(label+".json"), result)
        print(f"Evaluation {label}: {result['overall']}", flush=True)
        return result
    try:
        torch.set_num_threads(TORCH_THREADS)
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        # Freeze all existing game/opponent inputs, plus v3 map and training code.
        from ghost_champions_v2.rl.training import fingerprint, assert_frozen
        frozen = fingerprint(ROOT)
        for path in AI_ROOT.glob("*.py"):
            source = path.read_bytes()
            if path.resolve() == Path(__file__).resolve():
                # RESUME is execution control; changing it must not invalidate weights.
                source = re.sub(rb"(?m)^RESUME = (True|False)\r?$", b"RESUME = <execution control>", source)
            frozen[str(path.resolve())] = hashlib.sha256(source).hexdigest()
        frozen[str(BASELINE_CHECKPOINT)] = hashlib.sha256(BASELINE_CHECKPOINT.read_bytes()).hexdigest()
        manifest = dict(version=3, phase="defender_search_only", schema_hash=SCHEMA_HASH, config=config, frozen_inputs=frozen)
        def assert_inputs():
            source = re.sub(rb"(?m)^RESUME = (True|False)\r?$", b"RESUME = <execution control>", Path(__file__).read_bytes())
            if hashlib.sha256(source).hexdigest() != frozen[str(Path(__file__).resolve())]:
                raise RuntimeError("Search training source changed during the run")
            assert_frozen({p: h for p, h in frozen.items() if Path(p).resolve() != Path(__file__).resolve()})
        if args.resume:
            if json.loads((data_dir/"manifest.json").read_text(encoding="utf-8")) != json.loads(json.dumps(manifest)):
                raise ValueError("Resume configuration or frozen inputs changed; start a new run")
            with np.load(data_dir/"demonstrations.npz", allow_pickle=False) as arrays:
                demos = {name: arrays[name] for name in arrays.files}
            policy, payload = DefenderSearchPolicy.load(data_dir/"latest.pt")
            optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            step, round_index, best_score = payload["step"], payload["round_index"], payload["best_score"]
            rng.bit_generator.state = payload["numpy_rng"]
            torch.set_rng_state(payload["torch_rng"])
            random.setstate(payload["python_rng"])
            next_evaluation = payload["next_evaluation"]
            baseline_result = json.loads((log_dir/"baseline_v2.json").read_text(encoding="utf-8"))
        else:
            write_json(data_dir/"manifest.json", manifest)
            baseline_result = evaluation(None, "baseline_v2", baseline=True)
            rows, validation, episodes = [], [], []
            update(phase="collect_search_bc", collected=0, collection_total=len(OPPONENT_CODES)*len(TRAINING_SITES)*config["bc_rounds"])
            for ci, code in enumerate(OPPONENT_CODES):
                for si, site in enumerate(TRAINING_SITES):
                    for i in range(config["bc_rounds"]):
                        episode_rows, stats = episode(code, SEED+ci*10000+si*100+i, site=site)
                        rows.extend(episode_rows)
                        validation.extend([i == config["bc_rounds"]-1]*len(episode_rows))
                        episodes.append(stats)
                        update(collected=len(episodes), samples=len(rows), opponent=code, site=site)
                        print(f"BC {code}/{site} {i+1}/{config['bc_rounds']}: {len(episode_rows)} decisions", flush=True)
            if not rows:
                raise RuntimeError("No guard demonstrations collected")
            demos = dict(obs=np.stack([r["obs"] for r in rows]), mask=np.stack([r["mask"] for r in rows]),
                         teacher=np.array([r["teacher"] for r in rows], dtype=np.int64), validation=np.array(validation, dtype=np.bool_))
            np.savez_compressed(data_dir/"demonstrations.npz", **demos)
            write_json(data_dir/"demonstrations.json", dict(schema_hash=SCHEMA_HASH, episodes=episodes, samples=len(rows)))
            policy = DefenderSearchPolicy.from_v2(BASELINE_CHECKPOINT)
            optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
            training = np.flatnonzero(~demos["validation"])
            update(phase="train_search_bc")
            for epoch in range(config["bc_epochs"]):
                for indices in np.array_split(rng.permutation(training), max(1, math.ceil(len(training)/BATCH_SIZE))):
                    loss = bc_loss(policy, demos, indices)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite BC loss")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
                    optimizer.step()
                update(bc_epoch=epoch+1, loss=float(loss.detach()))
            held_out = np.flatnonzero(demos["validation"])
            with torch.no_grad():
                dist, _ = policy(torch.from_numpy(demos["obs"][held_out]), torch.from_numpy(demos["mask"][held_out]))
                accuracy = float((dist.logits.argmax(-1) == torch.from_numpy(demos["teacher"][held_out])).float().mean())
            update(bc_validation_accuracy=accuracy)
            policy.save(data_dir/"bc.pt", trained_phase="defender_search_only", step=0, verified=False)
            result = evaluation(policy, "after_bc")
            # Select the deployed model from PPO checkpoints so deployment heads are trained.
            best_score = -1.
            policy.save(data_dir/"best.pt", trained_phase="defender_search_only", step=0, evaluation=result["overall"], verified=False)
            step, round_index, next_evaluation = 0, 0, config["evaluate_every"]
        def save():
            policy.save(data_dir/"latest.pt", trained_phase="defender_search_only", step=step, round_index=round_index,
                        best_score=best_score, next_evaluation=next_evaluation, optimizer_state_dict=optimizer.state_dict(),
                        numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), python_rng=random.getstate(), verified=False)
        save()
        while step < config["steps"]:
            assert_inputs()
            update(phase="train_search_ppo", step=step)
            rows = []
            while len(rows) < min(ROLLOUT_DECISIONS, config["steps"]-step):
                # Deterministic rotation covers each opponent and site equally.
                code = OPPONENT_CODES[round_index % len(OPPONENT_CODES)]
                site = TRAINING_SITES[(round_index//len(OPPONENT_CODES)) % len(TRAINING_SITES)]
                episode_rows, stats = episode(code, int(rng.integers(1, 2**31)), site=site, policy=policy, mode="stochastic")
                rows.extend(episode_rows)
                round_index += 1
                with (log_dir/"rounds.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(stats)+"\n")
                update(phase="train_search_ppo", step=step, rollout_decisions=len(rows), round_index=round_index, opponent=code, site=site)
            loss = optimize(policy, optimizer, rows, demos, rng)
            step += len(rows)
            save()
            update(step=step, loss=loss)
            print(f"PPO defender search: {step}/{config['steps']} decisions; loss={loss:.4f}", flush=True)
            if step >= next_evaluation or step >= config["steps"]:
                result = evaluation(policy, f"evaluation_{step:07}")
                if result["overall"]["win_rate"] >= best_score:
                    best_score = result["overall"]["win_rate"]
                    policy.save(data_dir/"best.pt", trained_phase="defender_search_only", step=step, evaluation=result["overall"], verified=False)
                next_evaluation = (step//config["evaluate_every"]+1)*config["evaluate_every"]
                save()
        assert_inputs()
        best, _ = DefenderSearchPolicy.load(data_dir/"best.pt")
        final = evaluation(best, "final_best")
        write_json(log_dir/"comparison.json", dict(baseline_v2=baseline_result["overall"], candidate_v3=final["overall"],
                    win_rate_delta=final["overall"]["win_rate"]-baseline_result["overall"]["win_rate"],
                    matched_case_seeds=True, real_preplant_rounds=True, evaluation_stops_at_plant=True, full_match_evaluated=False))
        update(phase="complete", complete=True, step=step, checkpoint=str(data_dir/"best.pt"),
               training_complete=True, full_match_evaluated=False, verification=args.verification)
        print(f"Completed search training: {data_dir/'best.pt'}", flush=True)
    except Exception as exc:
        update(phase="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
