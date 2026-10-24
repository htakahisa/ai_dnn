"""PPO training with explicit contribution and potential-difference rewards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import numpy as np
import torch
from .gc3_policy import FreshPolicy, SCHEMA_HASH
from .gc3_rollout import play_phase
from .gc3_io import write_json
from .gc3_reward_phase import REWARD_VERSION, reward_settings
from .tools.gc3_training_compat import validate_resume_source, migrated_manifest
# GC3_LOGGING_ONLY_BEGIN
from .tools.gc3_training_metrics import record_training_metrics, without_logging_blocks, reconcile_resume_logs
# GC3_LOGGING_ONLY_END

MAX_CONSECUTIVE_EMPTY_EPISODES = 24


def empty_episode_reason(stats, phase):
    state = (stats.get("phase_start") or {}).get("state", {})
    attackers = [char for char in state.get("chars", ()) if char.get("team") == "A"]
    if (phase == "postplant" and stats.get("phase") == "postplant" and
            stats.get("decisions") == 0 and attackers and
            all(char.get("is_alive") is False for char in attackers)):
        return "no_alive_attacker_at_phase_start"
    raise RuntimeError(f"Episode supplied no policy decisions: phase={phase}, "
                       f"opponent={stats.get('opponent')}, seed={stats.get('seed')}; "
                       "not a post-plant entry with all attackers already dead")


def collect_training_batch(policy, config, logs, episode, decision_target):
    rows, empty_count = [], 0
    while len(rows) < decision_target:
        code = config["OPPONENT_CODES"][episode % len(config["OPPONENT_CODES"])]
        batch, stats = play_phase(code, config["SEED"]+episode,
                                  phase=config["PHASE"], policy=policy,
                                  gamma=config["GAMMA"], gae_lambda=config["GAE_LAMBDA"], **entry_settings(config))
        if not batch:
            reason = empty_episode_reason(stats, config["PHASE"])
            stats = dict(stats, training_skipped=True, training_skip_reason=reason)
        rows.extend(batch)
        with (logs/"episodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stats, ensure_ascii=False)+"\n")
        episode += 1
        if batch:
            empty_count = 0
        else:
            empty_count += 1
            print(f"No training actions: {code}, seed={stats['seed']}, "
                  f"reason={reason}; result recorded, continuing", flush=True)
            if empty_count >= MAX_CONSECUTIVE_EMPTY_EPISODES:
                raise RuntimeError(f"{empty_count} consecutive post-plant entries supplied no training actions")
    return rows, episode

def normalize_launch_settings(source):
    # These named settings affect the run location/target, not the learned task.
    # Allow ordinary source edits for additional steps without requiring CLI overrides.
    return re.sub(rb"(?m)^(RESUME|TRAINING_STEPS|RUN_NAME)\s*=.*$",
                  lambda match:match.group(1)+b" = <run setting>", source)

def source_hash():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    # These entry points belong to the isolated pre-plant pipeline and are
    # never imported by search/post-plant training. Their settings must not
    # invalidate an existing guard/search run.
    preplant_launchers = {"gc3_collect_attacker_preplant.py", "gc3_train_attacker_preplant_bc.py",
                          "gc3_train_attacker_preplant_rl.py", "gc3_collect_attacker_preplant_omoko.py",
                          "gc3_train_attacker_preplant_omoko_bc.py", "gc3_train_attacker_preplant_omoko_rl.py"}
    files = [path for path in root.glob("*.py") if path.name not in preplant_launchers]
    files += [root.parent/name for name in ("game_core.py", "abilities_los.py", "grid_lines.py", "battle_logic.py",
                                           "map_data.py", "map_data_defender_setup.py",
                                           "iq_perception.py", "character.py", "party_presets.py")]
    for path in sorted(files):
        if path.exists():
            digest.update(path.name.encode())
            source = path.read_bytes()
            # GC3_LOGGING_ONLY_BEGIN
            source = without_logging_blocks(source)
            # GC3_LOGGING_ONLY_END
            if path.name.startswith("gc3_train_"):
                source = normalize_launch_settings(source)
            digest.update(source)
    return digest.hexdigest()

def optimize(policy, optimizer, rows, config):
    advantages = np.array([r["advantage"] for r in rows], dtype=np.float32)
    advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
    losses = []
    policy.train()
    for _ in range(config["PPO_EPOCHS"]):
        order = np.random.permutation(len(rows))
        for start in range(0, len(rows), config["BATCH_SIZE"]):
            indices = order[start:start+config["BATCH_SIZE"]]
            batch = [rows[i] for i in indices]
            logp, entropy, values = policy.evaluate_records(batch)
            old = torch.tensor([r["logp"] for r in batch])
            target = torch.tensor([r["return"] for r in batch])
            adv = torch.from_numpy(advantages[indices])
            ratio = (logp-old).exp()
            actor = -torch.minimum(ratio*adv, ratio.clamp(1-config["PPO_CLIP"], 1+config["PPO_CLIP"])*adv).mean()
            loss = actor + .5*(values-target).square().mean()-config["ENTROPY_WEIGHT"]*entropy.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
    policy.eval()
    return float(np.mean(losses))

def evaluate(policy, config):
    rows = []
    with torch.no_grad():
        for index, code in enumerate(config["OPPONENT_CODES"]):
            for repeat in range(config["EVALUATION_ROUNDS_PER_OPPONENT"]):
                _, stats = play_phase(code, config["SEED"]+100000+index*1000+repeat,
                                      phase=config["PHASE"], policy=policy, **entry_settings(config))
                rows.append(stats)
    return dict(success_rate=sum(r["success"] for r in rows)/len(rows), rounds=rows,
                full_match_evaluated=False)


def entry_settings(config):
    if config["PHASE"] != "postplant":
        return {}
    return dict(entry_attacker_codes=config["POSTPLANT_ENTRY_ATTACKER_CODES"],
                entry_max_attempts=config["POSTPLANT_ENTRY_MAX_ATTEMPTS"])

def main(config):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=config["RUN_NAME"])
    parser.add_argument("--steps", type=int, default=config["TRAINING_STEPS"])
    parser.add_argument("--resume", action="store_true", default=config["RESUME"])
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.steps < 1:
        parser.error("run must be a directory name; steps must be positive")
    config = dict(config, RUN_NAME=args.run, TRAINING_STEPS=args.steps, RESUME=args.resume)
    torch.set_num_threads(config["TORCH_THREADS"])
    data, logs = config["DATA_ROOT"]/args.run, config["LOG_ROOT"]/args.run
    data.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    lock = data/"training.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        run(config, data, logs)
    finally:
        lock.unlink(missing_ok=True)

def run(config, data, logs):
    role = "defender" if config["PHASE"] == "search" else "attacker"
    manifest = data/"manifest.json"
    signature = source_hash()
    step, episode, next_eval, best_score = 0, 0, config["EVALUATE_EVERY_STEPS"], -1.
    if config["RESUME"]:
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        validate_resume_source(previous["source_hash"], signature)
        policy, saved = FreshPolicy.load(data/"latest.pt", role=role)
        step, episode = saved["step"], saved["episode"]
        next_eval, best_score = saved["next_eval"], saved["best_score"]
        # GC3_LOGGING_ONLY_BEGIN
        recovery = reconcile_resume_logs(logs, step, episode)
        if recovery is not None:
            print(f"Uncommitted log records preserved in {recovery}", flush=True)
        # GC3_LOGGING_ONLY_END
    else:
        if manifest.exists() or (data/"latest.pt").exists() or (data/"best.pt").exists():
            raise RuntimeError("Run already exists; set RESUME or choose a new RUN_NAME")
        policy = FreshPolicy.fresh(role, config["SEED"])
        policy.save(data/"initial.pt", step=0, trained=False, inherited_weights=False, reward_version=REWARD_VERSION)
        random.seed(config["SEED"])
        np.random.seed(config["SEED"])
        torch.manual_seed(config["SEED"])
        write_json(manifest, dict(source_hash=signature, schema_hash=SCHEMA_HASH,
                                 initialization="scratch", reward=REWARD_VERSION, reward_settings=reward_settings(),
                                 config={k: str(v) if isinstance(v, Path) else v for k, v in config.items()}))
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["LEARNING_RATE"])
    if config["RESUME"]:
        optimizer.load_state_dict(saved["optimizer"])
        random.setstate(saved["python_rng"])
        np.random.set_state((saved["numpy_rng"][0], np.array(saved["numpy_rng"][1], dtype=np.uint32),
                             *saved["numpy_rng"][2:]))
        torch.set_rng_state(saved["torch_rng"])
        if previous["source_hash"] != signature:
            write_json(manifest, migrated_manifest(previous, signature, step, episode))
            print("Resume source updated for a reviewed compatible change; "
                  "checkpoint weights and optimizer state retained", flush=True)
    def save():
        nrng = np.random.get_state()
        policy.save(data/"latest.pt", step=step, episode=episode, next_eval=next_eval,
                    best_score=best_score, trained=step>0, inherited_weights=False, reward_version=REWARD_VERSION,
                    optimizer=optimizer.state_dict(), python_rng=random.getstate(),
                    numpy_rng=(nrng[0], nrng[1].tolist(), *nrng[2:]), torch_rng=torch.get_rng_state())
        write_json(logs/"status.json", dict(step=step, episode=episode, best_score=best_score,
                                          training_started=True, initialization="scratch"))
    def evaluation():
        nonlocal next_eval, best_score
        # Evaluation seeds must not replace the training random stream.
        py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            report = evaluate(policy, config)
        finally:
            random.setstate(py_rng)
            np.random.set_state(np_rng)
            torch.set_rng_state(torch_rng)
        write_json(logs/f"evaluation_{step}.json", report)
        if report["success_rate"] > best_score:
            best_score = report["success_rate"]
            policy.save(data/"best.pt", step=step, trained=step>0, inherited_weights=False, reward_version=REWARD_VERSION)
        while next_eval <= step:
            next_eval += config["EVALUATE_EVERY_STEPS"]
        save()
    if not config["RESUME"]:
        initial = evaluate(policy, config)
        write_json(logs/"initial_evaluation.json", initial)
        random.seed(config["SEED"])
        np.random.seed(config["SEED"])
        torch.manual_seed(config["SEED"])
        save()
    if step >= next_eval:
        evaluation()
    while step < config["TRAINING_STEPS"]:
        rows, episode = collect_training_batch(
            policy, config, logs, episode,
            min(config["ROLLOUT_DECISIONS"], config["TRAINING_STEPS"]-step))
        # GC3_LOGGING_ONLY_BEGIN
        record_training_metrics(logs, step, episode, policy, rows, config["BATCH_SIZE"])
        # GC3_LOGGING_ONLY_END
        loss = optimize(policy, optimizer, rows, config)
        step += len(rows)
        save()
        print(f"step={step} episodes={episode} loss={loss:.4f}", flush=True)
        if step >= next_eval or step >= config["TRAINING_STEPS"]:
            evaluation()
    best, _ = FreshPolicy.load(data/"best.pt", role=role)
    report = evaluate(best, config)
    write_json(logs/"final_evaluation.json", report)
    write_json(logs/"status.json", dict(step=step, episode=episode, best_score=best_score,
                                      training_started=True, complete=True, initialization="scratch"))
