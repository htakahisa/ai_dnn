"""Explicit PPO for the macro only; imports and inference never train."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ..preplant.gc3_data_attacker_preplant import file_hash, source_signature as base_source_signature
from ..preplant.gc3_train_attacker_pipeline import run_directories, rng_state, restore_rng, seed_all, serial_config
from ..preplant.gc3_rollout_attacker_preplant import choose_roster
from ..preplant.gc3_observation_attacker_history import MatchHistory
from ..gc3_io import write_json
from ..tools.gc3_training_metrics import record_training_metrics, reconcile_resume_logs
from .gc3_policy_attacker_macro import MacroPolicy
from .gc3_rollout_attacker_macro import play_macro, load_lower_policies
from .gc3_reward_attacker_macro import reward_settings, REWARD_VERSION
from .gc3_opponents_attacker_macro import opponent_settings


def source_signature():
    digest = hashlib.sha256(base_source_signature().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.name.startswith("gc3_train_"):
            continue
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    game_root = Path(__file__).resolve().parents[2]
    for name in ("fnatic_v3", "frc_v1", "touyama_v2", "ghost_champions_v2", "gc_v1", "defender_v3", "attacker_v3"):
        for path in sorted((game_root/name).rglob("*.py")):
            if any(part in ("data", "logs", "test", "__pycache__") for part in path.relative_to(game_root/name).parts):
                continue
            digest.update(str(path.relative_to(game_root)).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def initialize(config, data):
    inputs = {key:file_hash(config[key]) for key in ("PREPLANT_CHECKPOINT", "POSTPLANT_CHECKPOINT")}
    initial_path = config.get("INITIAL_CHECKPOINT")
    if initial_path is not None:
        inputs["INITIAL_CHECKPOINT"] = file_hash(initial_path)
    if config.get("OPPONENT_MODE") == "major_main":
        report = json.loads(Path(config["OPPONENT_VALIDATION"]).read_text(encoding="utf-8"))
        if not report.get("complete") or report["source_hash"] != source_signature():
            raise RuntimeError("Run the frozen major-opponent verification for the current source first")
        if report["opponent_ais"] != config["OPPONENT_AIS"]:
            raise ValueError("Major opponent verification uses different AI assignments")
        inputs["OPPONENT_CHECKPOINTS"] = report["opponent_checkpoint_hashes"]
        if any(file_hash(path) != value for path,value in inputs["OPPONENT_CHECKPOINTS"].items()):
            raise RuntimeError("A main opponent checkpoint changed; verify opponents again")
    manifest = dict(source_hash=source_signature(), input_hashes=inputs, config=serial_config(config),
                    reward_settings=reward_settings(), lower_models_frozen=True,
                    initialization="warm_start_macro_only" if initial_path is not None else "fresh_macro_only")
    saved = None
    if config["RESUME"]:
        previous = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError("Macro source/settings/lower models changed; choose a new RUN_NAME")
        policy, saved = MacroPolicy.load(data/"latest.pt")
    else:
        seed_all(config["SEED"])
        if initial_path is None:
            policy = MacroPolicy.fresh("attacker", config["SEED"])
            inherited_step = 0
        else:
            policy, initial = MacroPolicy.load(initial_path)
            if not initial.get("trained") or initial.get("stage") != "macro_rl":
                raise ValueError("Macro continuation requires a trained macro RL checkpoint")
            if initial.get("lower_checkpoint_hashes") != {key:inputs[key] for key in ("PREPLANT_CHECKPOINT", "POSTPLANT_CHECKPOINT")}:
                raise ValueError("Macro continuation must keep the same frozen lower models")
            inherited_step = int(initial["step"])
        write_json(data/"manifest.json", manifest)
        policy.save(data/"initial.pt", trained=False, step=inherited_step, stage="macro_rl")
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["LEARNING_RATE"])
    if saved:
        optimizer.load_state_dict(saved["optimizer"]); restore_rng(saved)
    return policy, optimizer, saved


def evaluate(policy, preplant, guard, config):
    state = rng_state()
    try:
        rounds = []
        with torch.no_grad():
            for i, code in enumerate(config["OPPONENT_CODES"]):
                history = MatchHistory(); macro_history = None
                for repeat in range(config["EVALUATION_ROUNDS_PER_OPPONENT"]):
                    _, stats, macro_history = play_macro(code, config["SEED"]+100000+i*1000+repeat,
                        policy=policy, preplant=preplant, guard=guard, history=history, macro_history=macro_history,
                        gamma=config["GAMMA"], gae_lambda=config["GAE_LAMBDA"], **opponent_settings(config, code))
                    rounds.append(stats)
        n = len(rounds)
        return dict(rounds=rounds, success_rate=sum(r["success"] for r in rounds)/n,
            actual_round_win_rate=sum(r["success"] for r in rounds)/n, plant_rate=sum(r["planted"] for r in rounds)/n,
            reward_mean=sum(r["environment_reward"] for r in rounds)/n,
            no_defuse_bonus_rate=sum(r["reward_component_sums"].get("no_defuse_start", 0)>0 for r in rounds)/n,
            sampled=True, opponent_mode=config.get("OPPONENT_MODE", "fixed_logic"))
    finally:
        restore_rng(state)


def optimize(policy, optimizer, rows, config):
    advantages = np.array([r["advantage"] for r in rows], dtype=np.float32)
    advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
    losses = []
    policy.train()
    for _ in range(config["PPO_EPOCHS"]):
        indices = np.random.permutation(len(rows))
        for start in range(0, len(rows), config["BATCH_SIZE"]):
            slots = indices[start:start+config["BATCH_SIZE"]]
            batch = [rows[i] for i in slots]
            logp, entropy, values = policy.evaluate_records(batch)
            ratio = (logp-torch.tensor([r["logp"] for r in batch])).exp()
            advantage = torch.from_numpy(advantages[slots])
            actor = -torch.minimum(ratio*advantage, ratio.clamp(1-config["PPO_CLIP"], 1+config["PPO_CLIP"])*advantage).mean()
            returns = torch.tensor([r["return"] for r in batch])
            loss = actor+.5*(values-returns).square().mean()-config["ENTROPY_WEIGHT"]*entropy.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite macro PPO loss")
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
    policy.eval()
    return float(np.mean(losses))


def train(config):
    preplant, guard = load_lower_policies(config["PREPLANT_CHECKPOINT"], config["POSTPLANT_CHECKPOINT"])
    with run_directories(config) as (data, logs):
        policy, optimizer, saved = initialize(config, data)
        if config.get("OPPONENT_MODE") == "major_main":
            manifest = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
            config = dict(config, _expected_opponent_files=manifest["input_hashes"]["OPPONENT_CHECKPOINTS"])
        lower_identity = dict(lower_checkpoint_hashes={key:file_hash(config[key]) for key in ("PREPLANT_CHECKPOINT", "POSTPLANT_CHECKPOINT")})
        inherited_step = int(MacroPolicy.load(config["INITIAL_CHECKPOINT"])[1]["step"]) if config.get("INITIAL_CHECKPOINT") else 0
        step, episode, next_eval, best_rank = (saved["step"], saved["episode"], saved["next_eval"], tuple(saved["best_rank"])) if saved else (inherited_step, 0, inherited_step+config["EVALUATE_EVERY_STEPS"], (-1e30, -1., -1.))
        if saved:
            reconcile_resume_logs(logs, step, episode)

        def save():
            policy.save(data/"latest.pt", trained=step>0, stage="macro_rl", step=step, episode=episode,
                next_eval=next_eval, best_rank=list(best_rank), optimizer=optimizer.state_dict(), **lower_identity, **rng_state())
            write_json(logs/"status.json", dict(step=step, episode=episode, best_score=best_rank[0], complete=False))

        def evaluate_and_save(initial=False):
            nonlocal next_eval, best_rank
            report = evaluate(policy, preplant, guard, config); report["step"] = step
            write_json(logs/("initial_evaluation.json" if initial else f"evaluation_{step}.json"), report)
            rank = (report["reward_mean"], report["actual_round_win_rate"], report["plant_rate"])
            if not initial and rank > best_rank:
                best_rank = rank
                policy.save(data/"best.pt", trained=step>0, stage="macro_rl", step=step,
                            best_rank=list(best_rank), evaluation=report, **lower_identity)
            while next_eval <= step:
                next_eval += config["EVALUATE_EVERY_STEPS"]
            save()

        if not saved:
            evaluate_and_save(initial=True)
        while step < config["TRAINING_STEPS"]:
            rows = []; target = min(config["ROLLOUT_DECISIONS"], config["TRAINING_STEPS"]-step)
            while len(rows) < target:
                history = MatchHistory(); macro_history = None; first_episode = episode
                for _ in range(config["ROUNDS_PER_MATCH"]):
                    code = choose_roster(config["SEED"], episode, config["OPPONENT_CODES"])
                    batch, stats, macro_history = play_macro(code, config["SEED"]+episode, policy=policy,
                        preplant=preplant, guard=guard, history=history, macro_history=macro_history,
                        gamma=config["GAMMA"], gae_lambda=config["GAE_LAMBDA"], **opponent_settings(config, code))
                    if not batch:
                        raise RuntimeError("Macro round supplied no decisions")
                    rows.extend(batch)
                    with (logs/"episodes.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(stats, ensure_ascii=False)+"\n")
                    episode += 1
                path = data/"histories"/f"match_{first_episode:06d}.npz"
                if path.exists():
                    import os
                    path = path.with_name(path.stem+f"_retry_{os.getpid()}.npz")
                macro_history.save(path)
            record_training_metrics(logs, step, episode, policy, rows, config["BATCH_SIZE"])
            loss = optimize(policy, optimizer, rows, config)
            step += len(rows); save()
            print(f"Macro ST={step} rounds={episode} loss={loss:.4f}", flush=True)
            if step >= next_eval or step >= config["TRAINING_STEPS"]:
                evaluate_and_save()
        best, _ = MacroPolicy.load(data/"best.pt")
        write_json(logs/"final_evaluation.json", evaluate(best, preplant, guard, config))
        write_json(logs/"status.json", dict(step=step, episode=episode, best_score=best_rank[0], complete=True))
