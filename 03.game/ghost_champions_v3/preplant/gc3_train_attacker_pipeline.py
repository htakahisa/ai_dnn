"""Explicit BC and subsequent PPO stages; importing never starts training."""
import json
import importlib
import os
import random
import uuid
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from ..gc3_policy import FreshPolicy
from .gc3_policy_attacker_history import HistoryPolicy, SCHEMA_HASH, load_preplant_policy
from .gc3_policy_attacker_site import SiteHistoryPolicy, SCHEMA_HASH as SITE_SCHEMA, NAVIGATION_SCHEMA_HASH, RETRIEVAL_SCHEMA_HASH
from ..gc3_engine_actions import MODES
from .gc3_observation_attacker_history import MatchHistory
from ..gc3_io import write_json
from ..tools.gc3_training_metrics import record_training_metrics, reconcile_resume_logs
from .gc3_data_attacker_preplant import dataset_manifest, file_hash, load_shard, save_shard, source_signature
from .gc3_rollout_attacker_preplant import choose_roster, play_preplant, preplant_reward_settings, PREPLANT_REWARD_VERSION
from .gc3_state_attacker_resources import POSTPLANT_CHECKPOINT


def rollout_settings(config):
    return dict(postplant_checkpoint=config.get("POSTPLANT_CHECKPOINT", POSTPLANT_CHECKPOINT))


def check_preplant_runtime():
    """Check evaluation imports before BC updates weights or creates a run."""
    from ..gc3_rollout import engine_directory
    with engine_directory():
        importlib.import_module("run_game")
        importlib.import_module("run_competition_manager")


def policy_schema(config):
    if config.get("RETRIEVAL_FEATURES"):
        if not config.get("NAVIGATION_FEATURES") or not config.get("SITE_GOAL_POLICY"):
            raise ValueError("Dropped-spike navigation requires site and navigation features")
        return RETRIEVAL_SCHEMA_HASH
    if config.get("NAVIGATION_FEATURES"):
        if not config.get("SITE_GOAL_POLICY"):
            raise ValueError("Public navigation requires a learned shared site intention")
        return NAVIGATION_SCHEMA_HASH
    return SITE_SCHEMA if config.get("SITE_GOAL_POLICY") else SCHEMA_HASH


def rl_evaluation_rank(report):
    return report["actual_round_win_rate"], report["plant_rate"]


def rl_input_hash(config, bc_hash):
    import hashlib
    initial = config.get("INITIAL_CHECKPOINT")
    parts = [bc_hash, file_hash(config.get("POSTPLANT_CHECKPOINT", POSTPLANT_CHECKPOINT))]
    if initial is not None:
        parts.append(file_hash(initial))
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state():
    state = np.random.get_state()
    return dict(python_rng=random.getstate(), numpy_rng=(state[0], state[1].tolist(), *state[2:]),
                torch_rng=torch.get_rng_state())


def restore_rng(saved):
    random.setstate(saved["python_rng"])
    state = saved["numpy_rng"]
    np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
    torch.set_rng_state(saved["torch_rng"])


@contextmanager
def run_directories(config):
    for key in ("TORCH_THREADS", "LEARNING_RATE", "BATCH_SIZE", "EPOCHS", "TRAINING_STEPS", "ROUNDS_PER_MATCH", "DAGGER_ROUNDS_PER_EPOCH",
                "ROLLOUT_DECISIONS", "PPO_EPOCHS", "EVALUATE_EVERY_STEPS", "EVALUATION_ROUNDS_PER_OPPONENT"):
        if key in config and config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < config.get("MIN_BC_PLANT_RATE", .5) <= 1:
        raise ValueError("MIN_BC_PLANT_RATE must be between zero (exclusive) and one")
    if config.get("BC_KL_WEIGHT", 0.) < 0:
        raise ValueError("BC_KL_WEIGHT must be nonnegative")
    if config.get("BC_ORB_LABEL_WEIGHT", 1.) <= 0 or config.get("MIN_BC_ORB_COLLECTIONS", 0) < 0:
        raise ValueError("Orb label weight must be positive and minimum collections nonnegative")
    if "FIXED_OPPONENT_AI" in config:
        if config["FIXED_OPPONENT_AI"] != "default":
            raise ValueError("FIXED_OPPONENT_AI must be the fixed default rule controller")
        choose_roster(config["SEED"], 0, config["OPPONENT_CODES"])
    torch.set_num_threads(config["TORCH_THREADS"])
    data, logs = config["RUN_PATH"], config["LOG_PATH"]
    if not config["RESUME"] and data.exists():
        raise RuntimeError("Run already exists; set RESUME=True or select a new RUN_NAME")
    if not config["RESUME"] and logs.exists() and any(logs.iterdir()):
        raise RuntimeError("Run logs already exist; select a new RUN_NAME")
    data.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    lock = data/"training.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        yield data, logs
    finally:
        lock.unlink(missing_ok=True)


def serial_config(config):
    return {key:str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
            for key, value in config.items() if key not in ("RESUME", "EPOCHS", "TRAINING_STEPS")}


def initialize(config, data, stage, input_hash):
    if config.get("REQUIRE_FRESH_INITIALIZATION") and config.get("INITIAL_CHECKPOINT") is not None:
        raise ValueError("This fresh micro run cannot load an older checkpoint")
    signature = source_signature()
    settings = serial_config(config)
    saved = None
    if config["RESUME"]:
        manifest = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
        if (manifest["source_hash"] != signature or manifest["input_hash"] != input_hash or
                manifest["stage"] != stage or manifest["config"] != settings):
            raise RuntimeError("Source, task settings or stage input changed; choose a new RUN_NAME")
        policy, saved = load_preplant_policy(data/"latest.pt")
        expected_schema = policy_schema(config)
        if policy.schema_hash != expected_schema:
            raise ValueError("Previous absolute-movement model is inference-only; use a new RUN_NAME with the existing dataset")
        if saved.get("stage") != stage:
            raise ValueError("Wrong stage checkpoint")
    else:
        seed_all(config["SEED"])
        if stage == "bc":
            initial_path = config.get("INITIAL_CHECKPOINT")
            if initial_path is not None:
                initial_policy, metadata = load_preplant_policy(initial_path)
                if not metadata.get("trained") or not isinstance(initial_policy, HistoryPolicy):
                    raise ValueError("BC fine-tuning requires trained GC v3 public-history weights")
                policy = (initial_policy if isinstance(initial_policy, SiteHistoryPolicy) else SiteHistoryPolicy.from_history(initial_policy)) if config.get("SITE_GOAL_POLICY") else initial_policy
            else:
                policy = (SiteHistoryPolicy.fresh("attacker",config["SEED"],navigation=config.get("NAVIGATION_FEATURES",False),
                    retrieval=config.get("RETRIEVAL_FEATURES",False)) if config.get("SITE_GOAL_POLICY")
                    else HistoryPolicy.fresh("attacker",config["SEED"]))
        else:
            policy, initial = load_preplant_policy(config.get("INITIAL_CHECKPOINT") or config["BC_CHECKPOINT"])
            acceptable={policy_schema(config)}
            if config.get("NAVIGATION_FEATURES"):acceptable.add(SITE_SCHEMA)
            if config.get("RETRIEVAL_FEATURES"):acceptable.add(NAVIGATION_SCHEMA_HASH)
            if policy.schema_hash not in acceptable:
                raise ValueError("RL requires the corrected relative-movement BC model")
            expected_stage = "rl" if config.get("INITIAL_CHECKPOINT") else "bc"
            if initial.get("stage") != expected_stage or not initial.get("trained"):
                raise ValueError(f"RL initialization requires a trained preplant {expected_stage} checkpoint")
            if expected_stage == "rl" and initial.get("bc_checkpoint_hash") != file_hash(config["BC_CHECKPOINT"]):
                raise ValueError("Warm-start RL must retain the same frozen BC anchor")
        if config.get("NAVIGATION_FEATURES"):
            policy = SiteHistoryPolicy.with_navigation(policy)
        if config.get("RETRIEVAL_FEATURES"):
            policy=SiteHistoryPolicy.with_retrieval(policy)
        initial_step = int(initial["step"]) if stage == "rl" and config.get("INITIAL_CHECKPOINT") else 0
        write_json(data/"manifest.json", dict(source_hash=signature, input_hash=input_hash, stage=stage,
            initialization=("gc3_bc_finetune" if config.get("INITIAL_CHECKPOINT") else "scratch") if stage == "bc" else ("rl_warm_start" if config.get("INITIAL_CHECKPOINT") else "behavior_cloning"), config=settings,
            reward_settings=preplant_reward_settings() if stage == "rl" else None, inherited_step=initial_step,
            model_partition="one preplant model shared by all five attackers; separate existing postplant model"))
        policy.save(data/"initial.pt", stage=stage, trained=False, step=initial_step,
                    inherited_weights=stage == "rl" or config.get("INITIAL_CHECKPOINT") is not None)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["LEARNING_RATE"])
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved)
    return policy, optimizer, saved


def bc_log_prob(policy, rows):
    """A facing-only command teaches the probability of that direction, not a made-up cell."""
    logp, _, _ = policy.evaluate_records(rows)
    if isinstance(policy, HistoryPolicy):
        # The history policy already marginalizes facing-only labels.
        return logp
    indices = [i for i, row in enumerate(rows) if row.get("watch_options") is not None]
    if indices:
        context = policy.features(torch.from_numpy(np.stack([rows[i]["obs"] for i in indices])))
        batch = dict(phase="standard", hints=np.stack([rows[i]["hints"] for i in indices]),
                     masks={"watch":np.stack([rows[i]["masks"]["watch"] for i in indices])})
        dist = policy.distribution(context, batch, "watch")
        options = torch.from_numpy(np.stack([rows[i]["watch_options"] for i in indices]))
        selected = torch.tensor([rows[i]["selections"]["watch"] for i in indices])
        correction = (dist.probs*options).sum(-1).clamp_min(1e-12).log()-dist.log_prob(selected)
        logp = logp.index_add(0, torch.tensor(indices), correction)
    return logp


def bc_batches(dataset, manifest, split, batch_size, shuffle, extra_shards=()):
    shards = [shard for shard in manifest["shards"] if shard["split"] == split]
    if split == "train":
        shards += list(extra_shards)
    if shuffle:
        random.shuffle(shards)
    for shard in shards:
        rows = load_shard(Path(shard.get("root", dataset))/shard["file"])
        # Query order may shuffle without changing the stored chronological history.
        indices = np.arange(len(rows))
        if shuffle:
            np.random.shuffle(indices)
        for offset in range(0, len(rows), batch_size):
            yield [rows[i] for i in indices[offset:offset+batch_size]]


def reused_dagger_shards(config, dataset_hash):
    """Read only committed training labels; inherit no model/optimizer weights."""
    root = config.get("REUSE_DAGGER_RUN_PATH")
    if root is None:
        return []
    root = Path(root)
    saved = torch.load(root/"latest.pt", map_location="cpu", weights_only=True)
    previous = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if saved.get("stage") != "bc" or saved.get("dataset_hash") != dataset_hash or previous.get("input_hash") != dataset_hash:
        raise ValueError("Reused DAgger labels must come from this same teacher dataset")
    shards = list(saved.get("dagger_shards", ()))
    for shard in shards:
        if file_hash(Path(shard["root"])/shard["file"]) != shard["sha256"]:
            raise ValueError("Reused DAgger training shard changed")
    return shards


def train_bc(config):
    check_preplant_runtime()
    manifest, input_hash = dataset_manifest(config["DATASET_PATH"])
    reused = reused_dagger_shards(config, input_hash)
    run_input_hash = input_hash
    if reused or config.get("INITIAL_CHECKPOINT") is not None:
        import hashlib
        initial_hash = file_hash(config["INITIAL_CHECKPOINT"]) if config.get("INITIAL_CHECKPOINT") is not None else None
        run_input_hash = hashlib.sha256(json.dumps([input_hash, reused]+([initial_hash] if initial_hash else []), sort_keys=True).encode()).hexdigest()
    with run_directories(config) as (data, logs):
        policy, optimizer, saved = initialize(config, data, "bc", run_input_hash)
        epoch, step, best = (saved["epoch"], saved["step"], saved["best_loss"]) if saved else (0, 0, float("inf"))
        best_rank = tuple(saved["best_rank"]) if saved else ((-1., -1., -1., -float("inf")) if config.get("MIN_BC_ORB_COLLECTIONS", 0) else (-1., -1., -float("inf")))
        dagger_shards = list(saved.get("dagger_shards", ())) if saved else reused
        for shard in dagger_shards:
            if file_hash(Path(shard["root"])/shard["file"]) != shard["sha256"]:
                raise ValueError("DAgger shard changed; cannot resume")
        if saved:
            # Preserve abandoned epoch measurements; retain only checkpointed epochs.
            path = logs/"bc_metrics.jsonl"
            if path.exists():
                lines = path.read_text(encoding="utf-8").splitlines(True)
                kept = [line for line in lines if json.loads(line)["epoch"] <= epoch]
                if len(lines) != len(kept):
                    backup = logs/f"bc_metrics_uncommitted_{os.getpid()}.jsonl"
                    backup.write_text("".join(lines), encoding="utf-8")
                    path.write_text("".join(kept), encoding="utf-8")
        while epoch < config["EPOCHS"]:
            policy.train()
            total, count = 0., 0
            for rows in bc_batches(config["DATASET_PATH"], manifest, "train", config["BATCH_SIZE"], True, dagger_shards):
                per_row = -bc_log_prob(policy, rows)
                weights = per_row.new_tensor([config.get("BC_ORB_LABEL_WEIGHT", 1.) if r["selections"]["mode"] == MODES.index("COLLECT_ORB") else 1. for r in rows])
                loss = (per_row*weights).sum()/weights.sum()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite BC loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(rows)
                count += len(rows)
                step += len(rows)
            policy.eval()
            val_total, val_count = 0., 0
            with torch.no_grad():
                for rows in bc_batches(config["DATASET_PATH"], manifest, "validation", config["BATCH_SIZE"], False):
                    val_total += float(-bc_log_prob(policy, rows).sum())
                    val_count += len(rows)
            if not count or not val_count:
                raise ValueError("Empty train/validation split")
            epoch += 1
            validation = val_total/val_count
            report = evaluate_preplant(policy, config)
            write_json(logs/f"evaluation_epoch_{epoch}.json", report)
            orb_ready = report.get("orb_collections", 0) >= config.get("MIN_BC_ORB_COLLECTIONS", 0)
            rank = ((float(orb_ready),) if config.get("MIN_BC_ORB_COLLECTIONS", 0) else ())+(report["plant_rate"], report["success_rate"], -validation)
            ready = report["plant_rate"] >= config["MIN_BC_PLANT_RATE"] and orb_ready
            if rank > best_rank:
                best_rank = rank
                best = validation
                policy.save(data/"best.pt", stage="bc", trained=True, epoch=epoch, step=step,
                            validation_loss=best, dataset_hash=input_hash, evaluation=report,
                            rl_ready=ready, minimum_plant_rate=config["MIN_BC_PLANT_RATE"])
            with (logs/"bc_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(epoch=epoch, step=step, train_loss=total/count,
                    validation_loss=validation, plant_rate=report["plant_rate"],
                    success_rate=report["success_rate"], carrier_closest_site_distance=report.get("carrier_closest_site_distance"),
                    orb_collections=report.get("orb_collections", 0), orb_round_rate=report.get("orb_round_rate", 0.),
                    sites=report.get("sites", {}),
                    rl_ready=ready))+"\n")
            if config["ENABLE_DAGGER"] and epoch < config["EPOCHS"]:
                state = rng_state()
                try:
                    history, labelled, rounds = MatchHistory(), [], []
                    with torch.no_grad():
                        for index in range(config["DAGGER_ROUNDS_PER_EPOCH"]):
                            code = choose_roster(config["SEED"]+200000+epoch*1000, index, config["OPPONENT_CODES"])
                            batch, stats = play_preplant(code, config["SEED"]+200000+epoch*1000+index,
                                teacher=True, label_policy=policy, history=history,
                                teacher_kind=config.get("TEACHER_KIND", "concon"),
                                teacher_maps=config["DAGGER_TEACHER_MAPS"], fixed_opponent_ai=config["FIXED_OPPONENT_AI"])
                            labelled.extend(batch)
                            rounds.append(stats)
                    if not labelled:
                        raise RuntimeError("DAgger supplied no usable expert labels")
                    root = data/"dagger_train"
                    root.mkdir(exist_ok=True)
                    filename = f"epoch_{epoch:04d}_{uuid.uuid4().hex}.npz"
                    save_shard(root/filename, labelled)
                    dagger_shards.append(dict(root=str(root), file=filename, sha256=file_hash(root/filename)))
                    write_json(logs/f"dagger_epoch_{epoch}.json", dict(training_only=True, rounds=rounds,
                        labels=len(labelled), student_actions_sampled=True, file=filename))
                finally:
                    restore_rng(state)
            policy.save(data/"latest.pt", stage="bc", trained=True, epoch=epoch, step=step, best_loss=best,
                        best_rank=list(best_rank), evaluation=report, rl_ready=ready,
                        dagger_shards=dagger_shards,
                        dataset_hash=input_hash, optimizer=optimizer.state_dict(), **rng_state())
            write_json(logs/"status.json", dict(stage="bc", epoch=epoch, step=step,
                validation_loss=validation, plant_rate=report["plant_rate"], rl_ready=ready))
            print(f"BC epoch={epoch} train_loss={total/count:.4f} validation_loss={validation:.4f} "
                  f"plant_rate={report['plant_rate']:.1%} success_rate={report['success_rate']:.1%} "
                  f"completed_orbs={report.get('orb_collections', 0)} sites={report.get('sites', {})} RL_ready={ready}", flush=True)
        best_meta = load_preplant_policy(data/"best.pt")[1]
        write_json(logs/"status.json", dict(stage="bc", epoch=epoch, step=step, complete=True,
                    best_loss=best, best_plant_rate=best_meta["evaluation"]["plant_rate"], rl_ready=best_meta["rl_ready"]))


def evaluate_preplant(policy, config):
    state = rng_state()
    try:
        rounds = []
        with torch.no_grad():
            for index, code in enumerate(config["OPPONENT_CODES"]):
                history = MatchHistory()
                for repeat in range(config["EVALUATION_ROUNDS_PER_OPPONENT"]):
                    _, stats = play_preplant(code, config["SEED"]+100000+index*1000+repeat,
                        policy=policy, fixed_opponent_ai=config["FIXED_OPPONENT_AI"], history=history, **rollout_settings(config))
                    rounds.append(stats)
        measured = [r["carrier_closest_site_distance"] for r in rounds if r.get("carrier_closest_site_distance") is not None]
        return dict(success_rate=sum(r["success"] for r in rounds)/len(rounds),
                    actual_round_win_rate=sum(r["actual_round_success"] for r in rounds)/len(rounds),
                    success_definition="actual round win, including frozen postplant",
                    orb_collections=sum(r.get("orb_collections", 0) for r in rounds),
                    orb_round_rate=sum(r.get("orb_collections", 0)>0 for r in rounds)/len(rounds),
                    orb_ultimate_points_gained=sum(r.get("orb_ultimate_points_gained", 0) for r in rounds),
                    sites={site:dict(selected=sum(r.get("target_site") == site for r in rounds),
                                    planted=sum(r.get("planted_site") == site for r in rounds)) for site in ("A", "B")},
                    plant_rate=sum(r["planted"] for r in rounds)/len(rounds), rounds=rounds,
                    carrier_closest_site_distance=float(np.mean(measured)) if measured else None,
                    full_match_evaluated=False, phase="preplant", opponent_ai=config["FIXED_OPPONENT_AI"],
                    sampled=True, history_scope="same-opponent multi-round public history")
    finally:
        restore_rng(state)


def train_rl(config):
    if not config["BC_CHECKPOINT"].is_file():
        raise FileNotFoundError(f"Train the BC stage first: {config['BC_CHECKPOINT']}")
    input_hash = file_hash(config["BC_CHECKPOINT"])
    torch.set_num_threads(config["TORCH_THREADS"])
    anchor, bc_metadata = load_preplant_policy(config["BC_CHECKPOINT"])
    if config.get("REQUIRE_FRESH_BC"):
        bc_manifest=json.loads((config["BC_CHECKPOINT"].parent/"manifest.json").read_text(encoding="utf-8"))
        if (config.get("INITIAL_CHECKPOINT") is not None or anchor.schema_hash!=RETRIEVAL_SCHEMA_HASH
                or bc_manifest.get("initialization")!="scratch" or bc_manifest.get("source_hash")!=source_signature()):
            raise ValueError("Use the newly trained fresh retrieval BC; older BC/RL weights are not accepted")
    anchor_schemas={SITE_SCHEMA if config.get("SITE_GOAL_POLICY") else SCHEMA_HASH}
    if config.get("NAVIGATION_FEATURES"):anchor_schemas.add(NAVIGATION_SCHEMA_HASH)
    if config.get("RETRIEVAL_FEATURES"):anchor_schemas.add(RETRIEVAL_SCHEMA_HASH)
    if anchor.schema_hash not in anchor_schemas:
        raise ValueError("RL requires the corrected relative-movement BC model; reuse the existing teacher dataset for BC first")
    if bc_metadata.get("stage") != "bc" or not bc_metadata.get("trained"):
        raise ValueError("RL requires a trained history BC checkpoint")
    anchor.requires_grad_(False)
    # Check actual sampled performance again; never start PPO from a zero-plant BC.
    baseline = evaluate_preplant(anchor, config)
    if baseline["plant_rate"] < config["MIN_BC_PLANT_RATE"] or baseline.get("orb_collections", 0) < config.get("MIN_BC_ORB_COLLECTIONS", 0):
        write_json(config["LOG_PATH"].parent/"validation"/(config["RUN_PATH"].name+"_bc_gate.json"), baseline)
        raise RuntimeError(f"BC gate: plant={baseline['plant_rate']:.1%} (minimum {config['MIN_BC_PLANT_RATE']:.1%}), "
                           f"completed orbs={baseline.get('orb_collections', 0)} (minimum {config.get('MIN_BC_ORB_COLLECTIONS', 0)}); "
                           "PPO was not started. Improve BC first.")
    with run_directories(config) as (data, logs):
        policy, optimizer, saved = initialize(config, data, "rl", rl_input_hash(config, input_hash))
        inherited_step = int(load_preplant_policy(config["INITIAL_CHECKPOINT"])[1]["step"]) if config.get("INITIAL_CHECKPOINT") else 0
        step, episode, next_eval, best = (saved["step"], saved["episode"], saved["next_eval"], saved["best_score"]) \
            if saved else (inherited_step, 0, inherited_step+config["EVALUATE_EVERY_STEPS"], -1.)
        best_rank = tuple(saved["best_rank"]) if saved else (-1., -1.)
        if saved:
            reconcile_resume_logs(logs, step, episode)

        def save():
            policy.save(data/"latest.pt", stage="rl", trained=step>0, step=step, episode=episode,
                next_eval=next_eval, best_score=best, bc_checkpoint_hash=input_hash,
                best_rank=list(best_rank),
                optimizer=optimizer.state_dict(), reward_version=PREPLANT_REWARD_VERSION, **rng_state())
            write_json(logs/"status.json", dict(stage="rl", step=step, episode=episode, best_score=best))

        def evaluation(initial=False):
            nonlocal best, best_rank, next_eval
            report = baseline if initial and not config.get("INITIAL_CHECKPOINT") else evaluate_preplant(policy, config)
            report["step"] = step
            write_json(logs/("initial_evaluation.json" if initial else f"evaluation_{step}.json"), report)
            rank = rl_evaluation_rank(report)
            if rank > best_rank:
                best_rank = rank
                best = report["success_rate"]
                policy.save(data/"best.pt", stage="rl", trained=step>0, step=step,
                    best_score=best, best_rank=list(best_rank), evaluation=report,
                    bc_checkpoint_hash=input_hash, reward_version=PREPLANT_REWARD_VERSION)
            while next_eval <= step:
                next_eval += config["EVALUATE_EVERY_STEPS"]
            save()

        if not saved:
            evaluation(initial=True)
        while step < config["TRAINING_STEPS"]:
            rows = []
            target = min(config["ROLLOUT_DECISIONS"], config["TRAINING_STEPS"]-step)
            while len(rows) < target:
                history = MatchHistory()
                first_episode = episode
                for _ in range(config["ROUNDS_PER_MATCH"]):
                    code = choose_roster(config["SEED"], episode, config["OPPONENT_CODES"])
                    batch, stats = play_preplant(code, config["SEED"]+episode, policy=policy,
                        fixed_opponent_ai=config["FIXED_OPPONENT_AI"], gamma=config["GAMMA"],
                        gae_lambda=config["GAE_LAMBDA"], history=history, **rollout_settings(config))
                    if not batch:
                        raise RuntimeError("Pre-plant round supplied no decisions")
                    rows.extend(batch)
                    with (logs/"episodes.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(stats, ensure_ascii=False)+"\n")
                    episode += 1
                history_path = data/"histories"/f"match_{first_episode:06d}.npz"
                # A resumed interrupted batch must preserve its abandoned history.
                if history_path.exists():
                    history_path = history_path.with_name(history_path.stem+f"_retry_{os.getpid()}.npz")
                history.save(history_path)
            record_training_metrics(logs, step, episode, policy, rows, config["BATCH_SIZE"])
            loss = optimize_history(policy, anchor, optimizer, rows, config)
            step += len(rows)
            save()
            print(f"RL step={step} episodes={episode} loss={loss:.4f}", flush=True)
            if step >= next_eval or step >= config["TRAINING_STEPS"]:
                evaluation()
        best_policy, _ = load_preplant_policy(data/"best.pt")
        write_json(logs/"final_evaluation.json", evaluate_preplant(best_policy, config))
        write_json(logs/"status.json", dict(stage="rl", step=step, episode=episode, complete=True, best_score=best))


def optimize_history(policy, anchor, optimizer, rows, config):
    """PPO with exact causal context replay and a frozen BC distribution anchor."""
    advantages = np.array([r["advantage"] for r in rows], dtype=np.float32)
    advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
    losses = []
    policy.train()
    for _ in range(config["PPO_EPOCHS"]):
        # Whole chronological chunks retain full prefixes, recomputed with current weights.
        chunks = list(range(0, len(rows), config["BATCH_SIZE"]))
        random.shuffle(chunks)
        for offset in chunks:
            batch = rows[offset:offset+config["BATCH_SIZE"]]
            context = policy.contexts(batch)
            logp, entropy, values = policy.evaluate_contexts(batch, context)
            old = torch.tensor([r["logp"] for r in batch])
            target = torch.tensor([r["return"] for r in batch])
            adv = torch.from_numpy(advantages[offset:offset+len(batch)])
            ratio = (logp-old).exp()
            actor = -torch.minimum(ratio*adv, ratio.clamp(1-config["PPO_CLIP"], 1+config["PPO_CLIP"])*adv).mean()
            with torch.no_grad():
                anchor_context = anchor.contexts(batch)
            divergence = context.new_zeros(())
            for kind in ("mode", "position", "watch", "target", "ally", "site"):
                selected = [i for i,r in enumerate(batch) if kind in r["selections"]]
                if not selected: continue
                masks = {kind:np.stack([batch[i]["masks"][kind] for i in selected])}
                record = dict(phase="standard", obs=np.stack([batch[i]["obs"] for i in selected]),
                              goal_site=[batch[i].get("goal_site") for i in selected],
                              masks=masks, hints=np.stack([batch[i]["hints"] for i in selected]))
                if kind == "site" and getattr(policy, "navigation", None) is not None:
                    from .gc3_features_attacker_navigation import navigation_features
                    record.update(navigation_features=np.stack([navigation_features(batch[i], batch[i].get("goal_site")) for i in selected]),
                                  neutral_navigation_features=np.stack([navigation_features(batch[i], None) for i in selected]))
                baseline_dist = anchor.distribution(anchor_context[selected], record, kind)
                current_dist = policy.distribution(context[selected], record, kind)
                divergence = divergence+torch.distributions.kl_divergence(baseline_dist, current_dist).sum()/len(batch)
            loss = actor+.5*(values-target).square().mean()-config["ENTROPY_WEIGHT"]*entropy.mean()+config["BC_KL_WEIGHT"]*divergence
            if not torch.isfinite(loss): raise RuntimeError("Non-finite history PPO loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
    policy.eval()
    return float(np.mean(losses))
