"""Train a small tactical language model from random weights or its own checkpoint."""

# 通常設定はここで変更。CLI は検証時の一時的な上書き専用。
from pathlib import Path

AI_DIR = Path(__file__).resolve().parent
DATA_PATH = AI_DIR / "data" / "teacher.jsonl"
OUTPUT_DIR = AI_DIR / "data" / "checkpoints"
LOG_PATH = AI_DIR / "logs" / "training.jsonl"
TRAIN_STEPS = 1000  # 今回追加する optimizer ステップ数
BATCH_SIZE = 4
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.01
EVAL_INTERVAL = 100
VALIDATION_FRACTION = 0.2
SEED = 42
RESUME_TRAINING = True  # latest.pt があれば再開、なければランダム初期化
OVERWRITE_EXISTING = False  # 新規学習で既存出力を上書きする場合だけ True
DEVICE = "auto"
CPU_THREADS = 4
MAX_SEQUENCE_LENGTH = 2048
MODEL_WIDTH = 192
MODEL_LAYERS = 4
MODEL_HEADS = 6
MODEL_DROPOUT = 0.1
GRAD_CLIP = 1.0
PROGRESS_INTERVAL = 10
STOP_REQUEST_PATH = None  # GUI からの一時的な停止要求。通常のCLI実行では不要。
AS_OF_DATE = None  # None: 日本時間の今日を自動取得。
AUTO_RESET_CHANGED_DATA = True  # 変更・除外済みの旧知識を含むモデルは保管して新規学習。
AUTO_RESET_SEQUENCE_LENGTH = True  # 最大長変更時は旧モデルを保管して新規学習。

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from datetime import datetime, timezone
from uuid import uuid4

import torch

from tl1_data_tactics import load_records, scenario_assignments, split_records
from tl1_dates_tactics import KNOWLEDGE_POLICY_VERSION, checkpoint_issue, current_date, data_manifest, validate_date
from tl1_model_tactics import (ModelConfig, TacticalLM, batch_examples, example_tokens,
                              load_checkpoint, loss_sum, model_metadata)


def resolve_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA が使用できません。DEVICE='cpu' にしてください")
    return value


@torch.no_grad()
def evaluate_loss(model, examples, batch_size, device):
    if not examples:
        return None
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        inputs, labels = batch_examples(examples[start:start + batch_size], device)
        total += float(loss_sum(model(inputs), labels))
        count += int((labels != -100).sum())
    return total / count


def records_digest(records):
    payload = sorted((row["group"], row["context"], row["question"], row["answer"], row["reason"])
                     for row in records)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(args):
    if args.steps < 1 or args.batch_size < 1 or args.eval_interval < 1:
        raise ValueError("steps, batch-size, eval-interval は1以上にしてください")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate は0より大きくしてください")
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = resolve_device(args.device)
    as_of = validate_date(getattr(args, "as_of_date", AS_OF_DATE) or current_date())
    records = load_records(args.data, as_of)
    output = args.output
    latest = output / "latest.pt"
    best_path = output / "best.pt"
    checkpoint = None
    config = ModelConfig(args.max_length, args.width, args.layers, args.heads, args.dropout)
    # Validate all teacher text before creating models or archiving checkpoints.
    prepared = {row["id"]: example_tokens(row, config.max_length) for row in records}
    if args.resume and latest.is_file():
        model, checkpoint = load_checkpoint(latest, device)
        issue = checkpoint_issue(checkpoint, records)
        reset_allowed = AUTO_RESET_CHANGED_DATA
        length_changed = model.config.max_length != config.max_length
        same_architecture = (model.config.width, model.config.layers, model.config.heads, model.config.dropout) == (
            config.width, config.layers, config.heads, config.dropout)
        if not issue and length_changed and same_architecture:
            if not AUTO_RESET_SEQUENCE_LENGTH:
                raise ValueError("最大長が再開元と異なります。別の保存先で新規学習してください")
            issue = f"最大長を {model.config.max_length} から {config.max_length} に変更したため、新規学習します"
            reset_allowed = AUTO_RESET_SEQUENCE_LENGTH
        if issue:
            if not reset_allowed:
                raise ValueError(issue)
            archive = output / "archive" / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
            archive.mkdir(parents=True, exist_ok=True)
            for path in (latest, best_path):
                if path.exists():
                    shutil.copy2(path, archive / path.name)
            print(f"{issue}。旧モデルを保管: {archive}", flush=True)
            checkpoint = None
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            model = TacticalLM(config).to(device)
        elif model.config != config:
            raise ValueError("再開元とモデル設定が異なります。冒頭のモデル定数を戻すか、別の保存先で新規学習してください")
        if checkpoint and (checkpoint["seed"] != args.seed or checkpoint["validation_fraction"] != args.validation_fraction):
            raise ValueError("再開時の SEED / VALIDATION_FRACTION は変更できません")
        if checkpoint:
            print(f"自作モデルを再開: {latest}")
    else:
        if (latest.exists() or best_path.exists()) and not args.overwrite:
            raise ValueError("既存モデルがあります。再開するか OUTPUT_DIR を変更してください")
        model = TacticalLM(config).to(device)
        print("事前学習済みモデルを使わず、全パラメータをランダム初期化しました。")
    train_rows, validation_rows, assignment = split_records(
        records, args.validation_fraction, args.seed,
        checkpoint.get("group_assignment") if checkpoint else None,
        checkpoint.get("scenario_assignment") if checkpoint else None)
    scenarios = scenario_assignments(
        records, assignment, checkpoint.get("scenario_assignment") if checkpoint else None)
    examples = [prepared[row["id"]] for row in train_rows]
    heldout = [prepared[row["id"]] for row in validation_rows]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=WEIGHT_DECAY)
    step = checkpoint["step"] if checkpoint else 0
    digest = records_digest(validation_rows if heldout else train_rows)
    best = math.inf
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
            group["weight_decay"] = WEIGHT_DECAY
        rng.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.startswith("cuda") and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        if checkpoint.get("score_digest") == digest:
            best = checkpoint["best_loss"]
    output.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    print(f"基準日={as_of}, device={device}, parameters={sum(p.numel() for p in model.parameters()):,}, "
          f"train={len(examples)}, validation={len(heldout)}")
    if not heldout:
        print("評価用の独立した状況がありません。学習損失だけでは未知の質問への品質は分かりません。")
    run_id, started = uuid4().hex, time.monotonic()
    start_step = step
    end_step = step + args.steps
    stop_path = getattr(args, "stop_file", STOP_REQUEST_PATH)
    print(f"[進捗] 0/{args.steps}", flush=True)

    def save_and_log(current_step):
        nonlocal best
        train_loss = evaluate_loss(model, examples, args.batch_size, device)
        validation_loss = evaluate_loss(model, heldout, args.batch_size, device)
        score = validation_loss if validation_loss is not None else train_loss
        improved = score < best
        best = min(score, best)
        result = {"run_id": run_id, "at": datetime.now(timezone.utc).isoformat(),
                  "step": current_step, "train_loss": train_loss,
                  "validation_loss": validation_loss, "score_digest": digest,
                  "train_count": len(examples), "validation_count": len(heldout),
                  "seconds": round(time.monotonic() - started, 2)}
        saved = {**model_metadata(model), "model": model.state_dict(),
                 "knowledge_policy_version": KNOWLEDGE_POLICY_VERSION,
                 "as_of_date": as_of,
                 "data_manifest": data_manifest(records),
                 "optimizer": optimizer.state_dict(), "step": current_step,
                 "seed": args.seed, "validation_fraction": args.validation_fraction,
                 "group_assignment": assignment, "scenario_assignment": scenarios,
                 "score_digest": digest,
                 "best_loss": best, "metrics": result, "python_rng": rng.getstate(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None}
        atomic_save(saved, latest)
        if improved:
            atomic_save(saved, best_path)
        with args.log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
        val = f"{validation_loss:.4f}" if validation_loss is not None else "未評価"
        print(f"step={current_step} train_loss={train_loss:.4f} validation_loss={val}", flush=True)

    try:
        while step < end_step:
            if stop_path is not None and Path(stop_path).exists():
                if step > start_step:
                    save_and_log(step)
                    print("停止しました。現在の学習状態を保存しました。", flush=True)
                else:
                    print("学習の更新前に停止しました。", flush=True)
                return latest
            model.train()
            inputs, labels = batch_examples(rng.choices(examples, k=args.batch_size), device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_sum(model(inputs), labels) / (labels != -100).sum()
            if not torch.isfinite(loss):
                raise ValueError("損失が非有限値になりました。学習率やデータを確認してください")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            step += 1
            if (step - start_step) % PROGRESS_INTERVAL == 0 or step == end_step:
                print(f"[進捗] {step - start_step}/{args.steps}", flush=True)
            if step % args.eval_interval == 0 or step == end_step:
                save_and_log(step)
    except KeyboardInterrupt:
        print("\n中断しました。直前の保存済み latest.pt から再開できます。")
    return latest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--log", type=Path, default=LOG_PATH)
    for name, default in (("steps", TRAIN_STEPS), ("batch-size", BATCH_SIZE),
                          ("eval-interval", EVAL_INTERVAL), ("seed", SEED),
                          ("max-length", MAX_SEQUENCE_LENGTH), ("width", MODEL_WIDTH),
                          ("layers", MODEL_LAYERS), ("heads", MODEL_HEADS)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--validation-fraction", type=float, default=VALIDATION_FRACTION)
    parser.add_argument("--dropout", type=float, default=MODEL_DROPOUT)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=RESUME_TRAINING)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE_EXISTING)
    parser.add_argument("--stop-file", type=Path, default=STOP_REQUEST_PATH)
    parser.add_argument("--as-of-date", default=AS_OF_DATE)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from None
