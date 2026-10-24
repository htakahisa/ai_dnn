"""Export heldout responses alongside teacher answers for human tactical review."""

# 通常設定はここで変更。
from pathlib import Path

AI_DIR = Path(__file__).resolve().parent
DATA_PATH = AI_DIR / "data" / "teacher.jsonl"
CHECKPOINT_PATH = AI_DIR / "data" / "checkpoints" / "best.pt"
REPORT_PATH = AI_DIR / "logs" / "evaluation.jsonl"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 1024
MAX_REVIEW_EXAMPLES = 20
DEVICE = "auto"
CPU_THREADS = 4
AS_OF_DATE = None

import argparse
import json
import torch

from tl1_data_tactics import load_records, scenario_key
from tl1_model_tactics import example_tokens, generate, load_checkpoint, response_text
from tl1_train_tactics import evaluate_loss, resolve_device
from tl1_dates_tactics import checkpoint_issue, current_date, validate_date


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--as-of-date", default=AS_OF_DATE)
    args = parser.parse_args()
    as_of = validate_date(args.as_of_date or current_date())
    torch.set_num_threads(CPU_THREADS)
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    records = load_records(args.data, as_of)
    issue = checkpoint_issue(checkpoint, records, require_current_dates=True)
    if issue:
        raise ValueError(issue)
    # Only use groups deliberately withheld by this checkpoint's training run.
    heldout = [row for row in records
               if checkpoint["group_assignment"].get(row["group"]) == "validation"]
    training_contexts = {row["context"].strip() for row in records
                         if checkpoint["group_assignment"].get(row["group"]) == "train"
                         and row["context"].strip()}
    if any(row["context"].strip() in training_contexts for row in heldout):
        raise ValueError("データ変更により同じ状況が学習・評価の両方に存在します。再分割してください")
    if any(checkpoint.get("scenario_assignment", {}).get(scenario_key(row)) == "train"
           for row in heldout):
        raise ValueError("以前学習した状況が評価側に移っています。新規学習してください")
    if not heldout:
        raise ValueError("評価用グループがありません。複数の異なる状況を登録して学習してください")
    examples = [example_tokens(row, model.config.max_length) for row in heldout]
    loss = evaluate_loss(model, examples, BATCH_SIZE, device)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"type": "metrics", "checkpoint": str(args.checkpoint),
                               "step": checkpoint["step"], "validation_count": len(heldout),
                               "validation_loss": loss}, ensure_ascii=False) + "\n")
        for row in heldout[:MAX_REVIEW_EXAMPLES]:
            text, ended = generate(model, row["context"], row["question"],
                                   max_new_tokens=MAX_NEW_TOKENS, history=row.get("history"))
            file.write(json.dumps({"type": "review", "id": row["id"], "context": row["context"],
                                   "question": row["question"], "teacher": response_text(row),
                                   "history": row.get("history", []),
                                   "generated": text, "ended": ended}, ensure_ascii=False) + "\n")
    print(f"validation_loss={loss:.4f} / {len(heldout)} 件。回答比較: {args.report}")
    print("損失だけで戦術の正しさは判定できません。提案・理由・例外条件を正解と比較してください。")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from None
