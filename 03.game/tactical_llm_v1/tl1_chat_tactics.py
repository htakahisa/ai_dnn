"""Ask text-only tactical questions using the locally trained model."""

# 通常設定はここで変更。
from pathlib import Path

AI_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = AI_DIR / "data" / "checkpoints" / "best.pt"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0
DEVICE = "auto"
CPU_THREADS = 4
DATA_PATH = AI_DIR / "data" / "teacher.jsonl"
AS_OF_DATE = None

import argparse
import torch

from tl1_collect_tactics import read_multiline
from tl1_model_tactics import fit_history, generate, load_checkpoint
from tl1_train_tactics import resolve_device
from tl1_data_tactics import load_records
from tl1_dates_tactics import checkpoint_issue, current_date, knowledge_signature, validate_date
from tl1_conversations_tactics import history_for_knowledge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--as-of-date", default=AS_OF_DATE)
    args = parser.parse_args()
    as_of = validate_date(args.as_of_date or current_date())
    if not args.checkpoint.is_file():
        raise ValueError("学習済みモデルがありません。教師データ登録後に tl1_train_tactics.py を実行してください")
    torch.set_num_threads(CPU_THREADS)
    model, checkpoint = load_checkpoint(args.checkpoint, resolve_device(args.device))
    issue = checkpoint_issue(checkpoint, load_records(DATA_PATH, as_of), require_current_dates=True)
    if issue:
        raise ValueError(issue)
    print(f"自作モデル: {args.checkpoint} / step={checkpoint['step']}")
    print("回答は学習途上のモデルが生成します。ゲームの現在状態は自動取得しません。")
    if checkpoint["metrics"]["validation_loss"] is None:
        print("このモデルは独立した状況での評価がまだありません。")
    print("終了は質問欄で /quit、新しい会話は /new。前の会話を参照して続けて質問できます。")
    history, previous_context = [], ""
    while True:
        question = read_multiline("質問")
        if question == "/quit":
            break
        if question == "/new":
            history, previous_context = [], ""
            print("新しい会話を開始しました。")
            continue
        if not question:
            continue
        context = read_multiline("状況・条件（空欄なら前回の状況を引き継ぐ）") or previous_context
        try:
            as_of = validate_date(args.as_of_date or current_date())
            rows = load_records(DATA_PATH, as_of)
            issue = checkpoint_issue(checkpoint, rows, require_current_dates=True)
            if issue:
                raise ValueError(issue)
            signature = knowledge_signature(rows)
            included, omitted = fit_history(context, question, history_for_knowledge(history, signature),
                                            model.config.max_length, MAX_NEW_TOKENS)
            text, ended = generate(model, context, question, max_new_tokens=MAX_NEW_TOKENS,
                                   temperature=TEMPERATURE, history=included)
            if text:
                history.append({"question": question, "context": context, "answer": text,
                                "status": "completed", "knowledge_signature": signature})
            previous_context = context
            print("\n回答:\n" + (text or "（回答を生成できませんでした）"))
            if not ended:
                print("（生成長または入力＋出力の最大長に達しました）")
            if omitted:
                print(f"（古い会話 {omitted} 往復は長さ制限のため参照していません）")
        except ValueError as exc:
            print(str(exc))


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\n終了しました。")
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from None
