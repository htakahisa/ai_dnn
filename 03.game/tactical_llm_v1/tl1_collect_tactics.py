"""Register the user's tactical questions, correct answers, and explanations."""

# 通常設定はここで変更。tactical_llm_v1 を作業ディレクトリとして実行。
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "teacher.jsonl"
MULTILINE_END = "."

import json
from datetime import datetime, timezone
from uuid import uuid4

from tl1_data_tactics import SCHEMA_VERSION, default_group, validate_record
from tl1_dates_tactics import validate_date


def read_multiline(label):
    print(f"{label}（複数行可。最後に {MULTILINE_END} だけの行で確定）")
    lines = []
    while True:
        line = input()
        if line == MULTILINE_END:
            return "\n".join(lines).strip()
        lines.append(line)


def main():
    print("戦術の教師データを登録します。終了は質問欄で /quit。")
    print("ゲーム中に分からない敵の位置は、推測と明記してください。")
    while True:
        question = read_multiline("質問")
        if question == "/quit":
            return
        if not question:
            print("質問を入力してください。")
            continue
        context = read_multiline("状況・条件（省略可）")
        answer = read_multiline("あなたが正しいと考える回答")
        reason = read_multiline("判断理由・例外条件（省略可）")
        group = input("同じ状況のグループ名（空欄なら自動）: ").strip()
        valid_from = input("適用開始日（YYYY-MM-DD、空欄なら共通）: ").strip()
        try:
            if valid_from:
                valid_from = validate_date(valid_from)
        except ValueError as exc:
            print(f"保存しませんでした: {exc}")
            continue
        row = {"schema_version": SCHEMA_VERSION, "id": uuid4().hex,
               "context": context, "question": question, "answer": answer,
               "reason": reason, "group": group or default_group(context, question),
               "created_at": datetime.now(timezone.utc).isoformat(), "source": "user",
               "scope": "dated" if valid_from else "common", "valid_from": valid_from}
        try:
            validate_record(row)
        except ValueError as exc:
            print(f"保存しませんでした: {exc}")
            continue
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DATA_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"保存しました: {DATA_PATH} / id={row['id']}")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\n終了しました。入力途中のデータは保存していません。")
