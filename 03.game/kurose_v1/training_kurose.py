"""KUROSE用の軽量なQLoRA会話ファインチューニング。"""

import json
import random
from pathlib import Path

import torch
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from kurose_prompts import SYSTEM_PROMPT

# 通常使用する設定。対象AIディレクトリからオプションなしで実行する。
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = Path("data/train.jsonl")
EVAL_FILE = Path("data/eval.jsonl")
SOURCE_TEACHER_FILE = Path("../tactical_llm_v1/data/teacher.jsonl")
OUTPUT_DIR = Path("data/kurose_lora")
LOG_DIR = Path("logs/training")

MAX_LENGTH = 512
TEACHER_EVAL_RATIO = 0.2
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
CONTINUATION_LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
RANDOM_SEED = 42


def prepare_training_data():
    train_exists = TRAIN_FILE.is_file()
    eval_exists = EVAL_FILE.is_file()
    if train_exists and eval_exists:
        return
    if train_exists or eval_exists:
        raise FileExistsError(
            "data/train.jsonl と data/eval.jsonl は両方用意するか、"
            "両方削除して前作データから再生成してください"
        )
    if not SOURCE_TEACHER_FILE.is_file():
        raise FileNotFoundError(f"前作教師データがありません: {SOURCE_TEACHER_FILE}")

    grouped_examples = {}
    with SOURCE_TEACHER_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{SOURCE_TEACHER_FILE}:{line_number}: JSON形式を確認してください"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"{SOURCE_TEACHER_FILE}:{line_number}: JSON objectが必要です"
                )
            fields = ("context", "question", "answer")
            if any(
                not isinstance(record.get(field), str) or not record[field].strip()
                for field in fields
            ):
                raise ValueError(
                    f"{SOURCE_TEACHER_FILE}:{line_number}: "
                    "context・question・answerは空でない文字列にしてください"
                )

            group = str(record.get("group") or f"example-{line_number}")
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"状況: {record['context']}\n" f"質問: {record['question']}"
                        ),
                    },
                    {"role": "assistant", "content": record["answer"]},
                ]
            }
            grouped_examples.setdefault(group, []).append(example)

    if len(grouped_examples) < 2:
        raise ValueError("評価用に分離するため、2つ以上のgroupが必要です")

    group_names = list(grouped_examples)
    random.Random(RANDOM_SEED).shuffle(group_names)
    eval_group_count = min(
        len(group_names) - 1,
        max(1, round(len(group_names) * TEACHER_EVAL_RATIO)),
    )
    eval_groups = set(group_names[:eval_group_count])
    train_examples = [
        example
        for group, examples in grouped_examples.items()
        if group not in eval_groups
        for example in examples
    ]
    eval_examples = [
        example
        for group, examples in grouped_examples.items()
        if group in eval_groups
        for example in examples
    ]

    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_FILE.open("w", encoding="utf-8") as file:
        for example in train_examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")
    with EVAL_FILE.open("w", encoding="utf-8") as file:
        for example in eval_examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(
        "前作教師データを変換しました: "
        f"train={len(train_examples)}件, eval={len(eval_examples)}件"
    )


class ConversationDataset(Dataset):
    """1往復の会話を、回答部分だけ学習対象にするデータセット。"""

    def __init__(self, path: Path, tokenizer, max_length: int):
        self.examples = []

        if not path.is_file():
            raise FileNotFoundError(
                f"{path} がありません。JSONL形式の会話データを用意してください。"
            )

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    messages = record["messages"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"{path}:{line_number}: messages形式を確認してください"
                    ) from exc

                valid_messages = (
                    isinstance(messages, list)
                    and len(messages) in (2, 3)
                    and all(
                        isinstance(message, dict)
                        and isinstance(message.get("content"), str)
                        for message in messages
                    )
                    and messages[-2].get("role") == "user"
                    and messages[-1].get("role") == "assistant"
                    and (len(messages) == 2 or messages[0].get("role") == "system")
                )
                if not valid_messages:
                    raise ValueError(
                        f"{path}:{line_number}: "
                        "userとassistantの1往復（任意で先頭にsystem）にしてください"
                    )

                prompt_messages = messages[:-1]
                prompt_ids = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                full_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )

                if len(prompt_ids) >= len(full_ids):
                    continue

                full_ids = full_ids[:max_length]
                prompt_length = min(len(prompt_ids), len(full_ids))
                labels = [-100] * prompt_length + full_ids[prompt_length:]

                if not any(label != -100 for label in labels):
                    continue

                self.examples.append(
                    {
                        "input_ids": full_ids,
                        "attention_mask": [1] * len(full_ids),
                        "labels": labels,
                    }
                )

        if not self.examples:
            raise ValueError(f"{path} に学習可能な会話がありません")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def main():
    prepare_training_data()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA対応GPUが必要です。Google ColabなどのGPUランタイムで実行してください。"
        )

    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    adapter_config_path = OUTPUT_DIR / "adapter_config.json"
    if adapter_config_path.is_file():
        adapter_config = PeftConfig.from_pretrained(str(OUTPUT_DIR))
        if adapter_config.base_model_name_or_path != MODEL_NAME:
            raise ValueError(
                "既存アダプターのベースモデルが設定と一致しません: "
                f"{adapter_config.base_model_name_or_path!r} != {MODEL_NAME!r}"
            )
        model = PeftModel.from_pretrained(
            model,
            str(OUTPUT_DIR),
            is_trainable=True,
        )
        continuing_training = True
        print(f"既存LoRAアダプターを引き継いで追加学習します: {OUTPUT_DIR}")
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
        continuing_training = False
        print("新しいLoRAアダプターで学習を開始します")
    model.print_trainable_parameters()

    train_dataset = ConversationDataset(TRAIN_FILE, tokenizer, MAX_LENGTH)
    eval_dataset = ConversationDataset(EVAL_FILE, tokenizer, MAX_LENGTH)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        logging_dir=str(LOG_DIR),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=(
            CONTINUATION_LEARNING_RATE if continuing_training else LEARNING_RATE
        ),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        fp16=not use_bf16,
        bf16=use_bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        warmup_ratio=0.03,
        weight_decay=0.01,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"LoRAアダプターを保存しました: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
