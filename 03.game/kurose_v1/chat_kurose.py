"""学習済みKUROSE LoRAアダプターとの対話実行。"""

from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from kurose_prompts import SYSTEM_PROMPT

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = Path("data/kurose_lora")
MAX_NEW_TOKENS = 256
MAX_HISTORY_TURNS = 4
TEMPERATURE = 0.7
TOP_P = 0.9


def load_kurose():
    adapter_config_path = ADAPTER_DIR / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(
            f"学習済みアダプターがありません: {ADAPTER_DIR}。"
            "先にtraining_kurose.pyを実行してください。"
        )

    adapter_config = PeftConfig.from_pretrained(str(ADAPTER_DIR))
    if adapter_config.base_model_name_or_path != MODEL_NAME:
        raise ValueError(
            "アダプターのベースモデルが設定と一致しません: "
            f"{adapter_config.base_model_name_or_path!r} != {MODEL_NAME!r}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()
    return model, tokenizer


def generate_reply(model, tokenizer, messages):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_device = model.get_input_embeddings().weight.device
    prompt = prompt.to(input_device)

    generation_args = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if TEMPERATURE > 0:
        generation_args.update(
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    else:
        generation_args["do_sample"] = False

    with torch.inference_mode():
        output = model.generate(prompt, **generation_args)

    new_tokens = output[0, prompt.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    model, tokenizer = load_kurose()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("KUROSEとの対話を開始します。/reset で履歴を消去、/exit で終了します。")

    while True:
        try:
            user_text = input("あなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("会話履歴を消去しました。")
            continue

        messages.append({"role": "user", "content": user_text})
        history_limit = MAX_HISTORY_TURNS * 2
        active_messages = [messages[0], *messages[1:][-history_limit:]]
        assistant_text = generate_reply(model, tokenizer, active_messages)
        messages.append({"role": "assistant", "content": assistant_text})
        print(f"KUROSE> {assistant_text}")


if __name__ == "__main__":
    main()
