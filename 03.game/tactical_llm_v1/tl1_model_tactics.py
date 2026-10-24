"""Randomly initialized, byte-level causal Transformer. No pretrained weights."""

from dataclasses import asdict, dataclass
import codecs

import torch
from torch import nn
from torch.nn import functional as F

PAD, BOS, EOS, CONTEXT, QUESTION, ANSWER = range(6)
BYTE_OFFSET = 6
VOCAB_SIZE = 256 + BYTE_OFFSET
FORMAT_VERSION = 1

from tl1_conversations_tactics import validate_history


def encode(text):
    return [value + BYTE_OFFSET for value in text.encode("utf-8")]


def prompt_tokens(context, question, history=None):
    tokens = [BOS]
    for turn in validate_history(history or []):
        tokens.extend([CONTEXT, *encode(turn["context"]), QUESTION, *encode(turn["question"]),
                       ANSWER, *encode(turn["answer"]), EOS])
    return [*tokens, CONTEXT, *encode(context), QUESTION, *encode(question), ANSWER]


def fit_history(context, question, history, max_length, max_new_tokens):
    core = prompt_tokens(context, question)
    if len(core) >= max_length:
        raise ValueError("状況と質問が長すぎます。短くして再入力してください")
    history = list(validate_history(history or []))
    original = len(history)
    reserve = min(max_new_tokens, max_length - len(core))
    while history and len(prompt_tokens(context, question, history)) + reserve > max_length:
        history.pop(0)
    return history, original - len(history)


def response_text(record):
    return record["answer"] + (
        "\n判断理由：" + record["reason"] if record.get("reason") else ""
    )


def example_tokens(record, max_length):
    prompt = prompt_tokens(record["context"], record["question"], record.get("history"))
    tokens = prompt + encode(response_text(record)) + [EOS]
    if len(tokens) - 1 > max_length:
        raise ValueError(
            f"データ {record['id']}: {len(tokens) - 1} tokens > MAX_SEQUENCE_LENGTH={max_length}。"
            "文章を短くするか定数を増やしてください（日本語は通常1文字3 bytes）。"
        )
    # Predict only the teacher's response, including its first byte and EOS.
    labels = [-100] * (len(prompt) - 1) + tokens[len(prompt):]
    return tokens[:-1], labels


def batch_examples(examples, device):
    length = max(len(row[0]) for row in examples)
    x = torch.full((len(examples), length), PAD, dtype=torch.long, device=device)
    y = torch.full_like(x, -100)
    for index, (inputs, labels) in enumerate(examples):
        x[index, :len(inputs)] = torch.tensor(inputs, device=device)
        y[index, :len(labels)] = torch.tensor(labels, device=device)
    return x, y


@dataclass
class ModelConfig:
    max_length: int = 2048
    width: int = 192
    layers: int = 4
    heads: int = 6
    dropout: float = 0.1


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.dropout = config.dropout
        self.norm1 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.projection = nn.Linear(config.width, config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.ff = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(),
                                nn.Linear(4 * config.width, config.width),
                                nn.Dropout(config.dropout))

    def forward(self, x):
        b, t, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = [a.reshape(b, t, self.heads, width // self.heads).transpose(1, 2)
                   for a in (q, k, v)]
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        ).transpose(1, 2).reshape(b, t, width)
        x = x + F.dropout(self.projection(attended), self.dropout, self.training)
        return x + self.ff(self.norm2(x))


class TacticalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.width <= 0 or config.heads <= 0 or config.width % config.heads:
            raise ValueError("MODEL_WIDTH は MODEL_HEADS の正の倍数にしてください")
        if config.layers < 1 or config.max_length < 2 or not 0 <= config.dropout < 1:
            raise ValueError("モデル設定が不正です")
        self.config = config
        self.token = nn.Embedding(VOCAB_SIZE, config.width)
        self.position = nn.Embedding(config.max_length, config.width)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, VOCAB_SIZE, bias=False)
        self.apply(self._initialize)
        self.output.weight = self.token.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        if inputs.shape[1] > self.config.max_length:
            raise ValueError("入力がモデルの最大長を超えました")
        positions = torch.arange(inputs.shape[1], device=inputs.device)
        x = self.token(inputs) + self.position(positions)
        return self.output(self.norm(self.blocks(x)))


def loss_sum(logits, labels):
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                           ignore_index=-100, reduction="sum")


def load_checkpoint(path, device="cpu"):
    # Only load checkpoints you created/trust; optimizer/RNG tuples use pickle.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("対応していないチェックポイント形式です")
    model = TacticalLM(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint


@torch.no_grad()
def generate(model, context, question, *, max_new_tokens=512, temperature=0.0, stop_event=None, history=None):
    """Generate valid UTF-8 only, preserving the entire supplied prompt."""
    if max_new_tokens < 1 or temperature < 0:
        raise ValueError("生成長は1以上、temperature は0以上にしてください")
    model.eval()
    history, _ = fit_history(context, question, history, model.config.max_length, max_new_tokens)
    tokens = prompt_tokens(context, question, history)
    if len(tokens) >= model.config.max_length:
        raise ValueError("状況と質問が長すぎます。短くして再入力してください")
    output = bytearray()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    device = next(model.parameters()).device
    ended = False
    for _ in range(min(max_new_tokens, model.config.max_length - len(tokens))):
        if stop_event is not None and stop_event.is_set():
            break
        logits = model(torch.tensor([tokens], device=device))[0, -1].clone()
        state = decoder.getstate()
        allowed = []
        if not state[0]:
            allowed.append(EOS)
        for value in range(256):
            decoder.setstate(state)
            try:
                decoder.decode(bytes([value]), final=False)
                # Exclude non-printing ASCII control bytes except LF/tab.
                if value >= 32 and value != 127 or value in (9, 10):
                    allowed.append(value + BYTE_OFFSET)
            except UnicodeDecodeError:
                pass
        decoder.setstate(state)
        mask = torch.full_like(logits, float("-inf"))
        mask[allowed] = logits[allowed]
        next_token = (int(mask.argmax()) if temperature == 0 else
                      int(torch.multinomial(F.softmax(mask / temperature, dim=-1), 1)))
        tokens.append(next_token)
        if next_token == EOS:
            ended = True
            break
        value = next_token - BYTE_OFFSET
        output.append(value)
        decoder.decode(bytes([value]), final=False)
    # A length limit may interrupt a multibyte codepoint; discard that tail only.
    return output.decode("utf-8", errors="ignore"), ended


def model_metadata(model):
    return {"format_version": FORMAT_VERSION, "model_config": asdict(model.config)}
