"""Shared paths and helpers for the LoopInfer evaluation suite."""

import json
import os
import random
import re
from pathlib import Path

BASE = Path(os.environ.get(
    "LOOPINFER_BASE",
    "/data/tripplanner-gps-share/qyc/ccot/project/step-level/q38/loop",
))
REPO = BASE / "repo"
MODEL = BASE / "models" / "Ouro-1.4B"
GSM8K = BASE / "data" / "gsm8k_test.jsonl"
RESULTS = BASE / "results"

INSTRUCTION = (
    "Solve the problem step by step. "
    "End your reply with a final line exactly in the form: The answer is <number>"
)


def env():
    value = dict(os.environ)
    value.setdefault("PYTHONNOUSERSITE", "1")
    value.setdefault("CUDA_VISIBLE_DEVICES", "0")
    value.setdefault("HF_HOME", "/data/tripplanner/.cache/huggingface")
    value["HF_MODULES_CACHE"] = str(BASE / "cache" / "hf-modules")
    value["TOKENIZERS_PARALLELISM"] = "false"
    return value


NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_number(text):
    """Return the last number in the text, or None."""
    matches = NUMBER.findall(text or "")
    if not matches:
        return None
    raw = matches[-1].replace(",", "").rstrip(".")
    try:
        return str(float(raw)).rstrip("0").rstrip(".") if "." in raw else str(int(raw))
    except ValueError:
        return None


def gold_answer(answer):
    return parse_number(answer.split("####")[-1])


def load_gsm8k(count, seed=2026, path=GSM8K):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if count < len(rows):
        picks = sorted(random.Random(seed).sample(range(len(rows)), count))
        rows = [rows[index] for index in picks]
    return [
        {"question": row["question"], "gold": gold_answer(row["answer"])}
        for row in rows
    ]


def build_prompt(tokenizer, question):
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n\n{question}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
