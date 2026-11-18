"""Measure GSM8K accuracy with one engine (HF Transformers or LoopInfer)."""

import argparse
import json
import time
from pathlib import Path

from common import MODEL, RESULTS, build_prompt, env, load_gsm8k, parse_number, write_json

env()


def summarise(rows, elapsed, max_new):
    correct = sum(row["correct"] for row in rows)
    tokens = sum(row["tokens"] for row in rows)
    return {
        "engine": rows[0]["engine"] if rows else None,
        "questions": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 4) if rows else None,
        "max_new_tokens": max_new,
        "generated_tokens": tokens,
        "elapsed_seconds": round(elapsed, 2),
        "generated_tokens_per_second": round(tokens / elapsed, 2) if elapsed else None,
    }


def run_hf(items, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()

    rows, started = [], time.perf_counter()
    for start in range(0, len(items), args.batch):
        chunk = items[start:start + args.batch]
        prompts = [build_prompt(tokenizer, item["question"]) for item in chunk]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **encoded, max_new_tokens=args.max_new, do_sample=False, pad_token_id=0, eos_token_id=0,
            )
        prompt_width = encoded["input_ids"].shape[1]
        for index, item in enumerate(chunk):
            new_tokens = output[index, prompt_width:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            predicted = parse_number(text)
            row = {
                "engine": "hf", "index": start + index, "question": item["question"], "gold": item["gold"],
                "predicted": predicted, "correct": predicted == item["gold"],
                "tokens": int((new_tokens != 0).sum().item()), "text": text[-1200:],
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("index", "gold", "predicted", "correct")}), flush=True)
    return rows, time.perf_counter() - started


def run_loopinfer(items, args):
    from loopinfer import Engine, Sampling

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompts = [build_prompt(tokenizer, item["question"]) for item in items]
    started = time.perf_counter()
    with Engine(str(MODEL), device="cuda", loops=args.loops, context=512, max_batch=args.batch) as engine:
        results = engine.generate(prompts, Sampling(max_tokens=args.max_new, temperature=0.0))
    elapsed = time.perf_counter() - started

    rows = []
    for index, (item, result) in enumerate(zip(items, results)):
        predicted = parse_number(result.text)
        row = {
            "engine": "loopinfer", "index": index, "question": item["question"], "gold": item["gold"],
            "predicted": predicted, "correct": predicted == item["gold"],
            "tokens": len(result.token_ids), "ttft_seconds": round(result.first_token_seconds, 4),
            "elapsed_seconds": round(result.elapsed_seconds, 4), "text": result.text[-1200:],
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("index", "gold", "predicted", "correct")}), flush=True)
    return rows, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["hf", "loopinfer"], required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-new", type=int, default=192)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=RESULTS / "accuracy")
    args = parser.parse_args()

    items = load_gsm8k(args.count, seed=args.seed)
    runner = run_hf if args.engine == "hf" else run_loopinfer
    rows, elapsed = runner(items, args)
    summary = summarise(rows, elapsed, args.max_new)
    summary.update({"seed": args.seed, "batch": args.batch, "loops": args.loops})

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"gsm8k_{args.engine}.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(args.out / f"gsm8k_{args.engine}.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
