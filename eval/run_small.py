"""Run the small LoopInfer evaluation suite and write REPORT.md."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common import BASE, MODEL, RESULTS, env, write_json

PY = sys.executable
EVAL = Path(__file__).resolve().parent
LOGS = BASE / "logs"


def call(name, argv, timeout=5400):
    LOGS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    process = subprocess.run(argv, cwd=EVAL, env=env(), capture_output=True, text=True, timeout=timeout)
    (LOGS / f"eval_{name}.log").write_text(
        (process.stdout or "") + "\n[stderr]\n" + (process.stderr or ""), encoding="utf-8",
    )
    ok = process.returncode == 0
    print(f"[{name}] rc={process.returncode} in {time.time() - started:.0f}s", flush=True)
    if not ok:
        print((process.stderr or "")[-1200:], file=sys.stderr, flush=True)
    return ok


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else None


def parity_rows():
    rows = []
    for tag, label in (("fp32", "float32"), ("bf16", "bfloat16")):
        data = load(RESULTS / "parity" / f"verify_{tag}.json")
        if data:
            rows.append({
                "dtype": label, "positions": data["compared_positions"],
                "argmax_matches": data["argmax_matches"],
                "max_logprob_error": f"{data['max_all_logprobs_error']:.2e}",
                "max_hidden_error": f"{data['max_round_hidden_error']:.2e}",
            })
    return rows


def accuracy_rows():
    rows = []
    for engine in ("hf", "loopinfer"):
        data = load(RESULTS / "accuracy" / f"gsm8k_{engine}.json")
        if data:
            rows.append(data)
    return rows


def paired_agreement():
    paths = {engine: RESULTS / "accuracy" / f"gsm8k_{engine}.jsonl" for engine in ("hf", "loopinfer")}
    if not all(path.is_file() for path in paths.values()):
        return None
    tables = {}
    for engine, path in paths.items():
        tables[engine] = {
            row["index"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        }
    shared = sorted(set(tables["hf"]) & set(tables["loopinfer"]))
    if not shared:
        return None
    same = sum(tables["hf"][index]["text"] == tables["loopinfer"][index]["text"] for index in shared)
    same_correct = sum(
        tables["hf"][index]["correct"] == tables["loopinfer"][index]["correct"] for index in shared
    )
    same_predicted = sum(
        tables["hf"][index]["predicted"] == tables["loopinfer"][index]["predicted"] for index in shared
    )
    return {
        "questions": len(shared), "identical_text": same,
        "same_correctness": same_correct, "same_prediction": same_predicted,
    }


def perf_rows():
    rows = []
    for engine in ("hf", "loopinfer"):
        data = load(RESULTS / "perf" / f"perf_{engine}.json")
        if not data:
            continue
        for run in data["runs"]:
            rows.append({
                "engine": engine,
                "concurrency": run.get("concurrency"),
                "output_tokens_per_second": run.get("output_tokens_per_second"),
                "ttft_p50_ms": run.get("ttft_p50_ms", run.get("prefill_p50_ms")),
                "inter_token_p50_ms": run.get("inter_token_p50_ms"),
                "gpu_peak_mib": run.get("gpu_peak_mib"),
            })
    return rows


def report(parity, accuracy, agreement, perf):
    lines = [
        "# LoopInfer 小评测报告", "",
        f"- 模型：`{MODEL}`（Ouro-1.4B，4 loops，bf16）",
        "- 硬件：单卡 NVIDIA H20 96G",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}",
        "", "## 1. 数值对齐（vs 官方 Transformers）", "",
        "| dtype | 对比位置 | argmax 一致 | 最大 logprob 误差 | 最大隐状态误差 |",
        "|---|---|---|---|---|",
    ]
    for row in parity:
        lines.append(
            f"| {row['dtype']} | {row['positions']} | {row['argmax_matches']}/{row['positions']} | "
            f"{row['max_logprob_error']} | {row['max_hidden_error']} |"
        )
    lines += ["", "## 2. GSM8K 准确率", "", "| 引擎 | 题数 | 答对 | 准确率 | 生成 tok/s |", "|---|---|---|---|---|"]
    for row in accuracy:
        lines.append(
            f"| {row['engine']} | {row['questions']} | {row['correct']} | {row['accuracy']:.1%} | "
            f"{row['generated_tokens_per_second']:.1f} |"
        )
    if agreement:
        lines += [
            "", f"逐题配对（{agreement['questions']} 题）：输出文本完全相同 {agreement['identical_text']} 题，"
            f"判定结果相同 {agreement['same_correctness']} 题，抽取答案相同 {agreement['same_prediction']} 题。",
        ]
    lines += ["", "## 3. 性能（32 入 / 32 出）", "",
              "| 引擎 | 并发 | 输出 tok/s | 首 token 或 prefill p50 | 单 token p50 | GPU 峰值 |",
              "|---|---|---|---|---|---|"]
    for row in perf:
        lines.append(
            f"| {row['engine']} | {row['concurrency']} | {row['output_tokens_per_second']:.1f} | "
            f"{row['ttft_p50_ms']:.1f} ms | {row['inter_token_p50_ms']:.1f} ms | {row['gpu_peak_mib']} MiB |"
        )
    lines += [
        "", "## 4. 结论与局限", "",
        "- 数值：fp32 应与官方逐位一致；bf16 只要求 argmax 一致，logprob 偏差属累积误差。",
        "- 准确率：两引擎应落在同一水平；逐题配对差异才是有意义的信号。",
        "- 性能：HF 侧 prefill 与生成分别计时，LoopInfer 用客户端观测的 TTFT/TPOT，口径不完全等同，仅作量级参考。",
        "- 样本仅 100 题，绝对准确率置信区间约 ±5pp；用于比对两台引擎足够，不能当论文数字。",
    ]
    (BASE / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-new", type=int, default=192)
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--report-only", action="store_true", help="Regenerate REPORT.md from existing results")
    args = parser.parse_args()

    if args.report_only:
        report(parity_rows(), accuracy_rows(), paired_agreement(), perf_rows())
        return

    steps = {}
    if "parity" not in args.skip:
        for tag, dtype, prompts, steps_count in (("fp32", "float32", 4, 4), ("bf16", "bfloat16", 8, 8)):
            steps[f"parity_{tag}"] = call(
                f"parity_{tag}",
                [PY, str(BASE / "repo/tools/verify_ouro.py"), "--model", str(MODEL), "--revision", "main",
                 "--device", "cuda", "--dtype", dtype, "--loops", "4", "--steps", str(steps_count),
                 "--output", str(RESULTS / "parity" / f"verify_{tag}.json")],
            )
    if "accuracy" not in args.skip:
        for engine in ("loopinfer", "hf"):
            steps[f"accuracy_{engine}"] = call(
                f"accuracy_{engine}",
                [PY, "accuracy.py", "--engine", engine, "--count", str(args.count), "--max-new", str(args.max_new)],
            )
    if "perf" not in args.skip:
        for engine, extra in (("loopinfer", ["--concurrency", "1", "4", "8"]), ("hf", [])):
            steps[f"perf_{engine}"] = call(
                f"perf_{engine}",
                [PY, "perf.py", "--engine", engine, "--input-tokens", "32", "--output-tokens", "32"] + extra,
            )

    write_json(RESULTS / "run_small_status.json", {"steps": steps, "finished": time.strftime("%F %T")})
    report(parity_rows(), accuracy_rows(), paired_agreement(), perf_rows())
    print(json.dumps(steps, indent=2))


if __name__ == "__main__":
    main()
