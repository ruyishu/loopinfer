# LoopInfer on H1（单卡 H20）

LoopInfer v0.2 在这台机器上的落地与评测：能加载真实 Ouro-1.4B、能流式服务、数值与官方一致，解码走 CUDA Graph。

## 目录

| 路径 | 内容 |
|---|---|
| `repo/` | loopinfer 源码（v0.2.0）+ `eval/` 评测套件 |
| `repo/eval/` | 准确率、性能、报告脚本，见 `repo/eval/README.md` |
| `models/Ouro-1.4B/` | 真实权重 2.7G |
| `data/gsm8k_test.jsonl` | GSM8K test 1319 题 |
| `results/` | `parity/`、`accuracy/`、`perf/` |
| `logs/` | 各步骤日志 |
| `cache/hf-modules/` | `trust_remote_code` 缓存 |
| `scripts/setup.sh` | 初次安装与模型下载 |

## 环境

用现成 conda `py3.12`（`/opt/app/conda/envs/py3.12`），不需要 venv：

```bash
python -m pip install -e repo --no-deps --no-build-isolation
```

conda 的 `bin/` 不可写，控制台命令装不上，统一用 `python -m loopinfer`。

## 跑法

```bash
PY=/opt/app/conda/envs/py3.12/bin/python
$PY -m loopinfer generate models/Ouro-1.4B --device cuda --loops 4 \
  --prompt "The capital of France is" --max-tokens 16
$PY -m loopinfer serve models/Ouro-1.4B --device cuda --loops 4 --port 18080 \
  --context 512 --max-batch 8 --kv-mib 3072
$PY repo/eval/run_small.py --count 100 --max-new 192    # 全部评测，约 30 分钟
$PY repo/eval/run_small.py --report-only                # 只重生成 REPORT.md
```

## 最近结果（v0.3，2026-11-18）

数值对齐：fp32 32/32 argmax（logprob 误差 2.29e-05）；bf16 61/64（8 步后累积误差）。

GSM8K 100 题：HF 21/100，LoopInfer 20/100，与 HF 逐题判定一致 99/100。

性能（32 入 / 32 出）：

| 引擎 | 并发 1 | 并发 4 | 并发 8 | 单 token p50 |
|---|---|---|---|---|
| HF Transformers | 16.3 tok/s | 64.6 tok/s | 128.2 tok/s | 61–62 ms |
| LoopInfer v0.2（服务端） | 48.4 | 115.4 | 213.1 | 18–22 ms |
| LoopInfer v0.3（服务端） | 49.1 | 114.7 | 208.7 | 18–23 ms |
| LoopInfer v0.3（进程内） | 49.5 | 122.5 | 226.8 | — |

详细分析与分阶段耗时见 `repo/eval/REPORT_H1.md`。

## v0.3 这版做了什么

- **prefill/decode 拆分子批**：混入 prefill 不再把整批拖回 eager。
- **按需分页 + 抢占**：按实际增长分配页，池子不够时抢占最新请求重跑，抢占期间关闭准入。
- **页表在设备上构造**：每步只上传每行标量与补齐页表，槽位和因果掩码在 GPU 上生成。
- **接口**：`/v1/chat/completions`、`logprobs`、`stop`、`n`，chat 模板来自 checkpoint 或 `--chat-template`。
- **可观测**：`/metrics` 增加 `step_ms_*`（prepare/forward/sample/commit/step）、抢占与图计数。
- **实验开关**：`--attention triton`（分页注意力 kernel，数值正确但更慢，默认关闭）、`--graph-budget`。

## 已知问题

- 长上下文（2K 以上）每步耗时翻倍，主要花在逐层 gather + masked_fill；分页注意力 kernel 还没做到更快。
- 首 token 仍偏慢：prefill 走 eager，且新形状首次请求要付图捕获成本。
- HF 批量生成带 padding 会触发 Ouro 自定义注意力报错，准确率基线只能 `--batch 1`。
- 共享盘 `HF_HOME` 只读，需把 `HF_MODULES_CACHE` 指到 `cache/hf-modules`。
