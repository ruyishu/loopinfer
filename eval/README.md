# eval/ — LoopInfer 小评测套件

三件事：算得对不对、答得好不好、跑得快不快。全部对着官方 Transformers 实现比。

## 用法

```bash
cd /data/tripplanner-gps-share/qyc/ccot/project/step-level/q38/loop
PY=/opt/app/conda/envs/py3.12/bin/python

# 全部跑一遍（约 30 分钟），产出 REPORT.md
$PY repo/eval/run_small.py --count 100 --max-new 192

# 只跑某一部分
$PY repo/eval/run_small.py --skip accuracy perf
$PY repo/eval/accuracy.py --engine loopinfer --count 100
$PY repo/eval/perf.py --engine hf --concurrency 1 4 --input-tokens 32 --output-tokens 32
```

## 组成

| 文件 | 作用 | 产物 |
|---|---|---|
| `common.py` | 路径、环境变量、GSM8K 读取、答案抽取 | — |
| `accuracy.py` | 单引擎跑 GSM8K，贪心解码，最后数字比对 | `results/accuracy/gsm8k_<engine>.json(l)` |
| `perf.py` | LoopInfer 走 HTTP 压测；HF 走 prefill + 批量生成计时 | `results/perf/perf_<engine>.json` |
| `run_small.py` | 串起数值对齐 → 准确率 → 性能，生成 `REPORT.md` | `REPORT.md` |

数值对齐复用 `repo/tools/verify_ouro.py`：逐 token 喂同一参考 token，比较全词表 logprob 与每轮隐状态。

## 口径

- 模型固定 `models/Ouro-1.4B`，4 loops，bf16；准确率对比必须同 prompt、同 max_new_tokens、同贪心。
- 准确率样本默认 100 题（seed 2026），配对比较两引擎；绝对准确率噪声约 ±5pp。
- 性能统一 32 入 / 32 出；LoopInfer 报客户端观测的 TTFT/TPOT，HF 报 prefill 与 generate 耗时，
  两者口径不同，只作量级参考。
- fp32 用于判语义（要求 logprob 误差 < 1e-4），bf16 只要求 argmax 一致。

## 目录约定

```
q38/loop/
  repo/          代码（含本目录）
  models/        权重
  data/          GSM8K 等评测数据
  results/       parity / accuracy / perf / figures
  logs/          各步骤日志
  cache/         HF modules 缓存（共享盘 HF_HOME 只读）
  scripts/       环境安装脚本
  REPORT.md      最近一次评测结果
```
