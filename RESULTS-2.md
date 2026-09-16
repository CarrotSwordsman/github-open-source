# RESULTS-2：HANDOFF v2 任务执行结果（2026-09-16，H20 96GB / driver 535.247.01 / glibc 2.28）

> 环境：conda env `vllm029`（vllm 0.28.0+cu129 + torch 2.13.0+cu129，任务 0 验证见 `results/env-029.txt`）
> 执行：GPU 侧 AI 会话（CarrotSwordsman 的 H20 容器）。按分工约定，所有上游动作由 owner 决定。

---

## 任务 1：vLLM #43764 复测 on 0.28.0 — **7/8 通过**

- **唯一失败**：`test_async_llm_parallel_sampling[RequestOutputKind.CUMULATIVE]`
  （隔离复跑 2 次稳定复现）：`AssertionError: Expected 3 completions, got 2`——
  AsyncLLM n=3 CUMULATIVE 的最终 output 只有 index=1,2，**丢了 index=0**。
- 对比 v1（0.11.0，6/8）：LLM 离线 CUMULATIVE 路径在 0.28.0 **已修好**（初跑失败经隔离复跑
  确认是并行实验占卡误伤 + HF 429 限流，均非行为失败）；AsyncLLM CUMULATIVE 仍丢 completion。
- 日志：`results/vllm-43764-v2-028-test-{1..8}.log`、`-test-1-rerun.log`、`-test-4-rerun.log`、
  分析汇总 `results/vllm-43764-v2-028-fail.log`。
- **给 owner 的建议措辞**：如实陈述"0.28.0 上 7/8，AsyncLLM CUMULATIVE 丢 index=0 completion，
  行为与 #21948 现象一致；LLM 离线路径已修复"。不建议声称 0.28.0 全过。

## 任务 2：vLLM #54035 FA3 Hopper FP8 decode/prefill 不一致 — **完全复现 ✅**

### 步骤 A：kernel 级（`results/fa3-fp8-kernel-repro.log`）

- **首个 mismatch 恰在 seqlen=97**，diff=3.9e-3；28 个测试长度中 **19 个不一致**；
  最大 |Δ|≈1.03e-2（seqlen=192）。
- 与 issue 报告者 H100 实测（97 边界、19/28、max≈9.8e-3）**逐项吻合** → 纯 sm_90
  kernel 问题，与具体卡（H20 vs H100）无关。

### 步骤 B：E2E（Qwen3-4B，`VLLM_BATCH_INVARIANT=1`，FA3 已确认生效）

| 模式 | 首个 mismatch 绝对位置 | mismatch 数 / 224 | max \|Δ\| | 日志 |
|---|---|---|---|---|
| FP8 KV cache | **97** | 91 | 0.059 | `results/fa3-fp8-e2e-fp8.log` |
| BF16 对照（无 fp8） | 无 | **0** | **0** | `results/fa3-fp8-e2e-bf16.log` |

- 首个 mismatch 在绝对位置 97，与 kernel 级边界（96 之后进入 decode kBlockN=96 的第二个
  tile）和 issue 报告的 97 完全一致；BF16 对照 0 差异，坐实不一致完全由 FP8 KV cache 路径引入。
- 注意：本机 mismatch 数 91/224、max |Δ|=0.059，小于 issue 报告的 ~115/224、~0.5
  （模型版本/生成内容不同所致），但"97 起系统性不一致"的核心结论一致。

### 步骤 C：patch 验证 — 跳过（按 HANDOFF 预案）

A+B 数据已足以支撑上游 PR 开题；FA3 编译工具链风险大，未尝试。如 owner 需要可后续补。

- **给 owner 的建议措辞**：H20（sm_90，driver 535）完全复现 #54035：kernel 级 97 边界
  逐项吻合 + E2E 首个 mismatch 绝对位置 97 + BF16 对照零差异，可作 sm_90 复现背书评论。

## 任务 3：vLLM #56900 Qwen1.5-MoE + torch.compile 退化输出 — **未复现（负结果）**

- 严格按 issue 脚本（同 pinned revision `ec052fda`、同 16 prompts、同采样参数、
  独立进程），compile（mode=3 VLLM_COMPILE + FULL_AND_PIECEWISE + inductor，已确认生效，
  CUDA graph 捕获 51 sizes）与 eager（mode=0）均输出连贯，**16/16 无退化循环**。
- compile vs eager token 级对比：12/16 完全一致；4 个不一致 prompt 的分叉位置在
  token 14/31（32 token 的中后段，属 compile 正常数值漂移），两侧文本均连贯。
- 日志：`results/qwen15moe-compile-{eager,compile}.log`；issue 正文存档
  `results/issue-56900-body.txt`。
- 与报告者环境差异点：H100 + torch 2.13.0+**cu130**（PyPI 构建）vs 本机 H20 + torch
  2.13.0+**cu129**（GitHub release 官方 wheel）；两者均为 sm_90 / vllm 0.28.0。
- **给 owner 的建议措辞**（可选）：作为"未在 cu129 构建 + H20 复现"的数据点回复 issue，
  提示问题可能与 torch cu130 构建或硬件相关 kernel 选择有关；建议报告者提供
  `VLLM_LOGGING_LEVEL=DEBUG` 下 compiled graph dump。**注意**：这是负结果，措辞要克制
  （未复现 ≠ 不存在；我们只有 1 张 H20，硬件批次/驱动版本不同）。

---

## 执行环境备注（复现者注意）

- HF Hub 匿名 API 限流（429）频繁，模型已在本地缓存时务必 `HF_HUB_OFFLINE=1`。
- vllm 引擎默认吃 87GB KV cache，**不要并行跑两个 vllm 进程**（本次曾致一个测试误伤后隔离复跑澄清）。
- 测试模型：facebook/opt-125m（任务 1）、Qwen/Qwen3-4B（任务 2B）、
  Qwen/Qwen1.5-MoE-A2.7B-Chat@ec052fda（任务 3），均已入持久 HF cache。
