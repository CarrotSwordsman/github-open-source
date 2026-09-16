# H20 GPU 任务清单 v3（2026-09-16 晚，**双卡**）

> 执行环境：2× NVIDIA H20 96GB (sm_90) / driver 535.247.01 / glibc 2.28
> 账号：GitHub `CarrotSwordsman`
> **分工约定（沿用）**：GPU 侧只执行实验并把结果推回本仓库（`results/`）；所有上游动作（PR 评论、push、开 issue）由 owner 执行。
> v2 任务已全部完成（结果见 `RESULTS-2.md`：#43764 复测 7/8、#54035 完全复现、#56900 未复现）。

## 环境要点（v2 已验证）

- `vllm 0.28.0+cu129` wheel 在 driver 535 上完全可用（conda env `vllm029`，torch 2.13.0+cu129）
- **`vllm 0.29.0+cu129` wheel 也存在**（GitHub release），vLLM-Omni main 正好 pin v0.29.0
- HF Hub 匿名限流严重：模型已缓存后设 `HF_HUB_OFFLINE=1`
- 一个 vllm 进程默认吃 ~87GB KV cache；TP2 双卡任务天然单进程，无并行冲突问题

---

## 任务 1（主任务）：vLLM-Omni #6964 — MiniMax-H3 TP2 DLO 并发非法参数死锁复现

Issue：https://github.com/vllm-project/vllm-omni/issues/6964
无人认领、无并行 PR；模型 owner（david6666666）被 ping 未响应。
报告者环境是 2×RTX 5090 (sm_120)——我们在 sm_90 上复现可验证**架构无关性**（同 #54035 的 H20 背书模式）。

### 环境搭建

```bash
conda create -n omni-h3 python=3.12 -y && conda activate omni-h3
pip install https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
git clone https://github.com/vllm-project/vllm-omni.git && cd vllm-omni
pip install -e .   # 或按 repo README 的 GPU 安装方式；CPU-only 依赖按 requirements/common.txt
pip install py-spy  # 死锁栈抓取（本任务最高附加值的工具）
```

模型：`MiniMaxAI/MiniMax-H3`（先检查 HF 是否 gated；报告者 2×32GB 能跑，2×96GB 充裕；下载量大请提前拉取并缓存）。

### 启动配置（按 issue 环境段）

TP2 + DLO(no-AllGather) + resident=20 + enforce-eager + CUDNN_ATTN + VAE tile。
具体 flag 组装参考：
- `vllm_omni/deploy/minimax_h3_disaggregated.yaml`（MiniMax-H3 的默认 deploy config）
- `tests/e2e/online_serving/minimax_h3/_common.py`（e2e 测试的启动参数拼装）
- issue 只给了配置摘要，serve 命令需按上述两处拼出；`--text-encoder-tp-size` 按 #7564 的结论只影响 stage-0 AR TP

### 复现步骤（按序）

```bash
# 0. 启动 server（TP2 跨两张卡），等待 ready

# 1. 对照组：单个非法请求（duration=2 低于 H3 最小值 4s）
#    预期：~0.5s 内报错 "MiniMax H3 output duration must be in [4, 15] seconds, got 2.0"
curl -X POST http://localhost:8000/v1/videos -F "prompt=test" -F "size=832x480" -F "fps=24" \
  -F "num_inference_steps=20" -F "seconds=2" \
  -F 'extra_params={"task":"t2va","duration":2,"flow_shift":12,"audio_flow_shift":3,"aspect_ratio":"16:9"}'

# 2. 完全恢复/重启后：同一命令在两个终端立即并发（concurrency=2）
#    预期（死锁）：两个 HTTP 200 in_progress → 两个 worker 100% CPU (state R)、GPU 0% →
#    无 encode_prompt/diffuse 日志 → 后续任何请求全部挂起

# 3. 死锁发生时【关键动作】：抓两个 worker 的栈
ps aux | grep -E 'worker|dynamo'   # 找到两个 TP worker 的 PID
py-spy dump --pid <worker_pid_1> | tee results/omni6964-pyspy-worker1.txt
py-spy dump --pid <worker_pid_2> | tee results/omni6964-pyspy-worker2.txt
# 若 py-spy 需要权限：sudo py-spy dump --pid ... 或 py-spy dump --nonblocking

# 4. 验证"永久性"：死锁后再发一个正常请求（duration=5），确认也挂起
# 5. 记录 nvidia-smi / top -H 快照；重启 server 确认恢复
```

### 判定与回传

- **复现成功** → 日志存 `results/omni6964-*`（server 日志、py-spy 栈、快照）。
  py-spy 栈是 issue 最缺的诊断证据（NCCL busy-wait spin 的具体位置），价值最高。
- **未复现**（sm_90 上行为不同）→ 同样有价值，完整记录配置与观察。
- 不修改任何上游代码；复现数据由 owner 决定如何使用。

---

## 任务 2（条件任务，暂不执行）：#54035 patch 验证

已发复现背书（issue #54035 评论）。**触发条件**：报告者响应 / flash-attention fork 出现修复 PR。
到时：clone `vllm-project/flash-attention`，按 PR 或 issue 提示改 `hopper/tile_size.h`（decode 保留 kBlockM=64、FP8 kBlockN 对齐 prefill 的 192），按其 README 编译，重跑 kernel reproducer 确认 19/28 → 0。**先等上游信号再动手。**

---

## 任务 3（可选）：vLLM-Omni #7376 — 非流式 chat duplicate index-0 choices

Issue：https://github.com/vllm-project/vllm-omni/issues/7376（n=1 请求返回 2 个 choices，audio 被塞进 choices[1]，spec 客户端静默丢音频）。无 assignee、无 PR。
单卡即可（Qwen2.5-Omni-7B ~16GB）：

```bash
# 环境同任务 1（同一 omni-h3 env 可复用）
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091
# 按 issue 正文：非流式 chat completion，modalities=["text","audio"]，
# 官方 asset URL 的 mary_had_lamb.ogg 音频 + "What is recited in the audio?"
# 记录：len(resp.choices)==2? choices[0].message.audio is None? choices[1] 结构
```

日志存 `results/omni7376-*`。复现后 bug 大概率在 serving 层 choices 组装（CPU 可定位），修复由 owner 跟进。

---

## 结果回传（约定不变）

- 所有日志/结论提交到 `results/`，顶层写 `RESULTS-3.md` 汇总
- **禁止**：向上游仓库发任何评论/PR/push；修改任何上游分支
- 完成后 commit + push 本仓库，owner 接手上游动作

## 当前上游状态快照（供参考）

| 项 | 状态 |
|---|---|
| vLLM-Omni #7006 | APPROVED + MERGEABLE，等 linyueqian cycle ready → 即将合并 |
| vLLM-Omni #7652 | CI 跑着 + 自审已发，等 owner review（有 issue 语义确认背书） |
| vLLM #54035 | 复现背书已发（kernel+E2E+BF16 对照），等报告者响应 |
| vLLM #43764 | 0.28.0 数据（7/8）已补发，等维护者 |
| vLLM #56900 | 克制负结果已发，等报告者 graph dump |
| Dynamo #9819 | 等 tanmayv25 re-review + GPU CI |
| vLLM #56977 | 等 #56137/#56195 落地 |
