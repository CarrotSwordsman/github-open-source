# H20 GPU 任务清单 v4（2026-09-17 晚，**仅 H20 可执行**）

> 执行环境：2× NVIDIA H20 96GB (sm_90) / driver 535.247.01 / glibc 2.28
> 账号：GitHub `CarrotSwordsman`
> **分工约定（沿用）**：GPU 侧只执行实验并把结果推回本仓库（`results/`）；所有上游动作（PR 评论、push、开 issue）由 owner 执行。
> v2 已完成（`RESULTS-2.md`）；v3 未执行，其任务已并入本版。v4 变更：撤下 #54035/#184（作者已自行验证，无需我们背书），新增 #57092 与 #56564，增强 #6964 协议。
> **执行状态（2026-09-18）**：#56564 ✅ 完成 ｜ #7376 ✅ 完整复现 ｜ #57092 ✅ 核心交付完成（e2e 前后对照：干净基线 FAILED max abs diff 0.28 → PR patch PASSED bitwise equal；multigpu 补跑需双卡）｜ #6964 ⏸ 未复现（server 初始化失败 + 本机当前仅 1×H20，留待下次双卡会话）。详见 `RESULTS-3.md`。

## 环境要点（v2 已验证，直接复用）

- conda env `vllm029`（vllm 0.28.0+cu129 + torch 2.13.0+cu129）可用；`vllm 0.29.0+cu129` wheel 也存在（vLLM-Omni main pin 这个版本）
- HF 限流：模型已缓存后 `HF_HUB_OFFLINE=1`；一个 vllm 进程默认吃 ~87GB KV cache

---

## 任务 1（主任务）：vLLM-Omni #6964 — MiniMax-H3 TP2 DLO 并发非法参数死锁

Issue：https://github.com/vllm-project/vllm-omni/issues/6964（无人认领、无并行 PR；模型 owner 被 ping 未响应）。

**v4 协议增强**（吸收外部 review 意见）：
1. **先确认 current main 仍复现**：issue 报告基于 2026-09-01 的 main（`e51fe6e`），其后 #5810（H3 连续批处理）和 #5864（DLO DP 并发修复）已合并——先用最新 main 复现，若已修好则记录并关闭该任务
2. **NCCL busy-wait 是报告者的假设，不是结论**——不要带着预设找 NCCL，先取证再归因
3. 三组对照：单非法请求（预期 0.5s 报错）/ 双非法并发（预期死锁）/ 合法+非法混合（观察交叉行为）
4. 死锁时 **py-spy dump 两个 worker**（含 native 栈：`py-spy dump --pid X --native`，NCCL spin 需要 native 帧才看得见）
5. 死锁后发一个合法请求验证"永久性"，记录 top -H / nvidia-smi 快照
6. 复现成功后：定位 abort/cancel 路径在并发失败时的重入性（请求校验是否在所有 rank 一致执行、一个 rank 抛异常时其他 rank 是否已进 collective）

环境搭建、启动配置（`deploy/minimax_h3_disaggregated.yaml` + `tests/e2e/online_serving/minimax_h3/_common.py`）、复现 curl 命令：见 v3 版本的 git 历史（`git show 397f643:HANDOFF.md`）或 issue 正文。py-spy 提前 `pip install`。模型 `MiniMaxAI/MiniMax-H3` 需预下载（先确认 gated 状态）。

回传：`results/omni6964-*`（server 日志、py-spy 栈、快照、当前 main 的复现结论）。

## 任务 2（新增）：vLLM PR #57092 — FP8 MoE batch-invariance 的 kernel 组合测试验证

PR：https://github.com/vllm-project/vllm/pull/57092（修 #57016：在线 FP8 MoE 激活 scale 改 per-token + Triton launcher stride 修复）。

**缺口**：作者在其 WSL2 机器上**跑不了**模块化 kernel 组合测试（`CUDA error: invalid resource handle`，main 上也挂），这部分完全依赖 CI。H20 真机 Linux 可以补上。

```bash
# 复用 vllm029 env 或按 PR 分支要求建新 env
git clone https://github.com/vllm-project/vllm.git vllm-57092 && cd vllm-57092
git fetch origin pull/57092/head:pr57092 && git checkout pr57092
pip install -e .   # 若与 0.28 wheel 冲突，建独立 conda env

# 作者跑不了的套件（关键交付）
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/kernels/moe/test_modular_kernel_combinations.py -v 2>&1 | tee ../results/moe-57092-combinations.log

# 修复前对照（main 上应能看到 NYI skip；PR 分支上该组合不再 skip）
git checkout main && pip install -e .
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/kernels/moe/test_modular_kernel_combinations.py -v 2>&1 | tee ../results/moe-57092-main-baseline.log

# PR 自带的 e2e 回归（H20 sm_90 与作者的 sm_120 形成架构交叉点）
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/v1/determinism/test_batch_invariance.py::test_online_fp8_moe_logprobs_bitwise_bs1_vs_bsN -v 2>&1 | tee ../results/moe-57092-e2e.log
```

**判定**：三份日志存 `results/moe-57092-*`。组合套件在 PR 分支全过 → 这是给 review 的直接背书数据；有失败 → 完整 traceback 同样有价值（区分 PR 引入 vs main 预存，可用 stash/checkout 对照）。

## 任务 3（新增，快速评估）：vLLM #56564 — GLM-5.3-Flash 在 sm_90 上 MLA sparse 后端自动选择错误

Issue：https://github.com/vllm-project/vllm/issues/56564（0 评论、无平行 PR、无人认领）。

报告者环境 H100；**H20 同为 sm_90**，大概率可复现。两步走：

```bash
# 步骤 A（半小时内）：读 issue 拿复现命令（模型大小/serve 参数在其中），H20 上跑默认配置
# 观察：是否自动选择 FLASHINFER_MLA_SPARSE_SM90、吞吐是否显著低于预期
# 步骤 B：显式指定两个后端各跑一遍，得到 A/B 数据
#   --attention-backend FLASHINFER_MLA_SPARSE  vs 默认自动选择
```

回传：`results/glm53-flash-56564-{auto,explicit}.log`（含吞吐对比）。复现成功 → 后端选择逻辑的根因定位（大概率 CPU 可分析），修复 PR 由 owner 跟进；不复现（H20 显存/带宽与 H100 不同导致选型合理）→ 负结果同样记录。

## 任务 4（可选，沿用 v3）：vLLM-Omni #7376 — 非流式 chat duplicate index-0 choices

单卡即可（Qwen2.5-Omni-7B）。按 issue 正文步骤复现（非流式 chat + modalities=["text","audio"]，记录 `len(resp.choices)`、`choices[0].message.audio` 是否 None）。日志 `results/omni7376-*`。

## 明确排除 / 搁置

- **#54035 / flash-attention #184**：撤下——PR 作者已自行完成 H100 前后验证 + 回归测试，我们的验证不再关键
- **#56900（MoE 编译退化）**：**需 H100 才能做复现侧**（H20/cu129 不复现是已知结论，我们的负结果数据点已交付）。搁置，等 H100 接入任务清单后启动跨构建定位

## 结果回传（约定不变）

- 所有日志/结论提交到 `results/`，顶层写 `RESULTS-3.md` 汇总
- **禁止**：向上游仓库发任何评论/PR/push；修改任何上游分支
- 完成后 commit + push 本仓库，owner 接手上游动作

## 当前上游状态快照（供参考）

| 项 | 状态 |
|---|---|
| vLLM-Omni #7006 | **已合并**（第 2 个 merged） |
| vLLM-Omni #7652 | In Review（bug/core 标签 + Priority: high，等 owner review） |
| sglang #36691 | 已 rebase 激活（11/11 验证），等 review |
| vLLM #54035 | 复现背书已发，FA fork PR #184 在等 review |
| vLLM #43764 / #56900 | 数据点已补，等维护者/报告者 |
| Dynamo #9819 | 等 tanmayv25 re-review + GPU CI |
| vLLM #56977 | 等 #56137/#56195 落地 |
