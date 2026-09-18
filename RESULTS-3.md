# RESULTS-3 — HANDOFF v4 执行结果（2026-09-17 ~ 09-18）

> 环境：2×H20 96GB (sm_90) / driver 535.247.01 / glibc 2.28
> 分工：GPU 侧实验与结果回传；上游动作归 owner。
> 状态：#56564 ✅ ｜ #7376 ✅ ｜ #57092 ✅（multigpu 补跑遗留，需双卡）｜ #6964 ⏸ 未复现（server 初始化失败 + 当前仅 1×H20，等下次双卡会话）

---

## 任务 1（主任务）vLLM-Omni #6964 — MiniMax-H3 TP2 DLO 并发非法参数死锁

**状态：⏸ 未复现（等下次双卡会话）**

- 最新 main（`eb1abba`）复现**未开始**：模型已就位（t2va 走 FL2VA partition，HF hub 缓存 207GB），但 09-17 22:05 server 启动失败：`RuntimeError: Orchestrator initialization failed`（`results/omni6964-server.log` 尾部），随后会话中断
- 复现客户端：三组对照 + py-spy native 栈抓取（脚本就绪：`tools/omni6964-{serve.sh,client.py}`，在 GPU 机 `github-open-source/tools/` 下）
- 09-18 复查：本机当前仅可见 **1×H20**（09-17 为双卡），TP2 无法启动 → 按约定留待下次双卡会话
- 产物（完成后）：`results/omni6964-*`

## 任务 2 vLLM PR #57092 — FP8 MoE batch-invariance kernel 组合测试

**状态：✅ 核心交付完成（multigpu 补跑遗留，需双卡）**

方法说明（重要）：PR 基于的 main 与 0.28 wheel 结构接近，采用 **wheel patch 方案**——
vllm 0.28.0+cu129 wheel + PR 的 4 个改动文件 + 3 个依赖文件（`oracle/fp8.py`、
`utils/fp8_utils.py`、`model_executor/utils.py`，均取自 pr57092 分支）+ 1 个兼容 alias
（`_Fp8OnlineLinearBase = OnlineLinearBase`）。备份在 GPU 机 `tools/wheel57092-bak/`。
组合测试套件以 PR 分支测试文件在独立目录运行（含自建 `workspace_init` conftest）。

### run A（PR kernel + PR 测试，"修复后"）：`results/moe-57092-combinations.log`

**8 passed / 2 failed / 1 skipped，471s**

- ✅ 8 个通过：含 PR 修复的核心组合——TritonExperts（无量化）、quant_config1-7 的全部
  Triton FP8 路径（per-token/per-channel quant 配对，即 PR 解禁的组合）
- ❌ 2 个失败（dtype8/dtype9，DeepGemmExperts / TritonOrDeepGemmExperts）：
  `RuntimeError: (deepgemm compiler.hpp:187) NVCC version should be >= 12.3`
  → **环境墙**（本容器 nvcc 12.1 只读无法升级），与 PR 改动无关；CI（nvcc ≥ 12.3）不受影响
- ⏭ 1 skipped：multigpu 套件（vllm029 env 无 deep_ep/deep_gemm/flashinfer，gate 未满足）
  → 补充：omni-h3 env 的 flashinfer 0.6.16.post3 已验证满足 `has_flashinfer_cutlass_fused_moe`，
  待双卡 + 装入 vllm029 后补跑

### e2e（PR patch 环境，"修复后"）：`results/moe-57092-e2e.log`

**`test_online_fp8_moe_logprobs_bitwise_bs1_vs_bsN` PASSED（151s，09-17）**

- granite-3.1-1b-a400m（online FP8）+ VLLM_BATCH_INVARIANT=1，8 prompts bs1 vs bs8
  逐 token + logprobs bitwise equal

### run B（干净 0.28 wheel 基线 + PR 测试，"修复前"对照）：`results/moe-57092-main-baseline.log`

**已完成（09-17 22:21）：8 passed / 2 failed / 1 skipped，498s**

- 失败仍只有 dtype8/dtype9（DeepGemmExperts / TritonOrDeepGemmExperts，deep_gemm nvcc≥12.3 环境墙，与 run A 相同，非 PR bug）
- 基线甄别（09-18 复核）：wheel 备份恢复完整——7 个 patch 文件与 `tools/wheel57092-bak/` 逐字节一致（恢复时间 22:13，早于 run B 的 22:16-22:21），且与 run A 日志 md5 不同（两次独立运行）→ **run B 为干净有效的基线**
- 观察：PR 解禁的 Triton FP8 组合（quant_config1-7）在干净 0.28 基线上**也全部通过**——HANDOFF 预期的"main 上 NYI skip"未出现，即组合套件在 0.28 wheel 上无法展示"修复前"差异（0.28 wheel 与 PR 所基 main 的门控结构差异所致）

### e2e 基线对照（干净 0.28 wheel，"修复前"）：`results/moe-57092-e2e-baseline.log` ⭐

**FAILED（09-18，154s）——#57016 bug 在 H20 上的直接实证**

- `test_online_fp8_moe_logprobs_bitwise_bs1_vs_bsN`：**Prompt 0 logprobs not bitwise equal
  between BS=1 and BS=8，max abs diff 2.825e-01**
- 与 09-17 PR patch 环境下同测试 PASSED 构成完整前后对照：

  | 环境 | 结果 |
  |---|---|
  | 干净 0.28 wheel（修复前） | ❌ FAILED：logprobs 随 batch 组成漂移，max abs diff 0.28 |
  | 0.28 wheel + PR 7 文件（修复后） | ✅ PASSED：逐 token + logprobs bitwise equal |

- 结论：per-token dynamic activation scale 修复在 H20 sm_90 上彻底消除 batch 相关输出漂移，PR 核心主张在真实 Linux GPU 上成立（与作者 WSL2/sm_120 环境形成互补架构数据点）
- 运行注：独立目录跑 PR 测试文件需自建 conftest 预先 `import vllm.config`——否则 `utils.py` 把 `model_arch_config_convertor` 作为第一个 vllm 子模块导入，触发 0.28 wheel 的循环导入

### 遗留（需双卡，与 #6964 同批）

- multigpu 套件补跑：先给 vllm029 装 flashinfer 0.6.16.post3，再跑 `test_modular_kernel_combinations.py` multigpu 部分

## 任务 3 vLLM #56564 — GLM-5.3-Flash sm_90 后端自动选择

**状态：✅ 完成（静态根因定位 + 负结果记录）**

详见 `results/glm53-flash-56564-analysis.md`。要点：

- 根因链闭合：GLM-5.3-Flash NoPE（kv_lora_rank=512 + qk_rope_head_dim=0）→ head_size=512
  → `platforms/cuda.py` sm_90 分支 `insert(0, FLASHINFER_MLA_SPARSE_SM90)` 置顶 +
  FlashInfer ≥ 0.6.18 feature gate（`ckv_scale_arr` kwarg）放行 → 自动选择比
  FLASH_ATTN_MLA_SPARSE 慢 36-70% 的后端。引入 commit `98ed0856f3`（#53906），与 issue 指认一致
- H20 动态复现不可行：模型 FP8 328GB > 2×96GB；无更小官方量化版。性能 A/B 无法做
- 修复方向建议（供 owner）：sm_90 分支 SM90 FlashInfer sparse 从 insert(0) 改 append /
  recipe 文档化 / 按 batch size 自适应

## 任务 4 vLLM-Omni #7376 — 非流式 chat duplicate index-0 choices

**状态：✅ 完整复现**

- 配置：`vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091`（单卡 H20，
  `--stage-overrides '{"0": 0.5, "1": 0.3, "2": 0.15}'` 解决三 stage 单卡显存分配 OOM）
- 请求：非流式 chat，modalities=["text","audio"]，mary_had_lamb.ogg + "What is recited in the audio?"
- **结果与 issue 逐项吻合**：
  - `len(resp.choices) = 2`（请求 n=1）
  - `choices[0]`: index=0, content="The audio contains a recitation of 'Mary Had A Little Lamb'.", **audio=None**
  - `choices[1]`: **index=0（重复）**, content=None, audio={id, 42300 chars base64 ≈ 31KB wav}
- 产物：`results/omni7376-repro.log`（摘要）、`results/omni7376-response.json`（完整结构，
  base64 截断标注）、`results/omni7376-server.log`（server 日志）
- 最新 main（`eb1abba`，晚于 issue 报告的 `a7c295d5`）仍复现 → bug 未修

---

## 环境备注（跨任务）

- **磁盘（09-18 新增）**：share 70T 配额 100% 满（多用户共享，释放空间会被迅速抢占，写文件间歇性 QuotaExceeded，重要产物写完必须校验）；已清 pip 缓存（19G）与 wheelcache（5.8G）。MiniMax-H3 缓存 207G 保留（#6964 下次用）
- vllm029 env 当前为**干净 0.28 基线状态**（run B 后未重打 PR patch）；重打 PR patch 的文件清单与备份位置见任务 2 方法说明
- 本机 09-18 起仅可见 **1×H20**（09-17 为双卡）——**#6964（TP2）与 #57092 multigpu 补跑都需双卡，留待下次**
- MiniMax-H3 下载范围：t2va 任务实际走 **FL2VA/ partition**（顶层 modular_model_index.json
  存在时会误入 FastH3 modular 分支找 fastvideo_inference.json 而失败——已删除该文件）
- deep_gemm 在本容器不可用（JIT 断言 nvcc ≥ 12.3，本机 12.1）
- vllm-omni CLI：必须 `vllm serve ... --omni`（裸 `vllm-omni serve` 不加载 omni 参数组）
