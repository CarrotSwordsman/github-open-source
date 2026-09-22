# RESULTS-4 — HANDOFF v5 执行结果（2026-09-22，全部完成）

> 环境：2×H20 96GB (sm_90) / driver 535.247.01 / glibc 2.28
> 分工：GPU 侧实验与结果回传；上游动作归 owner。
> 状态：任务 1 ✅ ｜ 任务 2 ✅（**死锁不复现**）｜ 任务 3 ✅ 负结果交付（multigpu 本机不可达，gate 认知修正）

---

## 任务 1（单卡）vLLM-Omni PR #7879 — duplicate index-0 choices 修复 e2e 验证

**状态：✅ 完成，6/6 断言 PASS，前后对照实证**

- 环境：omni-h3 env（editable vllm_omni @ `github-open-source/vllm-omni`），Qwen2.5-Omni-7B，1×H20，port 8091
- 修复分支：`pr7879`（`ddf4d20`，与远端 PR head 同步；PR 实际改动 = `serving_chat.py` +48 行 + 新测试）
- 对照分支：`main`（`eb1abba`）
- **两分支用完全相同的 serve 命令**（干净 A/B）：

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091 \
  --stage-overrides '{"0": {"gpu_memory_utilization": 0.5}, "1": {"gpu_memory_utilization": 0.3}, "2": {"gpu_memory_utilization": 0.15}}'
```

⚠️ **stage-overrides 格式注意**：PR base（`5a93ec1`）与 current main（`eb1abba`）的
`parse_stage_overrides` 均要求对象格式（`stage_id -> {field: value}` → flat 成
`stage_<id>_gpu_memory_utilization`）。v4 的 float 简写 `{"0": 0.5}` 来自更早的 editable
版本（`g538eddd`），在当前树上会报 `must be an object, got float`。

- 请求：issue #7376 原始场景（非流式 chat，modalities=["text","audio"]，mary_had_lamb.ogg + "What is recited in the audio?"，n=1）
- **结果（可直接贴 PR 的 e2e 背书）**：

| | pr7879（`ddf4d20`） | main（`eb1abba`） |
|---|---|---|
| len(choices) | **1** | 2 |
| choices[0].index | 0 | 0 |
| choices[0].message.content | ✅ 文本（"The audio contains a recitation of 'Mary Had A Little Lamb'."） | 文本（audio=None） |
| choices[0].message.audio | ✅ base64 wav（data_len=42300） | None |
| choices[0].audio_metadata | ✅ present | absent |
| choices[1] | —（不存在） | **index=0（重复）**，content=None，audio=31KB wav |

- 产物：`results/omni7879-fixed.json`（43551 B）/ `results/omni7879-main.json`（43934 B，均完整 JSON 已校验）/ `results/omni7879-validate.log` / 双 server 日志
- 客户端脚本：GPU 机 `tools/omni7879-client.py`（label 参数化 fixed/main，断言内置）
- 备注：`mary_had_lamb.ogg` 原在 /tmp 已随容器重启丢失，从 issue 资产 URL 重下并备份到 `tools/`（md5 双份一致）

## 任务 2（双卡）vLLM-Omni #6964 — MiniMax-H3 TP2 DLO 并发死锁复现（重试）

**状态：✅ 完成 —— **死锁在 current main（`eb1abba`）上不复现**（双非法并发两轮验证均干净失败）**

### 2.1 上次初始化失败根因闭环（含新发现）

09-17 的失败链在 v5 重试中演化为**两道门**，均已解决：

1. **门 1（v4 已定位）**：顶层 `modular_model_index.json` 存在 → 误入 FastH3 modular 分支找
   `fastvideo_inference.json` → 删除该文件（v4 workaround，保持有效）
2. **门 2（本次新发现）**：删掉该文件后，current main 的 `omni_snapshot_download` 走
   全量 `snapshot_download(allow_patterns=["*"], require_all=True)` → `HF_HUB_OFFLINE=1`
   下校验报 `IncompleteSnapshotError`（137 个文件缺失：Ref2VA partition 等，本来就有意未下载）
   → **workaround：直接传本地 snapshot 路径**（`os.path.exists` 短路返回），t2va 正常走 FL2VA partition

### 2.2 第二道环境墙：vllm 0.29 wheel FA2 PTX JIT

合法请求首次执行时死于 `cudaErrorUnsupportedPtxVersion`（`torch.ops._vllm_fa2_C.varlen_fwd`，
vllm 0.29.0+cu129 wheel 的 FA2 kernel 在 driver 535 上 PTX JIT 不兼容）。CLI
`--attention-backend` 不作用于 diffusion attention（日志显示仍走 platform default FLASH_ATTN）。

- `--diffusion-attention-backend CUDNN_ATTN`：因 cuDNN FMHA 不支持 seq_len=1 失败
- **`--diffusion-attention-backend TORCH_SDPA`：成功**，合法请求（duration=5s, 832x480）完整生成（~118s）

### 2.3 复现协议执行结果（最终配置：TORCH_SDPA + 本地路径 + v4 原有 TP2/DLO 参数）

| 组 | 请求 | 结果 |
|---|---|---|
| A | 单非法（duration=2 < H3 min 4s） | ✅ 符合预期：~2s 快速失败（`MiniMax H3 output duration must be in [4, 15] seconds`） |
| B（run 1，FLASH_ATTN server） | 双非法并发 | **不复发死锁**：两请求各 1-2s 独立干净失败（duration 校验 + Orchestrator abort），无 hang 迹象 |
| B（run 2，TORCH_SDPA server，合法请求已验证可完整执行的健康 server） | 双非法并发 | **不复发死锁**：`ee131133`/`4f22d5ae` 分别 21:07:29/30 快速失败，worker 瞬时 CPU ~6%，GPU 0% |
| C | 合法+非法混合 | 合法请求正常 completed（~118s）+ 非法快速失败 + server 持续健康（HTTP 200） |
| D | 死锁后验证 | ⏭ 未执行（B 未死锁，无前提） |

py-spy 栈未抓取（无死锁可抓）。

### 2.4 时间线修正（供 owner 参考）

- issue #6964 报告于 **2026-09-03**
- #5864 [Diffusion] Fix DLO DP concurrent request execution：merged **2026-08-08**
- #5810 [Feature][MiniMax-H3] Support diffusion continuous batching：merged **2026-08-24**
- 即 HANDOFF v5 所述"两 PR 在 issue 报告 base 之后合并"与事实相反——**两个修复都早于 issue 报告**。
  报告者环境疑似旧版本（或触发路径另有差异）。本机 current main 同配置不复现与该时间线自洽。

- 产物：`results/omni6964-server-run{1,2,3}-*.log`（三轮 server 日志，run3 为最终有效配置）+ A/B/C 客户端日志 6 份
- serve 脚本改动（GPU 机 `tools/omni6964-serve.sh`）：本地路径 + `--diffusion-attention-backend TORCH_SDPA`

## 任务 3（双卡）vLLM PR #57092 — multigpu 套件补跑

**状态：🟡 进行中 —— gate 认知修正：真实阻塞是 deep_ep，不是 flashinfer**

已完成：
- flashinfer-python 0.6.16.post3 已在 vllm029（torch 2.13.0+cu129），`has_flashinfer_cutlass_fused_moe()` = True
- PR patch 重打完成（v4 同法）：pr57092 分支（`98fd10a17`，单 commit，3 源码文件 + 3 依赖文件：
  `fused_moe/fused_moe.py`、`quantization/fp8.py`、`quantization/online/fp8.py`、
  `fused_moe/oracle/fp8.py`、`quantization/utils/fp8_utils.py`、`model_executor/utils.py`）
  + `_Fp8OnlineLinearBase = OnlineLinearBase` 兼容 alias → import 全部通过
- 测试套件从 pr57092 分支重建（`/tmp/moetest57092`，含自建顶层 conftest 预 import vllm.config +
  补齐 `tests/kernels/{utils,quant_utils,allclose_default,quantization/}` 依赖）→ 11 tests collected

**卡点已查透，最终状态：✅ 负结果交付 —— multigpu 补跑在本机不可达**（三道墙）：

1. **参数集为空**：multigpu 用例 skip 理由是 `got empty parameter set`——multigpu 的全部
   prepare/finalize 类型（`DeepEPHT/LL/V2`、Mori）都注册自 DeepEP 家族，本机无 deep_ep →
   `MK_MULTI_GPU_PREPARE_FINALIZE_TYPES` 为空 → 参数集为空。**装 flashinfer 不解除 multigpu
   skip**（HANDOFF v5 的 gate 假设有误；flashinfer 只影响部分 expert 类型）
2. **deep_ep 无法安装**：PyPI `deep-ep 1.0.0` 仅 sdist，metadata 阶段即
   `AssertionError: Failed to find NVSHMEM`；且 DeepEP 依赖 **DeepSeek 私改版 NVSHMEM**
   （README 明示 "our modified NVSHMEM"，setup.py 静态链 `-l:libnvshmem.a
   -l:nvshmem_bootstrap_uid.so`），GitHub NVIDIA/nvshmem releases 无二进制资产、
   NVIDIA 下载直链 404 → 需从源码编私改 NVSHMEM（MPI 依赖 + 本机 nvcc 12.1 只读 +
   无 RDMA + share 磁盘 96%），不可行
3. **即使装上也只解一半**：multigpu 组合中的 DeepGemmExperts 部分仍会撞 deep_gemm JIT
   nvcc≥12.3 环境墙（v4 run A 与本次重跑的 dtype8/9 均实证）

### 补充：v4 "自建 workspace_init conftest" 的完整含义（本次踩坑）

重建套件时只拷 moe 目录 + 预 import vllm.config 不够——测试用 `workspace_init` fixture
（定义在 pr57092 的**根** `tests/conftest.py`），缺它所有 singlegpu 用例 ERROR at setup
（`fixture 'workspace_init' not found`）。完整版 conftest = 预 import vllm.config +
复刻该 fixture（`init_workspace_manager(torch.device(0))` / teardown `reset_workspace_manager`）。

### patch 环境健康度验证（singlegpu 全量对照 v4 run A）

`results/moe-57092-singlegpu-repatch.log`：**7 passed / 3 failed / 1 skipped，436s**
（v4 run A：8/2/1，471s）——等价：

- dtype8/9（DeepGemm nvcc 墙）与 v4 一致 FAILED；multigpu skip 与 v4 一致
- 唯一差异 dtype6（quant_config6 per-act-token FP8 + CutlassExpertsFp8）本次 FAILED 后
  **单独复跑 PASSED**——失败模式为 topk=4 场景 1-2 个元素超差（max abs diff 0.0334/0.0347
  vs 容差 0.03，仅超限 ~11%），属 FP8 量化噪声在容差边缘的抖动，非稳定回归

### 产物

`results/moe-57092-multigpu.log`（skip 记录）/ `results/moe-57092-singlegpu-repatch.log` /
`results/moe-57092-deepep-install-attempt.log`（安装尝试与三道墙记录）

---

## 环境备注（v5 新增，跨任务）

- **`set -x` + bootstrap.sh 会把 GH_TOKEN 打进日志**（GitHub push protection 拦截过一次）：
  脚本里 `source bootstrap.sh` 必须带 `2>/dev/null` 或去掉 `set -x`（GPU 机
  `tools/omni6964-serve.sh` 已修；本次三个 server 日志已脱敏）
- MiniMax-H3 两种 serve 姿势：① repo id + HF_HUB_OFFLINE=1 在 current main 上会触发
  IncompleteSnapshotError（FL2VA 部分下载被全量校验拒绝）→ 传本地 snapshot 路径；
  ② diffusion attention 必须显式 `--diffusion-attention-backend TORCH_SDPA`（CLI
  `--attention-backend` 不作用于 diffusion；FA2 有 PTX JIT 墙、CUDNN 不支持 seq_len=1）
- **multigpu MoE 测试（DeepEP 路径）在本机永久不可达**，除非解决 DeepSeek 私改 NVSHMEM +
  nvcc 12.3+ 两个硬墙——后续 HANDOFF 不应再排此类任务到本机
