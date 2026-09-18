# vLLM #56564 — GLM-5.3-Flash sm_90 MLA sparse 后端自动选择错误：根因定位（静态代码分析）

> 执行日期：2026-09-17 ｜ 分析基线：vllm upstream/main @ 75c71390d5（PR #53906 已合并）
> 结论：**后端选择逻辑根因已定位（CPU 侧决策，与显存无关）；H20 上吞吐 A/B 不可行（模型装不下），未做动态复现。**

## 1. H20 可行性评估（为何没有动态复现）

- 模型 `zai-org/GLM-5.3-Flash` HF 仓库本身就是 **FP8 量化版，328.4 GB**（73 files，非 gated）。
  无更小的官方量化版（zai-org 仅有 GLM-5.3 / GLM-5.3-Flash 的 BF16 与 FP8 变体）。
- 报告者配置 TP8 × H100 80GB = 640GB。本机 2×H20 = 192GB < 328GB → **权重都放不下，无法启动
  server 做 A/B 吞吐对比**。
- 后端选择发生在 CPU 侧（`platforms/cuda.py` 优先级表 + backend 的 `supports_combination`），
  与 GPU 显存无关 → 转为静态代码分析定位根因。以下分析完全复现了选择链。

## 2. 根因链（sm_90 + FlashInfer ≥ 0.6.18 → 选错后端）

### 2.1 模型侧输入（决定 head_size）

`zai-org/GLM-5.3-Flash` config.json（text_config）：

```
architectures      : ['Glm5NextForConditionalGeneration']
kv_lora_rank       : 512
qk_rope_head_dim   : 0      ← NoPE（无 RoPE 通道）
index_topk         : 2048   ← sparse attention（per-token top-k）
```

→ MLA head_size = kv_lora_rank + qk_rope_head_dim = **512 + 0 = 512**。

### 2.2 优先级表：head_size==512 时 SM90 FlashInfer 被插到 sparse 候选首位

`vllm/platforms/cuda.py`（`_get_backend_priorities`，`use_mla=True`，非 sm100/sm120 的
else 分支即 **sm_90 Hopper**，H100 与 H20 同为 major=9）：

```python
else:  # Hopper (major == 9)
    sparse_tail = [
        AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,
        AttentionBackendEnum.FLASHMLA_SPARSE,
    ]
    flashinfer_sparse = AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90
    if head_size == 512:
        sparse_tail.insert(0, flashinfer_sparse)   # ← head_size 512 时置顶
    else:
        sparse_tail.append(flashinfer_sparse)
    return [
        FLASH_ATTN_MLA, FLASHMLA, FLASHINFER_MLA, TRITON_MLA,
        *sparse_tail,
    ]
```

GLM-5.3-Flash 用 sparse MLA，前面 4 个非 sparse 后端全部被拒（requires sparse），
候选落到 `sparse_tail` → **head_size=512 恰好触发 `insert(0, ...)`，FLASHINFER_MLA_SPARSE_SM90
排在 FLASH_ATTN_MLA_SPARSE 之前**。

git blame：该 `insert(0, ...)` 逻辑由 commit `98ed0856f3` "[Model] add GLM-5.3-Flash
support (#53906)" 引入（`git log -S "FLASHINFER_MLA_SPARSE_SM90" -- vllm/platforms/cuda.py`），
与 issue 指认的 #53906（031c899d 模型侧 + e34f7f71 kernel 侧）一致。

### 2.3 版本 gate：FlashInfer 0.6.17 → 0.6.18 翻转选择

`vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py::supports_combination`：

```python
if not has_flashinfer_sm90_nope_mla():
    return "FLASHINFER_MLA_SPARSE_SM90 requires FlashInfer with SM90 MLA support
            (ckv_scale_arr in BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
if not use_sparse:
    return "..."
if hf.kv_lora_rank != 512:
    return "..."          # GLM-5.3-Flash = 512 ✓
if hf.qk_rope_head_dim not in (0, 64):
    return "..."          # GLM-5.3-Flash = 0 (NoPE) ✓
```

`has_flashinfer_sm90_nope_mla()`（`vllm/utils/flashinfer.py:359`）是 **feature 检测**：检查
`BatchMLAPagedAttentionWrapper.run` 是否带 `ckv_scale_arr` kwarg（FlashInfer 0.6.18 引入）。

- FlashInfer **0.6.17**：gate 返回 False → SM90 后端被拒 → 选中下一个候选
  **FLASH_ATTN_MLA_SPARSE**（issue 中旧构建 `glm-release` 分支的日志行为）
- FlashInfer **0.6.18**：gate 放行 + kv_lora_rank/qk_rope_head_dim 检查全过 →
  选中 **FLASHINFER_MLA_SPARSE_SM90**（issue 中新 nightly 的日志行为）

选择链与 issue 报告的两次日志完全吻合。

### 2.4 后果

#53906 的意图是让 GLM-5.3-Flash NoPE MLA 在 Hopper 上用 FlashInfer（FP8 KV in-kernel
dequant、half HBM traffic），但 H100 实测（issue 数据）：concurrency 1 慢 36%，concurrency 8
慢 70%（output tok/s 754→230），MTP accept len 完全一致 → 差距全部来自 per-step attention
kernel 本身。`FLASH_ATTN_MLA_SPARSE` 在 sm_90 上是更优选择，却排在被置顶的 FlashInfer 之后。

## 3. 判定与建议（供 owner 参考，上游动作由 owner 决定）

- **复现判定**：选择逻辑层面完全"复现"（静态推导与 issue 两次日志逐项吻合）；性能层面
  无法在 H20 上复现（显存墙，见 §1）。
- **根因**：`platforms/cuda.py` sm_90 分支的 `head_size == 512` 特例把
  FLASHINFER_MLA_SPARSE_SM90 无条件置顶，没有任何基准数据支撑该后端在 sm_90 上更快；
  FlashInfer 0.6.18 的 feature gate 恰好放行后，这个优先级错误就暴露了。
- **可能的修复方向**：
  1. 把 SM90 FlashInfer sparse 从 `insert(0, ...)` 改为 append（排在 FLASH_ATTN_MLA_SPARSE 之后）；
  2. 或保留置顶但在 Hopper recipe 文档化 `--attention-backend FLASH_ATTN_MLA_SPARSE`；
  3. 或按 batch size 自适应（issue 数据显示低并发差距更大）。
- 相关 issue：#56563（force FLASH_ATTN_MLA_SPARSE 时 TMA descriptor zero-extent 报错刷屏，
  输出正确）——修复时值得一并看。

## 4. 环境证据

- 本机 2×H20 96GB sm_90, driver 535.247.01
- vllm 0.28.0 wheel 中不存在 `FLASHINFER_MLA_SPARSE_SM90` / Glm5Next 注册（0.28 早于 #53906），
  分析基于 upstream/main @ 75c71390d5 源码（github-open-source/vllm-43764-v2 clone，
  upstream remote 已 fetch）
- GLM-5.3-Flash config.json 直接取自 HF（curl），见 §2.1 字段值
