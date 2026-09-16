# H20 GPU 任务清单 v2（2026-09-15）

> 执行环境：NVIDIA H20 96GB (sm_90) / driver 535.247.01 / glibc 2.28
> 账号：GitHub `CarrotSwordsman`
> **分工约定（沿用 v1）**：GPU 侧只执行实验并把结果推回本仓库（`results/` 目录）；所有上游动作（PR 评论、push、开 issue）由 owner 执行。
> 本清单取代 v1（v1 两个任务已完成，结果见 `RESULTS.md`）。

## ✅ 执行状态（2026-09-16，全部完成，详见 RESULTS-2.md）

| 任务 | 结果 | 一句话结论 |
|---|---|---|
| 0 cu129 环境验证 | ✅ PASS（09-15） | vllm 0.28.0+cu129 可用（`results/env-029.txt`） |
| 1 #43764 复测 0.28.0 | ✅ 7/8 | AsyncLLM CUMULATIVE 仍丢 index=0 completion（#21948 现象）；LLM 离线路径已修好 |
| 2 #54035 FA3 FP8 不一致 | ✅ **完全复现** | kernel 级 97 边界逐项吻合 + E2E 首个 mismatch 位置 97 + BF16 对照零差异 |
| 3 #56900 MoE compile 退化 | ⚠️ 未复现（负结果） | H20+cu129 构建下 compile 输出连贯、12/16 token 级同 eager；差异点疑为 cu130 构建 |

## 环境关键发现（v1 的硬墙已绕过）

v1 结论"只能跑 vllm 0.11.0"的墙是 **PyPI 默认 wheel 用 cu130（需 driver 580+）**。
vLLM GitHub release 提供官方 **cu129** wheel：CUDA 12.9 属于 12.x minor compatibility，**driver 535 应可运行**（与 Dynamo 环境跑 cu128 同理）。

```
vllm 0.28.0 release assets 里有: vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
```

---

## 任务 0（前置，必须先做）：cu129 环境验证

```bash
conda create -n vllm029 python=3.12 -y && conda activate vllm029
pip install https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl

# 验证 1：import + CUDA
python -c "import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 验证 2：smoke test（首次会下载 opt-125m，~250MB）
python -c "
from vllm import LLM
llm = LLM('facebook/opt-125m')
print(repr(llm.generate(['Hello, my name is'])[0].outputs[0].text))
"
```

**判定**：
- 两个验证都过 → 记录 torch/vllm 版本组合到 results/env-029.txt，继续任务 1/2/3
- 失败（driver 报错等）→ 完整 traceback 存 `results/env-029-failure.log`，**不要自行编译源码**，任务 1/2 改为跳过，仅尝试任务 3 的替代方案（见任务 3 备注）

---

## 任务 1：vLLM #43764 复测（用 0.28.0 重跑，取代 v1 的 0.11.0 结果）

背景：v1 用 0.11.0 复测 6/8（2 个失败恰好是 #21948 bug 的实证）。0.28.0 是 2026-08 的 release，与当前 main 接近，证据力强得多。若 8/8 全过，则证明测试与 current 兼容，owner 可在 PR 里补充这个数据。

```bash
git clone -b test/parallel-sampling-output-kinds https://github.com/CarrotSwordsman/vllm.git vllm-43764-v2
cd vllm-43764-v2
# 测试文件与 wheel 版本可能有小 API 漂移：先把测试文件复制出来跑（v1 的经验：避免源码目录 sys.path 污染）
mkdir -p /tmp/t43764 && cp tests/v1/engine/test_parallel_sampling_output_kinds.py /tmp/t43764/
cd /tmp/t43764

# v1 教训：共享进程会让首个失败的 EngineCore 泄漏连带后续失败。每测试独立进程。
for t in 1 2 3 4 5 6 7 8; do :; done  # 占位，实际按下面逐个跑
CUDA_VISIBLE_DEVICES=0 python -m pytest test_parallel_sampling_output_kinds.py -v -x --forked 2>&1 | tee ~/t43764-run.log
# 若 --forked 不可用（缺 pytest-forked），则逐 test 独立进程：
#   pytest --collect-only -q 拿到 test id 列表后循环: pytest "test_parallel_sampling_output_kinds.py::TEST_ID" -v
```

**判定**：
- 8/8 过 → 日志存 `results/vllm-43764-v2-028.log`，结论"0.28.0 全过"
- 有失败 → 日志存 `results/vllm-43764-v2-028-fail.log`，区分 API 漂移（ImportError/参数错误）vs 行为失败（断言失败）

---

## 任务 2（主任务）：vLLM #54035 复现 — FA3 Hopper FP8 decode/prefill 不一致

Issue：https://github.com/vllm-project/vllm/issues/54035（FP8 KV cache 下 decode 与 prefill rescore 的 logprobs 系统性不一致，根因在 vllm-project/flash-attention 的 `hopper/tile_size.h`：decode kBlockN=96 vs prefill kBlockN=192）。
H20 = sm_90，与报告者 H100 同架构，预期完全复现。

### 步骤 A：kernel 级复现（最重要，先做这个，5 分钟）

```python
# /tmp/fa3_repro.py — 来自 issue 报告者的最小复现
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

device = "cuda"
nheads_q, nheads_kv, head_dim = 12, 2, 128

def run(seqlen):
    torch.manual_seed(42)
    k = torch.randn(seqlen, nheads_kv, head_dim, device=device).to(torch.float8_e4m3fn)
    v = torch.randn(seqlen, nheads_kv, head_dim, device=device).to(torch.float8_e4m3fn)
    q = torch.randn(seqlen, nheads_q, head_dim, device=device).to(torch.float8_e4m3fn)
    descale = torch.ones(1, nheads_kv, device=device)

    def call(query):
        q_len = query.shape[0]
        return flash_attn_varlen_func(
            q=query, k=k, v=v,
            cu_seqlens_q=torch.tensor([0, q_len], dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor([0, seqlen], dtype=torch.int32, device=device),
            max_seqlen_q=q_len, max_seqlen_k=seqlen,
            softmax_scale=head_dim**-0.5, causal=True,
            k_descale=descale, v_descale=descale, q_descale=descale,
            block_table=None, fa_version=3,
        )

    prefill = call(q)[-1].float()
    decode = call(q[-1:])[0].float()
    return torch.equal(prefill, decode), (prefill - decode).abs().max().item()

for length in list(range(88, 112)) + [128, 160, 192, 256]:
    print(length, run(length))
```

**预期（H100 实测）**：FP8 首个 mismatch 在 seqlen=97；28 个测试长度中 19 个不同；97 处 diff≈3.9e-3，最大≈9.8e-3。
**H20 判定**：输出存 `results/fa3-fp8-kernel-repro.log`。若同样 97 边界 + 接近的比例 → 复现成功（这确认了与具体卡无关，纯 sm_90 kernel 问题）。

### 步骤 B（A 成功后）：E2E 复现（Qwen3-4B，约 8GB 权重）

```bash
export VLLM_BATCH_INVARIANT=1   # 必须在 import vllm 前设置
```

```python
# /tmp/e2e_repro.py — 按 issue 的 E2E 步骤
import os
assert os.environ.get("VLLM_BATCH_INVARIANT") == "1"
from vllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-4B", kv_cache_dtype="fp8", enable_prefix_caching=False,
          enforce_eager=True)  # FA3 默认后端；若日志显示非 FA3，显式指定 --async-scheduling off
prompt = "tell me a story"
sp = SamplingParams(temperature=0.0, max_tokens=224, logprobs=1)
out = llm.generate([prompt], sp)[0]
gen_texts = [o.text for o in out.outputs]
gen_lps = [o.logprobs for o in out.outputs]

# full-prefill rescore：prompt+output 一起送回去拿 prompt_logprobs
full = prompt + "".join(gen_texts)
sp2 = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
out2 = llm.generate([full], sp2)[0]
# 对比每个生成位置的 decode logprob vs rescore logprob，记录首个 mismatch 位置、
# mismatch 总数、max |Δ|
# 预期：首个 mismatch 在绝对位置 97 附近，约 115/224 mismatch，max |Δ|≈0.5
```

BF16 对照：同脚本去掉 `kv_cache_dtype="fp8"`（预期仅少量噪声差异）。
日志存 `results/fa3-fp8-e2e-{fp8,bf16}.log`。

### 步骤 C（可选，A+B 成功且时间充裕）：patch 验证

修复方向（issue 已给出）：`flash-attention` fork 的 `hopper/tile_size.h`，decode 保留 `kBlockM=64` 但 FP8 分支改用与 prefill 相同的 `kBlockN`。验证方式：clone `vllm-project/flash-attention`，改 tile_size.h，按其 README 编译，重跑步骤 A 确认 mismatch 消失。**编译 FA3 需要 nvcc + 较长构建时间，若工具链不完整就跳过并记录**——复现数据（A+B）本身就足以支撑上游 PR 的开题。

---

## 任务 3（次任务）：vLLM #56900 复现 — Qwen1.5-MoE + torch.compile 退化输出

Issue：https://github.com/vllm-project/vllm/issues/56900（Qwen/Qwen1.5-MoE-A2.7B-Chat 在 torch.compile 下产生退化输出，vLLM 0.28.0）。

```bash
# 取 issue 正文完整步骤（容器内若 gh 不可用，用 curl API）：
gh issue view 56900 -R vllm-project/vllm --json body -q .body
# 或: curl -s https://api.github.com/repos/vllm-project/vllm/issues/56900 | python3 -c "import json,sys;print(json.load(sys.stdin)['body'])"
```

按正文的复现命令跑（模型 A2.7B 很小，单卡几分钟），核心对照：eager（`--enforce-eager`）输出正常 vs compile 模式输出退化（重复/乱码）。日志存 `results/qwen15moe-compile-{eager,compile}.log`，附两边的生成文本样本。

**任务 0 失败时的备注**：此 issue 报告环境是 0.28.0；若 cu129 装不上，检查 0.27.1 release 是否有 cu129 wheel 可作降级尝试（`gh release view v0.27.1 -R vllm-project/vllm --json assets -q '.assets[].name'`）。

---

## 结果回传（约定）

- 所有日志/结论提交到本仓库 `results/` 目录，文件名按上文指定
- 顶层写 `RESULTS-2.md` 汇总（格式参考 `RESULTS.md`）
- **禁止**：向上游仓库发任何评论/PR/push；修改任何上游分支
- 完成后 commit + push 本仓库即可，owner 会接手所有上游动作

## 当前上游状态快照（供参考，无需动作）

| 项 | 状态 |
|---|---|
| Dynamo #9819 | CI 全绿 + review 意见解决 + H20 50/50 背书，等 tanmayv25 re-review + `/ok to test` |
| vLLM #43764 | 已发如实留言（0.11.0 下 6/8 + bug 实证 + 愿向 #48062 收敛），等维护者 |
| vLLM #56977（我们开的 issue） | 等 #56137/#56195 落地后提修复 PR |
| vLLM-Omni #7006 | 等 review（Sy0307 已标 P1 跟踪） |
| vLLM-Omni #7564 | 根因分析已发（两条丢失路径），等 module owner 拍板语义 |
