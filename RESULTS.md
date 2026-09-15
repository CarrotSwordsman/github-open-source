# H20 验证结果报告（2026-09-10）

> 执行者：临时 H20 容器上的 AI 助手（按 HANDOFF.md 执行）
> 环境：NVIDIA H20 96GB (sm_90) / driver 535.247.01 / glibc 2.28 / CUDA 12.1 toolkit
> 容器硬约束：**装不了 torch ≥ 2.11 (cu130) 的任何构建** → vLLM 只能用 0.11.0 wheel（2025-10-04），无法跑 2026-09 当前 main

---

## 任务 2：vLLM #43764 — H20 复测结果

**结论：6/8 通过；2 个失败为 vllm 0.11.0 上的已知 bug（issue #21948 所述），非 PR 代码问题。按 HANDOFF"测试失败不推送"规则，guard 补丁未 push、PR 留言未发。**

### 验证环境

- vllm **0.11.0 wheel**（PyPI，2025-10-04 上传）+ torch 2.8.0+cu128，Python 3.12（conda env `vllm-new`）
- 注意：PR 分支 base 为 2026-05-21 的 main（`0a54df2`），**比 0.11.0 wheel 新 7 个月**。本机因 glibc 2.28 / driver 535 无法装当前 main（torch 2.11+cu130 硬墙），只能用 0.11.0 近似
- 测试文件从 PR 分支 `test/parallel-sampling-output-kinds`（head `455adf8`）原样复制到仓库外独立目录运行（规避源码目录 `vllm._C` 缺失的 sys.path 污染问题）
- 每个测试用独立 pytest 进程串行跑（首轮 8 个测试共享进程时，首个失败的 EngineCore 未清理导致后续 7 个因显存不足连带失败——首轮日志已存 `results/vllm-43764-test-run1.log`）

### 逐测试结果（干净环境，每测试独立进程）

| # | 测试 | 结果 |
|---|---|---|
| 1 | `test_async_llm_parallel_sampling[CUMULATIVE]` | ❌ FAILED |
| 2 | `test_async_llm_parallel_sampling[DELTA]` | ❌ FAILED |
| 3 | `test_async_llm_parallel_sampling[FINAL_ONLY]` | ✅ PASSED |
| 4 | `test_llm_parallel_sampling_output_kinds[CUMULATIVE]` | ✅ PASSED |
| 5 | `test_llm_parallel_sampling_output_kinds[DELTA]` | ✅ PASSED |
| 6 | `test_llm_parallel_sampling_output_kinds[FINAL_ONLY]` | ✅ PASSED |
| 7 | `test_async_llm_parallel_sampling_delta_streaming` | ✅ PASSED |
| 8 | `test_async_llm_parallel_sampling_cumulative_monotonic` | ✅ PASSED |

逐测试日志：`results/vllm-43764-test-{1..8}.log`，汇总：`results/vllm-43764-each-test-summary.txt`

### 失败分析

两个失败均为同一断言：`AssertionError: Expected 3 completions, got 2` —— n=3 的
`AsyncLLM.generate()` 流式输出（CUMULATIVE / DELTA）最终 `RequestOutput.outputs`
只有 index=1,2，**index=0 的输出丢失**。

这与 issue #21948 正文明确吻合：

> @sethkimmel3 has reported that the `LLMEngine` + `CUMULATIVE` combination is not working properly

（实测在 0.11.0 上表现的是 AsyncLLM 流式路径的同类问题。）

**时间线推理**：
- 0.11.0 wheel（2025-10-04）：bug 存在 → 测试精确捕获，2 个失败
- PR 创建（2026-05-27）时在 H20 + 当时代码验证全过（HANDOFF 记载）→ bug 已在
  2025-10 ~ 2026-05 间于 main 修复
- 当前 main（2026-09）：本机硬约束无法验证（cu130 硬墙）

**含义**：这两个失败恰恰证明了 PR 测试的有效性——它能在未修复版本上捕获
issue #21948 描述的真实 bug，是合格的回归测试。

### 为什么没有 push guard 补丁 / 发 PR 留言

1. HANDOFF 明确规则："测试失败 → 不要推送任何修改，等这边分析"
2. HANDOFF 的留言模板声称 "re-validated the full test matrix on an H20 against the
   current code (8 tests, all passing)" —— 本机跑的 0.11.0 不是 current code，且
   6/8 而非全过。按此模板留言不符合事实
3. **查重发现竞品 PR**（HANDOFF 要求"留言前确认无并行 PR"的前提已不成立）：
   - **#44004**（wowohhh，2026-05-29，+127/-1）：仅 FINAL_ONLY 模式 2 个测试，覆盖窄
   - **#48062**（PSR94，2026-07-09，+164/-1，2 files）：**与本 PR 高度重叠** —— 覆盖全部
     output_kinds × AsyncLLM/LLMEngine 双路径，且断言更细（per-sample indexes、
     token accounting）
   - 竞品存在时，留言措辞需要重新评估差异化价值（vLLM AGENTS.md 也要求客观评估）
   → 此决定留给 PR owner

### 给 owner 的建议（可直接采用的后续动作）

1. 若要完成 guard 补丁：需在能跑当前 main 的机器上（torch 2.11+cu130，如
   vllm 官方镜像 `vllm/vllm-openai:nightly` 或带 GPU 的较新环境）复测 8/8 全过后再执行
   HANDOFF 步骤 1-4
2. 留言建议改为如实版本，例如：
   > Follow-up: re-ran the full test matrix on an H20 (vllm 0.11.0 wheel) — 6/8 pass;
   > the two AsyncLLM streaming (CUMULATIVE/DELTA) failures reproduce exactly the
   > bug this issue describes (missing index-0 output with n>1), confirming the
   > tests are effective regression coverage. Also added the CUDA module-level guard
   > used by sibling tests. Note: #48062 covers similar ground with finer assertions.
3. 或考虑在 #48062 review 中补充 H20 硬件验证数据（本报告的 6/8 通过 + bug 捕获证据），
   按"硬件验证差异化价值"思路参与

---

## 任务 1：Dynamo #9819 — GPU 测试验证结果

**结论：GPU 验证通过。重点测试 4/4 PASSED（含 HANDOFF 点名的两个断言修复用例），全文件回归 50/50 PASSED，0 失败 0 跳过。按 HANDOFF 规则无需改代码，是否在 PR 留言由 owner 决定。**

### 验证环境（2026-09-15，H20）

- conda env `dynamo-9819-rc24`（Python 3.12.14）：
  - **tensorrt_llm 1.3.0rc24**（pypi.nvidia.com wheel，`manylinux_2_28` x86_64，与 dynamo CI 容器版本一致，commit `362859393` 上的代码正是对着它写的）
  - **torch 2.11.0+cu128**（download.pytorch.org/whl/cu128）+ torchvision 0.26.0+cu128
  - transformers 5.5.4（trtllm rc24 的 pin，同时满足 dynamo `>=4.56`）
  - cu13 运行时库与 cu12 共存（nvidia-cublas 13.1 / cuda-nvrtc 13.0 / nccl-cu13 等 pip 包 + torch 自带的 cu12 系列，soname 不同互不冲突）
  - openmpi 4.1 + mpi4py（conda-forge，提供 libmpi.so.40）
- LD_PRELOAD shim：`tools/libdriver_shim.so`，补 driver 535 缺的 3 个 CUDA 12.3+ 内省符号
  （`cuKernelGetName` / `cuLibraryEnumerateKernels` / `cuLibraryGetKernelCount`，stub 返回 801，测试全程未触发真实调用）
- ⚠️ 2026-09-10 那轮的 1.0.0 环境路线是死胡同：dynamo main 代码需要 `TOKENIZER_ALIASES` 等 1.2+ 才有的 API。

### 结果

| 步骤 | 命令 | 结果 |
|---|---|---|
| 重点测试 | `-k "extra_engine_args or strip or postprocess"` | **4/4 PASSED**（含 `test_extra_engine_args_overwrite_is_warned`、`test_init_llm_worker_strips_num_postprocess_workers_from_extra_engine_args`） |
| 全文件回归 | `pytest test_trtllm_unit.py -v` | **50 passed, 0 failed, 0 skipped** in 37.77s |

日志：`results/dynamo-9819-test-full-rc24.log`

### 给 owner 的可选留言（如实版本）

> Re-ran the trtllm unit test suite on an H20 (tensorrt_llm 1.3.0rc24, torch 2.11.0+cu128,
> driver 535): all 50 tests pass, including the two tests affected by the assertion
> placement fix. No regressions on GPU.

（是否发、怎么发由 owner 决定；dynamo 仓库无 AGENTS.md 硬性要求。）

---

## 附录：本机执行环境备注

- conda env `dynamo-9819`（Python 3.12，torch 2.8.0+cu128，tensorrt_llm 1.0.0
  CUDA 12 版 from pypi.nvidia.com —— PyPI 上的 tensorrt_llm 只有 sdist 壳且 1.1+ 全部
  依赖 cuda-python>=13，1.0.0 是最后一个 CUDA 12 版本）
- 测试文件隔离运行方法（本机专用坑，见正文）
- GPU：单卡 H20 96GB，driver 535（不可升级）
