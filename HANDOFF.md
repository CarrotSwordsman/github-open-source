# 开源任务交接清单 v10（2026-10-08，计算资源侧开工版）

> 本轮唯一待启动的 GPU 任务是第 0 节 #58892 完整 CUDA pytest，状态为 READY FOR RESOURCE SIDE（尚未执行，不是 PASS）。四个 PR 的同步更新已提交并推送，但本交接清单的发布状态需单独核实。
> 用户安排另一计算资源会话执行；单张 H20 即可，第二张不必占用。不要按旧清单启动 CosyVoice 或重复死锁实验。
> GPU/CPU 侧负责实验和日志回传。修改上游或 fork 分支、提交、推送、发评论、修改 PR 正文、关闭 thread 均由 owner 列明内容并取得用户确认后执行。
> 运行前记录实际 GPU 数量、驱动、torch/vLLM/FlashInfer 版本与源码 SHA；本任务不下载模型，模型 revision 记为 N/A。旧环境信息不是当前机器保证。
> 2026-10-08 owner 会话检查：本机无 `nvidia-smi`，BCS 的 `ieg-gztechtke-aigc-h20-nj` 命名空间未发现名称含 `mershi` 的 Pod；这不代表另一会话没有资源。计算资源侧确认自己的测试环境，不占用或停止其他人的服务。
> SGLang #36691 的当前 main 兼容性检查不需要 GPU，留给 owner；Dynamo #9819 和 SGLang 的官方 CI 授权由维护者处理，不作为本地 GPU 实验派发。
> **[计算资源侧回传 2026-10-08 20:35] 第 0 节状态：BLOCKED（环境层，pytest 未运行，不是 PASS/FAIL）**。详见 `RESULTS-5.md`。

## 0. 现在开工：vLLM PR #58892 完整测试（单卡，最高优先）

### ⛔ 执行结果（计算资源侧回传，2026-10-08）：BLOCKED

- 固定 checkout 核对通过：HEAD `0a4c700103d047adda7c2faff710cf7c2cc66ead`，`HEAD^2` = `9367d8b96`；独立 `uv` 0.12.23 + `.venv`（CPython 3.12.15），未触碰 `vllm029` / `omni-h3`；2×H20 在位，driver 535.247.01（CUDA 12.2）。
- 阻断根因：`wheels.vllm.ai/9367d8b9…/` 只发布 **cu130** variant（无 cu129/cu128）；wheel pin `torch==2.13.0`，`_C_stable_libtorch.abi3.so` 依赖 `libcudart.so.13`。本机实测 libcudart 13 → `cudaErrorInsufficientDriver`；`torch 2.13.0+cu130` → `cuda.is_available()=False`（"driver too old, found version 12020"）。
- 安装尝试：标准命令 `--torch-backend=auto` 选中 cu129 → 404，setup.py 拒绝回退 root variant；显式 cu130 安装另卡 `llguidance==1.7.6` sdist 构建并超时终止（即便装完也过不了 CUDA 初始化）。
- v10 preflight 原样执行：`preflight_exit_code=1`（torch 未装成）→ 按规定未进入第三步。
- 覆盖情况：`test_sm90_sparse_backend_selection` 4 组合 **未运行**；`test_sm90_nope_fp8_ds_mla_resolves_to_flashmla` **未运行**；两文件完整 pytest、基线复核均未运行；GPU 数值/吞吐不在范围。
- 时长：19:31–20:35，超出 30 分钟限时（两次 cu130 安装尝试耗时）。
- 解锁条件（owner 判断）：① driver ≥ 580 的机器按原方案执行；② 上游为该 commit 发布 cu129 variant 后在本机重试（未验证）；③ 放宽约束允许 cu129 toolchain 源码编译（超出本轮授权）。
- 产物：`RESULTS-5.md`、`results/attention-58892-env*`（8 个）、`tools/remote_whl_inspect.py`。

### 目标与固定版本

- PR：https://github.com/vllm-project/vllm/pull/58892
- fork 分支：`CarrotSwordsman/vllm:fix/sm90-nope-sparse-backend-priority`
- 固定修复提交：`0a4c70010`（2026-10-08 核查仍是 PR head）。不要仅凭分支名检出随时变化的版本。
- 同版本对照基线：该提交第二父节点 `9367d8b96`，不是运行时的最新 main。
- 当前缺口：owner 机器没有 `vllm._C_stable_libtorch`，此前只完成隔离选择器逻辑验证；需要在有可用 CUDA 编译扩展的环境中运行完整 pytest 文件。
- 这是 CUDA 环境下的选择/兼容性回归测试，不是推理吞吐 benchmark；不下载 GLM 大模型，不将通过结果写成“性能提升已实测”。

### 第一步：环境检查（先做，失败则限时排查）

- 核对实际 H20 数量、显存和 driver；测试只使用一张卡。检查磁盘配额，重要日志写完校验；不要默认另一会话与 owner 使用相同挂载路径。
- 在独立 checkout 与独立 Python 环境中执行，不覆盖之前的 `vllm029` / `omni-h3` 环境，不修改驱动或共享 CUDA。遵循固定 checkout 的 `AGENTS.md`，使用 `uv` 管理环境，Python 命令显式使用 `.venv/bin/python`，不使用系统 Python 或裸 pip。
- 当前源码的 CUDA requirements 与旧发布 wheel 不同：固定版本的 `requirements/cuda.txt` 包含 `torch==2.13.0`、`flashinfer-python==0.7.0.post1`，以及 cu13 依赖。不要把 0.28/0.29 wheel 的少量 Python 文件拼接后报告为“完整 PR 测试”；以源码实际依赖和所选 wheel 元数据为准，出现不兼容即记录。
- 安装前阅读 `AGENTS.md`、`docs/contributing/incremental_build.md` 和 GPU 安装文档。预编译安装固定 `VLLM_PRECOMPILED_WHEEL_COMMIT=9367d8b9693c3d2fb6c3f99d8ccb4aa7673c0981`，并确认选定 wheel 的 CUDA variant、torch ABI 和本次仅 Python 差异兼容；不退回最新 main wheel。保存安装命令、wheel 完整来源及安装日志。
- driver 535 / 旧 nvcc 若无法支持所需运行时，最多排查 30 分钟，保存异常和环境信息后记录 BLOCKED；不通过改版本、打兼容 alias 或屏蔽 skip 把失败转成“通过”。若预编译 wheel 不存在或 ABI 不兼容，不盲目回退旧 wheel或启动不限时全量编译。

### 第二步：固定检出并验证实际导入

以下命令是执行方案，不要求在 owner 当前无 GPU 的会话运行。先从本交接仓库根目录建立输出目录：

```bash
RELAY_ROOT=$(git rev-parse --show-toplevel)
mkdir -p "$RELAY_ROOT/results"
OUT_DIR="$RELAY_ROOT/results"
```

在 GPU 侧专用实验目录执行（目录名已存在时先检查，不覆盖已有工作）：

```bash
git clone --single-branch --branch fix/sm90-nope-sparse-backend-priority https://github.com/CarrotSwordsman/vllm.git vllm-58892-check
cd vllm-58892-check
git checkout --detach 0a4c70010
git log -1 --oneline
git rev-parse --short HEAD^2
```

安装可用的独立开发环境后，在 checkout 根目录执行 preflight 并保存输出。`OUT_DIR` 必须沿用前面交接仓库下的绝对路径，而不是 checkout 自己的 `results/`：

```bash
set -o pipefail
(
  ENV_RC=0
  run_check() {
    "$@"
    RC=$?
    printf 'check=%s exit_code=%s\n' "$1" "$RC"
    if [ "$RC" -ne 0 ]; then ENV_RC=1; fi
  }
  run_check git rev-parse HEAD
  run_check git status --short
  run_check nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
  run_check .venv/bin/python -V
  run_check uv pip show --python .venv/bin/python torch vllm flashinfer-python pytest
  run_check .venv/bin/python -c 'import pathlib, torch, vllm, vllm._C_stable_libtorch; from vllm.platforms import current_platform; print("vllm source:", vllm.__file__); print("torch:", torch.__version__, "CUDA:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available(), "platform:", current_platform); assert pathlib.Path(vllm.__file__).resolve().is_relative_to(pathlib.Path.cwd().resolve()); assert torch.cuda.is_available(); assert current_platform.is_cuda()'
  exit "$ENV_RC"
) 2>&1 | tee "$OUT_DIR/attention-58892-env.log"
ENV_STATUS=("${PIPESTATUS[@]}")
printf 'preflight_exit_code=%s log_exit_code=%s\n' "${ENV_STATUS[0]}" "${ENV_STATUS[1]}" | tee "$OUT_DIR/attention-58892-env-exit.txt"
```

任一 preflight 检查或日志写入失败，先排查并按 30 分钟限制回传 BLOCKED，不直接开始测试。确认 `vllm.__file__` 指向固定 checkout，编译扩展实际可导入；另外记录 torch、vLLM、FlashInfer 版本，安装方式和所用 wheel 来源。不能只依赖最后一条命令的退出码。

### 第三步：运行完整两文件 + 单独确认关键用例

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_cuda_backend_probe_errors.py \
  tests/v1/attention/test_flashmla_nope_sm90_backend_selection.py \
  -v -rs --junitxml="$OUT_DIR/attention-58892-fixed.xml" \
  2>&1 | tee "$OUT_DIR/attention-58892-fixed.log"
TEST_STATUS=("${PIPESTATUS[@]}")
printf 'pytest_exit_code=%s log_exit_code=%s\n' "${TEST_STATUS[0]}" "${TEST_STATUS[1]}" | tee "$OUT_DIR/attention-58892-fixed-exit.txt"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_cuda_backend_probe_errors.py::test_sm90_sparse_backend_selection \
  tests/v1/attention/test_flashmla_nope_sm90_backend_selection.py::test_sm90_nope_fp8_ds_mla_resolves_to_flashmla \
  -v -rs 2>&1 | tee "$OUT_DIR/attention-58892-key-cases.log"
KEY_STATUS=("${PIPESTATUS[@]}")
printf 'pytest_exit_code=%s log_exit_code=%s\n' "${KEY_STATUS[0]}" "${KEY_STATUS[1]}" | tee "$OUT_DIR/attention-58892-key-cases-exit.txt"
```

- `test_sm90_sparse_backend_selection` 的 4 个组合必须真实执行；全 skip 不算完成。
- `test_sm90_nope_fp8_ds_mla_resolves_to_flashmla` 应执行而非因为 CPU 平台被跳过；环境/依赖拒绝需要完整 traceback。
- 任何其他失败或 skip 均列明原因，不预设总用例数，不删除失败用例。
- 当前测试会 mock 部分后端可用性，即使在 H20 上通过也不代表 attention kernel 数值或 GLM 吞吐已验证。

### 可选：基线复核（修复测试完成后，非必需）

如果安装开销可控，可另建同版本 `9367d8b96` 的独立 checkout，用相同环境跑它自身的两文件测试。两边的测试预期不同，因此“各自通过”只说明各自测试可运行，不证明修复有效；不混用发布 wheel 和 main，也不把上游已有排序测试当作修复前必失败的对照。要做同一回归用例的反向变异测试，先写明只恢复哪段优先级逻辑、预期哪一用例失败，并在本地实验拷贝中实施，不能推改 PR 分支。

### 验收与回传

- 输出：`results/attention-58892-{env,fixed,key-cases}*` 和 junit XML；日志包含完整退出码、passed/failed/skipped、固定 SHA 与真实依赖信息。
- 在 `RESULTS-5.md` 写汇总：PASS / FAIL / BLOCKED；明确关键 4 组合是否运行、fp8_ds_mla 用例是否运行、未覆盖 GPU 数值/吞吐。
- 不需要双卡 TP、不下载模型、不启动 #6964 或旧 #8240。第二张 H20 可留空。
- GPU 侧只交付实验结果。不向上游发评论，不改 PR/fork 分支；交接仓库的结果 commit/push 在展示将提交文件并取得确认后执行。

## 1. 立即暂停：CosyVoice3 #8235 / PR #8240 旧版前后对照

- #8235 已于 2026-10-04 关闭，维护者确认由 #8224（2026-09-29 合并，`1ea0ee3b38`）修复。
- 第三方 RTX 4090 验证：旧版 C8 串音 42/360（2 音色）与 90/360（8 音色）；#8224 与随后 main 对应组均 0/360。此数据属于第三方，不是我们的验证成果。
- #8240 仍为旧 tail-pad head `09db1c3802`，存在冲突，作者承诺的 request-id map 尚未推送。维护者要求作者判断是否重新定位为加固，或关闭。
- 不下载旧版本做原计划，不将 #8240 作为 #8235 的修复候选验证。
- 仅当作者给出 post-#8224 main 上仍失败的具体场景、新 head 和验证目标时，再定义 GPU 任务；基线应尽量使用该 PR 的 merge-base，保持版本、模型、配置和采样器相同。

## 2. 已发布的 PR 更新（2026-10-08）

### vLLM #58892：SM90 sparse 后端优先级

- 工作树：`/data/workspace/ai-infra/vllm-56564`
- 原有分支：`fix/sm90-nope-sparse-backend-priority`
- 同步基线：vLLM main `9367d8b96`，普通推送后的 PR head 为 `0a4c70010`。
- 已解决测试冲突，保留 #59246 新增的 fp8_ds_mla 测试，不覆盖上游支持逻辑。
- 精简源码注释，删除重复排序测试；修改现有 `test_cuda_backend_probe_errors.py` 中旧预期，改为 NoPE-512 / RoPE-576 的实际选择器首选与环境失败 fallback（4 个组合）。
- 静态验证：Ruff check / format、git diff --check 通过。
- 验证边界：本机没有 `vllm._C_stable_libtorch`，完整 CUDA pytest 未能收集；独立执行实际选择器函数、mock 后端可用性时 4 个组合通过，恢复 FI-first 选择行为会触发目标断言失败。不是 GPU 执行，也不是完整 CI。
- 已提交并普通推送，PR 描述及 ZJY0516 评审回复已更新；未自行关闭评审线程。当前 pre-run-check 仍被贡献者门槛阻塞，不是完整 CI 全绿。

### SGLang #36692：Mistral streaming

- 工作树：`/data/workspace/sglang-recon`
- 原有分支：`fix/mistral-streaming-second-call`
- 同步基线：SGLang main `0b635266d4`，包含 CI 指定的 `ddd4600197a4` 基线；已提交并普通推送 `da662a0ab3`。
- 实际 pytest：现有 Mistral + streaming 两文件共 28 passed；Ruff check / format 通过。
- 当前 streaming 测试为 10 个用例，各自 9 种切分（char/whole/halves/thirds + 5/13/40/65/100）。不将内部断言组合数量写作 pytest 测试数。
- 已更新测试说明、添加 apex-mochen 独立复现和验证署名。upstream CI 仍缺 run-ci 授权；本地通过不代表 upstream 全绿。

### SGLang #36688：GLM streaming

- 工作树：`/data/workspace/sglang-36688`
- 原有分支：`fix/glm-streaming-numeric-args`
- 同步 main `0b635266d4` 时两文件冲突已处理；GLM4.7 数值/auto 解析和 JSON 收尾采用当前 main 实现，不恢复旧代码，删除同步后未使用的 helper。
- 最终生产差异只在 `glm4_moe_detector.py`；保留 GLM4/GLM4.7 的 streaming 等价回归。
- 实际 pytest：40 passed；Ruff check / format、git diff --check 通过。
- 已提交并普通推送 `9dd91421d4`，PR 正文收窄到剩余 GLM4 修复；仍需维护者授权 CI，不需额外 GPU 实验。

### SGLang #36695：Step3 streaming

- 工作树：`/data/workspace/sglang-36695`
- 原有分支：`fix/step3-streaming-duplicate-call`
- main `0b635266d4` 已干净合并、提交并普通推送 `943baa6dcd`。
- 实际 pytest：16 passed；Ruff check / format、git diff --check 通过。
- 已更新 PR 验证段落，准确区分测试数和内部切分组合；仍需维护者授权 CI，不需额外 GPU 实验。

## 3. 等待维护者或其他作者

- Dynamo #9819：仍 OPEN。现有轻量检查成功不等于可信 GPU CI 完成。10/1 triage 要求 tanmayv25 对当前 head `cf10bdc5b7` 重新 `/ok to test`，并取得 TRT-LLM code owner 批准。不要自行触发受信任 CI。
- vLLM #57092：APPROVED + ready；当前 head 的轻量检查成功，未见对应最新 head 的 Buildkite 结果。由 PR 作者/维护者继续运行 CI。
- AutoRound #41835：已有 charan-rathore 的 PR #41877。我们的 `/data/workspace/sglang-41835` 两文件补丁未提交；不要开竞争 PR。
- #43764 / #44152 / #44273：旧 PR 仍等评审/贡献者 gate；已存在的 RTD 红叉未定位，不能直接归入贡献者门槛。
- sglang #36691：此前 CPU 验证结果不替代现在 main 的兼容性与 upstream CI；未在本轮同步。

## 4. 已结束的实验（不重复启动）

- v5 结果见 RESULTS-4.md，最后结果提交 `30c6de6`。
- #7879 e2e：已完成，PR 已合并。
- #6964 双 H20 非法请求并发：已执行，测试条件下未复现。第三方 H200 原始基线也未复现；不是我们修复了死锁，后续需报告者提供确切失败配置。
- #57092 multigpu：DeepEP/NVSHMEM/toolchain 受限，未执行成功；负结果已交付，不标为测试通过。

## 5. 回传和安全边界

- 新实验日志放 `results/`，汇总注明源码 SHA、环境、完整命令、测试退出码、通过/失败/跳过及未覆盖项。
- 本轮四个同步分支均已提交并推送，工作树干净；AutoRound 未提交候选仍需保留，不要重置覆盖。
- 计算资源侧现在执行第 0 节 #58892 完整 CUDA pytest；当前尚无 `RESULTS-5.md`，不代表已经开跑或完成。其他条目不新增 GPU 实验，按暂停、等待或已结束状态处理。
  - **[2026-10-08 回传] 已执行至 preflight → BLOCKED（driver 535 不支持 cu130-only 预编译 wheel），`RESULTS-5.md` 已提交。本机无法完成第 0 节，需 owner 决定解锁方案。**
- 用户已于 2026-10-08 授权发布本版清单。另一会话从 GitHub 拉取交接仓库后，先确认 `HANDOFF.md` 标题为 v10，再执行第 0 节；清单已发布不代表 GPU 测试已开始或完成。
- 未取得逐项确认前，不 commit/push、改 PR 正文、发评论、解决 thread；先展示完整 diff 或准确文本。
