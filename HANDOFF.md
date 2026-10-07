# 开源任务交接清单 v8（2026-10-08）

> 四个 PR 的同步更新已提交并推送；本清单取代远端 v7。不要按旧清单启动 CosyVoice 验证。
> GPU/CPU 侧负责实验和日志回传。修改上游或 fork 分支、提交、推送、发评论、修改 PR 正文、关闭 thread 均由 owner 列明内容并取得用户确认后执行。
> 运行前记录实际 GPU 数量、驱动、torch/vLLM/SGLang 版本、模型 revision 与源码 SHA；旧环境信息不是当前机器保证。

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
- 当前没有必须启动的新 GPU 实验；优先等待 code owner 评审和 CI 授权。
- 未取得逐项确认前，不 commit/push、改 PR 正文、发评论、解决 thread；先展示完整 diff 或准确文本。
