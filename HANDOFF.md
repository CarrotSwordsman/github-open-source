# 开源贡献 H20 验证任务交接清单

> 生成时间：2026-09-10（最近更新：2026-09-15）
> 执行环境：NVIDIA H20 GPU 机器（临时计算资源）
> 账号：GitHub `CarrotSwordsman`（fork owner；GPU 容器侧已配长期 PAT，git push 自动认证）
> 本文档原始地址：https://github.com/CarrotSwordsman/github-open-source
> RAW 直链（可直接 curl）：https://raw.githubusercontent.com/CarrotSwordsman/github-open-source/main/HANDOFF.md

**本仓库是任务中转站，双向工作流见 [README.md](README.md)**：owner 在此布置任务 → GPU 容器侧
AI 助手拉取执行 → 结果回写 RESULTS.md + results/ 并 push 回来。所有 PR 分支以 GitHub 远端为准，
无需从其他机器拷贝文件，按各任务的 clone 命令获取代码。

## 全局背景（必读）

当前有 3 个进行中的开源 PR：

| PR | 仓库 | 状态 | GPU 侧任务状态 |
|---|---|---|---|
| Dynamo #9819 | ai-dynamo/dynamo | review 进行中（维护者 tanmayv25） | ✅ **已完成（2026-09-15）**：50/50 全过，见 RESULTS.md；是否 PR 留言待 owner |
| vLLM #43764 | vllm-project/vllm | open 3.5 个月，等 `ready` 标签 | ✅ **已完成（2026-09-10）**：6/8 通过（2 失败为已知 bug 的有效捕获）；guard 补丁未推送，有竞品 PR #48062，如何措辞留言待 owner，见 RESULTS.md |
| vLLM-Omni #7006 | vllm-project/vllm-omni | CI 全绿，P1 跟踪 | 无 GPU 任务，本文档不涉及 |

以下两个任务的原始内容保留作历史参考（验收标准、命令、教训仍然有效，复跑时按此执行）。

---

## 任务 1（最高优先级）：Dynamo #9819 GPU 测试验证 ✅ 已完成（2026-09-15，50/50 通过，详见 RESULTS.md）

### 背景

PR 分支：`CarrotSwordsman/dynamo:fix/trtllm-user-config-preservation`（最新提交 `362859393`，已推送）

历史教训：上一轮维护者用 `/ok to test` 触发 NVIDIA 内部 GPU CI，暴露了一个**断言插错测试**的 bug（本地无 CUDA 时测试模块整体 skip，无法发现）。现已修复，需要 GPU 环境本地验证后确认不会再烧一轮失败 CI。

### 环境搭建

```bash
git clone -b fix/trtllm-user-config-preservation https://github.com/CarrotSwordsman/dynamo.git dynamo-9819
cd dynamo-9819
git log -1 --oneline   # 应显示 362859393

# 依赖安装（H20 = sm90，TRT-LLM 官方支持）
pip install tensorrt_llm
pip install -e components/src/dynamo
```

若 `pip install tensorrt_llm` 失败，改用 Docker 方式（仓库 `deploy/docker/` 下有 Dockerfile.template），或从 https://github.com/NVIDIA/TensorRT-LLM/releases 找对应 CUDA 版本的 wheel。

### 验证命令（按顺序）

```bash
# 1. 重点测试：断言错位修复的两个测试
pytest components/src/dynamo/trtllm/tests/test_trtllm_unit.py -v \
  -k "extra_engine_args or strip or postprocess"

# 2. 通过后全文件回归
pytest components/src/dynamo/trtllm/tests/test_trtllm_unit.py -v
```

### 预期与验收

- 步骤 1 重点测试**必须全过**，尤其是：
  - `test_extra_engine_args_overwrite_is_warned`
  - `test_init_llm_worker_strips_num_postprocess_workers_from_extra_engine_args`
- 步骤 2 全文件若有个别预存失败（非本 PR 引入），记录哪些失败即可

### 结果处理

- **全过** → 什么都不用改，回复用户"GPU 验证通过"，由用户决定是否在 PR 留言
- **有失败** → 保存完整 pytest 输出到 `/data/workspace/github_open_source/dynamo-9819-test-failure.log`，不要自行修改代码推送，等这边分析

---

## 任务 2：vLLM #43764 H20 复测 + 补丁推送 ⚠️ 已完成复测（2026-09-10，6/8，详见 RESULTS.md）；补丁推送/留言待 owner 决策

### 背景

PR 分支：`CarrotSwordsman/vllm:test/parallel-sampling-output-kinds`

该 PR 于 2026-05 创建，当时在 H20 验证通过。但 vLLM main 三个月来大改（8 月删除了 InputPreprocessor 等），需要确认测试与当前代码兼容。另外已知一处待补：测试文件缺少 CUDA 模块级保护（当前 main 的同类文件 `test_async_llm.py` 有）。

已验证（无 GPU 侧）：四个核心 import 在当前 main 均有效、`AsyncLLM.shutdown()` 存在、ruff 0.14.0 check/format 通过。

### 环境搭建

```bash
cd /data/workspace/github_open_source
git clone -b test/parallel-sampling-output-kinds https://github.com/CarrotSwordsman/vllm.git vllm-43764
cd vllm-43764
pip install -e .   # 或直接用 vllm/vllm-openai 官方镜像挂载此目录
```

### 验证命令

```bash
CUDA_VISIBLE_DEVICES=0 pytest tests/v1/engine/test_parallel_sampling_output_kinds.py -v
```

模型用 `facebook/opt-125m`（代码内置），单卡 H20 几分钟跑完。

### 通过后的动作（按序执行）

1. **加 CUDA 模块级保护**：编辑 `tests/v1/engine/test_parallel_sampling_output_kinds.py`，在 import 块结束后（`MODEL = ...` 之前）插入：

```python
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.", allow_module_level=True)
```

（注意 `import pytest` 已存在，无需重复。`from vllm.platforms import current_platform` 放到其他 vllm import 旁边按字母序排好。）

2. **格式检查**（vLLM 锁定 ruff 0.14.0）：

```bash
pip install ruff==0.14.0
ruff check tests/v1/engine/test_parallel_sampling_output_kinds.py
ruff format --check tests/v1/engine/test_parallel_sampling_output_kinds.py
```

3. **提交推送**：

```bash
git add tests/v1/engine/test_parallel_sampling_output_kinds.py
git commit -s -m "[Test] Add CUDA platform guard to parallel sampling tests"
git push origin test/parallel-sampling-output-kinds
```

（`-s` 生成 DCO sign-off，必须；commit message 遵循 vLLM 前缀规范。）

4. **在 PR 留言**（`gh pr comment 43764 -R vllm-project/vllm`）：

> Follow-up: re-validated the full test matrix on an H20 against the current code (8 tests, all passing), and added the CUDA module-level guard that sibling tests (e.g. `test_async_llm.py`) use, so collection on non-GPU runners skips cleanly. Ready for the `ready` label whenever a maintainer can trigger CI.

### 结果处理

- **测试失败** → 保存输出到 `/data/workspace/github_open_source/vllm-43764-test-failure.log`，**不要推送任何修改**，等这边分析（可能是 API 漂移，需要针对性适配）
- **注意**：vLLM 有 AGENTS.md，要求 AI 客观评估 PR 价值、做重复工作检查——留言前确认 #21948（本 PR 解决的 issue）仍 open 且无并行 PR（`gh pr list -R vllm-project/vllm --search "21948 in:body" --state open`）

---

## 重要注意事项（血泪教训）

1. **工具版本必须对照仓库锁定**：Dynamo pre-commit 用 black **23.1.0**（不是最新版！），vLLM 用 ruff **0.14.0**。格式化前先读各仓库 `.pre-commit-config.yaml`。
2. **禁止 force push** 到任何 PR 分支。
3. **修改代码前先跑一遍基线测试**，区分预存失败 vs 新引入失败。
4. **Dynamo 测试无 CUDA 时会静默 skip**（conftest 检查 tensorrt_llm + 模块级 CUDA guard）——本地全绿不代表 GPU 上绿，这就是任务 1 存在的原因。

---

## 本地资源位置（仅原开发机，临时机器可忽略）

| 路径 | 内容 | 状态 |
|---|---|---|
| `/data/workspace/github_open_source/dynamo-9819` | Dynamo #9819 工作副本（分支已 push） | 可直接用 |
| `/data/workspace/ai-infra/vllm-omni` | vLLM-Omni 主工作区（分支 fix/abort-final-stage = PR #7006，已 push；`.venv`/`.deps` 为无 GPU 测试环境） | 保留勿动 |
| `/data/workspace/ai-infra/vllm-omni-stage-cli` | PR #7007 工作副本 | 已合并，可忽略 |

所有 PR 分支在 GitHub fork 上都有完整备份，任何机器重新 clone 即可获得全部代码。
