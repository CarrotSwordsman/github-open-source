# H20 GPU 任务清单 v6（2026-09-24）

> 执行环境：H20（当前 1×96GB）/ driver 535.247.01
> 账号：GitHub `CarrotSwordsman`
> **分工约定（沿用）**：GPU 侧只执行实验并把结果推回本仓库（`results/`）；所有上游动作（PR 评论、push）由 owner 执行。
> v5 已完成（`RESULTS-4.md`）。v6 变更：sglang 两个闲置 PR 的保活激活；vLLM-Omni Qwen-Image-2.1 窗口期 issue 扫描（机会性，无固定任务）。

## 任务 1（CPU 即可，优先）：sglang #36688 / #36695 保活激活

背景：sglang bot 对 idle PR 有软上限，#36692 已有 reviewer 实质参与（mistral），#36691 已激活（9/17 rebase + 11/11 验证）。剩余两个 tool-call 修复 PR 自 8/28 闲置，有被关闭风险。**目标：rebase 到最新 main + 跑 detector 单测，推回 fork 分支（owner 已授权此操作）。**

激活 playbook（照 #36691 的模式）：

```bash
# 在 GPU 机的 sglang clone 中（sglang env，验证过的 python 环境）
# PR #36688 — GLM detectors: streamed tool-call arguments disassembly
git fetch origin && git checkout -b fix/glm-detectors-streaming-args-rebase \
  remotes/origin/pull/36688/head   # 或从 CarrotSwordsman fork 拉对应分支
git rebase origin/main
# 冲突处理：function_call 目录近期有变动，重点看 python/sglang/srt/function_call/
# 跑对应单测（test/registered/unit/function_call/ 下 GLM 相关文件）
python -m pytest test/registered/unit/function_call/ -k "glm" -v

# PR #36695 — step3: parameterless tool calls are dropped
# 同法：rebase origin/main + pytest -k "step3" -v
```

- 测试全过 → force-push 到 CarrotSwordsman fork 对应分支（这属于"推自己 fork"，不违反分工约定；**两个 PR 分支名从 PR 页面 headRefName 确认**）
- 有冲突/测试挂 → 保留原始分支不动，把 rebase 过程和失败输出记录到 `results/sglang-reactivate-{36688,36695}.log`，交给 owner 决策
- 产物：`results/sglang-reactivate-{36688,36695}.log`（rebase 结果 + 测试输出）

## 任务 2（机会性，无固定动作）：vLLM-Omni Qwen-Image-2.1 窗口期 issue 扫描

模型 9/20 发布，issue 洪峰进行中（已有 #8109/#8087 release CI 失败、#8076 feature 请求）。**筛选标准**（满足才报给 owner，不自行认领）：
- 单卡 H20 可复现（7B DiT 单卡 96GB 富余）
- 无 assignee、无并行 PR、非 release-CI 侧（A100/B200 CI 失败不碰）
- 优先级：调度/精度/显存类 bug > 文档/配置类

owner 侧每日检查时已在扫 `Qwen-Image` 关键词，GPU 侧只在拿到 owner 指定任务时介入。

## 明确不做

- **#56564 修复 PR**：搁置——vLLM 主仓门槛会让 PR 停车；根因分析已在 issue 里，等 owner 回应再启动
- **#6964**：已完成（不复现），等报告者
- **#57092 multigpu**：已完成（DeepEP 三道墙负结果，已贴 PR）

## 结果回传（约定不变）

- 日志提交 `results/`，汇总写 `RESULTS-5.md`
- 完成后 commit + push 本仓库

## 上游状态快照（2026-09-24）

| 项 | 状态 |
|---|---|
| vLLM-Omni | **4 merged**（#7006/#7007/#7652/#7879） |
| Dynamo #9819 | CI 全绿 + dmitry 双 approve，等 tanmayv25（9/24 已发第二次轻量 follow-up） |
| sglang #36692 | reviewer（apex-mochen）实质参与，定长 chunk 测试已补（`d15bc6e3`） |
| sglang #36691 | 已激活待 review |
| vLLM #57092 | yewentao256 review 中，作者已修完三点 |
| vLLM #52525/#56564/#43764/#44152/#44273 | 等 owner/label，无动作 |
