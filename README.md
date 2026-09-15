# github-open-source

开源贡献任务的 GitHub 中转站（task relay）。

- **[HANDOFF.md](HANDOFF.md)** — 任务清单：owner 在这里布置任务（含环境搭建、验证命令、验收标准）
- **[RESULTS.md](RESULTS.md)** — 实验结果：GPU 侧 AI 助手执行后回写的结论与日志索引
- RAW 直链（可直接 curl）：`https://raw.githubusercontent.com/CarrotSwordsman/github-open-source/main/HANDOFF.md`

## 工作流（双向循环）

```
owner 开发机                             临时 GPU 容器（AI 助手）
──────────────                          ──────────────────────
1. 在 HANDOFF.md 布置/更新任务 ─push─▶  2. pull 读取 HANDOFF.md
                                        3. 搭环境 → 跑实验（严格按任务里的验证命令）
                                        4. 结果写 RESULTS.md + 原始日志存 results/
       ◀────────────push─────────────   5. commit + push 回本仓库 main
6. review 结果，决定后续（留言/改代码/
   布置下一批任务）                       （如任务要求：去 upstream PR/issue 留言）
```

## GPU 侧 AI 助手接手流程（每次会话自动执行）

1. 读本 README + `HANDOFF.md`（任务清单与验收标准）+ `RESULTS.md`（已完成结果，避免重复执行）
2. `git pull`（owner 可能已更新任务）
3. 执行 HANDOFF 中**未完成**的任务，严格遵守各任务的"验证命令"和"结果处理"约定
4. 结果回写：
   - 结论写 `RESULTS.md` 对应小节（环境、命令、逐项结果、给 owner 的建议）
   - 原始日志存 `results/<task>-*.log`
   - commit message 格式：`docs: <任务名> verification results`
5. push 回 main；若任务明确要求 upstream 留言/推送，用容器持久盘上的长期 PAT
   （`git push` 走 credential helper 自动认证；API 用 `curl -H "Authorization: Bearer $GH_TOKEN"`）

## 约定

- **任务状态以 GitHub 远端为准**：执行前先 pull；本地不囤积未推送的改动
- **测试失败不擅自改代码推送**——保存日志，写明现象，留给 owner 分析（除非任务明确授权）
- 长任务跑完**立刻** commit + push，不要等会话结束（临时容器随时可能消失）
- 容器侧硬件/环境细节（H20 规格、glibc/driver 硬墙、可用软件栈上限、conda env 清单）
  由容器持久盘上的 `START.md` / `MEMORY.md` 维护，本仓库不重复记录，只记任务与结果
