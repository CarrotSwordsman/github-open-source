# H20 GPU 任务清单 v5（2026-09-20，仅 H20 可执行）

> 执行环境：2× NVIDIA H20 96GB (sm_90) / driver 535.247.01 / glibc 2.28（**9/18 起本机仅见 1×H20，任务按单/双卡标注**）
> 账号：GitHub `CarrotSwordsman`
> **分工约定（沿用）**：GPU 侧只执行实验并把结果推回本仓库（`results/`）；所有上游动作（PR 评论、push、开 issue）由 owner 执行。
> v4 已完成（`RESULTS-3.md`）：#56564 ✅ ｜ #7376 ✅ 复现 ｜ #57092 ✅ 核心交付 ｜ #6964 ⏸ 初始化失败+单卡。
> v5 变更：**新增 #7879 e2e 验证（单卡即可，优先做）**；#6964 带着已定位的初始化失败根因重试；#57092 multigpu 补跑；撤下已无 GPU 需求的项。

## 环境要点（沿用 v4，另见 RESULTS-3 环境备注）

- conda env `vllm029`（0.28.0+cu129 基线状态）与 `omni-h3` 可用
- **磁盘：share 70T 配额 100% 满**——重要产物写完必须校验；MiniMax-H3 缓存 207G 已保留（#6964 用）
- vllm-omni serve 必须 `vllm serve ... --omni`（裸 `vllm-omni serve` 不加载 omni 参数组）

---

## 任务 1（单卡，优先）：vLLM-Omni PR #7879 — duplicate index-0 choices 修复的 e2e 验证

PR：https://github.com/vllm-project/vllm-omni/pull/7879（owner 今天开的，issue #7376）
修复内容：text+audio 非流式请求的音频 final output 合并进匹配 index 的 text choice，不再产生重复 index=0 的双 choice。

**验证协议（前后对照）**：

```bash
# 1. 修复分支
cd /path/to/vllm-omni && git fetch origin && git checkout -b pr7879 origin/pr/7879/head
# 或 fetch CarrotSwordsman fork 的 fix/duplicate-index0-choices
# 若 omni-h3 env 是 editable 安装到 github-open-source/vllm-omni，直接在那棵树 checkout 分支即可

# 2. serve（单卡，v4 已验证的配置，见 RESULTS-3 任务4）
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091 \
  --stage-overrides '{"0": 0.5, "1": 0.3, "2": 0.15}'

# 3. 请求（同 issue #7376）：非流式 chat，modalities=["text","audio"]，音频输入+问句
# 4. 断言（修复后预期）：
#    - len(resp.choices) == 1
#    - choices[0].index == 0
#    - choices[0].message.content 为文本
#    - choices[0].message.audio 非 None（含 base64 wav）
#    - choices[0].audio_metadata 存在
# 5. 回到 main 重复请求（对照组，预期 len==2 且 audio 在 choices[1]）
```

产物：`results/omni7879-{fixed,main}.json`（完整响应）+ `results/omni7879-validate.log`。
**注意**：这组数据 owner 会贴到 PR 上作为 e2e 背书，请确保 JSON 完整（写完校验）。

## 任务 2（双卡）：vLLM-Omni #6964 — MiniMax-H3 TP2 DLO 并发死锁复现（重试）

**9/17 初始化失败的根因已从日志定位，且 workaround 已在会话中断前完成**：
- 失败点：DiffusionWorker_TP0/1 报 `FileNotFoundError: .../MiniMax-H3/snapshots/.../fastvideo_inference.json`
- 原因：模型快照顶层存在 `modular_model_index.json` 时误入 FastH3 modular 分支
- **workaround 已做**：删除该文件后 t2va 走 FL2VA partition（RESULTS-3 已记录）
- → **直接重试 `tools/omni6964-serve.sh`**；若仍失败，看新日志的 DiffusionWorker 段是否换成了别的错

复现协议（v4 原文，不变）：
1. 先确认 current main 仍复现（#5810/#5864 在 issue 报告 base 之后合并过）
2. NCCL busy-wait 是报告者假设不是结论
3. 三组对照：单非法（预期 0.5s 报错）/ 双非法并发（预期死锁）/ 合法+非法混合
4. 死锁时 `py-spy dump --pid <两个worker> --native`（NCCL spin 需要 native 帧）
5. 死锁后发合法请求验证永久性 + `top -H` / `nvidia-smi` 快照
6. 复现成功后定位 abort/cancel 路径的并发重入性

产物：`results/omni6964-*`（新 server 日志、py-spy 栈、快照、main 复现结论）。

## 任务 3（双卡，可与任务 2 同批）：#57092 multigpu 套件补跑

```bash
# 先给 vllm029 装 flashinfer（omni-h3 env 已验证 0.6.16.post3 满足 gate）
pip install flashinfer==0.6.16.post3 --torch 2.13.0cu129  # 按官方 wheel 索引
# 重打 PR patch（文件清单见 RESULTS-3 任务2 方法说明，备份在 tools/wheel57092-bak/）
# 然后跑 multigpu 部分
CUDA_VISIBLE_DEVICES=0,1 python -m pytest <组合套件路径> -v -k multigpu 2>&1 | tee results/moe-57092-multigpu.log
```

产物：`results/moe-57092-multigpu.log`（过/挂都完整保留 traceback）。

## 明确排除 / 搁置（不变）

- **#56900**（MoE 编译退化）：需 H100 复现侧，H20 数据点已交付
- **#43764 / #56977 / #52525 / #56137**：全在等维护者/reviewer，无 GPU 动作
- sglang 闲置 PR 处置：owner 侧决策，与 GPU 无关

## 结果回传（约定不变）

- 所有日志/结论提交到 `results/`，顶层写 `RESULTS-4.md` 汇总
- **禁止**：向上游仓库发任何评论/PR/push；修改任何上游分支（#7879 的分支由 owner 维护，只读 checkout）
- 完成后 commit + push 本仓库，owner 接手上游动作

## 当前上游状态快照（2026-09-20）

| 项 | 状态 |
|---|---|
| vLLM-Omni #7006 / #7007 | Merged（第 1、2 个） |
| vLLM-Omni #7652 | In Review（bug/core + Priority: high） |
| vLLM-Omni #7879 | **新开**（#7376 修复，等 CI/review） |
| Dynamo #9819 | dmitry 两轮 approve（含 force push 后 re-approve），CI 全绿，等 tanmayv25 write 权限批准 |
| vLLM #57092 | H20 验证数据已贴，作者确认价值（"evidence I could not produce myself"） |
| vLLM #52525 | 三架构证据汇总 + owner ping 已发；Bizuayeu 又补了 #52532 的 GB10 实测 |
| vLLM #43764 / #56900 / #56977 | 等 label / 等报告者 / 等平行 PR 落地 |
| sglang #36691 | 已激活（11/11 验证）；#36688/92/95 闲置待 owner 决策；bot 已关 3 个测试 PR |
