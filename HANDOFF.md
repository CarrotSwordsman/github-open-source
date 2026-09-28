# H20 GPU 任务清单 v7（2026-09-28）

> 执行环境：H20（当前 1×96GB）/ driver 535.247.01
> 账号：GitHub `CarrotSwordsman`
> **分工约定（沿用）**：GPU 侧只执行实验并把结果推回本仓库（`results/`）；所有上游动作（PR 评论、push）由 owner 执行。
> v6 未完成（sglang 激活任务仍在）。v7 变更：**新增最高优先任务 #8235/#8240 协作验证（已与作者 Sworol 约定，他会在 PR 里署名我们的验证）**；v6 的 sglang 任务保留。

## 任务 1（单卡，最高优先）：CosyVoice3 并发串音修复 PR #8240 的 GPU 前后对照验证

背景：issue #8235（并发下 CosyVoice3 返回其他请求的语音），作者 Sworol 已交修复 PR #8240（分支 `fix/cosyvoice3-voice-mixing`，fork `Sworol/vllm-omni`）+ CPU 回归测试，但没 GPU 做不了并发验证。**我们承诺的交付物**：stock 基线 vs 修复分支在并发 1/8/16 下的 speaker-embedding 矩阵，贴到 PR 线程（他明确说会 fold 进 review evidence 并 credit 我们）。

### 步骤

```bash
# 1. 两个环境/检出：
#    A（基线）：stock 发布版 vllm-omni（issue 在 0.28/0.30 都复现）
#    B（修复）：editable checkout Sworol 分支
git clone https://github.com/Sworol/vllm-omni.git -b fix/cosyvoice3-voice-mixing vllm-omni-8240
cd vllm-omni-8240 && pip install -e .

# 2. 模型（0.5B，极小）：FunAudioLLM/Fun-CosyVoice3-0.5B-2512
#    需要 campplus.onnx（在模型目录内）+ 两个官方音色（repro 脚本自己从 GitHub raw 下载）

# 3. serve（stock deploy 配置）：
vllm serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --omni --trust-remote-code --port 8091 \
  --served-model-name FunAudioLLM/Fun-CosyVoice3-0.5B-2512
#    配置：deploy/cosyvoice3.yaml 默认 async_chunk

# 4. 复现脚本：issue #8235 正文里的 repro_voice_mixing.py（完整贴在里面，复制下来）
#    依赖：onnxruntime / soundfile / torchaudio / requests
#    基线跑法：python repro_voice_mixing.py --model-dir <模型目录> --concurrency 8
#    报告者参考数据（0.30.0）：c1=0/64，c8=13/64 串音
# 5. 矩阵：A、B 两个版本各跑 concurrency 1 / 8 / 16（64 requests 每档）
#    预期：B 全 0 串音；A 的 c8/c16 有串音（c16 应比 c8 更多）
```

### 注意事项

- **vllm 版本**：报告者用 0.30.0+cu130；本机 driver 535 对 cu130 有风险（0.29/cu129 已验证可用）。修复与版本无关（payload 路由），若 0.30 起不来就用 vllm 0.29.0+cu129 + 该分支，在日志里注明实际版本
- 结果记录 `results/cosyvoice3-8240-{stock,fixed}-c{1,8,16}.log`，写 `RESULTS-5.md` 汇总
- 若 B 分支仍有串音 → 完整 traceback + sims 数据同样有价值，如实记录
- Sworol 提出可以先 rebase 到最新 main——**不需要**，直接在分支现状上验证即可

## 任务 2（CPU，v6 遗留）：sglang #36688 / #36695 保活激活

照 #36691 模板：rebase 到最新 main + 跑 detector 单测，推回 CarrotSwordsman fork 对应分支；冲突/挂测试则保留原分支并记录到 `results/sglang-reactivate-{36688,36695}.log`。命令细节见 v6 版本（`git show a475734:HANDOFF.md`）。

## 明确不做（不变）

- #56564：PR #58892 已开（owner 已发），等 review
- #6964 / #56900：等报告者/需 H100
- #57092 multigpu：负结果已交付

## 上游状态快照（2026-09-28）

| 项 | 状态 |
|---|---|
| vLLM-Omni | 4 merged（#7006/#7652/#7879/#7007） |
| #8240（Sworol） | 待我们 GPU 验证 → 贴结果 |
| #58892（我们 sm90 优先级） | 停在 label 门槛，等 |
| Dynamo #9819 | 等 tanmayv25 第 10 天；备选批准者 keivenchang/kthui/GuanLuo |
| #57092 | APPROVED，作者适配 #53585 后等合并 |
| sglang #36692 | apex-mochen 参与中；#36688/#36691/#36695 闲置 |
| Qwen-Image-2.1 窗口 | #8135（owner 已介入）；无新的无主 H20 可复现 bug |
