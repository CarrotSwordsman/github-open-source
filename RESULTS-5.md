# RESULTS-5：HANDOFF v10 第 0 节 vLLM #58892 完整 CUDA pytest

> 执行：2026-10-08 19:31–20:35（+08:00），计算资源侧 H20 容器
> **结论：BLOCKED**（环境层，未进入 pytest）。不是 PASS，也不是 FAIL。
> 注：实际耗时约 64 分钟，超出 v10 的 30 分钟排查限时（两次 cu130 安装尝试各约 20 分钟）。

## 1. 结论摘要

| 项 | 结果 |
|---|---|
| 固定 checkout | ✅ `0a4c700103d047adda7c2faff710cf7c2cc66ead`，`HEAD^2` = `9367d8b96`（与 v10 一致） |
| 预编译 wheel（`VLLM_PRECOMPILED_WHEEL_COMMIT=9367d8b9…`） | ⛔ 只有 **cu130** 一个 CUDA variant |
| 本机 driver | 535.247.01 = CUDA 12.2（`cudaDriverGetVersion` = 12020） |
| preflight | ⛔ `preflight_exit_code=1`（`.venv` 未装成，`torch` 不可导入） |
| `test_sm90_sparse_backend_selection` 4 组合 | ❌ **未运行** |
| `test_sm90_nope_fp8_ds_mla_resolves_to_flashmla` | ❌ **未运行** |
| 两文件完整 pytest | ❌ **未运行**（无 `attention-58892-fixed*` / `key-cases*` 日志） |
| 可选基线复核 `9367d8b96` | 未做（同一环境墙） |

阻断根因：固定提交的预编译扩展和它所 pin 的 `torch==2.13.0` 都要求 CUDA 13 运行时，**driver 535（CUDA 12.2）无法初始化 CUDA 13**。按 v10 规定不换 wheel、不拼接旧 wheel、不打 alias、不启动不限时全量编译。

## 2. 环境

- GPU：2×NVIDIA H20（97871 MiB），本任务只用 GPU 0；driver 535.247.01
- OS：TencentOS Server 3.2（glibc 2.28）；系统 nvcc 12.1（只读）
- 独立 checkout：`/tmp/vllm-58892-check`（未触碰 `vllm029` / `omni-h3`）
- 独立环境：`uv 0.12.23`（`/tmp/uvpkg`），`.venv` = CPython 3.12.15（uv 管理）
- 模型：N/A（不下载）
- 共享盘 `share_1227201` 100%（600G 余量），所有安装/cache 落 `/tmp`

## 3. 执行过程与证据

### 3.1 预编译 wheel 可用性（`attention-58892-env-wheel-metadata.log`）

- `https://wheels.vllm.ai/9367d8b9693c3d2fb6c3f99d8ccb4aa7673c0981/` 下只有 `cu130/`、`vllm/`、`xpu/`；`cu129/`、`cu128/` 为空。默认目录与 `cu130/` 指向同一文件 `vllm-0.31.1rc1.dev43+g9367d8b96-cp38-abi3-manylinux_2_28_x86_64.whl`（325,906,591 B）
- 通过 HTTP Range 远程读取 wheel 内容（未整包下载，脚本 `tools/remote_whl_inspect.py`）：
  - `Requires-Dist: torch==2.13.0`、`flashinfer-python==0.7.0.post1`、`nvidia-cutlass-dsl[cu13]==4.8.0`
  - `vllm/_C_stable_libtorch.abi3.so` 依赖 **`libcudart.so.13`**
- `download.pytorch.org` 上 `torch 2.13.0` 只有 cu129 / cu130，无 cu128

### 3.2 安装尝试

1. 文档标准命令（`attention-58892-env-install-auto.log`，exit 1）：
   ```
   VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=9367d8b9693c3d2fb6c3f99d8ccb4aa7673c0981 \
     uv pip install --python .venv/bin/python -e . --torch-backend=auto
   ```
   `setup.py` 按 nvidia-smi 的 CUDA 12.2 选中 variant `cu129` → 该 commit 下 404 →
   `RuntimeError: Failed to fetch precompiled wheel metadata for CUDA variant 'cu129' ... The root/default variant is not used as a fallback because its CUDA compatibility with the selected variant cannot be verified.`
   即**上游安装脚本自身拒绝给 CUDA 12 机器装这个 commit 的预编译 wheel**。
2. 显式指定唯一存在的 variant（`attention-58892-env-install-cu130.log` / `-retry.log`）：
   ```
   VLLM_MAIN_CUDA_VERSION=13.0 VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=9367d8b9693c3d2fb6c3f99d8ccb4aa7673c0981 \
     uv pip install --python .venv/bin/python -e . --torch-backend=cu130
   ```
   解析 198 个包、`vllm` editable 构建成功（预编译 wheel 拉取通过），但依赖 `llguidance==1.7.6` 在本机走 sdist（maturin/Rust）构建，首次 20 分钟 `timeout` exit 124，重试再挂起约 20 分钟，按 30 分钟限时终止。**此路径即便装完也无法通过 preflight**，见 3.3。

### 3.3 CUDA 13 运行时在本机的直接验证

- `attention-58892-env-cudart13-probe.log`（独立 probe venv，`nvidia-cuda-runtime==13.0.*`，ctypes 直调）：
  ```
  cudaDriverGetVersion rc=0 driver_cuda=12020
  cudaRuntimeGetVersion rc=0 runtime=13000
  cudaGetDeviceCount rc=35 cudaErrorInsufficientDriver
  ```
- `attention-58892-env-torch-cu130-probe.log`（同 probe venv，`torch==2.13.0+cu130`，即固定提交 pin 的版本）：
  ```
  torch 2.13.0+cu130 CUDA 13.0
  UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12020)
  cuda.is_available False
  RuntimeError The NVIDIA driver on your system is too old (found version 12020)
  ```
  → v10 preflight 的 `assert torch.cuda.is_available()` 与 `vllm._C_stable_libtorch`（libcudart.so.13）在本机必然失败。

### 3.4 v10 规定的 preflight（`attention-58892-env.log` / `attention-58892-env-exit.txt`）

按 v10 第二步原样执行：git SHA ✅、git status 干净 ✅、nvidia-smi ✅、Python 3.12.15 ✅、`uv pip show torch vllm flashinfer-python pytest` ⛔（未安装）、导入检查 ⛔ `ModuleNotFoundError: No module named 'torch'`。
`preflight_exit_code=1 log_exit_code=0`。按 v10 规定不进入第三步。

## 4. 未覆盖项

- 两个测试文件的任何用例（含关键 4 组合与 fp8_ds_mla 用例）均**未执行**；owner 侧此前的"隔离选择器 + mock"验证仍是唯一证据
- GPU 数值 / GLM 吞吐：本任务本就不覆盖
- 基线 `9367d8b96` 复核：未做

## 5. 解锁条件（供 owner 判断）

需满足其一：
1. 有 **driver ≥ 580**（支持 CUDA 13）的机器，按 v10 原方案直接跑（预期无其他障碍，预编译 wheel 已验证可拉取）；
2. 上游为 `9367d8b96` 发布 cu129 variant（当前 404），本机 driver 535 + torch 2.13.0+cu129 可能可用（未验证，cu12.9 在 driver 12.2 上依赖 minor version compatibility）；
3. 放宽 v10 约束，允许用 torch 2.13.0+cu129 + CUDA 12.9 toolchain **源码编译**（本机系统 nvcc 12.1 只读，需 pip 版 nvcc，工作量与时长不可控，超出本轮授权）。

## 6. 产物清单（`results/`）

| 文件 | 内容 |
|---|---|
| `attention-58892-env.log` / `attention-58892-env-exit.txt` | v10 规定 preflight 输出与退出码 |
| `attention-58892-env-wheel-metadata.log` | 预编译 wheel 索引、METADATA、`.so` CUDA 依赖 |
| `attention-58892-env-install-auto.log` | 标准安装命令（cu129 variant 404） |
| `attention-58892-env-install-cu130.log` / `-retry.log` | cu130 安装尝试（llguidance sdist 构建超时） |
| `attention-58892-env-cudart13-probe.log` | libcudart 13 直调：`cudaErrorInsufficientDriver` |
| `attention-58892-env-torch-cu130-probe.log` | torch 2.13.0+cu130：driver too old |
| `tools/remote_whl_inspect.py` | HTTP Range 远程读取 wheel 元数据脚本 |
