# 云端交接完整指南

## 1. 交接目标

交接者需要能完成三件事：

1. 查看任意一条 SBCSAE 主实验样本；
2. 从缓存重新训练我们的 shared A/B adapter；
3. 查看或重新运行 Talking Turns 官方 checkpoint 在同一 test set 上的结果。

## 2. 主实验快速包

主 adapter 训练不需要原始 WAV，也不需要 11.16 GiB 的 Qwen 权重。需要上传：

| 内容 | 路径 | 约大小 |
| --- | --- | ---: |
| 当前代码、测试、说明 | `code/`，排除 `code/history/` | 小于 1 MiB |
| 主样本和标签 | `data/processed/sbcsae_qwen_shared_ab_30s_causal_v1/` | 245.5 MiB |
| 事件源 | `data/processed/sbcsae_turn_events_v3/` | 24.0 MiB |
| 标签检查辅助文件 | `data/processed/sbcsae_vad_fiveclass_v2/` | 11.0 MiB |
| Qwen 特征缓存 | `artifacts/main_experiment/qwen_feature_cache/` | 748.5 MiB |
| 59 维 profile | `artifacts/main_experiment/profile_features/` | 2.1 MiB |
| 当前结果 | `artifacts/main_experiment/results/` | 12.3 MiB |
| 说明和审计 | `README.md`、`PROJECT_STRUCTURE.md`、`collaboration/` | 数 MiB |

未压缩来源约 1.03 GiB；其中 245.5 MiB 主要是 JSONL，压缩后会显著变小。可直接生成压缩包：

```powershell
.\.venv\Scripts\python.exe code\scripts\prepare_cloud_handoff.py package
```

已生成的压缩包位于 `artifacts/main_experiment/turn-taking-main-cloud.zip`，包含 104 个文件，约 768.4 MiB；旁边的 `.sha256` 文件保存本次压缩包的准确校验值。

## 3. Talking Turns 重新推理需要什么

只查看已经保存的结果，不需要模型或音频：

```powershell
.\.venv\Scripts\python.exe code\scripts\run_espnet_talking_turns_baseline.py --score-only --device cpu
```

重新推理还需要：

- test 三段原始 WAV：`SBC007.wav`、`SBC017.wav`、`SBC058.wav`，合计约 349.2 MiB；
- 官方 checkpoint：`valid.loss.ave.pth`，约 1,176.1 MiB；
- 与其匹配的 ESPnet 源码，约 60 MiB。

本机已有位置：

```text
data/sbcsae/openslr/WAV/
models/talking_turns/checkpoint/
models/talking_turns/espnet_source/
```

模型建议在云端从 Hugging Face 的 `espnet/Turn_taking_prediction_SWBD` 重新下载，不上传 GitHub。三段 SBCSAE 音频应通过私有云盘或云服务器文件上传，不公开发布。

## 4. 云端环境

当前本机验证环境：Python 3.12.13、PyTorch 2.11.0+cu128、NumPy 2.5.1、scikit-learn 1.9.0。主 adapter 只需要兼容 CUDA 的 PyTorch、NumPy 和 scikit-learn，不要求云端完全复制 Windows 环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e './code[dev]'
```

主训练：

```bash
python code/scripts/run_main_experiment.py check
python code/scripts/run_main_experiment.py train --device cuda
```

## 5. 云端路径

所有当前命令都从仓库根目录运行。缓存训练使用固定相对路径，不依赖本机盘符。

处理数据中的 `C:\Users\xiong\...` 是生成时的来源记录。查看器和 Talking Turns runner 如果找不到该路径，会自动使用 `data/sbcsae/openslr/WAV/<conversation_id>.wav`。因此换到 Linux 或另一台电脑不会因为旧盘符停止运行。

## 6. 不要上传什么

- `data/processed/history/`：约 5.79 GiB；
- `artifacts/history/`：约 8.98 GiB；
- `models/history/`：约 37.20 GiB；
- `models/qwen2.5-omni-3b-local/`：只在重新提 Qwen 特征时需要；
- `.venv/`、`.pytest*`、`tmp/`：本机环境或临时文件。

## 7. GitHub 与云端的区别

GitHub 建议只提交：代码、README、论文草稿、逐文件清单、少量 `code/examples/`、结果 CSV/JSON 摘要。不要把 SBCSAE 原始音频、大 NPZ cache、Qwen 权重或 ESPnet checkpoint 推到 GitHub。

云端训练机再接收主实验压缩包；如果要重跑 Talking Turns，再额外上传三段 test WAV，并在线下载官方 checkpoint。

## 8. 已验证状态

2026-09-03 完成：

- 逐样本查看器成功读取真实音频、转写、59 维 profile、标签及两种模型预测；
- 主实验 `check` 通过；
- 主实验从缓存完整重训成功，新旧 `aggregate.csv` 与 `profile_deltas.csv` 一致；
- 当前测试 `12 passed`；
- Talking Turns 的 1,938 条保存预测已成功重新评分。
