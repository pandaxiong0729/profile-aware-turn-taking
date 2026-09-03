# Profile-aware Turn-taking

这个仓库现在只保留两条当前实验链路：

1. **主实验**：冻结 Qwen2.5-Omni 特征，训练 profile-aware shared A/B adapter。
2. **Talking Turns 对照**：用论文公开的 ESPnet checkpoint，在同一批 SBCSAE test 样本上进行 audio-only 测试。

以前的 prompt、五分类 adapter、调参搜索和 smoke 代码/结果均已移入各目录的 `history/`，不参与当前结果。

## 第一次打开先做什么

```powershell
# 查看一条真实样本，以及两种模型分别看到了什么
.\.venv\Scripts\python.exe code\scripts\inspect_experiment_data.py --split test --index 0

# 检查主数据、标签、split 和 hidden/given/shuffled 配对
.\.venv\Scripts\python.exe code\scripts\run_main_experiment.py check

# 查看已经保存的主结果
.\.venv\Scripts\python.exe code\scripts\run_main_experiment.py show-results

# 查看 Talking Turns 已保存结果
.\.venv\Scripts\python.exe code\scripts\run_espnet_talking_turns_baseline.py --score-only --device cpu
```

## 文件从哪里看

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)：完整目录和当前/历史边界。
- [data/README.md](data/README.md)：原始数据、单条输入、标签和 16/60 会话范围。
- [code/README.md](code/README.md)：每个当前脚本的职责和命令。
- [artifacts/README.md](artifacts/README.md)：缓存、模型输出、逐样本结果怎么读。
- [models/README.md](models/README.md)：Qwen 与 Talking Turns 模型分别何时需要。
- [collaboration/PROJECT_FULL_FILE_INVENTORY.csv](collaboration/PROJECT_FULL_FILE_INVENTORY.csv)：逐文件清单；每个文件一行。
- [collaboration/CLOUD_HANDOFF_COMPLETE_GUIDE_ZH.md](collaboration/CLOUD_HANDOFF_COMPLETE_GUIDE_ZH.md)：交给云端实验者的操作说明。
- [code/examples/data_preview/synthetic_input.wav](code/examples/data_preview/synthetic_input.wav)：可直接试听的 4 秒合成示例音频；对应的输入、profile 和预测示例在同一目录。

## 最重要的实验约束

主输入始终是：预测点以前的因果音频、匹配的因果转写和 profile。比较 `hidden / given / shuffled` 时，只允许 profile 改变。预测点以后的音频、转写和标签不得进入模型。

SBCSAE 原始录音与大模型权重受许可和体积限制，不直接提交到 GitHub；仓库保存代码、说明、脱敏/合成示例、可试听的合成音频与结果摘要。
