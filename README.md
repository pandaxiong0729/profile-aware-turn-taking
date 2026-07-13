# Profile-aware Turn-taking

本仓库按实验说明和可运行代码分成两个目录：

| 目录 | 内容 |
|---|---|
| [`code/`](code/) | 模型、数据处理、训练、评测、自动测试和小型测试数据 |
| [`intro/`](intro/) | 实验总体思路、评测设计和 profile 输入示例 |

## 从哪里开始

- 想先理解实验：阅读 [`intro/profile_aware_turn_taking_intro_experiments.md`](intro/profile_aware_turn_taking_intro_experiments.md)。
- 想检查一条输入样本：查看 [`intro/sbcsae_profile_turntaking_training_example.json`](intro/sbcsae_profile_turntaking_training_example.json)。
- 想安装并运行：进入 [`code/`](code/) 并按照其中的 README 操作。

当前版本实现五分类 `C / BC / T / I / NA`，并支持同一 checkpoint 下的 profile `hidden / given / shuffled` 对照实验。

零训练的真实音频 MLLM pilot 见 [`code/docs/MLLM_PROMPT_BASELINE.md`](code/docs/MLLM_PROMPT_BASELINE.md)，本机结果见 [`code/reports/MLLM_PROMPT_QWEN2_5_OMNI_3B_REPORT.md`](code/reports/MLLM_PROMPT_QWEN2_5_OMNI_3B_REPORT.md)。
