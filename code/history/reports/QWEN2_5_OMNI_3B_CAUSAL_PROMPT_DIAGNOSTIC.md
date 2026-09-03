# Qwen2.5-Omni-3B 严格因果提示测试记录

日期：2026-07-19

## 结论

本轮只做现有 3B 模型的低成本提示测试，不是训练结果，也不是 profile 效果实验。

在当前本地 checkpoint 下，模型可以收到并转写音频，但尚不能稳定完成 `C / BC / T / I / NA` 五分类未来事件预测。所有已测 hidden 基线都只使用了 1–2 个类别，未通过非塌缩门禁。因此没有继续扩大到 50 或 500 条，也不能声称 given profile 有提升。

## 模型和输入

- 模型：Qwen2.5-Omni-3B，同一个模型，不是三个模型。
- 权重：`Qwen2.5-Omni-3B-Q4_K_M.gguf`。
- 推理程序：`llama.cpp` server。
- 每条输入：截止预测边界 `t` 的单声道音频、与该音频匹配且不晚于 `t` 的部分转写、profile。
- 每条输出：预测 `t` 之后指定时间范围内的 `C / BC / T / I / NA`。
- hidden、given、shuffled 三组只允许改变 profile；音频、转写、样本编号、预测边界、提示指令和解码设置保持相同。
- 10 条开发样本按自动候选标签平衡抽取，每类 2 条。它们是未人工复核的弱标签，只能用于排查流程。

## 音频是否真正输入模型

已确认音频正常传入：让模型直接转写因果音频时，可以识别出诸如 `I'm laughing...`、`being there I got it` 等实际语音。

在 400 ms 预测版本中，把真实音频替换成同长度静音后，10 条中有 4 条预测改变，且静音输入全部预测为 `NA`。这说明模型并非完全忽略音频，但它主要学到了粗粒度的“语音/静音”，没有稳定分出五种话轮事件。

## 主要提示版本结果

下面的 Macro-F1 只相对于自动弱标签计算，不能作为正式模型成绩。

| 版本 | 预测位置 | hidden 输出分布 | Macro-F1 | 非塌缩门禁 |
|---|---|---:|---:|---:|
| v3 | 未来 500 ms 内第一个事件，目标约在 400 ms 后 | T=4, NA=6 | 0.2333 | 未通过 |
| v4 | 强调“第一个话轮转换” | T=10 | 0.0667 | 未通过 |
| v5 | 目标约在 100 ms 后 | T=5, I=5 | 0.1714 | 未通过 |
| v6 | v5 加五类文字示例 | T=3, I=7 | 0.0889 | 未通过 |
| v7 | 精确预测 `[t+100,t+140) ms` | T=5, I=5 | 0.1143 | 未通过 |
| v8 | 精确预测 `[t+20,t+60) ms` | T=3, I=7 | 0.1244 | 未通过 |
| v9 | v8 改为最简提示 | C=5, T=5 | 0.1143 | 未通过 |

额外测试：

- 取消 JSON 约束：仍只输出 1–2 类。
- 先判断当前声学状态、再预测未来事件：仍塌缩为 `C/NA`。
- 为五类分别打分：真实音频、静音和错配音频最终都选 `C`。
- 为五个候选分别询问 YES/NO 并比较 token 概率：只得到 `C=6, T=4`，相对本组弱标签为 0/10。
- 将预测提前量从 20 ms 增加到 400 ms：没有恢复五类行为。

## 当前判断

问题不再是“音频没有传进去”，也不能只靠继续改一句提示词解决。当前 Q4 量化的 Qwen2.5-Omni-3B 在这个严格因果、细粒度五分类任务上表现为输出类别塌缩，尤其不能稳定区分 BC、T 和 I。

严格因果输入代码和门禁应保留；下一步若继续做正式实验，应先人工核验一小组目标标签，再选择更强的音频模型或对 3B 做监督训练。hidden 基线在人工核验集上表现正常后，才能运行 given/shuffled profile 对照。

## 可复核位置

- 主代码：`code/src/profile_turntaking/qwen25_omni_event_eval.py`
- 推理代码：`code/src/profile_turntaking/mllm_prompt_baseline.py`
- 命令入口：`code/scripts/run_qwen25_omni_event_eval.py`
- 当前配置：`code/configs/qwen25_omni_event_eval_v1.json`
- v3 结果：`artifacts/qwen25-omni-event-eval/micro-v3-lead400-horizon500/`
- v3 静音对照：`artifacts/qwen25-omni-event-eval/micro-v3-lead400-horizon500-silence/`
- v9 结果：`artifacts/qwen25-omni-event-eval/micro-v9-minimal-offset20-window40/`
- v9 音频/静音/错配打分对照：`artifacts/qwen25-omni-event-eval/micro-v9-minimal-offset20-window40/probability-control-v1/`
- 逐类 YES/NO 概率测试：`artifacts/qwen25-omni-event-eval/micro-v3-lead400-horizon500/one_vs_rest-logprobs-v1/`
