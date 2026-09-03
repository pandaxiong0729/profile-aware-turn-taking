# 音频多模态大模型 + Profile 话轮预测：500 条实验汇报

日期：2026-07-14

## 1. 实验目的

本实验评测一个现成音频多模态大模型：在输入相同音频和历史转写时，加入正确的说话人 profile，能否提高下一时刻的话轮状态预测准确率。

本轮不训练、不微调模型，只做 prompt 零样本评测。

## 2. 数据、模型和输入

### 数据

- 数据集：SBCSAE 双人自然对话；
- 测试样本：500 条；
- 当时的五类数量：C、BC、T、I、NA 各 100 条；
- 每条样本对应一个预测时刻 `t`。

原始数据位置：

- 音频：`data/sbcsae/openslr/WAV/`
- 时间转写：`data/sbcsae/openslr/TRN/`
- 会话和人物信息：`data/sbcsae/openslr/CHAT/`、`data/sbcsae/metadata/`

### 模型

- Qwen2.5-Omni-3B；
- Q4_K_M 量化；
- llama.cpp 推理；
- 本机 RTX 5060 Laptop GPU 8 GB；
- temperature=0，seed=13。

### 每条请求的输入

1. 预测时刻前 30 秒单声道音频 `[t-30s,t)`；
2. 截止 `t` 已结束的历史转写，包含 Speaker A/B 和时间；
3. 固定自然语言模板生成的 profile。

Profile 包括双方年龄段、性别、职业/社会角色、背景、relationship 和 situation。

模型输出 `[t,t+40ms]` 的一个标签：

- `C`：当前说话人继续；
- `BC`：听者简短反馈；
- `T`：发生话轮转移；
- `I`：两个人重叠说话；
- `NA`：无人说话。

## 3. 三种对照条件

同一条样本运行三次，共 1,500 次请求：

| 条件 | Profile 输入 |
|---|---|
| hidden | 不提供 profile |
| given | 提供当前会话的正确 profile |
| shuffled | 提供另一会话的错误 profile |

三种条件的音频、转写、预测边界、prompt、模型和解码参数完全相同，只改变 profile。

提示词核心是：结合预测边界前的音频、历史转写和 profile，预测未来 40 ms 的 C/BC/T/I/NA，只返回一个 JSON 标签。

实际提示词代码：

`code/src/profile_turntaking/mllm_prompt_baseline.py`

## 4. 评测指标

- Macro-F1：主要指标；
- Balanced Accuracy；
- Accuracy；
- 每类 Precision、Recall、F1；
- 混淆矩阵；
- hidden/given/shuffled 成对变化；
- exact McNemar 检验；
- 模型预测是否塌缩到少数类别；
- 推理延迟和输出有效率。

## 5. 500 条实验结果

1,500/1,500 个请求均得到有效输出。

### 总体结果

| 条件 | Macro-F1 | Balanced Accuracy | Accuracy | 正确数/500 |
|---|---:|---:|---:|---:|
| hidden | 0.0823 | 0.1940 | 0.1940 | 97 |
| given | 0.0803 | 0.2020 | 0.2020 | 101 |
| shuffled | 0.0746 | 0.2000 | 0.2000 | 100 |

given 的普通 Accuracy 比 hidden 高 0.8 个百分点，但主要指标 Macro-F1 从 0.0823 降到 0.0803。因此不能认为正确 profile 带来了稳定提升。

### 每类 F1

| 条件 | C | BC | T | I | NA |
|---|---:|---:|---:|---:|---:|
| hidden | 0.0774 | 0 | 0 | 0.3339 | 0 |
| given | 0.0650 | 0 | 0 | 0.3362 | 0 |
| shuffled | 0.0323 | 0 | 0 | 0.3409 | 0 |

### 预测分布

| 条件 | C | BC | T | I | NA |
|---|---:|---:|---:|---:|---:|
| hidden | 55 | 0 | 0 | 445 | 0 |
| given | 23 | 0 | 0 | 477 | 0 |
| shuffled | 24 | 0 | 1 | 475 | 0 |

模型严重偏向预测 `I`：hidden 中 89.0% 为 I，given 中 95.4% 为 I，shuffled 中 95.0% 为 I。BC、T、NA 基本无法识别。

### 成对比较

- hidden 与 given 有 66/500 条预测不同；
- hidden 与 shuffled 有 64/500 条预测不同；
- given 与 shuffled只有 38/500 条预测不同；
- exact McNemar p=0.503，没有显著证据说明 given 优于 hidden。

## 6. 阶段结论

在这次 500 条实验中：

1. 加入 profile 会改变部分预测；
2. 正确 profile 没有稳定优于不提供 profile；
3. 正确 profile 与 shuffled profile 的表现非常接近；
4. 当前模型输出严重塌缩到 I，无法形成可靠的五分类基线；
5. 因此本轮没有证明 profile 能提高现有大模型的话轮预测性能。

## 7. 结果文件和代码

500 条结果目录：

`artifacts/mllm-prompt-baseline/qwen2.5-omni-3b/audio-transcript-profile-test-100-per-class/`

主要文件：

- `metrics.json`：总体、每类指标和混淆矩阵；
- `diagnostics.json`：预测分布、塌缩和延迟；
- `predictions.csv`：500 条逐样本三条件预测；
- `profile_comparison.csv`：三条件汇总；
- `requests.jsonl`：1,500 个请求；
- `responses.jsonl`：1,500 个输出；
- `input_audit.json`：三条件输入一致性检查。

主要代码：

- 请求、提示词和推理：`code/src/profile_turntaking/mllm_prompt_baseline.py`
- Profile 模板和评分：`code/src/profile_turntaking/prompt_baseline.py`
- 主运行入口：`code/scripts/run_mllm_prompt_baseline.py`

## 8. 最后补充：当前已知问题和可能原因

后续审计发现，这 500 条存在旧弱标签错误、重复连续事件、只覆盖 3 个会话，以及 prompt 与标签语义不完全一致。因此这些数字可以作为本次阶段汇报和问题诊断，但不能作为论文最终结果。

可能原因主要有三点：

- 3B 量化通用模型不擅长未来 40 ms 的精细话轮预测；
- 模型没有真正理解正确 profile，更多是受到额外文本影响；
- 当时的数据标签和抽样方式降低了结果可靠性。

一句话总结：**500 条实验真实跑完了，但模型五分类输出严重塌缩，正确 profile 没有表现出可信的性能提升。**
