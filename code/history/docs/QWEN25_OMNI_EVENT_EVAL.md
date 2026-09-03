# Qwen2.5-Omni-3B Profile 话轮实验

这套代码不训练模型，使用本地 `Qwen2.5-Omni-3B-Q8_0` 检查 profile 是否能帮助预测下一步话轮事件。

## 每条模型输入

参考事件发生在 `e`，预测边界固定为 `t = e - 100 ms`。模型只能收到：

1. 截止到 `t` 的单声道对话音频；
2. 与这段音频匹配、同样截止到 `t` 的部分转写；
3. 音频结束时正在说话的 speaker 状态；
4. `hidden`、`given` 或 `shuffled` profile。

模型预测 `t+100 ms` 处开始的事件，并用随后 500 ms 区间区分：

- `C`：当前话轮持有者继续；
- `BC`：听者只给出短回应；
- `T`：原说话人让出话轮后，另一人接话；
- `I`：原说话人尚未让出话轮，另一人开始实质发言；
- `NA`：进入或保持静音。

`hidden / given / shuffled` 对照中，音频、转写、预测边界、问题、解码参数完全相同，只有 profile 内容变化。

## 为什么加入因果 ASR

原始转写按完整话语段保存。如果某句话跨过预测边界，整段文字都不能直接送入模型，否则会泄漏边界后的词。旧程序因此删掉了整句话，导致模型看不到音频末尾正在说的半句话。

新版先让同一个 Qwen 对“截止到预测点的音频”做一次转写，再把这份结果同时用于三种 profile 条件。ASR 没有未来音频，也不接收 profile 或答案。

## 为什么不用一次五选一

一次五选一会使 3B 模型固定输出 C。`Talking Turns: Benchmarking Audio Foundation Models on Turn-Taking Dynamics` 的附录使用独立 A/B 问题评测 Turn Change、Backchannel 和 Interruption。当前实现也拆成四个短问题：

1. `silence`：100 ms 后是否静音；
2. `listener_onset`：听者是否开始新回应；
3. `brief_response`：回应是否只是短 backchannel；
4. `yield`：实质回应开始前，原说话人是否已经让出话轮。

映射规则：

```text
静音 → NA
不静音，且没有听者新回应 → C
听者回应，且只是短回应 → BC
听者实质回应，原说话人先让出 → T
听者实质回应，原说话人尚未让出 → I
```

每个 A/B 问题都把选项正序、倒序各问一次，读取 A 和 B 的 token log-probability，再按语义平均。这样可以消除小模型总选第一个选项的位置偏差。

## 完整运行命令

在项目根目录执行：

```powershell
$python = ".venv\Scripts\python.exe"
$base = "artifacts\qwen25-omni-profile\q8-v11\gate50-base"
$run = "artifacts\qwen25-omni-profile\q8-v11\gate50-binary"

& $python code\scripts\run_qwen25_omni_event_eval.py prepare `
  --manifest data\processed\sbcsae_turn_events_v1\annotation_manifest.jsonl `
  --output-dir $base `
  --phase gate50 `
  --config code\configs\qwen25_omni_profile_pilot_v10.json `
  --prompt-style direct

& $python code\scripts\run_talking_turns_causal_asr.py `
  --run-dir $base `
  --model Qwen2.5-Omni-3B-Q8_0

& $python code\scripts\run_qwen25_omni_event_eval.py apply-causal-asr `
  --run-dir $base

& $python code\scripts\run_qwen25_omni_event_eval.py prepare-binary-hierarchy `
  --source-run-dir $base `
  --output-dir $run

& $python code\scripts\run_qwen25_omni_event_eval.py run-binary-hierarchy `
  --run-dir $run `
  --model Qwen2.5-Omni-3B-Q8_0

& $python code\scripts\run_qwen25_omni_event_eval.py aggregate-binary-hierarchy `
  --run-dir $run

& $python code\scripts\run_qwen25_omni_event_eval.py score `
  --run-dir $run
```

## 运行前自动检查

程序会检查：

- 五类样本数；
- 所有转写时间都不晚于音频结束点；
- 三种 profile 条件的音频 SHA-256 完全相同；
- 三种条件的转写和 ASR SHA-256 完全相同；
- 请求中没有参考答案和事件证据；
- 每个二问都有正序、倒序以及 hidden/given/shuffled 三种条件；
- 输出格式和请求数量完整。

检查结果分别写入 `input_audit.json` 和 `binary_input_audit.json`。

## 主要输出文件

- `binary_responses.jsonl`：每个二问的 A/B 输出和 A/B token 概率；
- `binary_predictions.jsonl`：四个问题合成的逐样本五分类结果；
- `responses.jsonl`：供统一评分器读取的逐样本结果；
- `metrics.json`：三种 profile 的 Accuracy、Macro-F1、Balanced Accuracy 和每类指标；
- `diagnostics.json`：输出分布、profile 改变次数、完整性和防塌缩检查；
- `predictions.csv`：每条样本的参考标签与三种预测；
- `bootstrap_95ci.json`：按会话 bootstrap 的 95% 置信区间；
- `binary_aggregation.json`：四个二问的答案分布和选项顺序一致性。

生成可听音频、可展开输入的网页：

```powershell
& $python code\scripts\build_qwen_binary_frontend.py `
  --run $run `
  --output artifacts\qwen25-omni-profile\q8-v11\review.html
```
