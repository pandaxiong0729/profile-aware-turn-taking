# Qwen2.5-Omni-3B 输入输出复查报告（2026-07-19）

## 结论

当前模型能够听到并转写 SBCSAE 音频，但还不能作为可靠的五分类 turn-taking 基线。

The 16-sample pilot reached 11/16 with semantic outputs, but only the interruption task produced both outcomes. After repairing the event rules, a newly sampled balanced 50-item hidden run scored 26/50 (52%); BC, floor-taking, and turn-change still emitted one constant answer per task. The checkpoint therefore fails the hidden-baseline gate and cannot support a profile-effect claim.

本次还确认旧自动事件规则确实存在标签问题。修复后复算 10,149 个非静音事件，有 79 个同一时间点的标签发生 C/T/BC 翻转；当前 16 条 pilot 中有 1 条 T 应改成 C。旧测试结果只保留为排错记录。

## 模型和运行方式

- 模型：Qwen2.5-Omni-3B，Q8_0 量化。
- 运行：llama.cpp，本机 RTX 5060 Laptop 8 GB。
- 输入顺序：先放文字指令，再放音频；反过来时模型曾把真实语音当成静音。
- 每条推理约 0.8–2.1 秒；这是小样本诊断，不是吞吐量基准。

## 模型实际收到什么

每条请求只包含预测时刻 `t` 之前的信息：

1. 最长 30 秒的单声道混合音频，音频结尾就是预测时刻 `t`；
2. 与这段音频匹配、且不越过 `t` 的带说话人时间戳的部分转写；
3. hidden 条件的 profile 占位文字；
4. 当前二分类任务的说明。

模型没有收到未来音频、未来转写、目标标签、自动标注依据或人工答案。输入审计见 `input_audit.semantic.json` 和 `input_audit.probability.json`。

四个任务不是一次选五类，而是按照 Talking Turns 分开预测：

| 任务 | 模型预测的问题 |
|---|---|
| Turn change | 停顿后当前说话人继续，还是另一人接话 |
| Backchannel | 紧接着是否出现简短回应 |
| Interruption | 另一人是否会在当前说话人结束前开始完整发言 |
| Floor-taking | 重叠开始后第二位说话人是否成功拿到话轮 |

当前最有效的提示词核心是：

```text
Listen to the attached two-speaker conversation.
The audio contains only the past and ends exactly at prediction time t.
Predict only what happens after t.

[一个具体的二分类问题]

Causal speaker-timed partial transcript; every listed unit ends no later than t:
[预测点之前的转写]

Profile condition:
Speaker profiles, relationship, and situation are unknown.

End with exactly one final conclusion from the two named outcomes.
```

## 输出怎么解释

测试了三种输出：

| 输出设计 | 结果 | 问题 |
|---|---:|---|
| A/B 二选一 | 6/16 | 交换 A/B 含义后只有 2/16 保持相同语义，存在严重字母位置偏置 |
| 固定语义 JSON | 8/16 | 每个任务内部仍输出同一个类别；turn-change 完全随答案排列改变 |
| 0–100 概率 | 旧弱标签 8/16 | BC 和 I 有变化，但 floor-taking、turn-change 概率全为 100 |
| 自然语言结论 | 修正规则后 11/16 | BC 4 条全为 NO，floor 4 条全为成功，turn-change 4 条全为继续；只有 I 非塌缩 |

自然语言结论的分任务结果：

| 任务 | 正确/总数 | 预测分布 | 是否可认为有效 |
|---|---:|---|---|
| Backchannel | 2/4 | NO_BACKCHANNEL=4 | 否，恒定答案 |
| Floor-taking | 2/4 | SECOND_SPEAKER_TAKES_FLOOR=4 | 否，恒定答案 |
| Interruption | 4/4 | INTERRUPTS=2，CONTINUES=2 | 只能认为值得扩样，样本太少 |
| Turn change | 3/4 | CURRENT_SPEAKER_CONTINUES=4 | 否，类别不平衡后恒定答案得到 75% |

这里的答案仍是自动结构规则产生的弱标签，不是人工金标，所以 4/4 也不能当正式准确率。

## Corrected v3 hidden-50 result

The new test set comes from the separate `sbcsae_turn_events_v3` directory and does not overwrite the old data. Each binary task is internally balanced. Every request still contains causal audio, its matching causal partial transcript, and the hidden-profile placeholder.

| Task | Correct/total | Prediction distribution | Gate |
|---|---:|---|---|
| Backchannel | 6/12 | NO_BACKCHANNEL=12 | Failed: constant answer |
| Floor-taking | 7/14 | SECOND_SPEAKER_TAKES_FLOOR=14 | Failed: constant answer |
| Interruption | 7/12 | INTERRUPTS=7, CONTINUES=5 | Non-collapsed, but accuracy remains low |
| Turn change | 6/12 | CURRENT_SPEAKER_CONTINUES=12 | Failed: constant answer |
| Overall | 26/50 (52%) | Four task-specific outputs | Overall diversity must not hide within-task collapse |

The 52% result is consistent with Talking Turns' finding that open audio foundation models are close to random on future turn-taking prediction. Because three tasks failed the hidden gate, given/shuffled profile runs are not performed or interpreted.

## 模型有没有真的听音频

有，但不同任务的依赖程度不同：

- Qwen 对 16 条因果音频全部生成了可辨认的英文转写，说明音频输入和模型通路工作正常。
- 把真实音频替换成等长数字静音，同时保持转写、profile、问题和预测时刻不变，16 条中 5 条概率改变。
- BC：4/4 改变；floor-taking：1/4 改变；interruption 和 turn-change：0/4 改变。

因此不能说模型完全没听音频，但当前 I/T 输出主要由文本或任务先验决定。

## 已发现并修复的数据规则错误

旧规则有两个明确问题：

1. 两个说话人同一时刻开始的短 IPU 可能互相被识别成 BC，真正的 floor speaker 会被错误排除；
2. 只有几十毫秒且没有转写的声音可能被当成完整话轮，造成 C 和 T 互换。

修复位置：

- `code/src/profile_turntaking/event_annotation.py`
- 规则版本：`sbcsae_event_candidates_v3_floor_safeguards`
- 回归测试：`code/tests/test_event_annotation.py`

全量静态复算摘要见 `event_rule_v3_audit.json`。旧的 `SBC044-event-0011479` 从 T 修正为 C；旧数据目录尚未被覆盖，避免破坏已有人工作。

## 与论文做法的关系

Talking Turns 对现成音频大模型也是分别做四个平衡二分类，而不是直接要求统一五分类；论文里的开源音频大模型在未来预测上基本接近 50%。达到约 75%–79% 的是用 Switchboard 监督训练的专用模型，不是 prompt-only Qwen。

2025–2026 年效果较好的工作也主要训练概率头或分类头：ACL 2025 的模型融合文本、音频和视觉后预测 turn/backchannel 概率；Prompt-Guided VAP 把文本提示 embedding 注入经过训练的 VAP；DualTurn 先做双声道生成式语音预训练，再微调显式 turn-taking 信号头。这些结果不支持“只改提示词就一定能得到很高准确率”的假设。

## Next step

The v3 directory and balanced hidden-50 run are complete. The hidden gate failed in three tasks, so the scientifically valid next action is to stop prompt-only profile comparisons. Use the 50-item package for human label checking, then either evaluate a stronger checkpoint or train a turn-taking probability/classification head before running hidden/given/shuffled profile conditions.

## 文件位置

- 可复用提示词诊断代码：`code/src/profile_turntaking/talking_turns_prompt_pilot.py`
- 命令入口：`code/scripts/run_talking_turns_prompt_pilot.py`
- 事件规则代码：`code/src/profile_turntaking/event_annotation.py`
- 16 条请求：`artifacts/talking-turns-paper-aligned/qwen2.5-omni-3b-q8/pilot16-causal-asr/requests.jsonl`
- Qwen 因果 ASR：同目录 `asr.jsonl`
- 自然语言输出：同目录 `responses.reasoned.jsonl`
- 修正规则后的结果：同目录 `metrics.reasoned.corrected_v3.json`
- 静音对照：同目录 `silence-probability-control/audio_sensitivity.json`
- 规则复算：同目录 `event_rule_v3_audit.json`
- New v3 event table: `data/processed/sbcsae_turn_events_v3/`
- Hidden-50 requests/results: `artifacts/talking-turns-paper-aligned/qwen2.5-omni-3b-q8/hidden50-v3-reasoned/`
- Hidden-50 preparation: `code/scripts/prepare_talking_turns_hidden50_v3.py`
- Hidden-50 inference/scoring: `code/scripts/run_talking_turns_hidden_requests.py`

## 参考

- Talking Turns（ICLR 2025）：https://openreview.net/pdf?id=2e4ECh0ikn
- ACL 2025 多模态 turn/backchannel 概率模型：https://aclanthology.org/2025.acl-long.743/
- Prompt-Guided Turn-Taking Prediction：https://arxiv.org/abs/2506.21191
- DualTurn：https://arxiv.org/abs/2603.08216
