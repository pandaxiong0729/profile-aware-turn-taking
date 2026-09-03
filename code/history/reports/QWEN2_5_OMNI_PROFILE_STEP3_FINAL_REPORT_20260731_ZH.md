# Qwen2.5-Omni Profile 话轮实验：第三步汇报

## 一句话结论

我们已经解决了“模型统一输出”的技术问题：最终方案在两组各 50 条实验中，hidden、given、shuffled 都实际输出了 `C / BC / T / I / NA` 五类，2,400 个二问输出全部有效。正确 profile 会明显改变模型判断，但在当前 99 个不同事件上还没有稳定优于错误 profile，因此现在可以说“模型确实对 profile 敏感”，还不能说“正确 profile 已经稳定提高性能”。

## 1. 我们要回答什么

这一步没有训练模型，只做低成本验证：

> 同一段对话音频和同一份部分转写，在不提供 profile、提供正确 profile、提供错误 profile 时，现成音频大模型对下一话轮事件的判断是否不同，正确 profile 是否更准？

三种条件是：

- `hidden`：profile 全部写为 unknown；
- `given`：使用当前会话的正确 profile；
- `shuffled`：使用另一段会话的错误 profile。

同一条样本中，三组只有 profile 不同。音频、转写、预测时间、问题顺序、解码参数完全相同。

## 2. 数据和模型

### 数据

使用 SBCSAE 双人会话事件数据。当前数据包含 16 个会话和 16 份完整 profile；关系有 5 类，场景有 5 类：

- relationship：romantic partners、family、colleagues、friends/peers、professional/client；
- situation：casual social、collaborative task、workplace/business、family/home、healthcare consultation。

本轮 prompt 开发只使用 4 个会话：`SBC005 / SBC010 / SBC041 / SBC047`。每组 50 条，五类各 10 条。两组只重合 1 条，因此共覆盖 99 个不同事件。

每段模型音频为预测点前 5.9 秒。参考事件在音频结束后 100 ms 开始。

### 模型

- 一个模型：`Qwen2.5-Omni-3B-Q8_0`；
- 推理框架：llama.cpp persistent server；
- 输入顺序：文字提示在前，音频在后；
- 没有训练、微调或 adapter；
- 温度为 0，所有组使用相同随机种子和解码设置。

这里不是三个模型。Qwen2.5-Omni-3B 是模型，Q8_0 是量化格式，llama.cpp 是运行模型的程序。

## 3. 最初为什么会统一输出

### 问题一：预测点附近正在说的半句话丢了

100 条检查样本的预测点都落在一个尚未结束的转写片段中。旧程序为了不泄漏未来文字，会把跨过预测点的整个片段删除。结果是模型虽然收到“部分转写”，但文字平均停在音频结束前约 0.45–0.62 秒，有些样本超过 2 秒；模型不知道最后正在说什么。

修复方法：

1. 保留预测点前已经完整结束的 speaker-timed transcript；
2. 只用截止预测点的同一段音频，让 Qwen 做一次因果 ASR；
3. 把 ASR 得到的末尾半句话同时用于 hidden/given/shuffled；
4. 额外告诉模型音频结束时哪些 speaker 正在发声，但不提供未来词、事件答案或标注证据。

两组 100 段因果 ASR 全部成功，无错误。ASR 每段平均约 0.83–0.85 秒。

### 问题二：一次五选一不适合这个 3B prompt 验证

修复转写以后，一次性让模型在五类中选一个，模型仍倾向固定选择 C。把五类分别打 0–100 分也不理想：模型大量给出 50 或 75 的并列分。

这说明问题不在请求是否成功，而在输入输出任务写法：一句话里同时要求模型判断静音、短回应、换人和打断，3B 模型会选择最安全的默认答案。

### 问题三：A/B 选项存在位置偏差

本地服务可以返回 A 和 B 两个 token 的真实 log-probability。检查发现模型经常偏向第一个选项。只看最终字母会把位置偏差误当成事件判断。

修复方法：每个二问都把选项正序、倒序各问一次，把两次结果换算到相同语义后平均 log-odds。两次 50 条实验中，分别有 319/600 和 349/600 个分支的硬答案受选项顺序影响，说明这项校准是必要的。

## 4. 最终送入模型的内容

每个请求只包含以下内容：

```text
1. 截止预测点 t 的 5.9 秒单声道音频
2. 在 t 前已经说完的 speaker-timed transcript
3. 从同一段因果音频得到的 ASR，包括末尾未说完的词
4. 音频结束时正在说话的 speaker 状态
5. hidden / given / shuffled 中的一种 profile
6. 一个简短 A/B 预测问题
```

不包含未来音频、未来转写、参考标签、事件类型或标注依据。

输入示意：

```text
Audio ends at time t.
Completed transcript before t: ...
Speaker activity at t: speaker_00 is currently audible.
Causal ASR: "... I got my last"
Profile: speaker ages/roles/background, relationship, situation

At exactly t+100 ms, which is more likely?
(A) No new listener response begins; the floor holder continues.
(B) The other participant begins a new response.
Output only A or B.
```

## 5. 四个二问怎样得到五分类

这个设计参考了 ICLR 2025 论文 *Talking Turns: Benchmarking Audio Foundation Models on Turn-Taking Dynamics* 附录第 27–28 页。该论文对 Turn Change、Backchannel、Interruption 分别使用 A/B 问题，而不是一次做多分类。

我们使用四个分支：

1. `silence`：100 ms 后是否静音；
2. `listener_onset`：听者是否开始新回应；
3. `brief_response`：回应是否只是简短 backchannel；
4. `yield`：实质回应开始前，原说话人是否已经让出话轮。

映射为：

```text
静音 → NA
不静音，且听者不开始新回应 → C
听者开始回应，且只是短回应 → BC
听者实质回应，原说话人先让出 → T
听者实质回应，原说话人未让出 → I
```

这是一个层次化输出接口，但最终评测仍是同一套五分类。

## 6. 运行前检查

两组运行都通过了以下检查：

- 每组 `C / BC / T / I / NA` 各 10 条；
- 因果转写时间不晚于音频结束点；
- hidden/given/shuffled 音频 SHA-256 相同；
- hidden/given/shuffled transcript 和 ASR SHA-256 相同；
- 请求中不存在参考答案或事件证据；
- 四个二问都有正序、倒序和三种 profile 条件；
- 每组 1,200 个二问输出完整，0 个无效响应。

## 7. 两组 50 条结果

### 主要指标

| 样本组 | 条件 | Accuracy | Macro-F1 | 输出分布 C/BC/T/I/NA |
| --- | --- | ---: | ---: | --- |
| seed137 | hidden | 24% | 0.194 | 20 / 4 / 2 / 3 / 21 |
| seed137 | given | 20% | 0.174 | 17 / 12 / 5 / 1 / 15 |
| seed137 | shuffled | 28% | 0.263 | 21 / 13 / 6 / 1 / 9 |
| seed237 | hidden | 16% | 0.141 | 18 / 9 / 5 / 3 / 15 |
| seed237 | given | 24% | 0.224 | 16 / 11 / 5 / 3 / 15 |
| seed237 | shuffled | 24% | 0.236 | 12 / 13 / 5 / 5 / 15 |

两组 hidden 都输出五类，最大类别占比分别为 42% 和 36%，防塌缩检查均通过。

### Profile 改变了多少答案

| 样本组 | hidden→given 改变 | given 改对 | given 改错 | given-hidden Macro-F1 差值 | 会话 bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| seed137 | 28/50 | 5 | 7 | -0.020 | [-0.139, 0.086] |
| seed237 | 33/50 | 8 | 4 | +0.083 | [-0.049, 0.194] |

第二组中 given 明显好于 hidden，但第一组相反；两组置信区间都跨过 0。

## 8. 合并 99 个不同事件

去掉两组重复的 1 条后：

| 条件 | Accuracy | Balanced Accuracy | Macro-F1 | C F1 | BC F1 | T F1 | I F1 | NA F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hidden | 20.2% | 20.1% | 0.168 | 0.345 | 0.121 | 0.077 | 0.077 | 0.218 |
| given | 22.2% | 22.1% | 0.201 | 0.302 | 0.279 | 0.138 | 0.083 | 0.204 |
| shuffled | 26.3% | 26.3% | 0.249 | 0.346 | 0.261 | 0.333 | 0.077 | 0.227 |

合并结果：

- given 比 hidden 的 Macro-F1 高 `+0.034`；
- given 比 shuffled 的 Macro-F1 低 `-0.048`；
- hidden→given 改变了 `61/99` 条；
- given 把 hidden 的错误改对 13 条，同时把正确答案改错 11 条；
- given 与 shuffled 的答案有 `50/99` 条不同。

## 9. 现在可以得出什么结论

### 已经确认

1. 输入输出链路已经正常。模型不再统一输出，两个 50 条组都覆盖五类。
2. 模型输出对 profile 文本明显敏感。只替换 profile，就有 56%–66% 的样本改变最终答案；这说明 profile 会影响决策，但不能单凭这一点断言模型正确理解了 profile。
3. 正确 profile 在一组提高性能，在另一组降低性能，当前结果不稳定。
4. 合并后 given 略优于 hidden，但 shuffled 更高，因此还不能证明模型理解了正确 profile 的含义。
5. BC 的 given F1 从 hidden 的 0.121 提高到 0.279，是当前最值得继续观察的类别；I 仍然最难。

### 不能得出的结论

- 不能说“正确 profile 已经稳定有效”；
- 不能把某一组的正向差值单独当成论文结论；
- 不能把 shuffled 较高解释成错误 profile 真有帮助，更可能说明现成 3B 模型对自然语言 profile 的使用还不稳定。

## 10. 当前速度

- 因果 ASR：每段约 0.83–0.85 秒，只做一次；
- 单个 A/B 请求：平均约 0.69 秒，P95 约 0.75 秒；
- 当前每个 profile 条件需要 4 个分支 × 2 个选项顺序，即 8 次请求，顺序执行约 5.5 秒；
- 这套方法适合低成本验证和分析，不是最终实时系统。正式 adapter 应一次前向直接输出层次化概率。

## 11. 下一步怎么做

1. **冻结当前事件输入协议。** 保留因果音频、因果 ASR、speaker 状态、profile 和精确的 `t+100 ms` 决策点，不再回到一次五选一 prompt。
2. **把四个二问变成一个可训练的层次化 adapter。** 主干一次编码音频、转写和 profile；输出 silence、listener onset、brief response、yield 四个概率，再推导五分类。这样既保留当前正常输出，又避免 8 次串行推理。
3. **训练时做 profile 对照。** 同一 checkpoint 在 hidden/given/shuffled 上评测；加入 profile dropout 和 shuffled-profile 负样本，使模型学习“正确 profile 与对话证据是否匹配”，而不是只对 profile 文字敏感。
4. **扩大 profile 多样性。** 当前 prompt 开发集只有 4 个会话 profile。正式实验应使用其余 12 个会话做独立测试，并按会话划分，避免同一 profile 同时进入训练和测试。
5. **重点看 BC 和 I。** BC 已出现可能的 profile 收益；I 的召回率仍低，需要更强的重叠声学特征和动态 relationship/situation 更新。

## 12. 文件在哪里

代码：

- `code/src/profile_turntaking/qwen25_omni_event_eval.py`：因果音频、转写、profile 请求和评分；
- `code/src/profile_turntaking/paper_binary_hierarchy.py`：四个二问、正反顺序 log-probability 校准和五分类映射；
- `code/scripts/run_talking_turns_causal_asr.py`：因果 ASR；
- `code/scripts/run_qwen25_omni_event_eval.py`：统一命令入口；
- `code/scripts/build_qwen_binary_frontend.py`：结果网页；
- `code/docs/QWEN25_OMNI_EVENT_EVAL.md`：完整运行指南。

结果：

- `artfile/q8-v11/gate50-paper-binary-calibrated-seed137/`；
- `artifacts/qwen25-omni-profile/q8-v11/gate50-paper-binary-calibrated-seed237/`；
- `artifacts/qwen25-omni-profile/q8-v11/combined_99_summary.json`；
- `artifacts/qwen25-omni-profile/q8-v11/review.html`。

网页包含 100 张样本卡、100 个可播放音频、参考标签、三种 profile 预测、四个二问答案与 log-odds、因果 ASR、profile 和实际提示词。静态检查已通过：100 个音频路径全部存在。
