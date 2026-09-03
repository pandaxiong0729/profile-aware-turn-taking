# Profile-aware / State-aware Turn-taking for Voice Agents：完整 intro 草稿

语音助手正在从简单的“用户说一句、系统答一句”走向更自然的连续对话。在真实的人类交流中，对话并不是严格轮流进行的。听者会用“嗯嗯”“对”“我在听”这样的简短反馈表示关注，也可能在对方卡住或犹豫时轻轻补一句。但这些行为并不总是合适的：同样一次停顿，对一个用户可能表示“我说完了”，对另一个用户却只是“我还在想”；同样一句补全，对一个用户可能是帮助，对另一个用户可能是打断。

现有 voice agent 已经开始支持 interruption、backchannel 和 barge-in。例如，有些系统允许用户打断 agent，也有系统会在用户说话时插入简短反馈，或者在用户长时间沉默时主动开启话题。这些方法让语音交互更接近真实对话，但它们通常依赖固定规则或手动设置的参数，例如固定等待时间、固定 backchannel 频率、固定打断阈值。这样的问题是，系统对所有用户几乎使用同一种节奏，不能根据不同用户的说话方式和实时状态调整行为。

我们关注的问题是：语音助手能不能不只听用户“说了什么”，还理解“谁在和谁说话”以及“这段对话发生在什么场景中”。同样一次短暂停顿，在朋友闲聊中可能只是共同思考，在陌生人对话中可能更容易被理解为话轮结束；同样一句简短的“yeah”或“right”，在日常聊天中可能只是 backchannel，在教学、咨询或访谈场景中却可能意味着对方准备接过话轮；同样一次重叠进入，在亲密关系中可能是合作性补充，在正式咨询场景中则更容易被感知为打断。因此，speaker 的角色、双方关系和当前 situation 都可能影响下一步最合适的话轮行为。

为此，我们希望构建一个带 profile 和动态用户状态的 turn-taking 方法。长期目标是维护一个动态用户状态 `s_u`，它不是固定用户画像，而是根据最近连续对话不断更新的状态描述，例如：“用户当前语速较慢，停顿较长，语气犹豫，且之前接受过助手补全。”这个状态会进入 turn-taking judge，帮助系统决定下一步该继续等待、给出 backchannel、正式接话、合作性补全，还是保持安静。

当前阶段，我们先做一个更可控的前置验证：在同一段音频和同一段因果转写完全不变的情况下，加入说话人 profile 是否能提高下一步话轮事件预测。如果固定 profile 已经能改变模型判断，并且正确 profile 比错误 profile 更好，那么后续再引入动态 `s_u` 就有更清楚的实验基础。

## 1. 数据集

我们当前使用 **SBCSAE（Santa Barbara Corpus of Spoken American English）**。选择这个数据集的原因是，它不仅有自然连续对话音频和人工转写，还有 speaker metadata、说话人时间戳、会话说明、人物关系和交谈场景。这些信息可以整理成 profile。

当前主要处理了 16 个核心双人会话，共约 6.43 小时音频。我们从中构建了两类数据：

### 1.1 逐帧五分类弱标签

第一类数据是 40 ms 逐帧标签。我们用 VAD、转写时间戳和规则，把每个 40 ms 时间格标成五类：

| 标签 | 含义 |
|---|---|
| C | 当前说话人继续说 |
| BC | 听者给出简短反馈 / backchannel |
| T | 自然换人 / turn change |
| I | 打断 / interruption |
| NA | 当前无人说话 |

规则优先级为：

```text
NA > BC > I > T > C
```

逐帧统计如下：

| 标签 | 40 ms 帧数 | 占比 |
|---|---:|---:|
| C | 346,016 | 59.79% |
| BC | 14,041 | 2.43% |
| T | 1,358 | 0.23% |
| I | 26,188 | 4.52% |
| NA | 191,083 | 33.02% |
| 合计 | 578,686 | 100% |

这套数据覆盖完整音频，但它是自动弱标签，不是人工金标签。

### 1心数据

后来我们发现，如果每 40 ms 都取样，连续说话片段会产生大量重复 C，很多样本其实不是独立事件。因此又构建了事件中心数据。它不是每 40 ms 取一次，而是围绕 IPU、Pause、Gap、Overlap 和短回应等结构找事件。

当前事件候选为 16,325 条：

| 候选标签 | 数量 |
|---|---:|
| C | 5,035 |
| BC | 1,315 |
| T | 2,585 |
| I | 1,301 |
| NA | 6,089 |
| 合计 | 16,325 |

每条事件都有一个最多约 10 秒的审核音频，包含目标事件前约 6 秒和后约 4 秒。这个数据后续可以用于人工标注和建立金标测试集。

### 1.3 当前主实验数据

当前 Qwen embedding 实验使用同一批 held-out semantic profile test set：

| split | 会话 | 样本数 | 每类样本数 |
|---|---|---:|---:|
| train | SBC005 / 006 / 009 / 010 / 024 / 041 / 044 / 045 / 047 / 060 | 1,500 | 300 |
| validation | SBC029 / 034 / 043 | 250 | 50 |
| test | SBC007 / 017 / 058 | 250 | 50 |

测试集共 250 条，五类均衡：

```text
C=50, BC=50, T=50, I=50, NA=50
```

每条样本都有三种 profile 条件：

| 条件 | 含义 |
|---|---|
| hidden | 不给 profile，profile 向量为 0 |
| given | 给正确 profile |
| shuffled | 给错误 profile |

三种条件中，音频、转写、预测边界、sample ID 和标签都不变，只改变 profile。

## 2. Profile 和后续动态状态

当前实验使用固定 profile，主要包括：

| 字段 | 含义 |
|---|---|
| speaker age / age group | 年龄或年龄段 |
| gender / demographic field | 数据集中记录的人口统计字段 |
| social role | 职业、社会角色、家庭角色或制度性角色 |
| background | 教育、地区、语言或其他背景信息 |
| relationship | 双方关系 |
| situation | 当前交谈场景 |

后续动态用户状态 `s_u` 可以看作 profile 的时间变化版本。固定 profile 描述“这个人是谁、双方什么关系、当前什么场景”；动态 `s_u` 描述“这个人现在怎么说话、最近怎么反应”。例如：

```text
Static profile:
speaker_00: adult, teacher, ...
speaker_01: student, ...
relationship: teacher-student
situation: tutoring

Dynamic state s_u:
user currently speaks slowly, has long pauses, sounds hesitant,
and accepted assistant completions earlier in the conversation.
```

当前实验先验证固定 profile 是否有用；如果成立，下一步再把 `relationship`、`situation`、语速、停顿、情绪、历史反应等信息做成动态状态。

## 3. 任务形式

最初任务是直接预测五分类：

```text
C / BC / T / I / NA
```

但直接五分类对当前本地 Qwen prompt 不稳定，模型容易只输出少数类别。因此后续把五分类拆成四个 A/B 二分任务：

| 二分任务 | A | B | 适用标签 |
|---|---|---|---|
| silence | NA / 没人说话 | 有人说话 | C/BC/T/I/NA |
| listener_onset | 当前说话人继续 | 另一人开始回应 | C/BC/T/I |
| brief_response | 简短 backchannel | 更实质的接话或打断 | BC/T/I |
| yield | 自然换人 | 打断 | T/I |

250 条测试样本在 A/B 形式下会产生：

```text
silence        250 条
listener_onset 200 条
brief_response 150 条
yield          100 条
合计           700 条

三种 profile 条件：
700 × 3 = 2100 条预测记录
```

## 4. 模型结构和实验

我们一共做了七类实验。前几类主要是诊断，最后两类是当前主结果。

### 4.1 Qwen prompt 直接五分类

输入：

```text
预测点之前的音频
+ 预测点之前的历史转写
+ profile 文本
+ 要求输出 C / BC / T / I / NA
```

模型：本地 Qwen2.5-Omni-3B Q4_K_M，通过 llama.cpp 推理，不训练。

输出：直接输出 JSON 标签，例如：

```json
{"label": "T"}
```

250 条测试结果：

| 条件 | Macro-F1 | Balanced Accuracy | Accuracy | 预测分布 |
|---|---:|---:|---:|---|
| hidden | 0.0868 | 0.1960 | 0.1960 | C=0, BC=0, T=226, I=24, NA=0 |
| given | 0.0935 | 0.2120 | 0.2120 | C=0, BC=0, T=236, I=14, NA=0 |
| shuffled | 0.0907 | 0.2040 | 0.2040 | C=0, BC=0, T=236, I=14, NA=0 |

结论：模型几乎只输出 T/I，直接 prompt 五分类不可靠。

### 4.2 500 条 MLLM prompt 诊断

早期还做过 500 条 prompt 实验，每类 100 条，每条运行 hidden / given / shuffled，共 1,500 次请求。

结果：

| 条件 | Macro-F1 | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|
| hidden | 0.0823 | 0.1940 | 0.1940 |
| given | 0.0803 | 0.2020 | 0.2020 |
| shuffled | 0.0746 | 0.2000 | 0.2000 |

模型主要预测 I，BC/T/NA 的 F1 为 0。后续也发现该批数据存在连续事件重复、NA 来源集中、prompt 和标签定义不一致等问题。因此这批结果只保留为诊断，不作为主证据。

### 4.3 MiniLM R2 五分类 embedding pilot

这个实验验证“把 profile 做成 embedding，再训练小模型”是否比 prompt 更稳。

输入：

```text
预测边界 t 以前的因果音频
+ t 以前的因果转写
+ 说话人活动状态
+ profile 文本
```

编码：

- `all-MiniLM-L6-v2` 把转写文本编码为 384 维向量；
- 同一个 MiniLM 把 profile 文本编码为 384 维向量；
- 音频使用 132 维多时间尺度声学与边界特征。

结构：

```text
音频特征 132 维 → Linear → 128
转写向量 384 维 → Linear → 128
profile 向量 384 维 → Linear → 128
音频+转写 shared state + gate × profile state
        ↓
Linear → 5
        ↓
C / BC / T / I / NA 概率
```

训练：

- MiniLM 冻结；
- 只训练音频分支、转写分支、profile 投影、gate 和五分类头；
- 损失函数是加权交叉熵；
- 三个随机种子取平均。

测试结果：

| Profile 条件 | Macro-F1 | Balanced Accuracy | Log Loss | Brier Score |
|---|---:|---:|---:|---:|
| hidden | 0.4037 | 0.4107 | 1.5541 | 0.7625 |
| given | 0.4072 | 0.4173 | 1.5460 | 0.7566 |
| shuffled | 0.4048 | 0.4160 | 1.5466 | 0.7569 |

结论：测试集上 given 略高，但提升很小，验证集不稳定。因此它说明 embedding 方向可行，但还不能作为最终证据。

### 4.4 Qwen hidden + profile adapter 五分类

这个实验把 MiniLM 换成 Qwen 向量。

输入：

- `qwen_context`：因果音频 + 因果转写的 Qwen hidden/context 向量；
- `profile_given`：正确 profile 的 Qwen embedding；
- `profile_shuffled`：错误 profile 的 Qwen embedding；
- hidden 条件下 profile 向量为 0。

结构：

```text
qwen_context 2048 维
profile_embedding 2048 维
        ↓
小型融合 adapter
        ↓
Linear → 5
        ↓
C / BC / T / I / NA 概率
```

训练：

- Qwen 冻结；
- 只训练后面的 adapter 和五分类头；
- 尝试 gate 融合和 concat/MLP 融合；
- 指标主要看 Macro-F1。

代表性结果：

| 方法 | hidden | given | shuffled | given-hidden | given-shuffled |
|---|---:|---:|---:|---:|---:|
| adapter_gate_pdrop050 | 0.3903 | 0.3965 | 0.3977 | +0.0061 | -0.0012 |
| adapter_concat_pdrop050 | 0.4163 | 0.4084 | 0.4094 | -0.0079 | -0.0010 |
| adapter_gate_pdrop025 | 0.3959 | 0.3935 | 0.3862 | -0.0024 | +0.0073 |
| adapter_concat_pdrop000 | 0.4197 | 0.4167 | 0.4051 | -0.0030 | +0.0116 |

结论：比 prompt 稳定，不再严重塌缩；但 given 没有稳定超过 shuffled。

### 4.5 A/B prompt 换算

这一步不是新模型，而是把 prompt 五分类结果换算成四个 A/B 任务。

结果：

| profile 条件 | silence | listener_onset | brief_response | yield | 平均 |
|---|---:|---:|---:|---:|---:|
| hidden | 0.8000 | 0.7500 | 0.6667 | 0.4900 | 0.6767 |
| given | 0.8000 | 0.7500 | 0.6667 | 0.5300 | 0.6867 |
| shuffled | 0.8000 | 0.7500 | 0.6667 | 0.5100 | 0.6817 |
| given-hidden | +0.0000 | +0.0000 | +0.0000 | +0.0400 | +0.0100 |
| given-shuffled | +0.0000 | +0.0000 | +0.0000 | +0.0200 | +0.0050 |

结论：profile 差异主要集中在 yield，其余任务基本没有变化。这推动我们改成可训练的 A/B adapter。

### 4.6 B 路线：Qwen embedding + shared A/B adapter

B 路线是当前最稳定的主结果。

输入：

```text
qwen_context: 2048 维
profile_embedding: 2048 维
```

结构：

```text
context branch:
qwen_context 2048
        ↓
Linear 2048 → 256
GELU + Dropout + LayerNorm
        ↓
shared

profile branch:
profile_embedding 2048
        ↓
Linear 2048 → 256
GELU + LayerNorm
        ↓
profile_state

gate:
[shared, profile_state]
        ↓
Linear 512 → 256
Sigmoid
        ↓
gate

fusion:
shared + gate × profile_state
        ↓
LayerNorm
        ↓
四个 Linear 256 → 2 heads
        ↓
silence / listener_onset / brief_response / yield 的 A/B 概率
```

训练：

- Qwen 冻结；
- 训练一个 shared adapter，同时服务四个 A/B 任务；
- 对适用 task 计算加权交叉熵，不适用 task 忽略；
- 加入轻量 profile 对照 margin，使正确 profile 对真实答案的 log-probability 高于 hidden 和 shuffled；
- 使用 contrastive shuffled；
- checkpoint 选择考虑 validation 上的 given-hidden 和 given-shuffled 差值。

最终结果：

| profile 条件 | silence | listener_onset | brief_response | yield | 平均 |
|---|---:|---:|---:|---:|---:|
| hidden | 0.6200 | 0.6850 | 0.5200 | 0.7800 | 0.6512 |
| given | 0.7080 | 0.7350 | 0.5533 | 0.8000 | 0.6991 |
| shuffled | 0.7000 | 0.7300 | 0.5467 | 0.7900 | 0.6917 |
| given-hidden | +0.0880 | +0.0500 | +0.0333 | +0.0200 | +0.0478 |
| given-shuffled | +0.0080 | +0.0050 | +0.0067 | +0.0100 | +0.0074 |

结论：B 路线满足当前六个要求：embedding、A/B 二分、Accuracy > 50%、测试集一致、given 高于 hidden 和 shuffled、接到 Qwen 模型。

### 4.7 A 路线：Qwen hidden space + Qwen A/B token head

A 路线更接近“让 Qwen 自己回答 A/B”。它不使用我们自己的 `Linear → 2` 输出头，而是最后使用 Qwen frozen `lm_head` 中 A/B token 的权重计算概率。

输入：

```text
qwen_context hidden vector: 2048 维
profile_embedding: 2048 维
```

结构：

```text
context_norm = LayerNorm(qwen_context)
profile_norm = LayerNorm(profile_embedding)

context_delta:
context_norm → Linear 2048 → 256 → GELU → Linear 256 → 2048

profile_delta:
profile_norm → Linear 2048 → 256 → GELU → Linear 256 → 2048

gate:
[context_norm, profile_norm] → Linear 4096 → 256 → GELU → Linear 256 → 2048 → Sigmoid

adjusted_hidden =
qwen_context
+ context_scale × context_delta
+ profile_scale × gate × profile_delta

logits =
adjusted_hidden · Qwen_lm_head[A/B token weights]
```

训练：

- Qwen 冻结；
- Qwen lm_head 冻结；
- 只训练 residual adapter；
- 每个 A/B task 单独训练；
- loss 包括交叉熵和 profile 对照 margin。

结果：

| profile 条件 | silence | listener_onset | brief_response | yield | 平均 |
|---|---:|---:|---:|---:|---:|
| hidden | 0.5040 | 0.5100 | 0.5600 | 0.6600 | 0.5585 |
| given | 0.6800 | 0.6800 | 0.6333 | 0.7400 | 0.6833 |
| shuffled | 0.6760 | 0.6750 | 0.6133 | 0.7300 | 0.6736 |
| given-hidden | +0.1760 | +0.1700 | +0.0733 | +0.0800 | +0.1248 |
| given-shuffled | +0.0040 | +0.0050 | +0.0200 | +0.0100 | +0.0098 |

结论：A 路线也满足六个要求。但当前 A 是 task-specific，结构比 B 复杂，因此更适合作为补充结果。

## 5. 当前完整结构

整体系统可以理解为三层：

```text
第一层：数据与状态
SBCSAE 音频 / 因果转写 / speaker profile / 后续动态状态 s_u

第二层：表示层
Qwen 编码音频+转写 → context embedding
Qwen 编码 profile → profile embedding

第三层：turn-taking judge
context embedding + profile embedding
        ↓
adapter / gate / Qwen lm_head
        ↓
四个 A/B 话轮判断
        ↓
继续听、backchannel、接话、打断等行为候选
```

当前已经完成的是第二层和第三层的离线验证。后续动态 `s_u` 可以接在 profile embedding 位置，作为一个随对话更新的用户状态。

## 6. 当前结论

目前可以形成三点结论：

1. 直接 prompt Qwen 做细粒度 turn-taking 五分类不稳定，容易输出塌缩。
2. 把 profile 做成 embedding，再训练轻量 adapter，明显更稳定。
3. 在 A/B 任务中，正确 profile 在同一测试集上高于 hidden 和 shuffled，说明 profile 有机会帮助模型做更个性化的话轮判断。

## 7. 当前贡献点

可以先整理成四个贡献：

1. 提出 profile-aware / state-aware turn-taking prediction；
2. 设计 hidden / given / shuffled paired profile control；
3. 构建 SBCSAE profile-turn-taking 数据处理流程；
4. 实现 Qwen-based profile embedding adapter，包括 B 路线 shared adapter 和 A 路线 Qwen lm_head adapter。

## 8. 下一步

下一步建议：

1. 建立人工金标测试集，至少 C/BC/T/I/NA 每类 50 条；
2. 锁定 B 路线配置，在新测试集上只跑一次正式评测；
3. 扩大 profile 数量和会话类型；
4. 引入动态用户状态 `s_u`，编码语速、停顿、语气、情绪和历史反应；
5. 将 A/B 预测映射到真实语音助手行为，例如继续听、backchannel、接话、合作性补全或避免打断。
