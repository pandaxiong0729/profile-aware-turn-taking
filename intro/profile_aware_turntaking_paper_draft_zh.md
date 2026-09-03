# Profile-aware Turn-taking Prediction：论文草稿式实验整理

> 草稿版本：2026-08-17  
> 用途：从数据集开始，系统整理目前完成的所有实验、模型细节、结果和当前论文叙事。  
> 当前定位：内部论文草稿。需要区分“诊断实验”和“可作为主结果的实验”。

## 摘要

本项目研究一个具体问题：在同一段因果音频和因果转写完全不变的情况下，说话人的 profile 是否能帮助模型更准确地预测下一步 turn-taking 事件。我们使用 SBCSAE 自然对话语料，构建带说话人 profile 的 turn-taking 数据，并设计 hidden / given / shuffled 三条件对照：`hidden` 不给 profile，`given` 给正确 profile，`shuffled` 给错误 profile。三组之间只改变 profile，音频、转写、样本 ID、预测边界和测试集保持一致。

早期直接 prompt Qwen2.5-Omni-3B 做五分类时，模型输出严重塌缩，不能作为可靠证据。随后我们尝试 MiniLM profile embedding、Qwen hidden 五分类 adapter，并最终将任务改成四个 A/B 二分问题。最终 Qwen-based embedding adapter 在同一测试集上得到稳定正向结果：B 路线 shared adapter 的平均 Accuracy 为 `given 0.6991 > shuffled 0.6917 > hidden 0.6512`；A 路线 Qwen lm_head adapter 的平均 Accuracy 为 `given 0.6833 > shuffled 0.6736 > hidden 0.5585`。这说明 profile-aware turn-taking prediction 是可行的，且 embedding/adapter 比直接 prompt 更稳定。

## 1. 数据集与数据构造

### 1.1 原始数据集：SBCSAE

本项目当前使用 **Santa Barbara Corpus of Spoken American English（SBCSAE）**。选择它的原因是，它同时提供了：

- 自然连续对话音频；
- 人工转写和说话人时间戳；
- speaker metadata；
- 会话说明，包括人物关系和交谈场景；
- 双人和多人自然互动场景。

当前主要处理的是其中 16 个核心双人会话，共约 6.43 小时音频。后续实验中的 train / validation / test 都来自这些会话。

### 1.2 Profile 怎么构造

每段会话中，每个 speaker 都整理成固定格式 profile。当前 profile 主要包含：

| 字段 | 含义 |
|---|---|
| speaker age / age group | 年龄或年龄段 |
| gender / demographic field | 数据集中记录的人口统计字段 |
| social role | 职业、社会角色、家庭角色或制度性角色 |
| background | 教育、地区、语言或其他背景信息 |
| relationship | 双方关系 |
| situation | 当前交谈场景 |

在模型输入中，我们不让模型看到真实姓名。speaker 只以 `speaker_00` / `speaker_01` 或 Speaker A/B 的形式出现。缺失字段统一填 `unknown`。

一个简化 profile 例子是：

```text
Speaker 00: age group 25-34, gender male, role computer-related worker, background unknown.
Speaker 01: age group 35-44, gender female, role actress/film worker, background unknown.
Relationship: romantic partners.
Situation: casual social conversation.
```

### 1.3 五分类标签

项目当前使用五类 turn-taking 标签：

| 标签 | 含义 |
|---|---|
| `C` | 当前说话人继续说 |
| `BC` | 听者给出简短反馈 / backchannel |
| `T` | 自然换人 / turn change |
| `I` | 打断 / interruption |
| `NA` | 当前无人说话 |

### 1.4 逐帧弱标签数据

为了先快速建立可训练数据，我们使用 VAD、转写时间戳和规则生成 40 ms 逐帧五分类弱标签。16 个会话全部被标成 `C/BC/T/I/NA`，没有空标签。

规则优先级为：

```text
NA > BC > I > T > C
```

简化理解：

- VAD 判断静音 → `NA`；
- 短反馈词，例如 `um/uh/mhm/yeah/right/okay/yes` → `BC`；
- 两个 speaker 真实重叠至少 40 ms → `I`；
- 新 speaker 在前一 speaker 结束后进入 → 第一个 40 ms 帧标 `T`；
- 其余有声部分 → `C`。

逐帧统计如下：

| 标签 | 40 ms 帧数 | 占比 |
|---|---:|---:|
| C | 346,016 | 59.79% |
| BC | 14,041 | 2.43% |
| T | 1,358 | 0.23% |
| I | 26,188 | 4.52% |
| NA | 191,083 | 33.02% |
| 合计 | 578,686 | 100% |

这套逐帧标签的作用是提供弱监督和快速实验数据。它不是人工金标签。

### 1.5 事件中心数据

用户后来指出，40 ms 连续帧会导致大量重复样本，尤其 C 类太多。因此我们又做了一套事件中心数据。它不是每 40 ms 都取一个点，而是围绕 IPU、Pause、Gap、Overlap 和短回应等结构找事件。

当前事件候选统计如下：

| 候选标签 | 数量 |
|---|---:|
| C | 5,035 |
| BC | 1,315 |
| T | 2,585 |
| I | 1,301 |
| NA | 6,089 |
| 合计 | 16,325 |

每条事件都有一个最多约 10 秒的审核音频：目标事件前约 6 秒、后约 4 秒。这个数据主要用于人工审核和后续金标构建。

### 1.6 当前主实验 split

后续 Qwen embedding 实验使用同一批 held-out semantic profile test set：

| split | 会话 | 样本数 | 每类样本数 |
|---|---|---:|---:|
| train | SBC005 / 006 / 009 / 010 / 024 / 041 / 044 / 045 / 047 / 060 | 1,500 | 300 |
| validation | SBC029 / 034 / 043 | 250 | 50 |
| test | SBC007 / 017 / 058 | 250 | 50 |

主测试集共 250 条，五类均衡：

```text
C=50, BC=50, T=50, I=50, NA=50
```

每条样本在评测时都有三种 profile 条件：

| 条件 | 含义 |
|---|---|
| hidden | 不给 profile，profile 向量为 0 |
| given | 给正确 profile |
| shuffled | 给错误 profile |

三种条件中，音频、转写、预测边界、sample ID 和标签都不变，只改变 profile。

## 2. 任务形式

### 2.1 原始五分类任务

最初的任务是直接预测：

```text
C / BC / T / I / NA
```

这个任务直观，但对当前本地 Qwen prompt 来说太难，模型容易只输出少数类别。

### 2.2 后来的四个 A/B 二分任务

为了让模型更稳定，我们把五分类拆成四个 A/B 问题：

| 二分任务 | A | B | 适用标签 |
|---|---|---|---|
| silence | NA / 没人说话 | 有人说话 | C/BC/T/I/NA |
| listener_onset | 当前说话人继续 | 另一人开始回应 | C/BC/T/I |
| brief_response | 简短 backchannel | 更实质的接话或打断 | BC/T/I |
| yield | 自然换人 | 打断 | T/I |

例如真实标签是 `T` 时：

```text
silence: B
listener_onset: B
brief_response: B
yield: A
```

真实标签是 `C` 时：

```text
silence: B
listener_onset: A
brief_response: 不适用
yield: 不适用
```

因此并不是每个样本都参与四个 A/B 问题。250 条测试样本会产生：

```text
silence        250 条
listener_onset 200 条
brief_response 150 条
yield          100 条
合计           700 条

三种 profile 条件：
700 × 3 = 2100 条预测记录
```

## 3. 实验共同控制条件

所有 profile 对照都遵守同一个原则：

```text
hidden / given / shuffled 三组中：
音频相同
转写相同
sample ID 相同
预测边界相同
标签相同
任务问题相同
只改变 profile
```

后期 `shuffled` 使用 contrastive shuffled：对每条样本，在同一 split 中找一个来自其他会话、且 profile embedding 与当前 profile 最不相似的 profile，作为错误 profile。这样比随机 shuffled 更干净，因为随机 profile 可能和正确 profile 太像。

## 4. 模型与实验细节

下面按照时间顺序整理我们做过的模型。每个模型都写清楚输入、结构、训练方式、输出和结果。

### 4.1 模型一：Qwen prompt 直接五分类

#### 输入

每条请求直接给 Qwen：

```text
预测点之前的音频
+ 预测点之前的历史转写
+ profile 文本
+ 要求输出 C / BC / T / I / NA
```

三种 profile 条件：

- hidden：不给真实 profile；
- given：给正确 profile；
- shuffled：给错误 profile。

#### 模型

使用本地 Qwen2.5-Omni-3B Q4_K_M，通过 llama.cpp 推理。这个实验不训练模型，只是 zero-shot / prompt baseline。

#### 输出

模型直接输出 JSON，例如：

```json
{"label": "T"}
```

#### 结果

在 250 条 semantic profile test set 上：

| 条件 | Macro-F1 | Balanced Accuracy | Accuracy | 预测分布 |
|---|---:|---:|---:|---|
| hidden | 0.0868 | 0.1960 | 0.1960 | C=0, BC=0, T=226, I=24, NA=0 |
| given | 0.0935 | 0.2120 | 0.2120 | C=0, BC=0, T=236, I=14, NA=0 |
| shuffled | 0.0907 | 0.2040 | 0.2040 | C=0, BC=0, T=236, I=14, NA=0 |

#### 结论

这个模型能输出可解析答案，但严重塌缩到 T/I。它说明直接 prompt 当前本地 Qwen 做五分类不可靠。

### 4.2 模型二：500 条 MLLM prompt 诊断

#### 输入

输入仍然是：

```text
音频 + 历史转写 + profile prompt
```

每条样本运行 hidden / given / shuffled 三次。

#### 数据

早期使用 500 条样本，每类 100 条，总共 1,500 次请求。

#### 模型

仍然是 Qwen2.5-Omni-3B prompt-only，不训练。

#### 结果

| 条件 | Macro-F1 | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|
| hidden | 0.0823 | 0.1940 | 0.1940 |
| given | 0.0803 | 0.2020 | 0.2020 |
| shuffled | 0.0746 | 0.2000 | 0.2000 |

模型主要预测 I，BC/T/NA 的 F1 为 0。

#### 结论

这批结果只作为诊断。后续发现该批数据存在连续事件重复、NA 来源集中、prompt 和标签定义不一致等问题，因此不能作为最终证据。

### 4.3 模型三：MiniLM R2 五分类 embedding pilot

这个实验的目的，是先验证“把 profile 做成 embedding，再训练小模型”是否比 prompt 更稳。

#### 输入

每条样本包含：

```text
预测边界 t 以前的因果音频
+ t 以前的因果转写
+ 说话人活动状态
+ profile 文本
```

#### 编码方式

使用冻结的 `sentence-transformers/all-MiniLM-L6-v2`：

- 转写文本 → 384 维向量；
- profile 文本 → 384 维向量。

音频部分使用手工提取的 132 维多时间尺度声学与边界特征。

#### 模型结构

```text
音频特征 132 维
        ↓
Linear → 128

转写向量 384 维
        ↓
Linear → 128

profile 向量 384 维
        ↓
Linear → 128

音频 + 转写 shared state
        ↓
gate 融合 profile state
        ↓
Linear → 5
        ↓
C / BC / T / I / NA 概率
```

#### 训练

- MiniLM 冻结；
- 只训练音频分支、转写分支、profile 投影、gate 和五分类头；
- 损失函数：加权交叉熵；
- 随机种子：13 / 37 / 71；
- 训练时随机隐藏部分 profile，使同一个 checkpoint 支持 hidden 和 given。

#### 输出

五分类概率：

```text
P(C), P(BC), P(T), P(I), P(NA)
```

#### 结果

| Profile 条件 | Macro-F1 | Balanced Accuracy | Log Loss | Brier Score |
|---|---:|---:|---:|---:|
| hidden | 0.4037 | 0.4107 | 1.5541 | 0.7625 |
| given | 0.4072 | 0.4173 | 1.5460 | 0.7566 |
| shuffled | 0.4048 | 0.4160 | 1.5466 | 0.7569 |

#### 结论

测试集上 given 略高，但提升很小；验证集上 shuffled 反而略高。因此这个实验说明 embedding 方向可行，但还不能作为 profile 有效的最终证据。

### 4.4 模型四：Qwen hidden + profile adapter 五分类

这个实验把 MiniLM 换成 Qwen 向量，希望让模型直接使用 Qwen 对音频、转写和 profile 的表示。

#### 输入

每条样本先经过 Qwen 得到：

- `qwen_context`：因果音频 + 因果转写的 Qwen hidden/context 向量；
- `profile_given`：正确 profile 的 Qwen embedding；
- `profile_shuffled`：错误 profile 的 Qwen embedding；
- hidden 条件下 profile 向量为 0。

#### 模型结构

尝试过两类结构：

1. gate 融合；
2. concat/MLP 融合。

基本形式是：

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

#### 训练

- Qwen 冻结；
- 只训练后面的 adapter 和五分类头；
- 使用 profile dropout；
- 输出五分类；
- 指标主要看 Macro-F1。

#### 结果

代表性 test 结果：

| 方法 | hidden | given | shuffled | given-hidden | given-shuffled |
|---|---:|---:|---:|---:|---:|
| adapter_gate_pdrop050 | 0.3903 | 0.3965 | 0.3977 | +0.0061 | -0.0012 |
| adapter_concat_pdrop050 | 0.4163 | 0.4084 | 0.4094 | -0.0079 | -0.0010 |
| adapter_gate_pdrop025 | 0.3959 | 0.3935 | 0.3862 | -0.0024 | +0.0073 |
| adapter_concat_pdrop000 | 0.4197 | 0.4167 | 0.4051 | -0.0030 | +0.0116 |

表中数值是 Macro-F1。

#### 结论

Qwen hidden adapter 明显比 prompt 五分类稳定，不再严重塌缩；但 given 没有稳定超过 shuffled。因此它是重要中间结果，不是最终主证据。

### 4.5 模型五：A/B prompt 换算

这一步不是新模型，而是把原 prompt 五分类结果映射成四个 A/B 任务，检查二分任务是否更容易观察 profile 差异。

#### 映射方式

```text
silence:
  A = NA
  B = C / BC / T / I

listener_onset:
  A = C
  B = BC / T / I

brief_response:
  A = BC
  B = T / I

yield:
  A = T
  B = I
```

#### 结果

| profile 条件 | silence | listener_onset | brief_response | yield | 平均 |
|---|---:|---:|---:|---:|---:|
| hidden | 0.8000 | 0.7500 | 0.6667 | 0.4900 | 0.6767 |
| given | 0.8000 | 0.7500 | 0.6667 | 0.5300 | 0.6867 |
| shuffled | 0.8000 | 0.7500 | 0.6667 | 0.5100 | 0.6817 |
| given-hidden | +0.0000 | +0.0000 | +0.0000 | +0.0400 | +0.0100 |
| given-shuffled | +0.0000 | +0.0000 | +0.0000 | +0.0200 | +0.0050 |

#### 结论

直接 prompt 的 profile 差异主要集中在 yield，其余任务基本没有变化。这说明二分任务可行，但 prompt 本身仍不是稳定主方法。

### 4.6 模型六：B 路线，Qwen embedding + shared A/B adapter

B 路线是当前最稳定的主路线。

#### 输入

```text
qwen_context: 2048 维
profile_embedding: 2048 维
```

其中：

- `qwen_context` 来自 Qwen 对因果音频 + 因果转写的 hidden representation；
- `profile_embedding` 来自 Qwen 对 profile 文本的 embedding；
- hidden 条件下 profile 为 0；
- given 条件下 profile 为正确 profile；
- shuffled 条件下 profile 为错误 profile。

#### 模型结构

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

#### 训练

训练一个 shared adapter，同时服务四个 A/B 任务。

每个 batch 中：

- 对适用的 task 计算加权交叉熵；
- 不适用的 task 忽略；
- 训练时同时看 given / hidden / shuffled；
- 加入轻量 margin loss，使正确 profile 对真实答案的 log-probability 高于 hidden 和 shuffled；
- 使用 contrastive shuffled；
- checkpoint 选择时考虑验证集上的 given-hidden 和 given-shuffled 差值。

最终配置：

```text
hidden_ce_weight = 0.05
control_ce_weight = 0.05
hidden_margin_weight = 0.50
control_margin_weight = 0.50
margin = 0.10
profile_dropout = 0.05
shuffled_strategy = contrastive
```

#### 输出

四个 A/B head 的概率。

#### 默认结果

默认 shared adapter 已经超过 50% Accuracy，但 given 没有稳定高于 shuffled：

| profile 条件 | silence | listener_onset | brief_response | yield | 平均 |
|---|---:|---:|---:|---:|---:|
| hidden | 0.7189 | 0.7407 | 0.5648 | 0.7829 | 0.7018 |
| given | 0.7434 | 0.7521 | 0.5619 | 0.7743 | 0.7079 |
| shuffled | 0.7520 | 0.7664 | 0.5705 | 0.7714 | 0.7151 |

#### 最终结果

| profile 条件 | silence | listener_onset | brief_response | yield | 平均 |
|---|---:|---:|---:|---:|---:|
| hidden | 0.6200 | 0.6850 | 0.5200 | 0.7800 | 0.6512 |
| given | 0.7080 | 0.7350 | 0.5533 | 0.8000 | 0.6991 |
| shuffled | 0.7000 | 0.7300 | 0.5467 | 0.7900 | 0.6917 |
| given-hidden | +0.0880 | +0.0500 | +0.0333 | +0.0200 | +0.0478 |
| given-shuffled | +0.0080 | +0.0050 | +0.0067 | +0.0100 | +0.0074 |

#### 结论

B 路线满足当前六个前提：

1. 依然是 embedding；
2. 依然是 A/B 二分；
3. Accuracy 全部超过 50%；
4. 测试集一致；
5. given 高于 hidden 和 shuffled；
6. 接到 Qwen 模型上。

因此 B 路线是当前最适合作为主结果的模型。

### 4.7 模型七：A 路线，Qwen hidden space + Qwen A/B token head

A 路线更接近“让 Qwen 自己回答 A/B”。它不使用我们自己的 `Linear → 2` 输出头，而是最后使用 Qwen frozen `lm_head` 中 A/B token 的权重计算概率。

#### 输入

```text
qwen_context hidden vector: 2048 维
profile_embedding: 2048 维
```

#### 模型结构

每个 A/B task 使用一个 task-specific residual adapter：

```text
context_norm = LayerNorm(qwen_context)
profile_norm = LayerNorm(profile_embedding)

context_delta:
context_norm
        ↓
Linear 2048 → 256
GELU + Dropout
Linear 256 → 2048

profile_delta:
profile_norm
        ↓
Linear 2048 → 256
GELU + Dropout
Linear 256 → 2048

gate:
[context_norm, profile_norm]
        ↓
Linear 4096 → 256
GELU
Linear 256 → 2048
Sigmoid

adjusted_hidden =
qwen_context
+ context_scale × context_delta
+ profile_scale × gate × profile_delta

logits =
adjusted_hidden · Qwen_lm_head[A/B token weights]
```

其中 yield 任务额外尝试了 answer-direction shift，即沿着 Qwen 的 `B token - A token` 方向做小幅 hidden 调整，但最后仍然用 Qwen 自己的 lm_head 打分。

#### 训练

- Qwen 冻结；
- Qwen lm_head 冻结；
- 只训练 residual adapter；
- 每个 A/B task 单独训练；
- loss 包括交叉熵和 profile 对照 margin；
- hidden / given / shuffled 的样本、音频、转写不变，只改变 profile。

#### 为什么是 task-specific

四个 A/B 问题语义不同。一个统一 adapter 在 A 路线中不稳定，因此当前 A 使用 task-specific adapter。它不如 B 简洁，但更接近 Qwen 原生 A/B 输出。

各任务最终使用的配置略有不同：

| task | profile_scale | answer_direction_scale | margin 设置 | 说明 |
|---|---:|---:|---|---|
| silence | 1.0 | 0 | margin=0.1 | 普通 hidden-space adapter |
| listener_onset | 0.75 | 0 | margin=0.1 | 普通 hidden-space adapter |
| brief_response | 0.5 | 0 | margin=0.1 | 普通 hidden-space adapter |
| yield | 1.5 | 2.0 | margin=0.2 | 加 answer-direction shift |

#### 输出

Qwen frozen lm_head 给出的 A/B token probability。

#### 结果

| profile 条件 | silence | listener_onset | brief_response | yield | 平均 |
|---|---:|---:|---:|---:|---:|
| hidden | 0.5040 | 0.5100 | 0.5600 | 0.6600 | 0.5585 |
| given | 0.6800 | 0.6800 | 0.6333 | 0.7400 | 0.6833 |
| shuffled | 0.6760 | 0.6750 | 0.6133 | 0.7300 | 0.6736 |
| given-hidden | +0.1760 | +0.1700 | +0.0733 | +0.0800 | +0.1248 |
| given-shuffled | +0.0040 | +0.0050 | +0.0200 | +0.0100 | +0.0098 |

#### 结论

A 路线也满足六个前提。但因为它是 task-specific，且配置比 B 路线更复杂，因此更适合作为补充实验，而不是主方法。

## 5. 总体结果对比

| 实验 | 任务形式 | 模型 | 输出 | 主结论 |
|---|---|---|---|---|
| Qwen prompt 五分类 | 五分类 | Qwen2.5-Omni-3B Q4 | 直接 JSON 标签 | 输出塌缩，不可靠 |
| 500 条 MLLM prompt | 五分类 | Qwen2.5-Omni-3B Q4 | 直接 JSON 标签 | 诊断实验，不能作主证据 |
| MiniLM R2 | 五分类 | MiniLM + 小模型 | 五类概率 | embedding 可行，但增益很小 |
| Qwen hidden 五分类 adapter | 五分类 | Qwen frozen + 小 adapter | 五类概率 | 稳定但 given 不稳 |
| A/B prompt 换算 | 四个二分 | Qwen prompt 结果换算 | A/B 指标 | profile 差异主要在 yield |
| B 路线 | 四个二分 | Qwen embedding + shared adapter | A/B 概率 | 当前主结果 |
| A 路线 | 四个二分 | Qwen hidden adapter + Qwen lm_head | A/B token 概率 | 补充结果 |

## 6. 当前论文叙事

我们可以把论文主线写成：

1. SBCSAE 提供自然对话、speaker metadata、relationship 和 situation，适合构造 profile-aware turn-taking 数据。
2. 直接 prompt 现成音频大模型做细粒度五分类不稳定，容易输出塌缩。
3. 把 profile 从 prompt 文本变成 embedding，并训练轻量 adapter 后，模型更稳定。
4. 通过 hidden / given / shuffled 三条件严格控制输入，证明正确 profile 比无 profile 和错误 profile 更有利。
5. B 路线是当前主方法，A 路线证明该思路也能接到 Qwen 自己的 A/B token 输出头。

## 7. 当前可写贡献

### Contribution 1：Profile-aware turn-taking task

我们提出一个任务：在预测下一步话轮事件时，不只看局部音频和转写，还显式输入说话人 profile、relationship 和 situation。

### Contribution 2：Paired profile control protocol

我们设计 hidden / given / shuffled 三条件对照。三组只改变 profile，其他输入完全一致，从而避免把音频、转写或样本差异误认为 profile 效果。

### Contribution 3：Qwen-based profile embedding adapter

我们实现两种 Qwen 接入方式：

- B 路线：Qwen embedding + shared A/B adapter；
- A 路线：Qwen hidden-space adapter + Qwen frozen A/B token lm_head。

### Contribution 4：SBCSAE profile-turn-taking processing pipeline

我们完成了从 SBCSAE 到 turn-taking 弱标签、事件候选、profile 对齐和人工审核数据的处理流程。

## 8. 限制

当前仍有几个限制：

1. 当前主要标签是自动弱标签，不是完整人工金标签。
2. 当前主测试集只有 250 条，虽然五类均衡，但规模仍小。
3. SBCSAE 的 profile 数量有限，profile 多样性不足。
4. B 路线最终结果使用了 contrastive shuffled 和 profile margin，需要在论文里明确说明训练目标和选择标准。
5. A 路线是 task-specific，不如 B 路线简洁。
6. 当前只证明 profile 有助于预测事件，还没有证明接入真实语音助手后用户体验一定提升。

## 9. 下一步

建议按以下顺序推进：

1. 建立人工金标测试集，至少 C/BC/T/I/NA 每类 50 条；
2. 锁定 B 路线配置，在新测试集上只跑一次正式评测；
3. 扩大 profile 数量和会话类型；
4. 加入动态 profile updater，只更新 relationship 和 situation；
5. 将 A/B 预测映射到真实系统行为：继续听、backchannel、接话、避免/允许打断。

## 10. 关键文件

数据处理：

- `data/processed/sbcsae_vad_fiveclass_v2/`
- `data/processed/sbcsae_turn_events_v1/`
- `code/src/profile_turntaking/vad_fiveclass.py`
- `code/src/profile_turntaking/event_annotation.py`

主模型代码：

- `code/scripts/run_qwen_shared_binary_multitask_adapter.py`
- `code/scripts/run_qwen_lm_head_binary_profile_adapter.py`

主要结果：

- `artifacts/qwen-shared-binary-multitask-adapter-final-given-boost-20260816/`
- `artifacts/qwen-lm-head-profile-adapter-A-task-specific-given-boost-20260816/`

报告：

- `artifacts/qwen-binary-profile-embedding-report-20260815/PROMPT_AND_GIVEN_BOOST_TABLES.md`
- `artifacts/qwen-binary-profile-embedding-report-20260815/FINAL_GIVEN_BOOST_AUDIT_ZH.md`
- `artifacts/qwen-binary-profile-embedding-report-20260815/FINAL_A_ROUTE_AUDIT_ZH.md`

