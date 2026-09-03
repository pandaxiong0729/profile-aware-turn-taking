# Profile 表征实验方案：自然语言 Prompt 与 Embedding 对比

日期：2026-08-10  
任务：使用相同的因果音频、相同的部分转写和相同的五分类标签，只改变 profile 的表示方式，判断哪种方式更适合话轮预测，以及动态 profile 应如何接入。

## 1. 一句话结论

可以使用 embedding 表征 profile，而且相关论文已经证明：把说话人、听者、关系或历史行为编码成向量，再与音频和文本特征融合，能够改善 backchannel 或 turn-taking 预测。

本项目最合理的正式实验不是把“现成 Qwen + 文字 Prompt”和“另一个训练模型 + embedding”直接比较，而是建立一个共享的音频—转写模型，在完全相同的输入、训练数据和参数量下，只替换 profile 编码器：

1. 不提供 profile；
2. 固定模板自然语言；
3. 自然语言语义 embedding；
4. 结构化 field embedding；
5. 结构化与自然语言混合 embedding。

Qwen2.5-Omni 的 Prompt 实验继续保留，但定位为“现成通用模型能否使用 profile”的低成本外部基线，不把它当作 Prompt 与 embedding 的严格公平对比。

---

## 2. 我们到底要证明什么

主问题是：

> 在下一话轮事件预测中，profile 是否提供了音频和部分转写之外的有效信息；如果有效，profile 用自然语言表示还是用 embedding 表示更好？

五分类保持不变：

- `C`：当前说话人继续；
- `BC`：听者开始简短回应，但不夺取话轮；
- `T`：当前说话人让出话轮，另一人自然接话；
- `I`：另一人在当前说话人尚未让出话轮时开始实质发言；
- `NA`：目标位置没有人说话。

每条样本的基本输入必须始终是：

```text
截止预测边界的因果音频
+ 与该音频完全匹配的因果部分转写
+ 音频结束时的说话人活动状态
+ 当前实验条件下的 profile
→ 预测边界后固定位置开始的 C / BC / T / I / NA
```

任何条件都不能看到未来音频、未来转写、参考标签或标注依据。

---

## 3. 论文调查：别人具体是怎么做的

### 3.1 与本项目最接近的工作

| 论文 | Profile/人物信息怎样表示 | 怎样与音频、文本融合 | 论文结果与本项目的关系 |
|---|---|---|---|
| Fukunaga et al., Interspeech 2025, *Backchannel prediction for natural spoken dialog systems using general speaker and listener information* | 说话人性别、年龄；听者性别、年龄；双方关系。每项先变成 128 维 embedding，再拼接并投影成 256 维人物特征 | BERT 转写向量、HuBERT 音频向量和人物向量拼接后分类 | 三分类中无人物信息为 Macro-F1 65.2，通用人物信息为 67.2；说明年龄、性别和关系 embedding 确实可以帮助 BC 预测。表格中 listener ID 更高，但 ID 难以推广到新说话人。 |
| Park et al., Interspeech 2024, *Backchannel prediction, based on who, when and what* | 不用身份文字，而是从历史对话计算 16 个可解释的行为统计，如平均话轮长度、词数、填充词、BC 频率等，再通过角色专属线性层得到 person embedding | person、对话进度、话题 embedding 与 CABP 的声学/语言特征拼接 | 基线 Macro-F1 50.1；person embedding 53.5；加对话进度后 54.2。这直接支持我们第二阶段用历史行为动态更新 profile。 |
| Ortega et al., 2023, *Listener-aware Backchannel Predictor*；*Modeling Speaker-Listener Interaction for Backchannel Prediction* | 为 speaker/listener 学习 ID embedding，并用求和、双线性或 neural tensor network 建模双方交互 | 与声学和词汇特征融合后预测 BC | 证明“分别编码当前说话人和听者，再显式建模关系”有效；但 ID embedding 主要记忆已见人物，只适合作为上限对照，不适合作为我们的主要方法。 |
| Wang et al., 2024, *Turn-taking and Backchannel Prediction with Acoustic and Large Language Model Fusion* | 不是 profile 论文，但清楚展示 HuBERT 音频向量与 LLM 部分转写向量的晚期融合 | HuBERT 均值池化后投影，与 GPT-2/RedPajama 的文本向量拼接；把 C/BC/T 拆成多个二分类任务 | 说明我们可以先建立稳定的音频＋因果转写共享骨干，再只替换 profile 支路。 |
| Zhang et al., ACL 2018, *Personalizing Dialogue Agents: I have a dog, do you have pets too?* | 一种方法把 persona 句子直接放在输入前；另一种方法把每条 persona 单独编码成 memory embedding，并由对话上下文注意力读取 | Prompt 式串接与 profile memory 在同一任务中比较 | 说明自然语言 persona 不一定只能直接拼进 Prompt，也可以先编码成独立向量；改写后的 persona 更难，提示我们要做模板改写鲁棒性测试。 |
| Li et al., ACL 2016, *A Persona-Based Neural Conversation Model* | 每个说话人一个可学习向量，或同时学习 speaker 与 addressee 向量 | 每个解码时间步注入人物向量 | 是 persona embedding 的早期明确实现；同样存在新人物没有训练过 ID 的泛化问题。 |
| Fu et al., EMNLP 2025, *PACHAT: Persona-Aware Speech Assistant for Multi-party Dialogue* | 用户 profile 以文字条件进入冻结 Llama；Whisper 编码语义，pyannote 编码说话人，Q-Former/线性层做对齐，仅训练 LoRA 和对齐模块 | 音频语义、说话人 embedding、profile 与历史一起构成 LLM 输入 | 支持“音频＋说话人 embedding＋文字 profile”的混合方案，但它是个性化语音对话生成，不是五分类，也没有做我们需要的严格 Prompt-vs-embedding 消融。 |

### 3.2 从论文中能得出的可靠结论

1. **结构化 embedding 可行。** 年龄、性别、关系等字段可以分别学习 embedding，再投影到统一维度。
2. **动态行为 embedding 可行。** 话轮长度、BC 频率、占用话语权比例等历史统计可以组成动态人物向量。
3. **人物双方要分别表示。** 当前说话人和听者对同一事件的作用不同，不能只把所有字段无序拼在一起。
4. **ID embedding 不是我们的主方案。** 它容易记住训练人物，在 speaker-disjoint 测试中无法处理新人物。
5. **自然语言 Prompt 与 embedding 各有优点。** Prompt 可直接利用预训练语言知识；结构化 embedding 更短、更快、字段清晰，但需要训练数据学习每个字段的意义。

### 3.3 目前尚未发现的直接工作

在本次检索到的直接相关工作中，没有发现一篇论文同时满足以下四点：

- 同一段因果音频和同一份因果部分转写；
- 同一共享模型；
- 同一套五类 `C / BC / T / I / NA`；
- 只把自然语言 profile 换成结构化 embedding，并同时做正确、隐藏、错误 profile 对照。

因此，这个**严格受控的 profile 表征比较**可以成为实验贡献的一部分。但它只是证据设计；论文的主要方法仍应是“可动态更新的 profile 表征和融合机制”。

---

## 4. 当前项目已有实现能不能直接用

当前代码已经有一个结构化 profile 原型：

- `code/src/profile_turntaking/model.py` 中的 `StructuredProfileEncoder`；
- 10 个字段各使用一个 `Embedding(512, 16)`；
- 10 个向量拼接后经过 MLP；
- profile 通过 adapter 和一个可学习 gate 加到音频—转写上下文上；
- `training.py` 已有 profile dropout，可用同一 checkpoint 测 `given/hidden`。

但是这个版本还不适合直接作为正式论文结果：

1. `features.py` 把字段字符串哈希到 512 个桶，同一字段的不同值可能碰撞；
2. `background` 整段字符串被当作单一类别，几乎一段会话一个值，难以推广；
3. `speaker_A/B` 是语料编号，不一定等于当前说话人/听者角色；
4. gate 只是全局标量，不能针对每条样本决定是否使用 profile；
5. 音频默认还有统计特征原型，正式实验需要稳定的预训练音频 encoder；
6. 尚未实现与其完全公平的自然语言 profile 编码支路。

### 当前数据中的 profile 规模

16 个会话中：

| 字段 | 当前取值情况 | 对 embedding 的影响 |
|---|---:|---|
| relationship | 5 类 | 适合显式类别 embedding |
| situation | 5 类 | 适合显式类别 embedding，也适合第二阶段动态更新 |
| age_group | 少量离散区间 | 适合类别 embedding |
| gender | 2 类加 unknown | 可做类别 embedding，同时要报告敏感字段消融 |
| social_role | A/B 两侧分别约 13 个不同原始值 | 样本稀疏，应先归一成较粗类别，同时保留文字支路 |
| background | A/B 两侧分别约 15 个不同原始字符串 | 不应整句作为类别；应拆为教育、地区等字段，或交给文字 embedding |

结论：**不能继续把完整原始字符串直接哈希后当成正式结构化 embedding。**

---

## 5. 正式实验的总体结构

整个实验分为两层，防止结论混淆。

### 层 A：现成 MLLM Prompt 基线

目的：回答“一个没有为本任务训练的通用音频大模型，能不能只靠提示词使用 profile”。

- 模型：当前 Qwen2.5-Omni-3B；
- 输入：因果音频＋匹配的因果转写＋固定模板 profile；
- 条件：`hidden / given / shuffled`；
- 不训练；
- 结果单独汇报为 zero-shot system baseline。

这层不能与 embedding adapter 做纯粹的表示方法胜负判断，因为模型、参数量和训练方式都不同。

### 层 B：同一模型下的受控表征对比

目的：只回答“同样的信息，用哪一种 profile 表征更好”。

所有实验共享：

- 完全相同的样本 ID；
- 完全相同的因果音频；
- 完全相同的因果部分转写；
- 完全相同的预测边界；
- 完全相同的五分类标签；
- 完全相同的音频 encoder、转写 encoder、分类器和训练参数；
- 近似相同的 profile 支路输出维度和可训练参数量。

唯一变化是 profile 如何编码。

---

## 6. 要比较的五种条件

### R0：Hidden

所有字段都替换为 `unknown`。这是无 profile 基线。

### R1：固定模板自然语言（Prompt-style token condition）

把 profile 用唯一固定模板写成自然语言，例如：

```text
The current speaker is an adult healthcare worker.
The listener is an adult student.
They have a professional-client relationship.
The situation is a healthcare consultation.
```

这段文字作为独立的 `profile_text` 段进入共享文本 encoder。这里叫 Prompt-style，是因为使用自然语言 token；它与当前“直接询问 Qwen”的 zero-shot Prompt 不是同一个实验。

### R2：自然语言语义 embedding

仍然使用与 R1 完全相同的自然语言文本，但不与转写直接串接：

1. 用冻结的句向量模型编码 profile；
2. 得到一个固定长度向量；
3. 用线性层投影为 128 维；
4. 与音频—转写上下文融合。

首选可复现模型：`sentence-transformers/all-MiniLM-L6-v2`，输出 384 维句向量，参数较小且可提前缓存。

这一条件回答：**保留自然语言语义，但不让长 profile 占用 Prompt token，效果如何？**

### R3：结构化 field embedding

每个字段使用明确词表，不再哈希：

```text
age_current_speaker  → 16d
gender_current_speaker → 16d
role_current_speaker → 16d
age_listener → 16d
gender_listener → 16d
role_listener → 16d
relationship → 16d
situation → 16d
```

拼接后经过两层 MLP，统一输出 128 维。

这一条件回答：**短、明确、可训练的字段向量能否比自然语言更稳定？**

### R4：Hybrid，推荐的最终方法候选

- 关系、场景、年龄段、性别等稳定离散字段：使用 field embedding；
- social role、background 等稀疏或自由文本字段：使用文本 embedding；
- 两部分拼接后投影为 128 维。

这一条件结合两边优点，但必须在 R1/R2/R3 之后运行，否则无法判断提升来自哪一部分。

### 可选上限：R-ID

为每名 speaker 学习 ID embedding，只作为“记住已见人物最多能达到什么程度”的上限。主结论不能依赖它，测试集仍按说话人隔离；未见人物统一使用 `unknown speaker`。

---

## 7. Profile 字段如何整理

### 7.1 统一角色方向

模型输入时不再只写语料中的 `speaker_A / speaker_B`，而是按预测边界转换成：

- `current_speaker`：边界前正在持有话轮的人；
- `listener`：另一位参与者。

如果边界处两人都说话，则保留双方身份，并增加 `both_active=1`。这样同一种字段在所有事件中语义一致。

### 7.2 显式词表代替哈希桶

每个字段使用独立词表：

- `0 = unknown`；
- 训练集出现的正常类别从 1 开始编号；
- 验证或测试中的新值映射为 `unknown`；
- 词表只根据训练集生成并保存到 checkpoint。

### 7.3 social_role 归一化

原始职业先保留，再映射为粗类别，例如：

```text
student
healthcare
service
business_finance
technical
creative_media
homemaker_family
other
unknown
```

R3 使用粗类别，R1/R2/R4 的文字部分可以保留清洗后的具体职业。

### 7.4 background 拆分

不要把如下整句当作一个类别：

```text
education=BA; home_state=CA; current_state=CA; ethnicity=white
```

至少拆成：

- `education_level`；
- `home_region`；
- `current_region`；
- `other_background_text`。

种族、性别等敏感信息可以按研究问题保留，但必须增加“移除该字段”的消融，避免把数据集偏差误写为普遍规律。

---

## 8. 共享模型怎么搭

### 8.1 输入编码

建议第一版使用较轻、容易在云端复现的结构：

```text
因果音频 ── HuBERT-base（先冻结） ── mean pooling ── 256d ┐
                                                        ├─ context fusion ── hc
因果转写 ── DistilRoBERTa（先冻结） ── last/mean pool ─ 256d ┘

profile ── R0/R1/R2/R3/R4 中的一种编码器 ── 128d ── zp

[hc; zp] ── 样本级 gate + residual adapter ── 五分类器
```

音频统一为 16 kHz。profile 文本与对话转写必须分段编码，防止模型把 profile 误当成对话内容。

### 8.2 融合公式

使用样本级 gate，而不是当前的一个全局标量：

```text
g = sigmoid(Wg · [hc ; zp])
h = LayerNorm(hc + g ⊙ Wp(zp))
logits = Wc(h)
```

含义：每条样本都能自行决定 profile 应该影响多少；如果某个事件主要由清楚的声学边界决定，gate 可以较小；如果 BC/T/I 需要人物关系帮助，gate 可以较大。

### 8.3 第一轮只使用单一五分类 loss

为隔离“profile 表征方式”这一变量，第一轮统一使用加权五分类交叉熵：

```text
L = WeightedCrossEntropy(C, BC, T, I, NA)
```

不要在同一张主表中同时修改层次化标签头、loss 或音频模型。找到最佳 profile 表征后，再单独测试层次化分类头。

---

## 9. 每条训练和评测样本具体是什么

建议正式的表征对比统一到当前事件中心协议，不与旧的 40 ms 连续帧实验混在同一张表中。

每条样本包含：

```json
{
  "sample_id": "SBC005-event-...",
  "conversation_id": "SBC005",
  "audio_path": "相对路径",
  "audio_start_s": 7.170,
  "audio_end_s": 13.070,
  "prediction_boundary_s": 13.070,
  "forecast_offset_ms": 100,
  "evaluation_window_ms": 500,
  "causal_transcript": "预测边界前的匹配转写",
  "boundary_speaker_state": "speaker_00 audible",
  "profile": {},
  "label": "T",
  "split": "train"
}
```

模型看到的是边界前的音频和转写；标签描述的是边界后 100 ms 处开始、并在固定 500 ms 窗口内判定的事件。具体窗口在正式生成清单时一次锁定，所有 R0–R4 完全相同。

实际提示词中不要只保留抽象字母 `t`。例如一段模型音频长 5.900 秒，应明确写：

```text
The provided audio ends at 5.900 seconds.
Predict the interaction state at 6.000 seconds, 100 ms after the audio ends.
```

原始长会话中的绝对时间保存在 JSONL 中供审计，不必告诉模型。

---

## 10. hidden / given / shuffled 怎样保证公平

对同一个 checkpoint、同一个样本产生三个请求：

- `hidden`：所有 profile 字段为 unknown；
- `given`：当前样本的正确 profile；
- `shuffled`：使用其他会话的 profile，并保证确实至少有关键字段不同。

三个请求中只有 profile 内容允许变化。以下内容必须逐字节相同：

- 音频文件和音频 SHA-256；
- 因果转写和转写 SHA-256；
- sample ID；
- 预测边界和预测窗口；
- 任务说明、标签定义和解码设置。

`shuffled` 第一阶段只在评测时使用。不要把错误 profile 与原标签一起大量训练，否则模型最容易学会完全忽略 profile。

---

## 11. 训练步骤

### Step 0：锁定数据协议

1. 固定事件中心样本清单；
2. 按 speaker-connected group 划分 train/val/test；
3. 检查五类数量；
4. 检查所有转写时间不超过预测边界；
5. 给音频、转写和 profile snapshot 保存 SHA-256；
6. 测试集在调参期间不打开。

完成物：一个不可修改的 manifest 和 audit 报告。

### Step 1：建立无 profile 的有效基线

先训练 R0 Hidden。只有满足下面条件，才能开始解释 profile：

- 输出至少覆盖多种标签，没有全部预测成同一类；
- Macro-F1 高于相应随机/多数类基线；
- 音频扰动控制表明预测对音频至少有一定增量敏感性；
- 因果输入审计全部通过。

### Step 2：训练 R1、R2、R3

- 从 R0 的同一个共享骨干初始化；
- 使用相同 batch、学习率、epoch、early stopping 和 class weight；
- profile 输出都固定为 128 维；
- profile dropout 使用同一个值，例如当前的 0.5；
- 至少运行 3 个随机种子，最好 5 个；
- 只在验证集选择 checkpoint。

### Step 3：选择表示方法

先比较 R1/R2/R3：

- 若 R3 最好：说明字段明确的结构化信息更适合当前小数据；
- 若 R2 最好：说明预训练语义能够缓解稀疏 profile；
- 若 R1 最好：说明模型需要 profile token 与转写直接交互；
- 若三者都不超过 Hidden：说明当前 profile 对这套标签没有可测增益，不能先假设需要更复杂 adapter。

### Step 4：训练 R4 Hybrid

只有在 R2/R3 至少一项表现出有效增益后，再组合成 R4。否则 Hybrid 的结果很难解释。

### Step 5：固定后只运行一次测试集

测试时对每个 checkpoint 同时跑 `hidden / given / shuffled`，不再根据测试结果修改模板、字段或超参数。

SBCSAE 当前只有 16 个核心双人会话，单个固定测试集只有 3 个会话，容易受会话内容影响。因此主表保留锁定测试集，另外增加 speaker-grouped 交叉验证作为稳健性结果；每一折都重新从训练折建立 profile 词表，不能让验证折或测试折决定类别编号和字段清洗规则。

---

## 12. 评测表应该怎样写

### 表 A：主要五分类结果

| 表征 | Profile 条件 | Macro-F1 | Balanced Acc. | C F1 | BC F1 | T F1 | I F1 | NA F1 | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 Hidden | hidden |  |  |  |  |  |  |  |  |  |
| R1 自然语言 token | hidden |  |  |  |  |  |  |  |  |  |
| R1 自然语言 token | given |  |  |  |  |  |  |  |  |  |
| R1 自然语言 token | shuffled |  |  |  |  |  |  |  |  |  |
| R2 文本 embedding | hidden/given/shuffled |  |  |  |  |  |  |  |  |  |
| R3 field embedding | hidden/given/shuffled |  |  |  |  |  |  |  |  |  |
| R4 Hybrid | hidden/given/shuffled |  |  |  |  |  |  |  |  |  |

### 表 B：Profile 是否真正有用

| 表征 | Given − Hidden Macro-F1 | Given − Shuffled Macro-F1 | 95% CI | 3/5 seeds 是否同方向 | 结论 |
|---|---:|---:|---|---|---|
| R1 |  |  |  |  |  |
| R2 |  |  |  |  |  |
| R3 |  |  |  |  |  |
| R4 |  |  |  |  |  |

必须同时看两项：

- `given > hidden`：profile 比没有 profile 好；
- `given > shuffled`：模型使用的是正确 profile，不只是因为多了一段信息。

### 表 C：鲁棒性

| 表征 | 原模板 F1 | 等义改写 F1 | 关系字段改错 | 场景字段改错 | 交换双方 profile | 未见类别 |
|---|---:|---:|---:|---:|---:|---:|
| R1 |  |  |  |  |  |  |
| R2 |  |  |  |  |  |  |
| R3 |  |  |  |  |  |  |
| R4 |  |  |  |  |  |  |

预期：固定模板 R1 可能对文字措辞最敏感；R3 不受同义改写影响，但对未见类别更弱；R4 应在二者之间取得较好平衡。

### 表 D：效率

| 表征 | 可训练参数 | Profile token 数 | 单步延迟 | 样本/秒 | 峰值显存 | 是否可缓存 |
|---|---:|---:|---:|---:|---:|---|
| R1 |  |  |  |  |  | 部分 |
| R2 |  |  |  |  |  | 是 |
| R3 |  | 0 |  |  |  | 是 |
| R4 |  |  |  |  |  | 是 |

统计检验使用会话级 bootstrap 95% CI，不能把同一会话中的大量事件当成互相独立的样本。

---

## 13. 怎样判断“Profile 有作用”

只有满足下面条件，才在论文中写“profile 提高了性能”：

1. Hidden 基线没有输出塌缩；
2. `given` 在 Macro-F1 上稳定高于 `hidden`；
3. `given` 也稳定高于 `shuffled`；
4. 提升不是只来自数量最多的 C，而是在 BC/T/I 中至少有目标类别改善；
5. 多个 seed 或 grouped fold 中大多数方向一致；
6. 会话级 95% CI 不显示结果方向高度不稳定；
7. profile 字段消融能解释提升主要来自哪里。

如果只发现 `given` 和 `shuffled` 都改变预测，却没有稳定差距，正确表述是：

> 模型对 profile 条件敏感，但尚无证据证明它理解并正确使用了 profile 语义。

---

## 14. 第二阶段：动态 Profile 怎样接入

先选出静态条件下最好的表征，再加入动态 updater；不要同时更换表征和 updater。

### 14.1 动态信息只从预测边界以前计算

参考 Park 2024 和 Meshorer & Heeman 2016，可以计算：

- 当前说话人和听者的平均/标准差话轮时长；
- 每话轮词数和填充词频率；
- BC 频率；
- C/BC/T/I/NA 的历史分布；
- 最近一段时间双方占用话语权比例；
- 最近若干话轮的关系/场景证据。

不能使用整段会话统计，因为其中包含预测点之后的信息。

### 14.2 Updater 输出统一 snapshot

```json
{
  "relationship": "colleagues",
  "relationship_confidence": 0.72,
  "situation": "collaborative_task",
  "situation_confidence": 0.81,
  "effective_time_s": 152.4
}
```

同一个 snapshot 同时供两种表示方式使用：

- 自然语言条件：按固定模板写出类别、置信度和生效时间；
- embedding 条件：类别 embedding 加数值 confidence/time 特征。

两边不能拿到不同信息。例如 embedding 如果拿完整概率分布，Prompt 也必须得到等价信息。

### 14.3 动态实验表

| Profile 来源 | 表征 | Macro-F1 | Given−Hidden | Given−Shuffled | BC F1 | T F1 | I F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Static gold | 最佳自然语言方案 |  |  |  |  |  |  |
| Static gold | 最佳 embedding 方案 |  |  |  |  |  |  |
| Dynamic predicted | 最佳自然语言方案 |  |  |  |  |  |  |
| Dynamic predicted | 最佳 embedding 方案 |  |  |  |  |  |  |
| Dynamic oracle | 最佳自然语言方案 |  |  |  |  |  |  |
| Dynamic oracle | 最佳 embedding 方案 |  |  |  |  |  |  |

`Dynamic predicted` 与 `Dynamic oracle` 的差距衡量 updater 还有多少改进空间。

---

## 15. 需要新增或修改的代码结构

```text
code/
├─ configs/
│  ├─ profile_repr_r0_hidden.yaml
│  ├─ profile_repr_r1_prompt_text.yaml
│  ├─ profile_repr_r2_text_embedding.yaml
│  ├─ profile_repr_r3_field_embedding.yaml
│  └─ profile_repr_r4_hybrid.yaml
├─ src/profile_turntaking/
│  ├─ profile_schema.py          # 字段清洗、显式词表、角色方向转换
│  ├─ profile_serialization.py   # 唯一自然语言模板及等义改写模板
│  ├─ profile_encoders.py        # R0/R1/R2/R3/R4 的统一接口
│  ├─ multimodal_backbone.py     # 共享音频和转写编码器
│  ├─ profile_fusion.py          # 样本级 gate 和 residual adapter
│  ├─ representation_dataset.py # 同一 manifest 生成所有表示条件
│  └─ representation_eval.py    # hidden/given/shuffled 与配对统计
├─ scripts/
│  ├─ build_profile_vocab.py
│  ├─ cache_backbone_features.py
│  ├─ train_profile_representation.py
│  ├─ run_profile_representation_eval.py
│  └─ audit_profile_representation_protocol.py
└─ tests/
   ├─ test_profile_schema.py
   ├─ test_representation_pairing.py
   ├─ test_no_future_leakage.py
   └─ test_profile_shuffle.py
```

所有 encoder 实现统一接口：

```python
profile_state = profile_encoder(profile_snapshot)
# shape 始终为 [batch, 128]
```

这样后续更换 encoder 不需要改音频、转写、训练循环或评测代码。

---

## 16. 一周内的执行时间线

| 时间 | 具体工作 | 当天必须得到的结果 |
|---|---|---|
| 第 1 天 | 锁定事件 manifest；清洗 profile；实现显式词表和当前说话人/听者转换；运行泄漏与配对审计 | 可重复的数据清单、profile vocab、审计报告 |
| 第 2 天                                              实现共享 HuBERT＋DistilRoBERTa 特征；缓存训练/验证/测试特征所有样本的固定音频/转写向量，避免每个实验重复编码 |
| 第 3 天 | 实现 R0/R1/R2/R3 统一接口、样本级 gate、训练与单元测试 | 四种条件都能在小批次端到端运行 |
| 第 4 天 | 正式训练 R0/R1/R2/R3，至少 3 seeds | 验证集主表和输出分布；确认无塌缩 |
| 第 5 天 | 选定设置，运行锁定测试集的 hidden/given/shuffled；生成混淆矩阵、CI、校准和效率表 | Prompt vs embedding 主结果 |
| 第 6 天 | 如果 R2/R3 有明确增益，训练 R4；做字段消融、模板改写和错误 profile 测试 | 方法消融和鲁棒性结果 |
| 第 7 天 | 汇总论文图表；只在静态结果成立后启动动态 updater 小实验 | 可向老师汇报的完整结果和下一阶段决定 |

如果算力有限，优先顺序是：`R0 → R1 → R2 → R3 → R4`。R4 和动态实验可以延后，但不能省掉 R0/R1/R2/R3 的公平对照。

---

## 17. 最终建议

1. **近期先做 R0/R1/R2/R3。** 这是最小但足以回答 Prompt 与 embedding 差异的正式实验。
2. **主方法优先考虑 R4 Hybrid。** SBCSAE 的关系、场景适合结构化 embedding，职业和背景过于稀疏，更适合文字 embedding。
3. **不要把当前哈希桶版本作为最终方法。** 先改成显式词表和清晰字段。
4. **不要拿 Qwen zero-shot 分数与 adapter 分数直接宣布 embedding 更好。** Qwen 结果只作为外部通用模型基线。
5. **动态关系和场景必须只从过去更新。** 同一个动态 snapshot 同时提供给 Prompt 和 embedding，才能公平比较。
6. **论文最有价值的结论不是“有 profile 时预测发生变化”。** 应证明正确 profile 比隐藏和错误 profile 都更好，并解释哪类信息、哪些话轮事件获得了改善。

---

## 18. 参考文献与可复现入口

1. Fukunaga et al. (2025), *Backchannel prediction for natural spoken dialog systems using general speaker and listener information*. [ISCA Archive](https://www.isca-archive.org/interspeech_2025/fukunaga25_interspeech.html)
2. Park et al. (2024), *Backchannel prediction, based on who, when and what*. [ISCA Archive](https://www.isca-archive.org/interspeech_2024/park24b_interspeech.html)
3. Ortega et al. (2023), *Oh, Jeez! or Uh-huh? A Listener-aware Backchannel Predictor on ASR Transcriptions*. [arXiv](https://arxiv.org/abs/2304.04478)
4. Ortega et al. (2023), *Modeling Speaker-Listener Interaction for Backchannel Prediction*. [arXiv](https://arxiv.org/abs/2304.04472)
5. Wang et al. (2024), *Turn-taking and Backchannel Prediction with Acoustic and Large Language Model Fusion*. [arXiv](https://arxiv.org/abs/2401.14717)
6. Zhang et al. (2018), *Personalizing Dialogue Agents: I have a dog, do you have pets too?* [ACL Anthology](https://aclanthology.org/P18-1205/)
7. Li et al. (2016), *A Persona-Based Neural Conversation Model*. [ACL Anthology](https://aclanthology.org/P16-1094/)
8. Fu et al. (2025), *PACHAT: Persona-Aware Speech Assistant for Multi-party Dialogue*. [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1492/)
9. Meshorer & Heeman (2016), *Using Past Speaker Behavior to Better Predict Turn Transitions*. [ISCA Archive](https://www.isca-archive.org/interspeech_2016/meshorer16_interspeech.html)
10. `all-MiniLM-L6-v2` model card. [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
11. `HuBERT-base-ls960` model card. [Hugging Face](https://huggingface.co/facebook/hubert-base-ls960)
12. `DistilRoBERTa-base` model card. [Hugging Face](https://huggingface.co/distilbert/distilroberta-base)

说明：Fukunaga 2025 的摘要措辞与正文表格存在不完全一致。上文数值采用正文表格：通用人物信息优于无人物信息，但略低于 listener ID embedding；因此本报告没有把它写成“通用信息优于 ID”。
