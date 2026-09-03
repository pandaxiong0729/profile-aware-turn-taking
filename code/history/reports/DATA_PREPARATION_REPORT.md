# SBCSAE 与 Persona-Dialogue 数据预处理报告

更新时间：2026-07-13

## 结论

SBCSAE 全量语料已经完成下载、解包、`.trn` 解析、说话人映射、结构化 profile 对齐、会话关系/场景映射、五分类弱标签生成和跨文件质量审计。当前可直接用于第一阶段的同 checkpoint `hidden / given / shuffled` 五分类实验。

Persona-Dialogue 的论文所述全量数据目前没有在论文页、项目页或项目页源码仓库中公开。现有代码只整理官方项目页的 4 个示例，明确标记为不可用于 40 ms turn-taking gold 评测，不能把它写成已经取得完整 PaChat 数据。

原始语料、处理后 manifest 和模型都保留在本地并被 Git 忽略；GitHub 只发布可复现代码、无版权测试 fixture、schema 和统计报告。

## 数据来源与许可

| 数据 | 实际取得内容 | 许可/发布状态 | 本实验用途 |
| --- | --- | --- | --- |
| SBCSAE | OpenSLR SLR155 完整归档，6,186,324,893 bytes | [OpenSLR 页面](https://www.openslr.org/155/)标注 CC BY-ND 3.0 US | 第一阶段训练、验证和测试 |
| Persona-Dialogue / PAChat | [官方项目页](https://persona-dialogue.github.io/)的 4 个演示案例 | 演示仓库未给出数据许可；不再分发 | 仅验证 profile 解析，不作 turn-taking 训练或评测 |
| Persona-Dialogue 论文 | [ACL Anthology 论文页](https://aclanthology.org/2025.emnlp-main.1492/)及 PDF | 论文可读，但未找到论文所述完整语料下载 | 核对论文统计和设计来源 |

OpenSLR 将 SBCSAE 描述为 60 段约 20 分钟的自然对话，含 22.05 kHz 录音、intonation-unit 级时间标记和参与者元数据。论文报告 Persona-Dialogue 有 21,760 段对话、159,933 个 turns 和 217 小时语音；这些数字是论文统计，不是本地已下载数据量。

## SBCSAE 全量统计

| 项目 | 结果 |
| --- | ---: |
| 会话 | 60 |
| 有效 `.trn` intonation units | 70,008 |
| WAV | 60 |
| WAV 总时长 | 23.314600 小时 |
| 立体声 22.05 kHz / 16-bit PCM | 56 |
| 单声道 22.05 kHz / 16-bit PCM | 4 |
| 元数据声明的双人会话 | 19 |
| 实际观察到恰好两位人类说话人的会话 | 18 |
| 第一阶段 core dyadic 会话 | 16 |
| core dyadic 音频 | 6.429755 小时 |
| core dyadic intonation units | 17,419 |
| 全语料 speaker-connected groups | 52 |

`core dyadic` 的可复现定义是：恰好观察到两位人类说话人；任一人的带时间 unit 占比不低于 5%；关系不是 `speaker_audience`。这个筛选避免把访谈中偶尔插话者或演讲者—听众会话误当作平衡双人训练数据。

### Speaker A/B 与 profile

每段 core dyadic 会话中，第一个实际出现的人类说话人映射为 `speaker_A`，另一个映射为 `speaker_B`。`ENV`、听众集合、动物和联合说话人标记不参与 A/B 分配。跨会话使用 metadata identity 构建 speaker-connected component，保证同一人的会话不会跨 train/val/test。

32 个 core dyadic 人类 profile 的匹配状态：

| 映射来源 | 数量 |
| --- | ---: |
| 唯一姓名匹配 `unique_name` | 14 |
| 人工高置信覆盖 `manual_high_confidence` | 12 |
| 会话提示匹配 `conversation_hint` | 6 |

core dyadic 中没有 `not_found` 或 `ambiguous_name`；只有 1 个 profile 的 `age_group` 为 `unknown`。profile 固定结构为：

```text
speaker_A.age_group / gender / social_role / background
speaker_B.age_group / gender / social_role / background
relationship / situation
```

关系分布为 romantic partners 4、family 4、colleagues 4、friends/peers 3、professional/client 1。场景分布为 casual social 5、workplace/business 4、family/home 4、collaborative task 2、healthcare consultation 1。关系和场景来自会话说明的规则映射，不应写成原语料人工 gold label。

## 五分类 manifest

第一阶段使用 `C / BC / T / I / NA`。每个样本包含预测时刻前 30 秒音频、只含已经结束 utterance 的因果文本前缀、结构化 profile，以及下一个 40 ms 的弱标签。56 个立体声文件在读取窗口时取两声道均值，统一作为单声道输入。

划分单位是 speaker-connected component，当前 16 个 core 会话分别落入 train 10、val 3、test 3；会话、group 和已解析 speaker UID 均无跨 split 泄漏。

### 全部 40 ms 弱标签帧

| Split | C | BC | T | I | NA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 295,659 | 5,684 | 596 | 22,180 | 27,901 |
| Val | 94,699 | 3,001 | 190 | 6,402 | 8,435 |
| Test | 90,651 | 2,932 | 125 | 5,213 | 2,426 |

### 当前抽样后的训练/评测 manifest

| Split | C | BC | T | I | NA | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 5,000 | 5,000 | 596 | 5,000 | 5,000 | 20,596 |
| Val | 5,000 | 3,001 | 190 | 5,000 | 5,000 | 18,191 |
| Test | 5,000 | 2,932 | 125 | 5,000 | 2,426 | 15,483 |
| 合计 | 15,000 | 10,933 | 911 | 15,000 | 12,426 | 54,270 |

另生成 11,290 个连续弱事件，供后续 ±200 ms event-level 评测或人工抽查。抽样策略保留稀有类并限制多数类规模；正式报告还应在自然分布测试集上复核，不能把这个平衡抽样测试集当成线上先验分布。

## 公平的 profile 主实验

主实验不是训练一个 Talking Turns 模型再与我们的模型比较。正确设计已经写进代码：

1. 使用同一份 manifest 和同一 train/val/test 划分训练一个模型；
2. 训练时以 `profile_dropout=0.5` 将部分样本替换为统一的 `unknown` profile，让一个 checkpoint 同时学会有/无 profile；
3. 加载这一个 checkpoint 一次，在完全相同的 test sample IDs 上运行 `hidden`、`given`、`shuffled`；
4. `hidden` 把全部字段设为 `unknown`，`given` 使用正确 profile，`shuffled` 按会话整体换成另一测试会话的 profile；
5. 音频、文本、样本顺序、模型权重和解码设置均保持不变。

因此 `given - hidden` 才能解释为“在当前系统中提供正确 profile 的增量”。`shuffled` 是负控制：如果错误 profile 仍有同样增益，说明模型可能只利用了额外参数或数据偏差。Talking Turns checkpoint 只可作为可选 backbone/外部参考，不是 profile 因果比较的主基线。

当前只完成了接口 smoke test，没有在全量 manifest 上完成论文级训练，任何 smoke 指标都不是实验结果。

## Persona-Dialogue / PAChat 实际取得情况

| 项目 | 本地结果 |
| --- | ---: |
| 官方 demo cases | 4 |
| profiles | 14 |
| turns / WAV | 29 |
| 音频总时长 | 4.973333 分钟 |
| 音频格式 | mono、16 kHz、32-bit IEEE float |

这些 WAV 是逐 turn 独立文件，没有原始连续时间轴，无法恢复真实 gap、overlap、PAUSE/GAP 或 40 ms 边界。因此它们只用于检查自然语言 profile 与结构化 profile 的解析，不进入 turn-taking manifest。官方 demo 中还发现 1 条跨案例完全重复的 transcript，详见问题报告。

## 自动质量审计

`audit-preprocessed` 的 19 项检查全部通过，包含：

- 60 段会话和 60 个有效音频头；
- utterance 计数一致、sample ID 唯一、profile schema 完整；
- 所有 manifest 音频路径存在；
- 文本前缀没有预测时刻之后的已结束时间；
- group、conversation、speaker UID 均无跨 split 泄漏；
- 五类在 train/val/test 中都有样本；
- PaChat demo 数量、音频有效性和“不具备 turn-taking gold 条件”的标记一致。

审计通过表示处理产物内部一致，不表示弱标签已经等价于人工 gold。

## 本地产物

| 路径 | 内容 |
| --- | --- |
| `data/processed/sbcsae_catalog/` | 60 会话目录、70,008 utterances、profile、问题记录和摘要 |
| `data/processed/sbcsae_mvp/manifest.jsonl` | 54,270 条可训练/评测样本 |
| `data/processed/sbcsae_mvp/weak_events.jsonl` | 11,290 个连续弱事件 |
| `data/processed/sbcsae_mvp/split_map.json` | speaker-connected group 划分 |
| `data/processed/pachat_demo/` | 官方 demo 的 cases/profiles/turns 和限制标记 |
| `data/processed/audit.json` | 19 项跨产物审计结果 |

字段解释见 `docs/DATA_SCHEMA.md`，完整问题和未解决限制见 `reports/DATA_PREPARATION_ISSUES.md`。

## 下一步的最短实验路线

1. 在训练 split 音频上补 VAD/overlap detector，保留当前 TRN 弱标签作为可追溯来源。
2. 人工复核 BC、I 和边界样本；优先制作 300–500 条七分类诊断集，而不是全量人工七分类。
3. 用 3 个或 5 个随机种子训练同一架构，并在每个 checkpoint 上做 paired `hidden/given/shuffled`。
4. 报告五分类 Macro-F1、Balanced Accuracy、每类 P/R/F1、混淆矩阵、bootstrap CI、校准和运行效率；七分类只在人工诊断集上报告。
5. 第二阶段再增加只更新 `relationship` 和 `situation` 的 causal Profile Updater，并比较结构化 field embedding 与同一 snapshot 的固定模板自然语言编码。
