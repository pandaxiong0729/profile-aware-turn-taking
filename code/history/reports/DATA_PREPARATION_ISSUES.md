# 数据预处理问题与未解决限制

更新时间：2026-07-13

本文件区分“已经恢复并记录的源数据异常”和“仍会影响实验结论的科学限制”。所有异常都保留在机器可读的 `issues.jsonl` 中，没有静默丢弃。

## SBCSAE 源数据异常

全量 catalog 记录 97 条问题：

| 原因 | 数量 | 当前处理 |
| --- | ---: | --- |
| 非正长度时间区间 `non_positive_interval` | 75 | 不进入有效 utterance；保留原文件、行号和诊断 |
| profile metadata 找不到 | 12 | profile 字段填 `unknown`，记录会话与说话人 |
| 姓名对应多条 metadata | 4 | 不猜测，填 `unknown` 并记录歧义 |
| 非人物说话人未在 header 声明 | 2 | 识别为环境/集合标签，不参与 A/B |
| metadata 控制字符 | 1 条记录，含 3 处字符 | 规范化为连字符并记录修复次数 |
| metadata 字段不足 | 1 | 缺失字段补 `unknown` |
| metadata 字段过多 | 1 | 按已知 schema 对齐并记录 |
| `.trn` 中出现未声明真实说话人 | 1 | 从转写恢复并记录来源 |

全 60 段会话中有 16 个 unresolved 人类 profile row。它们都不在当前 16 段 core dyadic manifest 中，因此不会污染第一阶段训练；如果未来扩展到非 core 会话，必须先人工消歧。

## SBCSAE 科学限制

### 标签不是人工 gold

- `.trn` 时间戳对应 intonation unit，不是 frame-accurate VAD。
- `BC` 依据反馈词表和持续时间生成，必须抽查讽刺、复述和真正抢话。
- `I` 只表示两个时间区间重叠，尚未区分 `I_COOP / I_COMP`。
- `NA` 合并说话人内部停顿与说话人之间 gap，尚未区分 `PAUSE / GAP`。
- `T` 在当前弱标签中很少：train 596、val 190、test 125，置信区间会较宽。
- 文本来自人工 `.trn` 的因果前缀代理，不是 streaming ASR；最终实验需要评估 ASR 延迟和错误的影响。

所以当前 54,270 行必须称为“自动弱标签 manifest”，不能称为五分类 gold dataset。

### 音频和说话人

- 56 个 WAV 为立体声、4 个为单声道；当前统一在加载时取声道均值。这符合单声道模型输入，但可能削弱利用声道区分说话人的能力。
- Speaker A/B 是会话内按首次出现顺序确定，不代表固定性别、用户/系统或主次角色。
- 跨会话 speaker identity 依赖 metadata 匹配。当前 core manifest 无已知 unresolved identity，但仍建议抽查人工 override。

### Profile 不是完全 gold

- 年龄、性别、职业和背景主要来自官方 metadata；core 中仍有 1 个年龄段未知。
- `relationship` 和 `situation` 是对会话说明的关键词规则映射，不是官方逐会话标签。
- `background` 目前是结构化字段的拼接文本；可能包含与话轮无关的信息，需要用 shuffled/field ablation 检查。
- 静态 profile 对整个会话恒定。动态 updater 只能在第二阶段使用当前时刻之前的信息更新 `relationship` 和 `situation`。

### 抽样与评测分布

当前 manifest 对每类设上限，以便一周内快速训练并保留稀有类。它适合模型开发和 macro 指标，但不代表真实线上类别先验。最终至少同时报告：

1. 当前类别覆盖较均衡的诊断集；
2. 不重采样的自然分布 test 结果；
3. 按会话 bootstrap 的置信区间。

## Persona-Dialogue / PAChat 限制

机器可读问题共 4 类：

| 问题 | 结论 |
| --- | --- |
| 未找到论文所述完整 release | 不能声称已取得 21,760 段对话或 217 小时数据 |
| demo 仓库未声明数据许可 | 不上传 demo 音频或 HTML 归档到本仓库 |
| 每个 turn 是独立 WAV | 没有连续 gap/overlap 时间，不能用于 turn-taking gold |
| 1 条 transcript 跨案例完全重复 | Case 4 turn 4 重复 Case 2 turn 4，使用内容前必须去重/核验 |

如果作者后续发布完整数据，重新运行前必须先检查：许可、连续多方音频、说话人时间轴、profile ID、train/test 官方划分以及是否存在合成模板泄漏。

## 尚未完成

- 没有训练全量研究模型；现有运行只是功能 smoke test。
- 没有 frame-accurate VAD/overlap gold。
- 没有人工七分类诊断集和 `I_COOP/I_COMP` Cohen's kappa。
- 没有多随机种子 Macro-F1、bootstrap CI、Brier/ECE、event-level F1 或推理效率结果。
- 没有动态 profile updater 训练数据。
- 没有取得 Persona-Dialogue 全量 release。

这些项目不会阻止第一阶段五分类代码训练，但会限制论文中可以作出的结论。
