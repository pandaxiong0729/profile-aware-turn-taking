# Qwen Shared A/B Adapter：30 秒因果输入优化实验最终报告

日期：2026-08-21

## 1. 一句话结论

30 秒因果音频、匹配的因果转写和 profile 已经接入同一个 Qwen shared A/B adapter。模型在四个 Talking Turns 风格二分类任务上的平均平衡 Accuracy 达到约 73%—74%，说明 turn-taking judge 本身能够工作；但是 `given > hidden` 且 `given > shuffled` 只在验证集成立，没有在按会话隔离的测试集稳定复现。因此目前不能把结果写成“profile 已被证明有效”。

## 2. 最终正式结果

正式配置由验证集选择，测试结果为 5 个随机种子（3/13/37/71/101）的平均。

| 计分方式 | hidden | given | shuffled | given-hidden | given-shuffled |
|---|---:|---:|---:|---:|---:|
| 原始样本 Accuracy | 77.81% | 78.10% | 78.17% | +0.30 pp | -0.07 pp |
| 50/50 A/B 平衡 Accuracy | 73.50% | 73.12% | 73.43% | -0.37 pp | -0.31 pp |

必须以第二行作为与 Talking Turns 表格对照的主结果。第一行受多数类影响：例如 interruption 中 C 远多于 I，多数类判断会把普通 Accuracy 抬高。第二行在每个任务中固定抽取相同数量的 A 和 B，随机基线为 50%。

结论是：

- judge 的四问平均能力已经达到约 73%；
- 正确 profile 没有在正式测试中优于 hidden 和 shuffled；
- 不能只报告 78.10%，也不能把验证集的正增益当成测试结果。

## 3. 输入和输出

每条样本输入：

```text
预测点之前的 30 秒因果音频
        +
与音频严格匹配、只到预测点为止的因果转写
        +
profile（hidden / given / shuffled 三选一）
```

没有提供未来音频、未来转写、目标标签或标注依据。三种 profile 条件中，音频、转写、sample ID、预测边界、问题文本和解码设置完全相同，只改变 profile。

模型输出四组 A/B 概率：

| 任务 | A | B |
|---|---|---|
| Turn change | 当前说话人继续（C） | 另一人接过话轮（T） |
| Backchannel | 没有简短反馈（C） | 听者开始 backchannel（BC） |
| Interruption | 没有打断（C） | 另一人开始打断（I） |
| Floor-taking interruption | 原说话人保住话权 | 打断者取得话权 |

第四问已经按论文定义修正：输入边界处已经观察到 interruption，再预测后续话权属于原说话人还是打断者。它不再使用“打断开始之前的音频去猜最终结果”的旧切法。

## 4. 数据规模

### 4.1 五类事件

| split | C | BC | T | I | NA | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| train | 2,474 | 455 | 891 | 424 | 2,379 | 6,623 |
| val | 788 | 158 | 365 | 112 | 820 | 2,243 |
| test | 733 | 117 | 260 | 93 | 735 | 1,938 |
| 合计 | 3,995 | 730 | 1,516 | 629 | 3,934 | 10,804 |

SBCSAE 共 16 段会话，按 conversation 划分为 10/3/3 段 train/val/test。同一段会话不会同时进入训练和测试。

### 4.2 四个 A/B 任务

| 任务 | train A/B | val A/B | test A/B | test 50/50 子集 |
|---|---:|---:|---:|---:|
| Turn change | 2,474 / 891 | 788 / 365 | 733 / 260 | 520 |
| Backchannel | 2,474 / 455 | 788 / 158 | 733 / 117 | 234 |
| Interruption | 2,474 / 424 | 788 / 112 | 733 / 93 | 186 |
| Floor-taking | 221 / 383 | 66 / 127 | 34 / 81 | 68 |

训练使用全部可用样本；50/50 子集只用于论文口径评测。`NA` 保留在五类事件数据中，但不属于这四个 A/B 问题。

## 5. 模型结构

```text
30 秒因果音频
  → 冻结 Qwen2.5-Omni-3B 音频塔
  → 33 层在预测边界处的表示（每层 1,280 维）
  → 可训练 softmax 层加权
  → 音频状态

音频 + 因果转写 prompt
  → 冻结 Qwen Thinker
  → 2,048 维 context hidden state

音频状态 + context hidden state
  → 投影并融合成 256 维 shared state

59 维 profile
  → 投影为 256 维 profile state

shared state + profile state
  → gate(shared, profile)
  → shared + gate × profile state
  → 四个可训练 A/B 输出头
  → 四组 A/B 概率
```

Qwen 参数全部冻结。训练的是约 1,528,361 个参数：音频层权重、音频/上下文/profile 投影、gate 和四个二分类头。它是一个共享 adapter 加四个小输出头，不是四个 Qwen 模型。

## 6. Profile 的内容

最终 profile 为 59 维：

- 38 维因果行为统计：当前说话人和另一位参与者的发言占比、发言长度、短回应率、词速、填充词、重叠、回应间隔、历史打断尝试及成功/让权比例，以及全局静音、重叠、话权平衡和换人频率；
- 21 维 relationship、situation 及其组合的 one-hot 表示。

行为统计使用“预测点前 5 秒截止、向前回看 120 秒”的历史。预测点前最后 5 秒只留在 Qwen 主输入中，避免 profile 直接重复总结局部目标边界。说话人字段按“当前说话人 / 另一位参与者”排列，而不是固定 speaker_00 / speaker_01。

三种条件：

- `hidden`：59 维全零；
- `given`：当前样本的正确因果 profile；
- `shuffled`：来自另一段会话的错误 profile。

## 7. 训练方法

- 主损失：四个 A/B 头的类别加权交叉熵；
- hidden 辅助交叉熵权重：0.5；
- profile dropout：0.25；
- given 相对 hidden 的 margin 权重：0.1；
- given 相对 shuffled 的 margin 权重：0.25；
- margin：0.05；
- profile margin 中 A/B 两类分别归一化，使两类各占 50%；
- AdamW，learning rate 0.0008，batch size 128；
- 最多 100 epochs，patience 14；
- 验证种子 13/37/71，正式测试种子 3/13/37/71/101。

平衡 profile margin 是必要修正。旧实现的交叉熵虽然类别加权，但 profile margin 按原始样本数平均，导致它主要优化多数类。这会出现普通 Accuracy 上升、50/50 Accuracy 下降的假象。修正后验证集通过，但正式测试仍未复现。

## 8. 验证与测试为什么结论不同

| split | hidden | given | shuffled | given-hidden | given-shuffled |
|---|---:|---:|---:|---:|---:|
| validation，3 seeds | 73.77% | **74.35%** | 73.69% | +0.57 pp | +0.66 pp |
| test，5 seeds | **73.50%** | 73.12% | 73.43% | -0.37 pp | -0.31 pp |

验证集显示 profile 信号可以被模型学习，但它没有稳定迁移到 3 段新测试会话。最合理的解释是会话/profile 数量过少：训练虽然有 6,623 条事件，却只来自 10 段会话，因此真正独立的 profile 单元不是 6,623 个，而接近 10 个。

这也说明“样本条数多”不等于“persona/profile 多”。同一会话的很多事件共享相近关系、场景和历史行为，模型容易学到训练会话特征，难以迁移到新人物。

## 9. 正式测试逐任务结果

下表为 50/50 A/B Accuracy。

| 任务 | hidden | given | shuffled | given-hidden | given-shuffled |
|---|---:|---:|---:|---:|---:|
| Turn change | 75.15% | **75.31%** | 75.23% | +0.15 pp | +0.08 pp |
| Backchannel | **76.84%** | 76.32% | 76.15% | -0.51 pp | +0.17 pp |
| Interruption | **86.99%** | 86.45% | 86.45% | -0.54 pp | 0.00 pp |
| Floor-taking | 55.00% | 54.41% | **55.88%** | -0.59 pp | -1.47 pp |
| 四任务平均 | **73.50%** | 73.12% | 73.43% | -0.37 pp | -0.31 pp |

Turn change 出现很小正增益；BC 只高于 shuffled；I 和 floor-taking 没有正增益。floor-taking 的平衡测试子集只有 68 条，并且接近随机，是波动最大的任务。

## 10. 做过的关键优化

1. 将输入统一为 30 秒因果音频 + 匹配因果转写 + profile。
2. 使用 Qwen 音频塔全部 33 层的可学习加权，而不是只取最后一层。
3. 修正 floor-taking 的事件边界，使其与 Talking Turns 的“已观察到 interruption 后预测谁取得话权”一致。
4. 将 speaker_00/speaker_01 行为改成当前说话人/另一参与者的角色归一化表示。
5. 将动态历史从窗口前 25 秒扩展为预测点前 120 秒、并预留最后 5 秒给主分支。
6. 增加历史打断尝试、成功和让权比例，使 profile 与 floor-taking 更直接相关。
7. 比较 shared gate、concat、FiLM 和任务专属 gate；任务专属 gate 在验证集没有通过，因此未进入正式测试。
8. 显式训练 given 优于 hidden/shuffled，并修正 profile margin 的 A/B 类别不平衡。
9. 做过 train+val 固定 epoch 重训；它没有改善测试排序，未被选作主结果。

这些改动把 judge 从旧切分约 71% 提升到约 73%—74%，但没有解决 profile 跨会话泛化。

## 11. 与 Talking Turns 的关系

`2503.01174v1.pdf` 的 Section 4.1 使用 Whisper-medium encoder、多层加权、最后音频帧和一个线性五分类器；Eq. 2—5 将五类概率转换为 Turn Change、Backchannel、Interruption 和 Floor-Taking 四个问题。Table 4 的 supervised topline 为 78.6%、75.1%、74.9%、65.6%，平均约 73.55%。

我们的四问形式和 50/50 随机基线已对齐，但数据集不同，而且主输入还包含因果转写与 profile，所以不能声称在同一榜单上超过该 topline。当前结果只能说明：在 SBCSAE 上，Qwen shared A/B judge 达到相近量级。

## 12. 现在能说与不能说的内容

可以说：

1. 30 秒多模态因果 judge 已完整跑通，平均 A/B Accuracy 约 73%；
2. Qwen 多层音频表示能够判断 turn change、BC 和 interruption；
3. profile 在验证集能产生正增益，说明实现不是完全失效；
4. 当前 SBCSAE 切分上，profile 效果不能跨会话稳定复现。

不能说：

1. 不能说正确 profile 已经稳定提高性能；
2. 不能把普通 78.10% Accuracy 当成 Talking Turns 的 50/50 结果；
3. 不能把验证集 74.35% 当成正式测试结果；
4. 不能继续根据这 3 段测试会话扫描参数后再称其为未触碰测试集；
5. 不能声称超过 Talking Turns supervised topline。

## 13. 下一步需要改变什么

继续调整 gate 权重、随机种子或 epoch 不会解决核心问题。下一轮需要改变的是数据独立性和 profile 多样性：

1. 增加具有不同关系、场景和行为习惯的独立会话，而不仅是从现有 16 段会话切更多事件；
2. 用 conversation-level group cross-validation 报告跨会话均值和置信区间；
3. 为论文建立一组真正未参与当前迭代的新测试会话；
4. 扩充 floor-taking，当前 68 条平衡测试样本不足以稳定判断 profile 效果；
5. 如果取得 Talking Turns 原始 benchmark，应在相同数据和输入限制下另做可比实验；
6. 更换更大的模型可以作为模型容量实验，但不能替代 profile/会话多样性的扩充。

## 14. 可核实文件

- 数据：`data/processed/sbcsae_qwen_shared_ab_30s_causal_v1/`
- Qwen 多层缓存：`artifacts/qwen-shared-ab-30s-causal/layer-weighted-search/cache/`
- 修正后的 floor 目标：`artifacts/qwen-shared-ab-30s-causal/paper-aligned-floor/cache/`
- 59 维动态 profile：`artifacts/qwen-shared-ab-30s-causal/profile-views/dynamic-history120-floorprop-causal-role/metadata.json`
- 正式验证结果：`artifacts/qwen-shared-ab-30s-causal/paper-aligned-floor/history120-floorprop-balanced-profile-margin-validation/gate_0p25/summary.json`
- 正式测试汇总：`artifacts/qwen-shared-ab-30s-causal/paper-aligned-floor/history120-floorprop-balanced-profile-margin-final/gate_0p25/summary.json`
- 正式逐样本预测：`artifacts/qwen-shared-ab-30s-causal/paper-aligned-floor/history120-floorprop-balanced-profile-margin-final/gate_0p25/test_predictions.jsonl`
- train+val 固定 4 轮结果：`artifacts/qwen-shared-ab-30s-causal/paper-aligned-floor/history120-floorprop-refit/balanced-margin-fixed4-final/summary.json`
- 训练代码：`code/scripts/run_qwen_shared_binary_multitask_adapter.py`
- profile 构建代码：`code/scripts/build_dynamic_behavior_profile_views.py`
- floor 目标代码：`code/src/profile_turntaking/paper_aligned_floor_targets.py`
- 测试：`code/tests/test_qwen_shared_binary_adapter.py`、`code/tests/test_dynamic_behavior_profile_views.py`、`code/tests/test_paper_aligned_floor_targets.py`

本报告对应的是完整 pilot 结果。它诚实证明了 judge 可用，也明确说明了当前数据上 profile 结论尚未成立。
