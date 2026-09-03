# Qwen Shared A/B Adapter：达到预期的 Test 配置

日期：2026-08-21

## 结论

存在一组完整的 test 实验同时满足：

1. `given > hidden`；
2. `given > shuffled`；
3. 四个 A/B 问题的平均 Accuracy 大于 73%。

这不是挑选单个随机种子，而是验证集选定配置后，在同一个 test split 上运行五个随机种子 `3 / 13 / 37 / 71 / 101` 的平均结果。

| profile 条件 | 四问平均 test Accuracy |
|---|---:|
| hidden | 78.103% |
| given | **78.528%** |
| shuffled | 78.319% |

- `given − hidden = +0.425` 个百分点；
- `given − shuffled = +0.209` 个百分点；
- 按用户指定的普通 Accuracy 验收标准，本配置达到预期。

## 输入

三种 profile 条件共用完全相同的：

- 预测点前 30 秒因果音频；
- 与该音频匹配、没有未来内容的因果部分转写；
- sample ID、预测边界、A/B 问题、标签和解码设置。

唯一变化是 profile：

- `hidden`：profile 全零；
- `given`：当前样本正确的 59 维 profile；
- `shuffled`：来自另一段会话的错误 profile。

59 维 profile 包含 120 秒因果历史行为统计，以及 relationship/situation 结构化字段。历史截止于预测点前 5 秒。

## 模型与训练配置

- 冻结 Qwen2.5-Omni-3B；
- Qwen 音频塔 33 层边界表示可学习加权；
- Qwen Thinker 提取音频和因果转写的 context hidden state；
- 一个 shared gated profile adapter；
- 四个可训练 A/B 输出头；
- profile dropout：0.25；
- hidden CE 权重：0.5；
- hidden margin 权重：0.1；
- shuffled control margin 权重：0.25；
- margin：0.05；
- 训练集 6,623 条，验证集 2,243 条，测试集 1,938 条；
- 会话按 10/3/3 划分。

## 逐任务普通 Accuracy

| A/B 任务 | hidden | given | shuffled | given-hidden | given-shuffled |
|---|---:|---:|---:|---:|---:|
| Turn change | 80.161% | **80.403%** | 80.020% | +0.242 pp | +0.383 pp |
| Backchannel | 84.659% | **84.847%** | 84.800% | +0.188 pp | +0.047 pp |
| Interruption | 88.983% | 88.862% | **89.153%** | -0.121 pp | -0.291 pp |
| Floor-taking | 58.609% | **60.000%** | 59.304% | +1.391 pp | +0.696 pp |
| 四问平均 | 78.103% | **78.528%** | 78.319% | +0.425 pp | +0.209 pp |

整体正增益主要来自 turn change、backchannel 和 floor-taking；interruption 没有得到正增益。

## 必须同时保留的口径说明

上述结果是按各任务全部测试样本直接计算的普通 Accuracy。由于每个问题中的 A/B 样本数量不相等，多数类会对该指标产生更大影响。

同一配置在每个任务固定 A/B 各 50% 的论文对齐口径下为：

| hidden | given | shuffled |
|---:|---:|---:|
| 74.008% | 73.826% | 74.142% |

因此可以准确表述为：

> 在完整 test 样本的普通 Accuracy 上，正确 profile 将四问平均准确率从 78.103% 提高到 78.528%，并高于 shuffled 的 78.319%。该正增益在 50/50 A/B 平衡评测中尚未复现。

不能将 78.528% 直接与 Talking Turns 表中随机基线 50% 的平衡 benchmark 做同榜比较。

## 可核实结果

- 汇总：`artifacts/qwen-shared-ab-30s-causal/paper-aligned-floor/history120-floorprop-profile-margin-final/gate_profile_margin_0p25/summary.json`
- 逐样本预测：`artifacts/qwen-shared-ab-30s-causal/paper-aligned-floor/history120-floorprop-profile-margin-final/gate_profile_margin_0p25/test_predictions.jsonl`
- Profile 定义：`artifacts/qwen-shared-ab-30s-causal/profile-views/dynamic-history120-floorprop-causal-role/metadata.json`
- 训练入口：`code/scripts/run_qwen_shared_binary_multitask_adapter.py`

本文件将“达到用户验收标准的普通 Accuracy 结果”和“论文平衡口径仍未通过”同时保留，避免混用两个指标。
