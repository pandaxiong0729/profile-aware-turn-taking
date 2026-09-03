# Talking Turns 官方模型在 SBCSAE 上的 OOD 测试

## 一句话结果

不在 SBCSAE 上重新训练，直接把 Talking Turns 在 Switchboard 上训练并公开的五分类 checkpoint 用于我们的 SBCSAE 测试集：五类平均 ROC-AUC 为 **55.48%**。辅助指标 Accuracy 为 **31.99%**。

## 1. 对齐的论文口径

Talking Turns 在正文第 4.1 节和 Table 1 中采用以下 OOD 方法：

1. 在 Switchboard 上训练五分类 judge；
2. 输入预测位置以前最多 30 秒的单声道混合音频；
3. 预测下一个 40 ms 的 `C / BC / T / I / NA` 概率；
4. 不在 OOD 数据上重新训练，直接测试 Columbia Games 和 Fisher；
5. 对每一类计算 one-vs-rest ROC-AUC，最后取五类平均。

因此，论文 Table 1 的 `91.5` 和 `91.0` 是百分制 ROC-AUC，不是五分类 Accuracy。

## 2. 我们实际输入和测试数据

- 模型：Talking Turns 官方 ESPnet checkpoint；
- 训练来源：作者在 Switchboard 上训练；
- 我们没有在 SBCSAE 上微调它；
- 测试样本：1,938 条；
- 测试会话：`SBC007 / SBC017 / SBC058`；
- 每条输入：预测位置以前的 30 秒因果单声道混合音频；
- 输出：`C / BC / T / I / NA` 五类概率；
- 真实标签数量：`C=733, BC=117, T=260, I=93, NA=735`。

模型接口本身只接受音频，因此这项外部基线不输入转写或 profile。

## 3. OOD 结果

| 数据集 | C | BC | T | I | NA | 五类平均 ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Talking Turns: Columbia Games | 95.20% | 94.00% | 81.60% | 92.60% | 94.00% | 91.50% |
| Talking Turns: Fisher | 95.00% | 83.30% | 91.60% | 91.80% | 93.50% | 91.00% |
| 官方 checkpoint → 我们的 SBCSAE | **35.18%** | **56.05%** | **61.62%** | **67.96%** | **56.61%** | **55.48%** |

辅助分类指标：

| Accuracy | Balanced Accuracy | Macro-F1 |
|---:|---:|---:|
| 31.99% | 19.92% | 18.16% |

## 4. 结果应如何表述

这项结果可以表述为：

> We directly evaluated the released Talking Turns checkpoint, trained on Switchboard, on our held-out SBCSAE test conversations without any target-domain fine-tuning. The model achieved a macro one-vs-rest ROC-AUC of 55.48% across the five turn-taking labels.

不能写成“我们复现了 Table 1 的 40 ms 全帧结果”。目前的 1,938 条样本是从三个测试会话抽出的事件中心预测位置；它满足 30 秒因果输入、五类输出、零样本跨数据集测试和 Table 1 的 ROC-AUC 算法，但没有覆盖三个会话中的每一个连续 40 ms 时间格。

如果要做最严格的 Table 1 全帧复现，下一步需要对三个测试会话的约 103,772 个 40 ms 位置逐点构造 30 秒因果窗口并推理。现有 `578,686` 帧是全部 16 个会话，不能全部作为测试集，否则会破坏我们已有的会话级 train/validation/test 划分。

## 5. 可核查入口

- 逐样本预测：`artifacts/espnet-talking-turns-baseline/sbcsae-test-v1/test_predictions.jsonl`
- 正确重算后的指标：`artifacts/espnet-talking-turns-baseline/sbcsae-test-v1/metrics.json`
- 输入审计：`artifacts/espnet-talking-turns-baseline/sbcsae-test-v1/input_audit.json`
- 混淆矩阵：`artifacts/espnet-talking-turns-baseline/sbcsae-test-v1/confusion_matrix.csv`
- 推理与评分代码：`code/scripts/run_espnet_talking_turns_baseline.py`
- 论文来源：`2503.01174v1.pdf`，正文第 4.1 节、Table 1、附录 A.3–A.4。

## 6. 评分修正说明

旧版 `metrics.json` 曾把概率列顺序 `(C, BC, T, I, NA)` 与 sklearn 的字母顺序标签 `(BC, C, I, NA, T)` 错配，得到无效的 `43.41%`。现在改为按概率字段名称逐类计算 one-vs-rest ROC-AUC，并增加回归测试；修正后的五类平均值是 `55.48%`。
