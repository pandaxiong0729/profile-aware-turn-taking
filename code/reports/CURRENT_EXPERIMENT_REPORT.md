# 当前实验结果

## 数据

主实验使用 SBCSAE 的 16 段双人会话，共 10,804 个事件样本。train/val/test 分别为 6,623/2,243/1,938 条，三组会话不重叠。

我们的模型输入是 30 秒因果音频、匹配的因果转写和 profile；Talking Turns 对照读取同一 test 样本的同一段 30 秒音频，但不读取转写或 profile。

## 我们的 shared A/B adapter

下表为五个随机种子的 paper-balanced Accuracy 均值：

| 任务 | hidden | given | shuffled | given-hidden | given-shuffled |
| --- | ---: | ---: | ---: | ---: | ---: |
| Turn change | 75.15% | 75.31% | 75.23% | +0.15 | +0.08 |
| Backchannel | 76.84% | 76.32% | 76.15% | -0.51 | +0.17 |
| Interruption | 86.99% | 86.45% | 86.45% | -0.54 | 0.00 |
| Floor-taking | 55.00% | 54.41% | 55.88% | -0.59 | -1.47 |
| 四任务平均 | 73.50% | 73.12% | 73.43% | -0.37 | -0.31 |

当前模型可以完成四个 A/B 任务，但 `given` 没有稳定高于 `hidden` 和 `shuffled`，所以这份结果不能作为 profile 已经有效的证据。

## 同架构 Qwen audio-only 对照

它只读 Qwen 音频塔的 33 组表示，不读转写和 profile。三个 profile 名称在输出文件中只是为了复用统一评测格式，实际预测完全相同。

| 任务 | paper-balanced Accuracy |
| --- | ---: |
| Turn change | 76.03% |
| Backchannel | 77.21% |
| Interruption | 88.53% |
| Floor-taking | 52.45% |
| 四任务平均 | 73.55% |

## Talking Turns 官方 checkpoint

官方 ESPnet checkpoint 在同一 SBCSAE test 上的结果：

| 指标/任务 | 结果 |
| --- | ---: |
| 五分类 Accuracy | 31.99% |
| 五分类 Balanced Accuracy | 19.92% |
| 五分类 Macro-F1 | 18.16% |
| 五分类 Macro ROC-AUC | 55.48% |
| Turn change A/B | 49.81% |
| Backchannel A/B | 44.44% |
| Interruption A/B | 54.84% |
| Floor-taking A/B | 51.47% |
| 四任务平均 A/B | 50.14% |

这里是该 checkpoint 从 Switchboard 到 SBCSAE 的跨语料测试。它和 Talking Turns 论文在原始数据上的 Table 1/Table 4 数字不是同一个测试集。

## 整理后的运行验证

2026-09-03 已从新目录运行：

```powershell
.\.venv\Scripts\python.exe code\scripts\run_main_experiment.py train --device cuda --output-dir artifacts/main_experiment/rerun
```

训练正常完成，生成 9 个结果文件；`aggregate.csv` 和 `profile_deltas.csv` 与固定主结果逐行一致。当前自动测试为 `12 passed`，Talking Turns 保存结果也已重新评分成功。
