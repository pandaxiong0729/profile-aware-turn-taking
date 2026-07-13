# Reviewable data and output preview

这个目录可以直接提交到 GitHub，目的是让审阅者不用下载 SBCSAE，也能看懂一条输入如何进入模型、目标标签是什么，以及训练后输出长什么样。

## 文件

| 文件 | 内容 | 数据状态 |
| --- | --- | --- |
| `sbcsae_manifest_sample.json` | 一条完整 manifest row：时间窗、因果 transcript、profile 和五分类目标 | schema-faithful、完全脱敏、文本为合成 |
| `profile_sample.json` | 与 manifest 中一致的独立结构化 profile | 完全合成，无个人数据 |
| `synthetic_input.wav` | 与示例时间轴对应的短单声道音频 | 代码生成，16 kHz PCM，不来自 SBCSAE |
| `pachat_demo_sample.json` | 一个按官方项目页内容改写的 case/profile/turn schema 示例 | 不含音频；文本已改写；官方 demo 仓库未声明数据许可 |
| `smoke_training_history.json` | 每个 epoch 的 loss 和验证指标 | 功能 smoke，不是论文结果 |
| `smoke_predictions.csv` | 同一批 17 个 test samples 的 hidden/given/shuffled argmax 预测 | 功能 smoke，不是论文结果 |
| `profile_comparison.csv` | 三种 profile 条件的摘要与 given-hidden 差值 | 功能 smoke，不是论文结果 |

## 一条样本怎样读

`sbcsae_manifest_sample.json` 中：

```text
输入 = synthetic_input.wav 的 [0.00s, 2.10s]
     + 2.10s 之前已经结束的 transcript_prefix
     + profile

目标 = 2.10s 之后 40 ms 的 BC
```

正式实验的音频窗口是 30 秒；这里为了控制仓库体积而缩短。字段名和 profile 结构与正式 manifest 一致，但 conversation ID、文本、profile 和音频都是合成内容，不是 SBCSAE 原始数据。

## 输出怎样读

`smoke_predictions.csv` 每行是同一个 test sample 在三种条件下的预测：

- `hidden_prediction`：所有 profile 字段为 `unknown`；
- `given_prediction`：正确 profile；
- `shuffled_prediction`：错误 profile 负控制。

这三列来自同一个 checkpoint。由于 smoke 只有一段合成会话，`shuffled` 安全退化为 hidden，所以这些数值只能验证代码链路，不能用于论文结论。

## 没有放进 GitHub 的内容

- 6.67 GB SBCSAE WAV；
- 54,270 行真实弱标签 manifest；
- PaChat demo 音频；
- smoke checkpoint；
- 任何全量正式训练结果。

完整语料由仓库中的预处理命令在本地生成，数据统计和限制见 `code/reports/`。
