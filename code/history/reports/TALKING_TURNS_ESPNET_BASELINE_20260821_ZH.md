# Talking Turns 官方基线与本地对照

这份报告只回答一件事：我们把 Talking Turns 的公开模型跑在 SBCSAE 测试集上，再用同样的二分方式换算，和我们自己的结果放在一起看，作为 baseline 对照。

## 1. 这次跑的是什么模型

我们下载并运行的是 Talking Turns 论文公开发布的 ESPnet checkpoint：

- 代码来源：`models/espnet/source-cea64ab`
- 模型权重：`models/espnet/Turn_taking_prediction_SWBD/exp/asr_train_asr_whisper_turn_taking_raw_en_word/valid.loss.ave.pth`
- 官方模型卡：`espnet/Turn_taking_prediction_SWBD`

这个模型是论文里的公开基线，不是我们自己训练的模型。

## 2. 这次怎么测

我们保留了 SBCSAE 的同一批测试样本、同一段因果音频窗口和同一套五类标签：

- `C`
- `BC`
- `T`
- `I`
- `NA`

然后把官方模型的五类输出，按 Talking Turns 的二分口径换算成四个问题：

- turn change: `C` vs `T`
- backchannel: `C` vs `BC`
- interruption: `C` vs `I`
- floor taking: `A` vs `B`

其中 `floor taking` 的二分判定仍按我们前面已经对齐好的 source kind 口径来算，不改样本，不改音频，不改转写，不改边界，只改 profile 时才是 profile 实验；这次官方 baseline 不看 profile。

## 3. 官方模型在 SBCSAE 上的结果

五分类结果：

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.3199 |
| Balanced Accuracy | 0.1992 |
| Macro-F1 | 0.1816 |
| Macro OVR ROC-AUC | 0.5548 |

逐类 one-vs-rest ROC-AUC：

| C | BC | T | I | NA | 平均 |
|---:|---:|---:|---:|---:|---:|
| 0.3518 | 0.5605 | 0.6162 | 0.6796 | 0.5661 | 0.5548 |

注意：这 1,938 条是事件中心测试样本，不是对连续音频每 40 ms 全覆盖的 Table 1 严格复现。完整口径说明见 `code/reports/TALKING_TURNS_OOD_ON_SBCSAE_20260903_ZH.md`。

四个二分类结果：

| 任务 | Accuracy | Balanced Accuracy | Macro-F1 | ROC-AUC | paper-balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| turn change | 0.5015 | 0.4886 | 0.4654 | 0.4665 | 0.4981 |
| backchannel | 0.7294 | 0.4552 | 0.4571 | 0.4289 | 0.4444 |
| interruption | 0.8717 | 0.5381 | 0.5446 | 0.5322 | 0.5484 |
| floor taking | 0.4696 | 0.5211 | 0.4655 | 0.5730 | 0.5147 |

四个任务的 paper-balanced Accuracy 平均值是 `0.5014`。

## 4. 我们自己的结果

### 4.1 Qwen audio-only A/B baseline

这是我们不加 profile 的音频-only 基线，和官方模型一样走二分口径。

| 任务 | paper-balanced Accuracy |
|---|---:|
| turn change | 0.7603 |
| backchannel | 0.7721 |
| interruption | 0.8853 |
| floor taking | 0.5245 |
| 平均 | 0.7355 |

对应文件：

- `artifacts/qwen-audio-only-ab-baseline/paper-aligned-v1/summary.json`

### 4.2 Qwen shared A/B adapter

这是我们加入 profile 的主实验。三种 profile 条件在同一测试集上比较：

| 条件 | 平均 paper-balanced Accuracy |
|---|---:|
| hidden | 0.7401 |
| given | 0.7383 |
| shuffled | 0.7414 |

对应文件：

- `artifacts/qwen-shared-ab-30s-causal/paper-aligned-floor/history120-floorprop-profile-margin-final/gate_profile_margin_0p25/summary.json`

## 5. 该怎么理解

这个 baseline 说明两件事：

1. 官方 Talking Turns 模型在我们这份 SBCSAE 测试集上是能正常跑通的，但它的五分类和二分表现都不高，尤其五分类很弱。
2. 我们的 Qwen 音频-only 基线已经明显高于这个官方公开模型；而 profile adapter 这条线目前在 SBCSAE 上没有稳定证明 `given > hidden > shuffled`，所以它现在更像一个方法验证，而不是已经完成的 profile 证据。

## 6. 可复核文件

- 官方 baseline 推理与汇总：`artifacts/espnet-talking-turns-baseline/sbcsae-test-v1/`
- 官方 baseline 逐样本结果：`artifacts/espnet-talking-turns-baseline/sbcsae-test-v1/test_predictions.jsonl`
- 官方 baseline 指标：`artifacts/espnet-talking-turns-baseline/sbcsae-test-v1/metrics.json`
- 我们的音频-only baseline：`artifacts/qwen-audio-only-ab-baseline/paper-aligned-v1/summary.json`
- 我们的 profile adapter：`artifacts/qwen-shared-ab-30s-causal/paper-aligned-floor/history120-floorprop-profile-margin-final/gate_profile_margin_0p25/summary.json`
