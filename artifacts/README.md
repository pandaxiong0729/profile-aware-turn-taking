# 实验产物说明

## 主实验 `main_experiment/`

### `qwen_feature_cache/`

| 文件 | 内容 |
| --- | --- |
| `train.qwen-hidden.npz` | 6,623 条训练样本的 Qwen context `[2048]`、33 层音频末端表示 `[33,1280]` 和四任务标签 |
| `val.qwen-hidden.npz` | 2,243 条验证特征 |
| `test.qwen-hidden.npz` | 1,938 条测试特征 |
| `*.meta.json` | 各 split 的来源、shape 和哈希 |
| `metadata.json` | 三个缓存的总说明 |

三份 NPZ 是云端快速重训最重要的数据，合计约 748.5 MiB。

### `profile_features/`

| 文件 | 内容 |
| --- | --- |
| `train.profile-view.npz` | train 的 given/shuffled 59 维向量 |
| `val.profile-view.npz` | val 的 given/shuffled 59 维向量 |
| `test.profile-view.npz` | test 的 given/shuffled 59 维向量 |
| `metadata.json` | 59 个字段、词表、归一化统计和因果历史范围 |

`hidden` 不单独存文件，运行时把同形状 profile 向量置零。

### `results/`

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 五个随机种子的完整汇总 |
| `aggregate.csv` | 每任务 hidden/given/shuffled 指标表 |
| `profile_deltas.csv` | given-hidden、given-shuffled 差值 |
| `test_predictions.jsonl` | 每个 seed、任务、profile 条件、样本的 A/B 概率 |
| `seed-3/13/37/71/101.json` | 每个随机种子的训练和评测详情 |

### `audio_only_baseline/`

同架构关闭转写和 profile 后的 Qwen audio-only 对照。它和主结果分开，不能当成 hidden 条件；hidden 仍保留音频与转写，只把 profile 置零。

## Talking Turns `talking_turns/sbcsae_test/`

| 文件 | 内容 |
| --- | --- |
| `test_predictions.jsonl` | 官方 checkpoint 对 1,938 条 SBCSAE test 样本的五类概率 |
| `binary_predictions.csv` | 换算后的四个 A/B 任务逐样本结果 |
| `metrics.json` | 五分类、ROC-AUC、四任务 Accuracy |
| `confusion_matrix.csv` | 五分类混淆矩阵 |
| `input_audit.json` | 音频窗口、输入模态、checkpoint 和输出顺序检查 |

## `data-preview/`

少量可直接打开的示例，不是正式结果。用于在 GitHub 上展示数据和输出格式。

## `history/`

保存旧 prompt、旧 adapter、调参搜索、smoke、诊断和中间缓存。当前命令不会读取这里。确认不需要追溯后可整体删除。
