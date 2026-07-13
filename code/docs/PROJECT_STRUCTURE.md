# 项目结构

```text
turn-taking/
├── README.md
├── intro/
│   ├── profile_aware_turn_taking_intro_experiments.md
│   └── sbcsae_profile_turntaking_training_example.json
├── code/
│   ├── configs/                  # 轻量 smoke 与 Whisper 训练配置
│   ├── docs/                     # 数据 schema、项目结构和云端说明
│   ├── examples/
│   │   ├── smoke.trn             # 最小合成测试转写
│   │   ├── smoke_profile.json    # 最小合成 profile
│   │   └── data_preview/         # GitHub 可直接查看的数据/输出示例
│   ├── reports/                  # 数据统计、问题清单和验证报告
│   ├── scripts/                  # 下载与数据预览工具
│   ├── src/profile_turntaking/   # 数据、模型、训练和评测实现
│   └── tests/                    # 自动测试
├── data/                         # 本地数据，Git 忽略
│   ├── sbcsae/
│   │   ├── openslr/
│   │   │   ├── WAV/             # 60 段训练用原始音频
│   │   │   ├── TRN/             # 60 份带时间转写
│   │   │   ├── CHAT/            # 60 份会话 header/参与者信息
│   │   │   └── docs/            # 语料说明
│   │   └── metadata/             # 4 份 profile CSV
│   ├── pachat/
│   │   └── official_site/        # 官方项目页的 4 个 demo
│   └── processed/
│       ├── sbcsae_catalog/       # 全 60 会话的规范化目录
│       ├── sbcsae_mvp/           # 54,270 条五分类 manifest
│       ├── pachat_demo/          # 4 cases / 14 profiles / 29 turns
│       └── audit.json            # 19 项数据审计
└── artifacts/                    # 本地运行产物，Git 忽略
    ├── paired-profile-smoke-final/
    │   ├── model.pt              # 功能 smoke checkpoint
    │   ├── model.train.json      # 每个 epoch 的训练记录
    │   ├── samples.jsonl         # smoke 输入样本
    │   └── evaluation/
    │       ├── metrics.json
    │       ├── predictions.json
    │       └── profile_comparison.csv
    └── data-preview/
        ├── sbcsae/               # 一条真实 SBCSAE 输入/目标与 30 秒 WAV
        ├── pachat/               # 一条官方 demo、profile 和 WAV
        └── training_output/      # 一条训练后预测和聚合结果示例
```

## 从原始数据到结果

```text
WAV + TRN + CHAT + metadata
        ↓ prepare-sbcsae-corpus
sbcsae_catalog
        ↓ prepare-sbcsae-manifests
manifest.jsonl + split_map.json + weak_events.jsonl
        ↓ train（一次）
model.pt + model.train.json
        ↓ evaluate（同一 checkpoint、同一 test sample IDs）
hidden / given / shuffled
        ↓
metrics.json + predictions.json + profile_comparison.csv
```

## 一个 SBCSAE manifest row

一行不是一整段对话，而是一个预测时刻：

- 音频输入：`[t-30s, t]` 的单声道窗口；
- 文本输入：在 `t` 之前已经结束的 transcript；
- profile 输入：Speaker A/B、relationship、situation；
- 目标输出：下一个 40 ms 的 `C / BC / T / I / NA` 弱标签。

`data/processed/sbcsae_mvp/manifest.jsonl` 有 54,270 行。使用 `scripts/export_data_preview.py --sample-id ...` 可以把任意一行变成容易阅读的 JSON 和可直接试听的 30 秒 WAV。

## 训练后的文件

| 文件 | 含义 |
| --- | --- |
| `model.pt` | 模型参数和配置，可被 `evaluate` 重新加载 |
| `model.train.json` | train/val 样本数、每个 epoch loss 和验证指标 |
| `metrics.json` | hidden/given/shuffled 的 Macro-F1、Balanced Accuracy、每类指标和混淆矩阵 |
| `predictions.json` | 每个 test sample 在三种 profile 条件下的 target 与 argmax prediction |
| `profile_comparison.csv` | 最适合放进实验表的三条件摘要与 `given-hidden` 差值 |

当前 `paired-profile-smoke-final` 只是功能 smoke。全量 SBCSAE 模型训练后应输出同样的目录结构，但其中的 checkpoint 和指标才是正式实验候选。

## GitHub 中可直接审阅的示例

`code/examples/data_preview/` 是刻意保留在 Git 中的小型审阅包。它包含一条 schema-faithful 的脱敏合成 SBCSAE manifest、独立 profile、4 秒合成音频、一条不含音频的 PaChat JSON，以及 smoke 的训练历史、17 条逐样本预测和 profile 比较表。

这个目录只展示接口和输出格式。真实 SBCSAE、PaChat 音频、checkpoint 和全量 manifest 仍由 `.gitignore` 排除。
