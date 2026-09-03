# 项目结构

> 2026-09-03：最新、最简的合作结构和主实验命令已统一到 [`../../collaboration/README.md`](../../collaboration/README.md)。本文件以下内容保留历史探索路径，复现正式结果请以合作入口为准。

## 当前 Qwen 第三步结果

- 最终汇报：`code/reports/QWEN2_5_OMNI_PROFILE_STEP3_FINAL_REPORT_20260731_ZH.md`
- 完整使用说明：`code/docs/QWEN25_OMNI_EVENT_EVAL.md`
- 结果网页：`artifacts/qwen25-omni-profile/q8-v11/review.html`
- 两组实验：`artifacts/qwen25-omni-profile/q8-v11/gate50-paper-binary-calibrated-seed137/` 和 `gate50-paper-binary-calibrated-seed237/`
- 合并统计：`artifacts/qwen25-omni-profile/q8-v11/combined_99_summary.json`
- 输入与因果 ASR：`code/src/profile_turntaking/qwen25_omni_event_eval.py`
- 四个二问与五分类映射：`code/src/profile_turntaking/paper_binary_hierarchy.py`
- 统一命令入口：`code/scripts/run_qwen25_omni_event_eval.py`
- 结果页生成器：`code/scripts/build_qwen_binary_frontend.py`

```text
turn-taking/
├── README.md
├── intro/
│   ├── profile_aware_turn_taking_intro_experiments.md
│   └── sbcsae_profile_turntaking_training_example.json
├── code/
│   ├── configs/                  # smoke、Whisper 与锁定的 prompt pilot 协议
│   ├── docs/                     # 数据 schema、项目结构和云端说明
│   ├── examples/
│   │   ├── smoke.trn             # 最小合成测试转写
│   │   ├── smoke_profile.json    # 最小合成 profile
│   │   └── data_preview/         # GitHub 可直接查看的数据/输出示例
│   ├── reports/                  # 数据统计、问题清单和验证报告
│   │   ├── EXPERIMENT_PRESTART_REVIEW.md # 当前实验协议与开跑前验收结论
│   │   ├── LABEL_AUDIT_AND_V2_REBUILD.md # 弱标签问题与 v2 重建记录
│   │   └── MLLM_PROMPT_QWEN2_5_OMNI_3B_REPORT.md # 已作废旧运行的问题记录
│   ├── scripts/                  # 数据准备、审计、复核与实验入口
│   │   ├── audit_prompt_pilot_data.py # 深度检查输入、标签、profile 和 split
│   │   ├── select_prompt_review_set.py # 按会话平衡并隔开预测边界
│   │   ├── verify_prompt_protocol.py # 拒绝数据、prompt 或运行参数漂移
│   │   ├── review_labels.py     # 生成/导入 50/500 条人工复核页面
│   │   └── run_mllm_prompt_baseline.py # 音频+因果转写+profile 零训练基线
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
│       ├── sbcsae_catalog_v2/    # 修正转写清洗、说话人及关系映射后的目录
│       ├── sbcsae_mvp_v2/        # v2 五分类帧、事件及 onset manifest
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
    ├── mllm-prompt-baseline/     # 三路输入的 hidden/given/shuffled prompt 实验
    │   └── qwen2.5-omni-3b/
    │       ├── onset-balanced-500-review-required/ # 当前候选集；尚未运行推理
    │       │   ├── audio_clips/  # 只到预测时刻 t 的因果 WAV
    │       │   ├── review_clips/ # 仅供人工标注、含 t 后信息；绝不进模型
    │       │   ├── requests.jsonl # hidden/given/shuffled 三条件请求
    │       │   ├── gold.jsonl     # 当前保存自动候选标签；复核前不是正式 gold
    │       │   ├── review.html   # 500 条人工复核页面
    │       │   ├── input_audit.json
    │       │   └── preflight_audit.json
    │       ├── onset-balanced-lowrisk-50-review-required/ # 第一阶段人工复核门禁
    │       │   ├── review.html   # 五类各 10 条；先复核这 50 条
    │       │   ├── requests.jsonl # 150 个待审 paired requests；尚未推理
    │       │   └── preflight_audit.json
    │       ├── onset-500-review-required/ # 会话/profile 分布审计未通过
    │       └── audio-transcript-profile-test-100-per-class/ # 已作废旧运行
    └── data-preview/
        ├── sbcsae/               # 一条真实 SBCSAE 输入/目标与 30 秒 WAV
        ├── pachat/               # 一条官方 demo、profile 和 WAV
        └── training_output/      # 一条训练后预测和聚合结果示例
```

## 从原始数据到结果

低成本 prompt 验证（本轮）与正式 adapter 训练共用 v2 数据底座，但不是同一条
实验流水线：

```text
WAV + TRN + CHAT + metadata
        ↓ prepare-sbcsae-corpus
sbcsae_catalog_v2
        ↓ prepare-sbcsae-manifests
manifest.jsonl + event_manifest.jsonl + event_onset_manifest.jsonl + split_map.json

prompt 验证：event_onset_manifest.jsonl
        ↓ select_prompt_review_set（会话平衡、边界间隔至少 5 秒）
prompt_review_balanced_500.jsonl
        ↓ 固定低风险五类各 10 条
prompt_pilot_lowrisk_50.jsonl
        ↓ review_labels + pilot50 protocol gate
reviewed_pilot_50.jsonl → 50 条模型/静音门禁
        ↓ 仅在门禁通过后，导入 50 条已有复核并完成剩余 450 条
reviewed_500.jsonl + full500 protocol gate
        ↓ 同一现成 MLLM 推理；不训练
hidden / given / shuffled
        ↓
metrics.json + predictions.json + bootstrap_95ci.json

正式 adapter：manifest.jsonl + speaker-connected split
        ↓ streaming ASR/VAD 数据补齐后 train（一次）
model.pt + model.train.json
        ↓ evaluate（同一 checkpoint、同一 test sample IDs）
hidden / given / shuffled
        ↓
metrics.json + predictions.json + profile_comparison.csv
```

## 一个 SBCSAE manifest row

一行不是一整段对话，而是一个预测时刻：

- 音频输入：`[t-30s, t)` 的单声道窗口；
- 文本输入：在 `t` 之前已经结束的 transcript；
- profile 输入：Speaker A/B、relationship、situation；
- 目标输出：下一个 40 ms 的 `C / BC / T / I / NA` 弱标签。

当前权威的完整弱标签时间轴位于 `data/processed/sbcsae_vad_fiveclass_v2/frame_labels.jsonl.gz`：16个核心双人会话的578,686个40ms帧全部具有 `C/BC/T/I/NA` 标签，不含review或空值。`label_spans.jsonl` 提供连续区间，`verification.json` 证明100%覆盖。旧 `semantic_review_queue.jsonl` 与旧500条prompt候选只保留作历史诊断，不得作为当前训练标签来源。

`review_clips/` 是标注工具专用音频，允许包含预测点之后最多 2 秒，以便人判断事件类型；`requests.jsonl` 引用的 `audio_clips/` 严格止于预测点，两者不可混用。

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
