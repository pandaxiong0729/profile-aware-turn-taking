# Profile-aware Turn-taking MVP

这是一个可本地跑通、可迁移到云端的五分类话轮预测工程。模型在每个预测点只读取截止当前时刻的单声道音频、已经完成的转写前缀和结构化 profile，并预测下一 40 ms 的：

```text
C / BC / T / I / NA
```

第一实验只训练一个 checkpoint。训练时随机隐藏 profile；评测时对同一测试集分别运行：

- `hidden`：所有 profile 字段设为 `unknown`；
- `given`：使用正确 profile；
- `shuffled`：使用被打乱的 profile 作为负控制。

核心结果写入 `profile_comparison.csv`，直接对应研究文档中的五分类评测表。

这里的主比较始终是同一个模型、同一个 checkpoint 和同一批样本，只改变 profile 条件。Talking Turns 等公开模型可以作为可选骨干或外部参考，但不替代这个 paired ablation。

## 当前实现

- 解析 SBCSAE `.trn` 时间戳并映射 Speaker A/B；
- 自动生成五分类候选标签；
- 只把在预测点前已经结束的转写放入输入，防止未来文字泄漏；
- 支持真实 PCM WAV，也支持根据真实时间戳生成确定性合成单声道音频；
- 结构化 field embedding profile encoder；
- gated residual profile adapter；
- 带类别权重的 Cross Entropy；
- Macro-F1、Balanced Accuracy、每类 Precision/Recall/F1 和混淆矩阵；
- 同 checkpoint 的 hidden/given/shuffled 成对评测；
- 本地统计音频编码器和可选 Hugging Face Whisper 编码器；
- 多会话 manifest 按 `split_group` 切分，避免同一说话人连通组跨训练和测试。

## 快速安装

先进入代码目录：

```powershell
cd code
```

建议使用 Python 3.10-3.13。Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Linux/macOS：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

## 一条命令跑通

```powershell
.\.venv\Scripts\profile-turntaking smoke --work-dir artifacts/smoke
```

如果本地存在 `data/sbcsae/openslr/TRN/SBC041.trn`，命令会解析真实 SBC041 转写和 profile，并根据真实时间戳生成合成单声道音频。克隆后的仓库没有 SBCSAE 时，会使用 `examples/` 中自带的无版权合成 fixture。

若要强制使用自带 fixture：

```powershell
.\.venv\Scripts\profile-turntaking smoke --bundled-fixture
```

输出目录：

```text
artifacts/smoke/
├── samples.jsonl
├── samples.summary.json
├── smoke.wav
├── model.pt
├── model.train.json
└── evaluation/
    ├── metrics.json
    ├── predictions.json
    └── profile_comparison.csv
```

注意：单会话 smoke run 会为保证每类都能被测试而采用样本级分层切分，只能证明代码链路可运行，不能作为论文结果。

## 使用真实 SBCSAE 音频

SBCSAE 官方页面提供转写、时间戳以及 WAV 下载入口：

- UCSB：https://www.linguistics.ucsb.edu/research/santa-barbara-corpus-spoken-american-english
- TalkBank：https://talkbank.org/ca/access/SBCSAE.html
- OpenSLR 完整镜像：https://www.openslr.org/155/

将下载的 WAV 放到 `data/sbcsae/audio/`，然后执行：

```powershell
profile-turntaking prepare-sbcsae `
  --trn ../data/sbcsae/openslr/TRN/SBC041.trn `
  --profile ../intro/sbcsae_profile_turntaking_training_example.json `
  --audio ../data/sbcsae/openslr/WAV/SBC041.wav `
  --manifest ../data/processed/SBC041.jsonl `
  --conversation-id SBC041 `
  --split-group speakers-KRISTIN-PAIGE `
  --context-seconds 30 `
  --horizon-ms 40
```

如果不提供 `--audio`，程序会生成 synthetic WAV，但摘要中的 `audio_source` 会明确写成 `synthetic_from_real_timestamps`。

## 全量 SBCSAE 预处理

下面命令从 OpenSLR 下载完整归档并可断点续传。命令假设当前位于 `code/`，下载数据不会进入 Git：

```powershell
New-Item -ItemType Directory -Force ../data/sbcsae/archives | Out-Null
python scripts/download_resumable.py `
  --url https://us.openslr.org/resources/155/SBCSAE.tar.gz `
  --output ../data/sbcsae/archives/SBCSAE.tar.gz `
  --expected-bytes 6186324893 `
  --workers 8

New-Item -ItemType Directory -Force ../data/sbcsae/openslr | Out-Null
tar -xf ../data/sbcsae/archives/SBCSAE.tar.gz -C ../data/sbcsae/openslr
```

先规范化全 60 段会话、metadata、说话人和 profile：

```powershell
profile-turntaking prepare-sbcsae-corpus `
  --trn-dir ../data/sbcsae/openslr/TRN `
  --chat-dir ../data/sbcsae/openslr/CHAT `
  --metadata-dir ../data/sbcsae/metadata `
  --audio-dir ../data/sbcsae/openslr/WAV `
  --output-dir ../data/processed/sbcsae_catalog
```

再从 16 段 core dyadic 会话建立 speaker-disjoint manifest：

```powershell
profile-turntaking prepare-sbcsae-manifests `
  --catalog-dir ../data/processed/sbcsae_catalog `
  --output-dir ../data/processed/sbcsae_mvp `
  --context-seconds 30 `
  --horizon-ms 40 `
  --frame-stride-ms 40 `
  --evaluation-stride-ms 200 `
  --max-train-per-class 5000 `
  --max-evaluation-per-class 5000 `
  --seed 13
```

如果已经取得官方 PaChat 项目页源码，可以单独规范化 demo；它不会进入 turn-taking 训练：

```powershell
profile-turntaking prepare-pachat-demo `
  --site-dir ../data/pachat/official_site/<checkout-directory> `
  --output-dir ../data/processed/pachat_demo
```

最后执行跨产物审计：

```powershell
profile-turntaking audit-preprocessed `
  --sbcsae-catalog-dir ../data/processed/sbcsae_catalog `
  --sbcsae-manifest ../data/processed/sbcsae_mvp/manifest.jsonl `
  --pachat-demo-dir ../data/processed/pachat_demo `
  --output ../data/processed/audit.json
```

本次完整统计见 [reports/DATA_PREPARATION_REPORT.md](reports/DATA_PREPARATION_REPORT.md)，异常与未解决限制见 [reports/DATA_PREPARATION_ISSUES.md](reports/DATA_PREPARATION_ISSUES.md)，字段定义见 [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md)，目录与数据流见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)。

## 文本大模型 Prompt 辅助基线

在正式训练前，可以不更新任何模型参数，先把同一批验证/测试样本分别组织成 `hidden / given / shuffled` profile prompt，调用现成文本大模型做五分类。代码会把真实 API 请求与 gold 文件分离，并支持断点续跑和成对评分：

```powershell
python scripts/run_prompt_baseline.py prepare `
  --manifest ../data/processed/sbcsae_mvp/manifest.jsonl `
  --output-dir ../artifacts/prompt-baseline/val-20 `
  --split val `
  --max-per-class 20
```

完整的接口调用、样本规模和结果判断说明见 [docs/PROMPT_BASELINE.md](docs/PROMPT_BASELINE.md)。这是看不到音频声学信息的文本 prompt baseline，不替代正式模型训练。

本机 Qwen3-4B 的 500 条正式测试结果见 [reports/PROMPT_BASELINE_QWEN3_4B_REPORT.md](reports/PROMPT_BASELINE_QWEN3_4B_REPORT.md)。该实验得到负/不确定结果：`given` Macro-F1 为 0.1957，低于 `hidden` 的 0.2371；这说明文本 profile 会改变预测，但尚未证明它能稳定提升 40 ms 话轮判断。

## 音频 + 因果转写 + Profile MLLM 低成本验证

更接近当前研究问题的 prompt 实验使用现成音频 MLLM，每个请求同时输入 `[t-30s, t]` 单声道 WAV、截止 `t` 已完成的部分转写和 fixed-template 自然语言 profile，输出未来 40 ms 的五分类。每个样本做 `hidden / given / shuffled` 三条件对照；自动审计保证三次请求的音频、转写、预测边界和任务提示完全相同，只改变 profile：

```powershell
python scripts/run_mllm_prompt_baseline.py prepare `
  --manifest ../data/processed/sbcsae_mvp/manifest.jsonl `
  --output-dir ../artifacts/mllm-prompt-baseline/qwen2.5-omni-3b/audio-transcript-profile-test-100-per-class `
  --split test `
  --max-per-class 100
```

完整的模型服务、审计、断点续跑和评分命令见 [docs/MLLM_PROMPT_BASELINE.md](docs/MLLM_PROMPT_BASELINE.md)。本机 Qwen2.5-Omni-3B Q4 的 500 条平衡测试结果见 [reports/MLLM_PROMPT_QWEN2_5_OMNI_3B_REPORT.md](reports/MLLM_PROMPT_QWEN2_5_OMNI_3B_REPORT.md)：1,500/1,500 请求有效，`given` accuracy 比 `hidden` 高 0.8 个百分点，但 Macro-F1 更低、配对检验不显著，而且 hidden 有 89% 输出为 `I`。因此该 checkpoint 没有提供可信的正向 profile 证据。

## 查看一条真实数据和训练输出

如果只想在 GitHub 上直接查看一个无需下载语料的示例，打开 [`examples/data_preview/`](examples/data_preview/)。其中包含脱敏合成 manifest、profile、短音频、PaChat 文本 demo 和 smoke 输出；每个文件都明确标注是否为合成数据以及是否可以作为研究结果。

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe code\scripts\export_data_preview.py
```

程序会在 `artifacts/data-preview/` 导出：

- 一条真实 SBCSAE test 样本的 30 秒单声道 WAV、因果 transcript、完整 profile 和五分类目标标签；
- 一条 PaChat 官方 demo 的 WAV、case、自然语言/结构化 profile 和 turn 文本；
- 最后一次 smoke 的训练历史、同一样本 `hidden/given/shuffled` 预测、聚合指标和比较表。

默认选择 test split 的第一条 `I`。查看指定样本时传入 manifest 中的 `sample_id`：

```powershell
.\.venv\Scripts\python.exe code\scripts\export_data_preview.py `
  --sample-id SBC058-000716560 `
  --output-dir artifacts/data-preview-SBC058
```

这只复制一个 30 秒窗口，不会复制整段 20 分钟录音。PaChat demo 是逐 turn 独立音频，因此预览中没有 turn-taking target；训练输出仍是功能 smoke，不是全量 SBCSAE 论文结果。

## 多会话云端数据

每段会话先生成独立 manifest。具有共同说话人的会话必须传入相同的 `--split-group`。完成后统一合并并重新划分：

```powershell
profile-turntaking merge-manifests `
  --inputs data/processed/SBC005.jsonl data/processed/SBC007.jsonl data/processed/SBC009.jsonl `
  --output data/processed/all.jsonl
```

少于三个 `split_group` 时命令会拒绝生成所谓“科学划分”。

## 训练和评测

本地轻量配置：

```powershell
profile-turntaking train `
  --manifest data/processed/all.jsonl `
  --checkpoint artifacts/run/model.pt `
  --config configs/smoke.json

profile-turntaking evaluate `
  --manifest data/processed/all.jsonl `
  --checkpoint artifacts/run/model.pt `
  --output-dir artifacts/run/evaluation
```

`evaluate` 只加载一次指定 checkpoint，并用相同 `sample_id` 顺序依次生成 `hidden / given / shuffled` 三组预测。训练中的 profile dropout 与评测 `hidden` 都使用完全相同的全 `unknown` profile 编码；`shuffled` 按会话整体换成另一会话的 profile。

Whisper 云端配置：

```powershell
python -m pip install -e ".[whisper]"
profile-turntaking train `
  --manifest data/processed/all.jsonl `
  --checkpoint artifacts/whisper/model.pt `
  --config configs/cloud_whisper.json
```

更详细的换模型、配置和云端运行说明见 [docs/CLOUD_AND_MODEL_SWAP.md](docs/CLOUD_AND_MODEL_SWAP.md)。

## 标签规则与限制

五类自动标签优先级为：

1. 没有说话活动：`NA`；
2. 短反馈词且不取得话轮：`BC`；
3. 两位说话人同时发声且不是 BC：`I`；
4. 新说话人在当前 40 ms 开始并从上一说话人接过话轮：`T`；
5. 其他情况：`C`。

当前 `.trn` 时间戳是 intonation-unit 级，不能替代正式 VAD。全量 manifest 也是自动弱标签，不是人工 gold。正式大规模实验必须用真实 WAV 重新运行 VAD/overlap detection，并抽查 BC、I 和时间边界。

## 测试

```powershell
.\.venv\Scripts\python -m pytest
```

## 数据许可

仓库不发布 SBCSAE 音频或完整转写。SBCSAE 由原作者以 CC BY-ND 3.0 US 提供，使用时必须遵循官方许可并引用语料。`examples/` 是为测试代码自行编写并合成的 fixture。
