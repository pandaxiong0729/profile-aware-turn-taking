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

如果本地存在 `data/sbcsae/transcripts_trn/TRN/SBC041.trn`，命令会解析真实 SBC041 转写和 profile，并根据真实时间戳生成合成单声道音频。克隆后的仓库没有 SBCSAE 时，会使用 `examples/` 中自带的无版权合成 fixture。

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
  --trn data/sbcsae/transcripts_trn/TRN/SBC041.trn `
  --profile ../intro/sbcsae_profile_turntaking_training_example.json `
  --audio data/sbcsae/audio/SBC041.wav `
  --manifest data/processed/SBC041.jsonl `
  --conversation-id SBC041 `
  --split-group speakers-KRISTIN-PAIGE `
  --context-seconds 30 `
  --horizon-ms 40
```

如果不提供 `--audio`，程序会生成 synthetic WAV，但摘要中的 `audio_source` 会明确写成 `synthetic_from_real_timestamps`。

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

当前 `.trn` 时间戳是 intonation-unit 级，不能替代正式 VAD。Smoke run 的目的仅为验证系统。正式大规模实验必须用真实 WAV 重新运行 VAD/overlap detection，并抽查 BC、I 和时间边界。

## 测试

```powershell
.\.venv\Scripts\python -m pytest
```

## 数据许可

仓库不发布 SBCSAE 音频或完整转写。SBCSAE 由原作者以 CC BY-ND 3.0 US 提供，使用时必须遵循官方许可并引用语料。`examples/` 是为测试代码自行编写并合成的 fixture。
