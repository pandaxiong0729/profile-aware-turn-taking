# 云端运行与换模型说明

## 1. 不变的数据契约

模型替换时不要改变 JSONL manifest。每行包含：

```json
{
  "sample_id": "SBC041-0000123",
  "conversation_id": "SBC041",
  "split_group": "speakers-KRISTIN-PAIGE",
  "split": "train",
  "prediction_time_s": 46.46,
  "horizon_ms": 40,
  "window_start_s": 16.46,
  "window_end_s": 46.46,
  "audio_path": "/data/SBC041.wav",
  "transcript_prefix": "[speaker_A ...] ...",
  "profile": {},
  "label": "BC"
}
```

`audio_path + window_start_s + window_end_s` 指向单声道输入窗口。源 WAV 即使是双声道，也会在加载时平均为单声道并重采样到 16 kHz。

## 2. 配置文件

模型和训练参数均来自 JSON：

```json
{
  "model": {
    "audio_backend": "statistical",
    "hidden_dimension": 64
  },
  "train": {
    "epochs": 6,
    "batch_size": 32,
    "profile_dropout": 0.5
  }
}
```

本地使用 `configs/smoke.json`，云端 Whisper 使用 `configs/cloud_whisper.json`。

## 3. 切换 Whisper 型号

只需修改：

```json
{
  "model": {
    "audio_backend": "whisper",
    "whisper_model": "openai/whisper-small",
    "freeze_audio_encoder": true
  }
}
```

可以替换为兼容的 Hugging Face Whisper checkpoint，例如 tiny/base/small/medium。首次运行会下载模型。默认冻结 Whisper，只训练投影层、profile encoder、adapter 和分类头；冻结 backbone 不会重复写入 checkpoint。

## 4. 接入其他音频模型

新增编码器时需要满足一个接口：

```text
输入：batch 单声道 waveform，形状 [B, samples]
输出：context embedding，形状 [B, hidden_dimension]
```

在 `src/profile_turntaking/model.py` 中：

1. 新建 `nn.Module`；
2. 在 `ProfileTurnModel.__init__` 根据新的 `audio_backend` 实例化；
3. 在 `ManifestDataset` 中让该 backend 返回 waveform；
4. 在配置文件中指定 backend 名称。

Profile adapter、分类头、训练和评测不需要修改。

## 5. Profile 编码与未来动态更新

当前 `StructuredProfileEncoder` 为每个固定字段建立独立 embedding。所有 `unknown` 都映射到 padding bucket 0。训练中的 `profile_dropout=0.5` 会随机把整份 profile 设为 unknown，因此同一 checkpoint 可以公平运行 hidden/given。

第二阶段动态更新时保持数据接口不变：

```text
长历史 -> Profile Updater -> 新的 relationship/situation snapshot
```

只替换 manifest 或实时状态中的 `relationship` 和 `situation`；年龄、性别、角色和背景不更新。若要比较自然语言表示，应把同一份结构化 snapshot 按固定模板序列化，避免把 updater 差异与编码方式混在一起。

## 6. 云端推荐运行顺序

1. 下载真实 WAV 并校验时长、采样率和许可。
2. 对每段会话生成独立 manifest。
3. 为共享说话人的会话设置同一 `split_group`。
4. 使用 `merge-manifests` 产生最终 train/val/test。
5. 先用 `configs/smoke.json` 在真实数据上跑一轮，验证数据。
6. 再安装 `[whisper]` 并切换云端配置。
7. 保存 `samples.summary.json`、训练历史和 `profile_comparison.csv`。

## 7. 正式实验必须确认

- 测试集保留自然类别分布；
- 训练集可以平衡抽样；
- 同一说话人不得跨 split；
- ASR 文本只能包含预测时间之前已经可用的内容；
- `Profile hidden/given/shuffled` 使用同一 checkpoint 和相同测试样本；
- synthetic audio 的结果不能作为论文数字；
- `.trn` 标签必须用真实音频 VAD 和人工抽查复核。
