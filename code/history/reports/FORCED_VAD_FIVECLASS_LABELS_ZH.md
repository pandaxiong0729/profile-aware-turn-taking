# SBCSAE VAD＋时间规则强制五分类标注报告

更新时间：2026-07-15

## 结论

16个核心双人会话、共6.4298小时已经全部标完。每40ms恰好有一个 `C/BC/T/I/NA` 标签；不存在空标签、`UNCERTAIN`、声学复核或语义复核状态。旧的7,126条语义候选队列没有参与本次标注，已退出正式流程。

## 自动规则

规则优先级为 `NA > BC > I > T > C`：

1. `NA`：VAD多数票为静音。
2. `BC`：VAD有声，并且时间对齐转写是1.5秒以内的短反馈词，例如 `um/uh/mhm/mm-hm/yeah/right/okay/yes`；BC覆盖重叠规则。
3. `I`：VAD有声，不同说话人的时间区间真实重叠至少40ms，而且没有先被标成BC。整个重叠区间标I。
4. `T`：VAD有声，非BC的新说话人在上一位说话人结束后进入，并且没有达到40ms有效重叠。只把新说话人起始所在的第一个40ms帧标T，后续持续讲话标C。
5. `C`：其余所有VAD有声帧。

15个双声道会话使用左声道、右声道和混合声道三路Silero VAD多数票；单声道 `SBC047` 使用单路结果。转写只提供说话人时间和短词，不产生空标签。

## 全量统计

| 标签 | 40ms帧数 | 时长（秒） | 占比 |
|---|---:|---:|---:|
| C | 346,016 | 13,840.64 | 59.79% |
| BC | 14,041 | 561.64 | 2.43% |
| T | 1,358 | 54.32 | 0.23% |
| I | 26,188 | 1,047.52 | 4.52% |
| NA | 191,083 | 7,643.32 | 33.02% |
| 合计 | 578,686 | 23,147.44 | 100% |

覆盖检查：

- 会话：16/16。
- 已标帧：578,686/578,686。
- 空标签：0。
- 非五分类标签：0。
- review字段：0。
- 帧号断裂：0。
- 时间缺口：0。
- 没有两个定时说话人的I帧：0。
- 连续标签区间与逐帧计数不一致：0。

机器验证结果为 `verified=true`。

## 输出文件

- `data/processed/sbcsae_vad_fiveclass_v2/frame_labels.jsonl.gz`：训练标签主文件，每行对应一个40ms预测时刻。
- `data/processed/sbcsae_vad_fiveclass_v2/label_spans.jsonl`：把相邻同标签帧合并后的连续区间，便于试听和统计。
- `data/processed/sbcsae_vad_fiveclass_v2/summary.json`：总计数、每会话计数和完整规则。
- `data/processed/sbcsae_vad_fiveclass_v2/verification.json`：独立完整性检查。
- `data/processed/sbcsae_vad_fiveclass_v2/examples.jsonl`：每类三个逐帧实例。

逐帧行示例：

```json
{
  "sample_id": "SBC005-frame-0000256",
  "conversation_id": "SBC005",
  "frame_index": 256,
  "prediction_time_s": 10.24,
  "start_s": 10.24,
  "end_s": 10.28,
  "horizon_ms": 40,
  "label": "T",
  "label_id": 2,
  "label_source": "nonoverlap_speaker_change_onset",
  "vad_state": "speech",
  "vad_votes": 3,
  "vad_source_count": 3,
  "active_speakers": ["speaker_00", "speaker_01"],
  "rule_version": "vad_trn_forced_fiveclass_v1"
}
```

`active_speakers`在一个T帧里可能包含两个人，因为两段不重叠的语音可能先后落在同一个40ms格子内；实际重叠不足40ms时仍按已确认规则标T。

## 复现命令

在项目根目录运行：

```powershell
$env:PYTHONPATH='code/src'
.venv\Scripts\python.exe code/scripts/build_vad_fiveclass_labels.py
.venv\Scripts\python.exe code/scripts/verify_vad_fiveclass_labels.py
```

实现位置：

- `code/src/profile_turntaking/vad_fiveclass.py`
- `code/scripts/build_vad_fiveclass_labels.py`
- `code/scripts/verify_vad_fiveclass_labels.py`
- `code/tests/test_vad_fiveclass.py`

