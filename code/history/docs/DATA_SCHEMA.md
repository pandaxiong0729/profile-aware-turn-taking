# 数据 schema 与输入输出

## 处理链路

```text
SBCSAE TRN + CHAT + metadata CSV + WAV
    -> sbcsae_catalog/{conversations, utterances, issues}.jsonl
    -> speaker-connected train/val/test split
    -> sbcsae_mvp/{manifest, weak_events}.jsonl
    -> train one checkpoint
    -> evaluate the same sample IDs as hidden / given / shuffled
```

## `conversations.jsonl`

每行对应一段 SBCSAE 会话。关键字段：

| 字段 | 含义 |
| --- | --- |
| `conversation_id` | `SBC001` 等稳定 ID |
| `duration_s` | `.trn` 时间轴末端 |
| `participants` | header 参与者、stable speaker UID、profile 和匹配来源 |
| `declared_dyadic` | CHAT header 是否声明两位人类说话人 |
| `observed_dyadic` | 有效 `.trn` 中是否实际出现两位人类说话人 |
| `core_dyadic` | 是否符合第一阶段双人训练筛选 |
| `relationship` / `situation` | 会话说明的规则映射 |
| `split_group` | 共享说话人的会话连通组 |
| `audio_path` / `audio_info` | 本地 WAV 路径和编码信息 |

## `utterances.jsonl`

每行是一个有效的带时间 intonation unit：

```json
{
  "conversation_id": "SBC005",
  "start_s": 12.34,
  "end_s": 13.10,
  "speaker": "speaker_00",
  "speaker_uid": "sbcsae:metadata1.csv:17",
  "is_person": true,
  "text": "..."
}
```

环境、集合和动物标签可以保留在 catalog 中，但不进入 core dyadic A/B 训练 utterances。

## `manifest.jsonl`

每行是一个独立预测点，模型输入只允许读取 `window_end_s` 之前的信息。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `sample_id` | string | `conversation_id + prediction_time_ms`，全局唯一 |
| `conversation_id` | string | 来源会话 |
| `split_group` | string | speaker-connected component |
| `split` | `train/val/test` | 固定划分 |
| `prediction_time_s` | float | 预测时刻 `t` |
| `horizon_ms` | int | 当前为 40 ms |
| `window_start_s` / `window_end_s` | float | 音频窗口，当前为 `[t-30s, t]` |
| `audio_path` | string | 本地 WAV 绝对路径 |
| `audio_channel_policy` | string | 立体声在读取时取均值 |
| `transcript_prefix` | string | 仅包含在 `t` 前已经结束的人工 TRN units |
| `text_source` | string | 明确标记文本是 streaming ASR 代理 |
| `profile` | object | A/B 结构化 profile、relationship、situation |
| `profile_provenance` | object | profile 和上下文映射来源 |
| `label` | string | `C/BC/T/I/NA` |
| `label_source` | string | 自动弱标签规则版本 |
| `gold_label` | boolean | 当前始终为 `false` |

### Profile 对象

```json
{
  "speaker_A": {
    "age_group": "25-34",
    "gender": "female",
    "social_role": "teacher",
    "background": "education=BA; home_state=CA; current_state=CA; ethnicity=white"
  },
  "speaker_B": {
    "age_group": "35-44",
    "gender": "male",
    "social_role": "engineer",
    "background": "education=MA; home_state=OR; current_state=CA; ethnicity=white"
  },
  "relationship": "friends_or_peers",
  "situation": "casual_social_conversation"
}
```

Speaker A 是该会话中第一个实际出现的人类，Speaker B 是另一个人。`hidden` 使用相同字段但所有值为 `unknown`。`shuffled` 对一整段会话使用另一段测试会话的完整 profile，不逐帧改变。

## 模型输入和输出

单个训练样本的输入：

- 单声道 30 秒音频窗口；
- 截止当前时刻的文本前缀；
- 10 个结构化 profile fields（A/B 各 4 个，加 relationship、situation）。

模型输出为五个 logits/probabilities，标签顺序由 `constants.py` 的 `LABELS` 唯一规定。训练使用一个五分类 Cross Entropy。当前代码不伪造七分类：后续只有在人工细标可用后，才增加 `NA -> PAUSE/GAP` 和 `I -> I_COOP/I_COMP` 两个条件头。

## `weak_events.jsonl`

把相邻同类 40 ms 帧合并成事件：

```json
{
  "conversation_id": "SBC005",
  "split": "test",
  "start_s": 100.0,
  "end_s": 100.24,
  "label": "BC",
  "source": "automatic_weak_chunk_state_from_trn_timestamps_v2",
  "gold_label": false
}
```

它可用于人工抽样和开发 event-level 评测，但仍不是人工 gold。

## `event_manifest.jsonl`

每个 `weak_events.jsonl` 事件只保留一个接近中点、且确实位于原 40 ms
时间网格上的代表帧，并增加 `weak_event_id`、`weak_event_start_s`、
`weak_event_end_s` 和 `event_representative=true`。低成本 prompt 验证从这个
文件抽样，避免把同一段连续 BC/I/NA 当成多条独立测试数据。自动标签完成
人工复核前，`gold_label` 仍为 `false`。

`event_onset_manifest.jsonl` 使用相同事件，但把预测边界放在事件起始
网格点，`event_representative_policy="onset"`。低成本“预测”验证优先使用
这份文件，使 BC/I 等目标尚未出现在模型输入音频中；中点版本只用于状态
识别诊断。
