# 重新实验前的数据验收与实验协议

日期：2026-07-14

## 1. 当前结论

这次准备运行的是 **现成音频 MLLM 的低成本五分类 prompt 验证**，不是正式
profile adapter 训练。

- 数据/代码结构验收：通过，自动审计没有发现 ID 重复、标签器不一致、
  Speaker/Profile 错配、音频错配或 split 泄漏。
- 科学标签验收：尚未通过。当前 onset 候选集的 500 条目标仍全部是自动弱
  标签，必须听取包含 `t` 后证据的 annotation-only 音频后复核。
- 模型推理：尚未开始。代码已经增加硬门禁，未复核标签默认不能运行或评分。
- 正式 adapter 实验：尚未就绪；还缺 speaker-aware VAD/diarization 和真正的
  streaming-ASR prefix。

因此现在不能说“数据已经是 gold”，但可以明确说：已知的软件映射错误已经
修复，剩余问题被审计器显式列出，不会再被当作正式结果。

## 2. 这轮实验究竟回答什么

研究问题是：在完全相同的历史音频和历史转写上，加入正确的静态 profile，
是否能让同一个现成 MLLM 更准确地预测下一 40 ms 的五分类话轮状态？

模型不训练、不微调。500 个事件各发送三次，共 1,500 个请求：

| 条件 | Profile 内容 | 作用 |
|---|---|---|
| `hidden` | 所有信息不可用 | 无 profile 基线 |
| `given` | 当前会话正确 profile | 测量正确 profile 的增量 |
| `shuffled` | 另一会话的完整 profile | 检查模型是否只受额外文字影响 |

三次请求的 sample ID、音频、转写、预测边界、任务说明、输出 schema、模型、
temperature 和 seed 完全相同；只替换 profile 文本。

## 3. 数据在哪里

| 内容 | 本机位置 |
|---|---|
| 原始 WAV | `data/sbcsae/openslr/WAV/` |
| 原始 TRN 时间戳转写 | `data/sbcsae/openslr/TRN/` |
| 原始 CHAT comment/参与者 | `data/sbcsae/openslr/CHAT/` |
| 原始 profile metadata | `data/sbcsae/metadata/` |
| 修复后的 60 会话 catalog | `data/processed/sbcsae_catalog_v2/` |
| 逐 40 ms 弱标签与事件 | `data/processed/sbcsae_mvp_v2/` |
| 全部事件起点 manifest | `data/processed/sbcsae_mvp_v2/event_onset_manifest.jsonl` |
| 当前会话平衡的 500 条 | `data/processed/sbcsae_mvp_v2/prompt_review_balanced_500.jsonl` |
| 500 条待复核请求 | `artifacts/mllm-prompt-baseline/qwen2.5-omni-3b/onset-balanced-500-review-required/` |
| 深度审计 | 上述目录的 `preflight_audit.json` |
| 人工复核页面 | 上述目录的 `review.html` |

仓库不会上传完整 SBCSAE 音频、转写或生成的 30 秒 clips。

## 4. 500 条是怎样得到的

1. 从 60 个 SBCSAE 会话建立 catalog。
2. 只保留 16 个核心双人会话。
3. 在每个 40 ms 网格点生成 `C/BC/T/I/NA` 候选弱标签。
4. 把相邻同类帧合并为 10,555 个连续事件。
5. 每个事件生成两种代表点：
   - `midpoint_grid`：事件中点，适合“识别正在发生的状态”；
   - `onset`：事件起始网格点，适合“预测下一块将出现什么”。
6. 第一版从每类随机抽 100 条，但审计发现 NA 的 54% 来自 SBC024、37% 来自
   SBC029；静态 profile 因而可能充当“会话 ID→类别先验”，该候选集已在推理
   前拒绝。
7. 当前固定随机种子 13，选择 C/BC/T/I 各 110 条、NA 60 条，共 500 个不同
   `weak_event_id`。每个会话每类最多 10 条，同一会话的任意两个预测边界至少
   相隔 5 秒。C/BC/T/I 均覆盖 16 会话，NA 覆盖 13 会话；单个会话在任何类中
   的最大占比为 16.7%。

使用全部 16 个会话不会造成模型训练泄漏，因为这个 checkpoint 没有在
SBCSAE 上训练。后续 adapter 训练仍必须使用 speaker-connected train/val/test。

## 5. 一条请求的准确输入和输出

预测边界为 `t`。

### 模型输入

1. **音频**：`[t-30s,t)`，16 kHz 单声道 WAV。原 WAV 为双声道时取双声道
   均值。本轮没有 `t` 之后的声音。
2. **转写**：只包含 `end_s <= t` 的人工 TRN 单元，带 `speaker_A/B` 和起止
   时间；不包含未来单元。
3. **Profile**：固定模板自然语言，不是 embedding。字段包括双方年龄段、
   数据集性别、职业/社会角色、背景、relationship 和 situation。

当前文本是“已经结束的人工转写单元”，不是 streaming ASR。它满足因果性，
但会省略在 `t` 时仍未结束的当前单元，因此只能称为低成本 proxy。正式
adapter 必须重新生成实际 streaming-ASR prefix。

### 模型输出

只允许一个硬标签 JSON：

```json
{"label": "C|BC|T|I|NA"}
```

标签表示 `[t,t+40ms]` 这一块的状态：

| 标签 | 当前实验定义 |
|---|---|
| `C` | 一位参与者继续持有当前话轮 |
| `BC` | 听者短 backchannel 出现，但另一方保持话轮 |
| `T` | 话轮在该块转移给另一位参与者 |
| `I` | 双方同时说话且重叠不是 backchannel |
| `NA` | 两位参与者都没有说话 |

第一轮只做五分类。`PAUSE/GAP` 和 `I_COOP/I_COMP` 没有人工细标前不进入本轮。

## 6. Profile 如何形成

Speaker A/B 按该会话中两位人类参与者首次出现的顺序固定，再通过 local
speaker ID 连接到 metadata。模型不接收姓名。

16 个核心会话的 relationship/situation 已逐一根据官方 CHAT `@Comment`
复核，并记录为 `manual_core_chat_comment_review_v1`。本次复核纠正了两类典型
关键词错误：

- `SBC029` 是工程师给住户报价，应为 `professional_client`，不是同事；
- `SBC043` 是母女，应为 `family`，不是同事。

审计结果中，500/500 的 profile 与 canonical participant 映射一致；正确
profile 与 shuffled profile 没有一条完全相同。部分原 metadata 职业字段本身
是缩写或截断文本，这会作为数据来源限制保留，不由模型或代码猜测补全。

## 7. 当前提示词

实际模板由 `build_audio_prompt()` 生成。花括号部分按每条数据替换：

```text
You are a strict audio turn-taking event classifier.
The attached {duration}-second mono conversation audio contains only
information available before the prediction boundary and ends exactly at time t.
Classify the conversational state during the next 40-millisecond chunk,
[t, t+40 ms].

Labels:
- C: exactly one participant is speaking and keeps the current floor.
- BC: a short listener backchannel is present while the other participant keeps the floor.
- T: exactly one participant is speaking after the floor transfers to that participant in this chunk.
- I: both participants speak in this chunk and the overlap is not a backchannel.
- NA: neither participant speaks in this chunk.

Use audible speech activity, pauses, overlap, turn-final prosody, and backchannel cues.
The recording is mono and may contain both participants; do not invent a speaker identity.
The transcript below contains completed causal transcript units; every listed
unit ended no later than time t.
Use it together with the audio; do not infer or invent future words.

Completed causal transcript units available before time t:
{causal_transcript}

Profile condition:
{hidden_or_given_or_shuffled_profile}

Return exactly one JSON object with the single key "label".
Its value must be exactly one of C, BC, T, I, NA.
```

Prompt 中没有目标标签示例、未来文本或 annotation evidence。模型调用使用
`temperature=0`、`seed=13`、`max_tokens=16` 和严格 JSON schema。

## 8. 已完成的数据验收

`preflight_audit.json` 当前记录：

| 检查 | 结果 |
|---|---:|
| event manifest 行数 / 唯一 event ID | 10,555 / 10,555 |
| 选中样本 | 500，覆盖 16 会话 |
| 候选类别 | C/BC/T/I 各 110；NA 60 |
| 类别覆盖的会话数 | C/BC/T/I 各 16；NA 13 |
| 单会话最大类别占比 | 16.7%（NA） |
| 同会话最小预测边界间隔 | 5.08 秒 |
| 重新计算标签与 manifest 不一致 | 0 |
| Speaker/Profile 映射不一致 | 0 |
| 源音频映射不一致 | 0 |
| speaker-group split 泄漏 | 0 |
| conversation split 泄漏 | 0 |
| given 与 shuffled 完全相同 | 0 |
| 请求中未来转写或 target 泄漏 | 0 |
| 三条件非 profile 字段变化 | 0 |

仍需人工复核的风险：

- 500/500 仍是弱标签；
- 81 个 onset 窗口含活动的非词汇人声单元，例如呼吸/笑声标记；
- 24 个窗口同时存在环境或非人物标注，其中 22 个是 NA 候选；
- 只有 2 个 `I` 候选在 onset 边界前已经存在重叠，需要重点改时点或改标签；
- 当前转写不是 streaming ASR。

复核页面播放的是 `[t-3s,t+2s]` annotation-only 音频，并明确显示边界；其中
`t` 后音频只用于人工确定 gold，不会写入 `requests.jsonl`。页面同时显示边界
附近、以 `t` 为零点的 annotation-only 转写，并把非词汇人声和环境单元等
高风险条目排在前面；这些字段只存在于 `review_items.json/review.html`。

## 9. 本轮实际测评什么

| 内容 | 指标/检查 |
|---|---|
| 五分类总体 | Macro-F1（主指标）、Balanced Accuracy、Accuracy（次要） |
| 每类 | C/BC/T/I/NA 的 Precision、Recall、F1、support |
| 错误结构 | 五分类混淆矩阵 |
| Profile 增益 | `given-hidden`、`given-shuffled` Macro-F1 差值 |
| 成对显著性 | given 对 hidden 的 exact McNemar |
| 不确定性 | 按 `conversation_id` 聚类的 2,000 次 bootstrap 95% CI |
| 模型是否塌缩 | hidden 至少预测 3 类，最大类别占比不超过 80% |
| 是否使用音频 | hidden 原音频与等长静音音频的成对诊断 |
| 工程性能 | 单请求 mean/median/p95 latency、有效输出率 |

本轮不报告 Brier、ECE、ROC-AUC，因为模型只输出硬标签而非五类概率；也不
报告 ±200 ms event-level F1，因为本轮只抽事件起点，没有解码完整时间线。
这些指标在正式 streaming adapter 输出逐帧概率后计算。

支持“profile 有帮助”的最低证据预先规定为：

1. hidden 基线不塌缩且对音频敏感；
2. `given` Macro-F1 同时高于 `hidden` 和 `shuffled`；
3. `given-hidden` 会话 bootstrap 95% CI 不跨 0；
4. exact McNemar 不支持“改善与破坏数量相同”；
5. 改善不能只来自 C，必须查看 BC/T/I/NA 每类结果。

任何一项失败，都只能报告“当前 checkpoint 未提供可靠正向证据”，不能说
profile 普遍无效。

## 10. 正式运行顺序

1. 在 `review.html` 完成 500 条 onset 标签复核，`U` 样本重新裁决。
2. 导出 `reviewed_labels.json`，生成固定的 `reviewed_500.jsonl`。
3. 重新运行 preflight；只有 `prompt_pilot_ready=true` 才继续。
4. 从 reviewed set 按类取 10 条，先跑 50 样本/150 请求 pilot。
5. 检查 JSON 有效率、hidden 塌缩和静音音频敏感性；失败则停止，不跑 500。
6. 门禁通过后，对固定 500 sample ID 跑 1,500 个请求。
7. 生成 metrics、混淆矩阵、配对变化、bootstrap CI、延迟和错误案例。
8. 报告中同时写清弱点：小模型、量化、单声道、TRN proxy 与会话数限制。

## 11. 代码位置

| 功能 | 文件 |
|---|---|
| SBCSAE/metadata/profile 规范化 | `src/profile_turntaking/sbcsae_corpus.py` |
| 五分类弱标签和 onset manifest | `src/profile_turntaking/sbcsae_manifest.py` |
| 三输入 prompt、审计、推理和门禁 | `src/profile_turntaking/mllm_prompt_baseline.py` |
| Profile 固定模板和评分 | `src/profile_turntaking/prompt_baseline.py` |
| 深度数据 preflight | `src/profile_turntaking/experiment_preflight.py` |
| 人工复核与 reviewed manifest | `src/profile_turntaking/label_review.py` |
| 主命令 | `scripts/run_mllm_prompt_baseline.py` |
| 复核命令 | `scripts/review_labels.py` |
| 深度审计命令 | `scripts/audit_prompt_pilot_data.py` |

当前自动化测试为 45/45 通过。
