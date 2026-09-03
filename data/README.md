# 数据说明

## 1. 原始数据怎么看

`data/sbcsae/openslr/` 保存下载的 SBCSAE 语料：

- `WAV/SBCxxx.wav`：原始录音；
- `TRN/SBCxxx.trn`：带时间戳的转写；
- `CHAT/SBCxxx.cha`：参与者和会话信息；
- `docs/`：语料官方说明。

`data/sbcsae/metadata/` 保存用于建立 speaker/profile 映射的元数据。

全语料共 60 段、约 23.29 小时。当前主实验使用其中 16 段、约 6.43 小时。剩余 44 段并非没有下载或没有处理：42 段不是严格双人会话，另 2 段分别因为一位说话人占比低于 5% 和 `speaker_audience` 场景而没有进入当前双人协议。

## 2. 当前主实验数据

主目录：`data/processed/sbcsae_qwen_shared_ab_30s_causal_v1/`

总计 10,804 条事件：train 6,623、val 2,243、test 1,938。每一条样本对应一个预测点，不是一整段对话，也不是连续每 40 ms 重复取样。

每个 split 都有四个文件：

| 文件 | 一行代表什么 | 用途 |
| --- | --- | --- |
| `selected_inputs.jsonl` | 一条样本的录音、30 秒窗口和预测时刻 | 确定输入音频 |
| `requests.jsonl` | 同一样本的 hidden/given/shuffled 三行描述 | 审计三条件只有 profile 不同，并保存因果转写 |
| `reference_labels.jsonl` | 五分类标签及四个 A/B 目标 | 训练和评分 |
| `input_audit.json` | 该 split 的数量、时间和哈希检查 | 开跑前核对 |

根目录的 `summary.json` 保存三组数量和会话划分。

## 3. 一条主实验样本到底输入什么

以 `SBC007-event-0001917` 为例：

```text
预测边界：原录音 30.600 秒
音频输入：[0.600, 30.600) 的 30 秒音频
转写输入：只包含 30.600 秒以前已经结束的转写单元
profile 输入：与该样本对齐的 59 维动态交互状态
预测对象：边界后 100 ms 开始、随后 500 ms 内的话轮事件
标签：五类参考标签 + 适用的四个 A/B 目标
```

直接查看任意一条：

```powershell
.\.venv\Scripts\python.exe code\scripts\inspect_experiment_data.py --split test --index 0
```

按 ID 查看并导出实际 30 秒输入音频：

```powershell
.\.venv\Scripts\python.exe code\scripts\inspect_experiment_data.py `
  --split test `
  --sample-id SBC007-event-0001917 `
  --extract-audio artifacts\data-preview\SBC007-event-0001917.wav
```

## 4. profile 在哪里

自然语言人物资料保存在 `requests.jsonl` 的 `profile_text`，用于看原始资料。当前主 adapter 真正读取的是：

`artifacts/main_experiment/profile_features/{train,val,test}.profile-view.npz`

每行 59 维：当前说话人 16 维历史行为、另一位参与者 16 维、双方互动 6 维、relationship/situation 类别 21 维。字段顺序在 `artifacts/main_experiment/profile_features/metadata.json`。

## 5. 事件标签源

- `sbcsae_turn_events_v3/ipus.jsonl`：合并后的连续发声单元；
- `sbcsae_turn_events_v3/event_candidates.jsonl`：16,144 个 C/BC/T/I/NA 事件；
- `sbcsae_turn_events_v3/summary.json`：统计；
- `sbcsae_turn_events_v3/review.html`：事件查看页。

主实验从这 16,144 个事件中按会话划分并选择 10,804 条。`sbcsae_vad_fiveclass_v2/` 是早期逐 40 ms 完整时间轴，当前 adapter 不用它训练，只在标签审计中交叉核对。

## 6. Talking Turns 的每条输入

Talking Turns 对照没有单独复制一份数据。它读取主实验 test split 的同一个 `selected_inputs.jsonl` 和 `reference_labels.jsonl`：

```text
输入：完全相同的 30 秒因果音频
不输入：转写、profile
原生输出：C / BC / T / I / NA 五类概率
对比输出：再按固定规则换算为四个 A/B 任务
```

因此两种模型的测试样本 ID、音频窗口和参考标签可以逐条对齐。

## 7. 云端路径

代码优先使用仓库相对位置。JSON 中少量 `C:\Users\xiong\...` 是生成时留下的来源记录；当前查看器和 Talking Turns runner 在该路径不存在时会自动转到 `data/sbcsae/openslr/WAV/<conversation_id>.wav`。

只用缓存重新训练主 adapter 时，不读取这些原始音频路径。
