# SBCSAE：VAD 预标与语音模型语义标注状态

更新时间：2026-07-15

> **状态更新：** 本文原先的“高置信/声学复核/语义复核/7,126候选”是已经退出主流程的保守方案。当前16个会话已按确认后的确定性规则实现100%逐帧五分类，正式结果请以 `FORCED_VAD_FIVECLASS_LABELS_ZH.md` 和 `data/processed/sbcsae_vad_fiveclass_v2/` 为准；旧候选队列不得用于训练。

## 一句话结论

VAD＋时间规则的强制五分类已经完成：578,686个40ms帧全部得到 `C/BC/T/I/NA`，覆盖率100%，不含任何复核或空标签。Qwen试验仅保留为历史诊断，不参与本轮弱标签生成。

## 1. 当前处理的数据

- 数据集：SBCSAE（Santa Barbara Corpus of Spoken American English）。
- 当前实验范围：16 个核心双人会话，共 6.4298 小时。
- 会话：`SBC005/006/007/009/010/017/024/029/034/041/043/044/045/047/058/060`。
- 15 个会话是双声道；`SBC047` 原始文件是单声道。
- 时间分辨率：40 ms 一帧，与 Talking Turns 的设置一致。

## 2. VAD 实际做了什么

使用 `silero-vad 6.2.1 ONNX` 检测物理上的“有人说话/无人说话”。双声道文件分别检测左声道、右声道和混合声道，只有三路一致时才允许自动判定；单声道 `SBC047` 保留单路来源，不能伪造三路一致性。

VAD 只负责它能确定的部分：

- `C`：一致检测到语音、时间信息只对应一个说话人，并且不靠近短回应、换人、重叠或边界。
- `NA`：一致检测到静音，并且远离 VAD 边界。
- `BC/T/I`：一律不由 VAD 直接赋值，进入语义复核。

| 结果 | 40 ms 帧数 | 占比 |
|---|---:|---:|
| 高置信自动标签 | 271,848 | 46.98% |
| 需要声学复核 | 83,549 | 14.44% |
| 需要语义复核 | 223,289 | 38.59% |
| 合计 | 578,686 | 100% |

高置信自动标签中：`NA=145,143` 帧，`C=126,705` 帧。这些仍叫“自动银标签”，正式使用前应抽样人工审计误差率，不能称为人工金标签。

## 3. 语义事件队列

第一版曾把邻近事件合并成长区间，最长超过二十秒，可能令模型一次听到多个目标。该问题已修复：现在每条记录只锚定一个短回应、换人起点或重叠起点，同时保留事件前 12 秒和后 8 秒上下文。

- 事件总数：7,126。
- 目标事件长度：最短 0.04 秒，中位数 1.821 秒，90 分位 2.36 秒，最长 2.5 秒。
- 候选组：`BC/I=3,243`，`BC/C/T=2,643`，`BC/T=1,240`。
- 触发原因可重叠：短发言 6,174，换人 4,168，双人重叠 3,243。

这里的“候选组”只用于抽样和排队，不作为给模型的现成答案。

## 4. Qwen-Omni 试标怎么测

模型：本地 `Qwen2.5-Omni-3B`，`Q4_K_M` 量化，通过 llama.cpp 音频接口运行。

旧提示方式让模型直接输出标签，真实测试出现了明显矛盾：例如选择 `T`，理由却说“简短反馈、不占话轮、原说话人继续”，按定义应更接近 `BC`。因此没有继续全量运行。

修正版先让模型分别听目标前、目标和目标后三段音频，并进行了以下检查：

1. 将每个样本明确拆成连续的 `BEFORE / TARGET / AFTER` 三段原始音频。
2. 发现双声道简单平均可能发生相位抵消：60条中有7条混合后能量低于最响声道的一半，最严重只剩约24.5%。现已改成相位安全的响度保持混合，并增加反相立体声测试。
3. 第一轮不提供 profile、转写、候选标签或旧弱标签；第二轮增加不含标签的时间对齐说话人转写，帮助模型定位目标说话人。
4. 同一事件使用两种问法，只有两次同标签、高置信且都不要求复核，才允许成为“模型银标签”。
5. 另测试了简化输出，只要求 `label/confidence/reason`，避免让3B模型同时填写过多逻辑字段。

截至2026-07-15的真实结果：音频事实版60个事件、120次请求全部成功，但银标签为0；修复声道混合后对同一60条重跑，仍为0。加入部分转写后模型开始输出多类标签，但事实字段仍矛盾；把任务简化成直接判断后，前10次有9次塌缩为相同的 `T` 和重复理由。因此失败瓶颈已定位为本机 `3B Q4 + llama.cpp实验性音频接口` 的判断质量，而不是VAD、事件时间或空音频。当前不能用它全量自动标7,126条。

## 5. 图片中的“最好模型”到底是什么

图片来自 Talking Turns 的 Table 4。表里的数值是四个平衡二分类数据集上的准确率，不是完整五分类 Macro-F1。

- `Supervised Topline` 才是表中最好的方法：Turn Change 78.6、Backchannel 75.1、Interruption 74.9、Floor-Taking Interruption 65.6。
- 它不是一个可直接提示的通用大模型。论文方法是：Whisper-medium 音频编码器、隐藏层加权、最后音频帧接线性五分类头，在 Switchboard 上用监督标签训练。
- SALMONN、Qwen2-Audio-Instruct/Chat、Whisper+GPT-4o 在大多数任务上接近随机水平。这正说明不能默认“更大的通用语音模型就能直接标好 turn-taking”。

官方实现与权重：

- 论文页面：https://machinelearning.apple.com/research/talking-turns
- ESPnet 实现：https://github.com/espnet/espnet/pull/5948
- 预训练权重：https://huggingface.co/espnet/Turn_taking_prediction_SWBD

## 6. 接下来应该怎么做

建议按以下顺序，不直接让任何单一模型制造“金标签”：

1. 保留当前 VAD 高置信 `C/NA`，先人工分层抽查约 200 条，估计自动标签误差率。
2. 在 Linux/云端部署论文官方 `Turn_taking_prediction_SWBD` checkpoint，对 16 个会话每 40 ms 输出五类概率。
3. 用该 checkpoint 对 7,126 个语义候选排序和给出第一教师意见，但不直接当真值，因为它的 BC 定义本身也带有启发式噪声，而且训练域是 Switchboard。
4. 本机 3B Qwen 不再全量运行。若有更大的全精度音频模型，只在候选事件上使用当前“三段音频＋事实问题”协议，并先做 50 条人工金标对照；达不到门槛就停。
5. 自动接受条件建议为：专用模型高置信、语音模型双问一致、规则无冲突；其余交给人工。对 `BC/I` 和单声道 `SBC047` 提高人工复核比例。
6. 最终测试集必须人工确认，并报告双人标注一致性；模型生成标签只能用于扩大训练集，不能同时拿来当测试真值。

这样做能把人工量集中在真正需要语义判断的部分，同时避免让 VAD 或当前 3B Qwen 伪造可靠性。

## 7. 文件位置

- 汇总：`data/processed/sbcsae_vad_v1/summary.json`
- VAD 区间：`data/processed/sbcsae_vad_v1/vad_segments.jsonl`
- 逐帧状态：`data/processed/sbcsae_vad_v1/frame_annotations.jsonl.gz`
- 高置信区间：`data/processed/sbcsae_vad_v1/high_confidence_spans.jsonl`
- 语义复核队列：`data/processed/sbcsae_vad_v1/semantic_review_queue.jsonl`
- Qwen 事实检查 pilot：`artifacts/mllm-annotation/qwen2.5-omni-3b/vad-semantic-pilot-60-v2/`
- 相位安全重跑：`artifacts/mllm-annotation/qwen2.5-omni-3b/vad-semantic-pilot-60-v3-robust-mix/`
- 音频＋转写直接判断：`artifacts/mllm-annotation/qwen2.5-omni-3b/vad-semantic-pilot-60-v5-simple-judgement/`
- VAD 代码：`code/src/profile_turntaking/vad_annotation.py`
- Qwen 语义标注代码：`code/src/profile_turntaking/mllm_annotation.py`
- 命令入口：`code/scripts/build_vad_annotations.py`、`code/scripts/run_mllm_annotation.py`
