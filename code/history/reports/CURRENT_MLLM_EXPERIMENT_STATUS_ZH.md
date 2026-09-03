# 现有多模态大模型话轮预测实验：当前情况说明

更新时间：2026-07-14

## 一、先说结论

我们现在做的是：**评测一个现成的音频多模态大模型，在“音频 + 历史转写”之外加入人物 profile，能不能把下一时刻的话轮状态预测得更准。**

此前已经真实跑完 **500 条数据**，每条分别测试三种 profile 条件，共
**1,500 次模型请求**。本报告以这次 500 条结果为主体。后来又跑的低风险
50 条只用于复现和排查问题，不替代 500 条历史结果。

500 条历史运行的结果是：

- 当前使用的 Qwen2.5-Omni-3B 量化模型，五分类能力很差；
- 加入正确 profile 后没有变准，反而略差；
- 正确 profile 和打乱 profile 的表现几乎相同；
- 模型主要把样本预测成 `I`（重叠/打断），说明它没有真正理解 profile 的含义；
- 因此，这次结果只能说明 **这个 3B checkpoint 和当时的数据/提示词组合不适合当前任务**，不能说明“profile 思路无效”。

还必须注意：500 条运行完成后，我们发现它的数据标签、抽样和 prompt 定义
存在系统性问题，所以这些数字是“真实运行过的历史结果”，但已经被正式标记
为无效，不能当作论文最终结果。

我们不能继续在同一批 50 条上反复修改提示词，直到 profile 结果变好。那样属于在测试集上调参，结果不能相信。

## 二、我们现在评测的到底是什么

当前任务不是训练 profile adapter，也不是训练一个新模型。

当前任务是一个低成本验证：

> 找一个已经训练好的、能够听音频的多模态大模型，直接通过 prompt 让它预测话轮类别，然后比较有无 profile 的区别。

研究问题是：

> 在音频和历史转写完全相同的情况下，给模型正确的说话人 profile，是否比不给 profile、给错误 profile 更准确？

## 三、数据集是什么

### 3.1 原始数据

使用的数据集是 **Santa Barbara Corpus of Spoken American English（SBCSAE）**。

本机原始数据位置：

| 内容 | 本机位置 | 作用 |
|---|---|---|
| WAV 音频 | `data/sbcsae/openslr/WAV/` | 模型听到的真实双人对话 |
| TRN 转写 | `data/sbcsae/openslr/TRN/` | 带时间区间和说话人的人工转写 |
| CHAT 文件 | `data/sbcsae/openslr/CHAT/` | 会话标题、参与者和场景说明 |
| Profile metadata | `data/sbcsae/metadata/` | 年龄、性别、职业、教育等人物资料 |

完整 SBCSAE 有 60 个会话。当前 prompt 实验只使用其中 16 个比较可靠的核心双人会话。

### 3.2 处理后的数据

| 内容 | 本机位置 |
|---|---|
| 修复后的标准化语料 | `data/processed/sbcsae_catalog_v2/` |
| 全部五分类帧和事件 | `data/processed/sbcsae_mvp_v2/` |
| 10,555 个不同事件起点 | `data/processed/sbcsae_mvp_v2/event_onset_manifest.jsonl` |
| 会话平衡的 500 条候选 | `data/processed/sbcsae_mvp_v2/prompt_review_balanced_500.jsonl` |
| 本次实际运行的 50 条 | `data/processed/sbcsae_mvp_v2/prompt_pilot_lowrisk_50.jsonl` |

### 3.3 五个标签是什么意思

模型预测的是预测边界 `t` 之后的 40 ms，即 `[t,t+40ms]`：

| 标签 | 简单解释 |
|---|---|
| `C` | 当前说话人继续说 |
| `BC` | 听者说“嗯、对、mhm”等简短反馈，但没有抢走话轮 |
| `T` | 话轮从一个人转移给另一个人 |
| `I` | 两个人同时说，且不是简短 backchannel |
| `NA` | 两个人都没有说话 |

### 3.4 当前标签是否完全可靠

还不是。

这些标签主要根据 TRN 的时间区间自动生成，所以叫“弱标签”，不是人工 gold label。已经修复的程序错误包括：

- 以前错误删除方括号里的真实词，例如 `[Mhm]`；
- 以前会把先后发生、没有真正重叠的两句话误判为 `I`；
- 以前部分粘连的 TRN 行没有正确恢复；
- 以前 T 的判断会受文件行顺序影响；
- 以前从事件中间取样，BC/I 已经发生后才让模型预测。

现在上述软件错误已经修复，重新计算标签、音频、Speaker A/B、profile 和数据划分的自动审计均为 0 错误。

但 TRN 是 intonation-unit 时间区间，不是精确 VAD，所以最终正式结论仍需要人工复核。目前 500 条人工复核数量是 0/500。

## 四、为什么没有直接从 500 条里随便抽 50 条

第一版 500 条虽然每类 100 条，但后来发现 NA 有 91% 来自 SBC024 和 SBC029。

这会产生一个严重问题：profile 在一个会话内不变，模型可能把 profile 当作“会话 ID”，然后猜这个会话经常出现什么标签。这样得到的 profile 提升是假的。

因此第一版 500 条已废弃。当前 500 条重新按会话平衡：

- C、BC、T、I 各 110 条；
- NA 60 条；
- 每个会话每个类别最多 10 条；
- 同一会话的预测边界至少相隔 5 秒；
- C/BC/T/I 都覆盖 16 个会话，NA 覆盖 13 个会话。

本次实际运行的 50 条又做了一层更严格筛选：

- 五类各 10 条；
- 覆盖全部 16 个会话；
- 一个“会话 × 类别”最多 1 条；
- 同一会话的边界至少相隔 9.24 秒；
- 排除了所有自动发现的非词汇人声、环境声音和 BC/I 边界风险样本。

注意：你浏览器中当前打开的
`artifacts/.../onset-500-review-required/review.html` 是已经被会话分布审计淘汰的旧页面，不要继续标它。

当前正确的 500 条复核页面是：

`artifacts/mllm-prompt-baseline/qwen2.5-omni-3b/onset-balanced-500-review-required/review.html`

## 五、模型是什么

当前实际运行的模型：

| 项目 | 内容 |
|---|---|
| 模型 | Qwen2.5-Omni-3B |
| 参数规模 | 约 3B |
| 模型量化 | Q4_K_M |
| 音频 projector | Q8_0 mmproj |
| 推理框架 | llama.cpp server |
| 本机 GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8 GB 显存 |
| 解码 | temperature=0，seed=13，max_tokens=16 |

模型文件：

- `models/huggingface/Qwen2.5-Omni-3B-GGUF/Qwen2.5-Omni-3B-Q4_K_M.gguf`
- `models/huggingface/Qwen2.5-Omni-3B-GGUF/mmproj-Qwen2.5-Omni-3B-Q8_0.gguf`

## 六、模型每次输入什么

设预测时间为 `t`。每条请求包含三个输入：

### 6.1 音频

- `[t-30s,t)` 的 30 秒音频；
- 16 kHz；
- 单声道；
- 原音频如果是双声道，就取两个声道的平均；
- 绝对不包含 `t` 后的声音。

### 6.2 历史转写

- 只输入 `end_s <= t`、在 `t` 前已经结束的人工 TRN 单元；
- 保留 `speaker_A / speaker_B` 和时间；
- 不输入未来词语；
- 目前是人工 TRN 的因果代理，不是真实 streaming ASR。

### 6.3 Profile

Profile 目前按固定模板转成自然语言，不是 embedding。

字段包括：

- Speaker A：年龄段、性别、职业/社会角色、背景；
- Speaker B：年龄段、性别、职业/社会角色、背景；
- relationship；
- situation。

relationship 和 situation 已根据 16 个会话的官方 CHAT comment 人工复核过。

## 七、三种实验条件

同一条样本发送三次：

| 条件 | 输入的 profile |
|---|---|
| `hidden` | profile 全部不可用 |
| `given` | 当前会话的正确 profile |
| `shuffled` | 另一个会话的完整错误 profile |

三次请求的音频、转写、预测边界、prompt、输出格式、模型和解码参数完全相同，只改变 profile 文本。

因此：

- `given > hidden`：说明加入 profile 可能有帮助；
- `given > shuffled`：说明帮助来自正确 profile 的内容，而不只是多了一段文字；
- given 和 shuffled 一样：说明模型只是受“存在 profile 文本”影响，没有理解正确内容。

## 八、提示词是什么样

实际提示词由下面的代码生成：

`code/src/profile_turntaking/mllm_prompt_baseline.py` 中的 `build_audio_prompt()`。

核心内容可以简单理解为：

```text
你是一个严格的音频话轮事件分类器。
音频只包含预测时间 t 之前的信息，并在 t 结束。
请预测接下来 40 ms 是 C、BC、T、I、NA 中的哪一类。

请结合：
- 音频中的说话、静音、重叠、句尾语调和 backchannel 线索；
- 截止 t 已结束的历史转写；
- 当前 profile 条件。

不能编造未来词语。
只返回一个 JSON，例如标签字段的值必须是 C/BC/T/I/NA 之一。
```

正式锁定的完整 prompt 没有给任何目标标签示例，也没有未来音频、未来转写或人工标注证据。

第一次运行发现模型只预测 T/I。为了排查原因，我们在同一 50 条开发集上只做过一次 prompt 修正：要求主要关注音频最后 500 ms，不要把 30 秒历史中更早的重叠当成目标重叠。这个修正只用于诊断，没有被当成新的正式结果。

## 九、怎么测评

主要指标：

- Macro-F1：五类同等重要，当前最主要指标；
- Balanced Accuracy；
- 普通 Accuracy，只作为辅助；
- 每类 Precision、Recall、F1、support；
- 五分类混淆矩阵；
- `given-hidden` Macro-F1 差值；
- `given-shuffled` Macro-F1 差值；
- given 与 hidden 的 exact McNemar 成对检验；
- 按 conversation 聚类的 2,000 次 bootstrap 95% CI；
- 模型是否只预测少数类别；
- 原音频换成静音后预测是否变化；
- 输出有效率和延迟。

本轮模型只输出一个硬标签，所以不计算 ROC-AUC、Brier、ECE。也没有解码完整时间线，因此不计算 ±200 ms event-level F1。

## 十、之前 500 条实验的真实结果（本报告主体）

### 10.1 运行规模

- 样本数：500；
- 当时按旧标签每类 100 条；
- 每条运行 hidden、given、shuffled 三次；
- 总请求数：1,500；
- 1,500/1,500 都得到可解析输出；
- 运行时间约 22.3 分钟。

### 10.2 总体指标

| 条件 | Macro-F1 | Balanced Accuracy | Accuracy | 正确数/500 |
|---|---:|---:|---:|---:|
| hidden | 0.0823 | 0.1940 | 0.1940 | 97 |
| given | 0.0803 | 0.2020 | 0.2020 | 101 |
| shuffled | 0.0746 | 0.2000 | 0.2000 | 100 |

表面上看，given 的普通 Accuracy 比 hidden 高 0.8 个百分点。但主要指标
Macro-F1 反而从 0.0823 降到 0.0803，因此不能说正确 profile 提高了性能。

### 10.3 每类 F1

| 条件 | C | BC | T | I | NA |
|---|---:|---:|---:|---:|---:|
| hidden | 0.0774 | 0 | 0 | 0.3339 | 0 |
| given | 0.0650 | 0 | 0 | 0.3362 | 0 |
| shuffled | 0.0323 | 0 | 0 | 0.3409 | 0 |

BC、T、NA 三类的 F1 全是 0。模型只有 I 类看起来有一定 F1，但这是因为它
几乎把所有数据都猜成 I。

### 10.4 模型到底预测了什么

| 条件 | C | BC | T | I | NA | 最大类别占比 |
|---|---:|---:|---:|---:|---:|---:|
| hidden | 55 | 0 | 0 | 445 | 0 | I=89.0% |
| given | 23 | 0 | 0 | 477 | 0 | I=95.4% |
| shuffled | 24 | 0 | 1 | 475 | 0 | I=95.0% |

这说明 hidden 基线已经严重塌缩。加入正确 profile 和加入错误 profile 后，
模型都更倾向于猜 I；正确 profile 并没有表现出独立作用。

### 10.5 成对变化和显著性

- hidden 和 given 有 66/500 条预测不同；
- hidden 和 shuffled 有 64/500 条预测不同；
- given 和 shuffled 只有 38/500 条预测不同；
- given 修复 12 个 hidden 错误，但也破坏 8 个 hidden 正确样本；
- exact McNemar p=0.503，没有显著证据说明 given 优于 hidden。

所以，“profile 让预测发生变化”是真的，但“正确 profile 提供了有用信息”
没有被证明。错误 profile 也产生了几乎同样的变化。

### 10.6 当时的静音对照

从 500 条中的 hidden 条件抽 50 条，把真实音频换成同长度数字静音：

- 只有 7/50 条预测变化，即 14%；
- 原音频分布：I=45、C=5；
- 静音分布：I=44、C=6。

这说明该模型/提示词组合对音频的敏感性很弱。

### 10.7 为什么 500 条结果后来被判定无效

这 500 条确实运行过，数字也真实保存在本地，但后续审计发现：

1. 只来自 3 个会话，不代表完整 16 个会话；
2. 多行其实是同一连续事件的重复 40 ms 帧，不是 500 个独立事件；
3. 100 条 NA 只对应 11 个连续事件；
4. 旧清洗器会删除 `[Mhm]` 等真实重叠词；
5. 旧重叠逻辑会把 A 结束、B 随后开始误判为 I；
6. 两条粘连 TRN 行没有正确恢复；
7. prompt 问“BC/I 是否开始”，而标签表示“该 40 ms 中是什么状态”；
8. 完成单元 TRN 被描述得像 streaming transcript，定义不准确。

因此 500 条结果必须保留并说明，但不能写成“profile 有效/无效”的最终证据。

### 10.8 500 条结果文件在哪里

结果目录：

`artifacts/mllm-prompt-baseline/qwen2.5-omni-3b/audio-transcript-profile-test-100-per-class/`

主要文件：

- `metrics.json`：上面的总体、每类指标和混淆矩阵；
- `diagnostics.json`：预测分布、塌缩检查和延迟；
- `predictions.csv`：500 条逐样本三条件预测；
- `profile_comparison.csv`：三条件对比；
- `requests.jsonl`：1,500 个真实请求，不含 target；
- `responses.jsonl`：1,500 个模型输出；
- `input_audit.json`：当时的三条件输入一致性审计。

仓库中的历史报告：

`code/reports/MLLM_PROMPT_QWEN2_5_OMNI_3B_REPORT.md`

### 10.9 后来的低风险 50 条只用于确认问题

在修复标签、事件抽样和会话分布后，我们又从 16 个会话选择了五类各 10 条
低风险数据。这不是本报告的主体结果，而是用来检查 500 条失败是否只是旧
标签造成的。

原锁定 prompt 的 50 条结果：

| 条件 | Macro-F1 | Accuracy | 主要预测 |
|---|---:|---:|---|
| hidden | 0.1333 | 0.2400 | T=18，I=32 |
| given | 0.1060 | 0.2000 | T=8，I=42 |
| shuffled | 0.1074 | 0.2000 | T=7，I=43 |

一次边界聚焦 prompt 诊断后：

| 条件 | Macro-F1 | Accuracy | 主要预测 |
|---|---:|---:|---|
| hidden | 0.2035 | 0.2600 | I=29，NA=11，T=8，C=2 |
| given | 0.1615 | 0.2400 | I=44 |
| shuffled | 0.1584 | 0.2400 | I=43 |

50 条再次证明：时间聚焦可以稍微改善 hidden，但 given 和 shuffled 仍几乎
一样，正确 profile 语义没有被模型使用。

## 十一、现在的问题是什么

### 问题 1：当前 3B 模型本身不适合这个任务

这是一个非常专门的任务：听 30 秒对话，预测未来 40 ms 的五种话轮状态。当前 3B 量化通用模型并没有为这种任务训练过。

### 问题 2：模型没有真正利用 profile 语义

如果模型理解 profile，正确 profile 应该优于 shuffled profile。现在两者几乎完全一样，说明它主要受到“多了一段详细文字”的影响。

### 问题 3：40 ms 预测非常难

BC、T、I 通常需要极细的时间和说话人活动判断。通用 MLLM 更擅长理解音频内容，不一定擅长 40 ms 级控制预测。

### 问题 4：标签仍然是弱标签

虽然已经修复程序错误并选择了低风险 50 条，但没有人工确认，所以当前准确率只能用于排查模型，不能写成正式论文结果。

### 问题 5：单声道削弱说话人分离

音频被合成单声道。模型听到两个人，但不容易精确区分哪一个是 Speaker A、哪一个是 Speaker B，也不容易判断真正的双人重叠。

## 十二、这次结果能说明什么，不能说明什么

可以说明：

- 当前 Qwen2.5-Omni-3B Q4 checkpoint 不适合直接做这个零样本五分类实验；
- 原提示词存在时间关注问题；
- 加入详细 profile 会改变预测，但模型没有理解正确 profile 的内容；
- 继续在同一 50 条上调 prompt 不可信。

不能说明：

- 不能说明 profile 理论无效；
- 不能说明训练后的 profile adapter 不会有效；
- 不能把当前准确率作为论文正式结果；
- 不能声称 profile 提升，因为当前数据明确没有提升。

## 十三、接下来怎样继续“评测现有大模型”

当前任务仍然是评测现有大模型，不立刻切换成 adapter 训练。合理的下一步只有下面这条：

1. 停止使用当前 3B Q4 checkpoint；
2. 换一个更强、原生支持音频输入的现成模型；
3. 先使用与当前 50 条不重合的少量开发集检查它是否：
   - hidden 至少预测 3 类；
   - 能识别 C、NA 和 BC，而不只预测 I/T；
   - 对真实音频和静音音频有明显不同；
4. 固定 prompt 后，只在一份新的、人工复核过的 50 条测试集上运行一次；
5. 仍然比较 hidden/given/shuffled；
6. 无论结果正负都如实报告，不能再调到 `given` 赢为止。

只有同时满足以下条件，才可以说“现有大模型实验支持 profile 有帮助”：

- hidden 基线不塌缩；
- given Macro-F1 高于 hidden；
- given Macro-F1 高于 shuffled；
- 改善不只来自 C；
- 会话 bootstrap 的 profile 增益具有稳定方向；
- 标签已经人工复核。

## 十四、代码和结果在哪里

### 14.1 主要代码

| 功能 | 文件 |
|---|---|
| 音频 MLLM prompt 和请求生成 | `code/src/profile_turntaking/mllm_prompt_baseline.py` |
| Profile 自然语言模板和评分 | `code/src/profile_turntaking/prompt_baseline.py` |
| 数据与 profile 深度审计 | `code/src/profile_turntaking/experiment_preflight.py` |
| SBCSAE 解析和 profile 映射 | `code/src/profile_turntaking/sbcsae_corpus.py` |
| 五分类标签与事件起点 | `code/src/profile_turntaking/sbcsae_manifest.py` |
| 500 条人工复核页面 | `code/src/profile_turntaking/label_review.py` |
| 会话平衡抽样 | `code/scripts/select_prompt_review_set.py` |
| 主运行命令 | `code/scripts/run_mllm_prompt_baseline.py` |
| 锁定实验协议 | `code/configs/mllm_prompt_pilot_locked.json` |

### 14.2 本次 50 条结果

原提示词结果目录：

`artifacts/mllm-prompt-baseline/qwen2.5-omni-3b/onset-balanced-lowrisk-50-diagnostic/`

其中：

- `metrics.json`：三条件总体和每类指标；
- `diagnostics.json`：预测分布、塌缩检查和延迟；
- `predictions.csv`：每条数据的 target 和三种预测；
- `profile_comparison.csv`：hidden/given/shuffled 对比表；
- `bootstrap_95ci.json`：按会话 bootstrap 置信区间；
- `input_audit.json`：三条件输入一致性和泄漏检查；
- `requests.jsonl`：模型实际收到的请求，不含 target；
- `responses.jsonl`：模型原始输出和解析结果。

静音对照目录：

`artifacts/mllm-prompt-baseline/qwen2.5-omni-3b/onset-balanced-lowrisk-50-silenced/`

一次边界 prompt 修正的开发诊断目录：

`artifacts/mllm-prompt-baseline/qwen2.5-omni-3b/onset-balanced-lowrisk-50-prompt-v2-dev/`

### 14.3 GitHub

- 私有仓库：`pandaxiong0729/profile-aware-turn-taking`
- 分支：`agent/full-data-prep`
- Draft PR：`https://github.com/pandaxiong0729/profile-aware-turn-taking/pull/3`

GitHub 不包含真实 SBCSAE 音频、全量 manifest、原始模型或本地请求结果，因为这些文件体积大或受数据许可限制。

## 十五、最终一句话

**数据和三条件对照流程现在基本排清了；当前失败的核心是 Qwen2.5-Omni-3B 这个现成模型不能稳定完成 40 ms 五分类，也没有真正理解 profile。下一步应该换更强的现有音频大模型，在新的人工复核 50 条上只评测一次，而不是继续在同一批数据上把 prompt 调到正结果。**
