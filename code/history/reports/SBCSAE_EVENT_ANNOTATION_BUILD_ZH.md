# SBCSAE 事件中心标注数据生成报告

更新时间：2026-07-15

## 当前结果

16段核心双人会话、约6.43小时音频已经重新处理为“一个事件一条数据”。新流程没有读取旧的40 ms五分类文件。

| 项目 | 数量 |
|---|---:|
| IPU（同一人的连续语音段） | 9,218 |
| 对话结构记录 | 15,095 |
| 五分类事件候选 | 16,325 |
| 对应审核音频 | 16,325 |

| 候选标签 | 数量 |
|---|---:|
| C | 5,035 |
| BC | 1,315 |
| T | 2,585 |
| I | 1,301 |
| NA | 6,089 |

这些是机器候选标签，不是人工金标签。`human_label`目前全部为空，等待人工判断。

## 每条事件音频

- 格式：16 kHz、单声道、16-bit PCM WAV。
- 长度：最多约10秒。
- 内容：目标事件前约6秒、后约4秒。
- 双声道原音频采用相位安全混音，避免普通平均造成声音抵消。
- manifest同时保存事件在原会话中的时间、在短音频中的时间以及音频SHA-256。

## 文件位置

- `data/processed/sbcsae_turn_events_v1/ipus.jsonl`：连续语音段。
- `data/processed/sbcsae_turn_events_v1/interaction_structures.jsonl`：Pause、Gap、Overlap、短回应和共同静音。
- `data/processed/sbcsae_turn_events_v1/event_candidates.jsonl`：未带音频信息的事件候选。
- `data/processed/sbcsae_turn_events_v1/annotation_manifest.jsonl`：人工标注主文件。
- `data/processed/sbcsae_turn_events_v1/audio_clips/`：一条事件一个审核音频。
- `data/processed/sbcsae_turn_events_v1/review.html`：可直接双击打开的人工标注页面。
- `data/processed/sbcsae_turn_events_v1/review_data/`：页面使用的16个会话数据文件；不包含机器候选答案。
- `data/processed/sbcsae_turn_events_v1/summary.json`：数量和每会话统计。
- `data/processed/sbcsae_turn_events_v1/verification.json`：完整性检查。

## 在另一台电脑标注

把整个 `sbcsae_turn_events_v1` 文件夹复制到另一台电脑，然后用现代版 Chrome 或 Edge 双击打开 `review.html`。页面只使用相对路径，不依赖本机盘符、Python或网络服务器。每个会话标完后点击“导出本会话结果”，把导出的JSON文件交回即可。

## 已完成检查

- 16个会话全部存在。
- 事件编号没有重复。
- 每个事件只有一个五分类候选。
- 五个类别全部存在。
- 每个事件都有对应音频。
- 每个音频的SHA-256与manifest一致。
- 所有音频都是16 kHz、单声道、16-bit PCM WAV，时长与manifest一致。
- 每个目标位置都在对应音频范围内。
- 人工标签没有被提前填写。
- 旧40 ms标签未参与事件生成。
- 页面脚本和17个会话索引/数据脚本语法通过检查。
- 页面中的16,325条音频路径全部是相对路径，0条缺失，且页面数据不显示机器候选标签。

机器验证结果：`verified=true`。

## 仍需人工完成

BC和I涉及对话作用，不能只根据时间和短词确定；T、C和NA也可能受VAD边界或转写时间误差影响。正式训练和评测前，必须通过人工界面填写 `human_label`，并保留机器候选与人工答案之间的差异。
