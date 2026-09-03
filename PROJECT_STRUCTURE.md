# 项目结构

## 当前目录

```text
turn-taking/
├── README.md                      # 总入口和四条最常用命令
├── PROJECT_STRUCTURE.md           # 本文件
├── data/
│   ├── README.md                  # 数据结构与单条样本说明
│   ├── sbcsae/                    # SBCSAE 原始 60 段语料
│   ├── processed/                 # 当前主实验处理数据
│   │   ├── sbcsae_catalog_v2/     # 60 段会话统一目录
│   │   ├── sbcsae_turn_events_v3/ # 16 段双人会话的事件源
│   │   ├── sbcsae_qwen_shared_ab_30s_causal_v1/ # 10,804 条主样本
│   │   ├── sbcsae_vad_fiveclass_v2/ # 早期完整 40 ms 时间轴，供核对
│   │   └── history/               # 旧处理版本，不进入当前主实验
│   └── reference/                 # PaChat 等参考数据，不进入主训练
├── code/
│   ├── README.md                  # 当前代码逐文件说明
│   ├── scripts/                   # 当前命令入口
│   ├── src/profile_turntaking/    # 当前事件/音频公共函数
│   ├── tests/                     # 当前两条链路的自动测试
│   ├── examples/                  # 不依赖大数据的小示例
│   ├── docs/                      # 当前使用说明
│   ├── reports/                   # 当前结果说明
│   └── history/                   # 旧实验代码、报告、测试和配置
├── artifacts/
│   ├── README.md
│   ├── main_experiment/           # 主缓存、profile、结果、audio-only 对照
│   ├── talking_turns/             # 论文 checkpoint 在 SBCSAE 上的输出
│   ├── data-preview/              # 小型可查看示例
│   └── history/                   # 旧实验产物
├── models/
│   ├── README.md
│   ├── qwen2.5-omni-3b-local/     # 重新提特征时使用
│   ├── talking_turns/             # 官方 checkpoint 与匹配 ESPnet 源码
│   └── history/                   # 旧模型和重复下载缓存
├── intro/                         # 论文草稿
├── history/runtime/               # 测试缓存、临时文件和旧根目录材料
└── collaboration/                 # 交接指南、审计、逐文件清单
```

## 什么算“当前”

只有满足下面任一条件的文件留在当前目录：

- 主实验训练、检查或查看结果必需；
- Talking Turns 对照运行或评分必需；
- 解释当前数据、模型、结果或云端交接必需；
- 不依赖受限语料的最小示例。

其余文件先进入 `history/`。这一步没有删除历史实验，方便确认后再决定是否永久删除。

## 每个最小文件在哪里查

逐文件清单位于 `collaboration/PROJECT_FULL_FILE_INVENTORY.csv`。它不是只列文件夹，而是每个文件一行，包含：

```text
path, bytes, category, main_experiment_status, purpose
```

按 `main_experiment_status` 过滤即可看到：`required`（主实验必须）、`recommended`（复核或对照推荐）、`feature-rebuild-only`（重新提特征才需要）、`history`（旧实验）和 `optional`（示例或辅助材料）。

## 两条实验的数据关系

```text
同一条 SBCSAE test 事件
        ├── 我们的模型：30秒音频 + 因果转写 + profile → 四个 A/B 概率
        └── Talking Turns：同一段30秒音频 → 五类概率 → 转成同四个 A/B 指标
```

这样可以看出模型结构差异，但不能称为在 Talking Turns 原始 Switchboard benchmark 上刷新 SOTA；目前是在我们的 SBCSAE benchmark 上比较。
