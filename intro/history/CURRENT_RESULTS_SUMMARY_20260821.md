# 当前实验结果汇总

## 一句话结论

纯音频基线已跑到与 Talking Turns 四任务平均相近的 **73.55%** 平衡 Accuracy；带 profile 的缓存特征 adapter 已有完整结果，但 profile 的 `given` 还没有在平衡口径下稳定超过 `hidden` 和 `shuffled`；真正插入 Qwen 的 soft-prompt adapter 已真实训练跑通，但目前只做了 8 条训练样本的技术 pilot，不能报告 profile 效果。

## 三条路线不是同一个结构

| 路线 | 实际输入 | Qwen 在哪里 | 训练什么 | 输出 | 当前作用 |
|---|---|---|---|---|---|
| A. audio-only baseline | 预测点前 30 秒音频 | 只取冻结 Qwen 音频塔的 33 层边界向量 | 层权重、音频 MLP、4 个 A/B 头 | 四个 A/B 概率 | 对齐 Talking Turns 的纯音频基线 |
| B. cached-feature profile adapter | 30 秒因果音频＋因果转写＋profile | Qwen 先离线生成 context/profile 向量；训练时不再跑 Qwen | 音频/转写融合、profile gate、4 个 A/B 头 | 四个 A/B 概率 | 快速、可重复的 profile 对照实验 |
| C. Qwen soft-prompt adapter | 30 秒因果音频＋因果转写＋profile | profile 变为 soft token，真实进入冻结 Qwen Transformer | profile→soft-token MLP、task soft token | Qwen 原始 `lm_head` 的 A/B token 概率 | 最终希望使用的“真正 Qwen 内部 adapter” |

因此：A 与 B **不是同一结构**。A 没有转写、没有 profile、没有 Qwen context 分支；B 有音频、转写、profile 三部分，但 Qwen 只作为冻结特征提取器。C 才是 profile 直接进入 Qwen prompt 内部的版本。

## A. 音频-only baseline：正式结果

| 指标 | 结果 |
|---|---:|
| 普通 Accuracy（自然类别比例） | 76.58% |
| 论文式 50/50 A/B 平衡 Accuracy | **73.55%** |
| Turn Change | 76.03% |
| Backchannel | 77.21% |
| Interruption | 88.53% |
| Floor-Taking Interruption | 52.45% |

Talking Turns Table 4 的 Supervised Topline 四项平均也是 73.55%。但两边数据集和音频编码器不同，所以只能说总体量级相近，不能写成“超过论文 SOTA”。

## B. 音频＋转写＋profile cached-feature adapter：现有正式对照

这里的三组只改变 profile：

- `hidden`：不给 profile；
- `given`：给正确 profile；
- `shuffled`：给错误的其他会话 profile。

| 指标 | hidden | given | shuffled | given-hidden | given-shuffled |
|---|---:|---:|---:|---:|---:|
| 普通 Accuracy | 78.10% | **78.53%** | 78.32% | +0.43 | +0.21 |
| 论文式平衡 Accuracy | **74.01%** | 73.83% | 74.14% | -0.18 | -0.32 |

所以目前可以说：加入 profile 后普通 Accuracy 有很小提升，但在更公平的平衡 Accuracy 下没有稳定改善，不能作为“profile 已有效”的主结论。

## C. 真正 Qwen soft-prompt adapter：训练到什么程度

已经完成：

- 真实 Qwen2.5-Omni-3B 载入；
- 音频＋因果转写＋A/B 问题真实送进 Qwen；
- profile 变成 4 个 2048 维 soft token，继续送入 Qwen；
- 原始 Qwen `lm_head` 输出 `A` / `B` 的 logits；
- Qwen 参数冻结、没有梯度；adapter 获得梯度并更新；
- 关闭 adapter 时直接调用原始 Qwen，因此不会影响普通 Qwen 功能；
- 最小端到端训练完成：8 条训练、8 条验证、8 条测试，1 epoch；adapter 权重已保存。

最小 pilot 的结果：validation `hidden/given/shuffled=50/50/50%`；test `37.5/50/50%`。样本只有 8 条，数值没有统计意义，只能说明训练和三组对照流程正常。

## 当前最重要的下一步

扩大 C 的训练/验证/测试样本，固定 split 后再比较 `given` 是否同时高于 `hidden` 和 `shuffled`。这是可以支撑论文 profile 主张的结果；A 与 B 目前主要用于证明纯音频基线和快速迭代结构已经可用。

## 文件

- A 结果：`artifacts/qwen-audio-only-ab-baseline/paper-aligned-v1/summary.json`
- B 结果：`artifacts/qwen-shared-ab-30s-causal/paper-aligned-floor/history120-floorprop-profile-margin-final/gate_profile_margin_0p25/summary.json`
- C pilot：`artifacts/qwen-soft-prompt-adapter/pilot-1-per-class/summary.json`
- C 真实模型前向/反向验证：`artifacts/qwen-soft-prompt-adapter/smoke-one/report.json`
