# 音频-only 基线与 Qwen Soft-Prompt Adapter

## 1. 这次补做了什么

本次实验分成两部分，二者不能混为一谈。

1. **音频-only baseline**：只输入预测点以前 30 秒音频，不输入转写，也不输入 profile。目的只是尽量对齐 *Talking Turns* 的音频预测设定。
2. **Qwen soft-prompt adapter**：输入因果音频、匹配的因果转写和 A/B 问题；profile 被转换为可训练 soft token，真正送进冻结的 Qwen Transformer，再由 Qwen 原始语言模型输出头给出 A/B。

第一部分检验纯音频话轮预测能力。第二部分才是后续的 profile 方法。

## 2. 音频-only baseline

### 2.1 输入和输出

每条输入：

- 预测点以前 30 秒单声道音频；
- 不含转写；
- 不含 profile；
- 不含未来音频、标签和标注解释。

输出沿用 *Talking Turns* 的四个二分问题：

- Turn Change：A=继续，B=自然换人；
- Backchannel：A=不是 BC，B=出现 BC；
- Interruption：A=没有打断，B=出现打断；
- Floor-Taking Interruption：A=打断者没有取得话轮，B=打断者取得话轮。

### 2.2 模型结构

```text
预测点前 30 秒音频
        ↓
冻结的 Qwen2.5-Omni-3B 音频塔
        ↓
卷积输入层 + 32 个 Transformer 层
在预测边界各取一个向量，共 33 层
        ↓
训练一个 softmax 权重，自动组合 33 层
        ↓
Linear 1280→256 + GELU + LayerNorm
        ↓
4 个 A/B 输出头
        ↓
四个问题各输出 P(A)、P(B)
```

Qwen 音频塔保持冻结；重新训练的是层权重、256 维投影层和四个二分类头。代码中的转写/context 分支和 profile 分支被真正绕过，并非仅把某个混合向量随意清零。

### 2.3 数据审计

| split | 样本数 |
|---|---:|
| train | 6,623 |
| validation | 2,243 |
| test | 1,938 |

审计确认：音频均截至预测边界；请求中不含未来信息和标签；同一样本的音频哈希一致。

### 2.4 结果

下表报告确定性的 50/50 A/B 平衡测试子集 Accuracy，和 *Talking Turns* Table 4 的口径相同。

| 方法 | Turn Change | Backchannel | Interruption | Floor-Taking | 四项平均 |
|---|---:|---:|---:|---:|---:|
| Talking Turns Supervised Topline（论文） | 78.60% | 75.10% | 74.90% | 65.60% | 73.55% |
| 本实验：Qwen 音频-only | 76.03% | 77.21% | 88.53% | 52.45% | **73.55%** |

本实验在全部自然分布测试样本上的普通 Accuracy 为 **76.58%**。这个数受到类别比例影响，不能代替上表的 50/50 平衡 Accuracy。

这里可以说明我们的音频-only pipeline 已达到相近的总体水平，但**不能据此声称超过论文 SOTA**，因为双方使用的数据集、音频编码器和训练标签并不相同。真正严格的 SOTA 比较仍需在 Talking Turns 的同一数据集和 split 上运行。

## 3. 真正的 Qwen Soft-Prompt Adapter

### 3.1 为什么它不同于原来的 adapter

原来的 LM-head adapter 是在 Qwen 已经得到一个最终 hidden vector 后再调整该向量。新版本把 profile 转成连续的 soft token，并把它们真正送入 Qwen Transformer。因此 Qwen 的注意力层会处理 profile 信息，最后仍由 Qwen 自己的语言模型头输出词表中的 `A` 或 `B`。

### 3.2 完整结构

```text
因果音频 + 匹配的因果转写 + 自然语言 A/B 问题
                    ↓
          冻结的 Qwen2.5-Omni
                    ↓
        原始多模态 prompt 的 KV 状态

profile 的 Qwen embedding + 当前问题类型
                    ↓
     小型 MLP：2048→256→(K×2048)
                    ↓
          K 个 profile soft token
                    ↓
接在原始 prompt 后，继续通过冻结的 Qwen Transformer
                    ↓
             Qwen 原始 lm_head
                    ↓
             token A / token B 概率
```

训练时只更新：

- soft token 的基础参数；
- profile→soft token 的小型 MLP；
- 四类问题的 task soft token。

Qwen 的音频塔、Transformer 和 lm_head 全部冻结。

### 3.3 为什么不影响 Qwen 的其他功能

adapter 是显式开关：

- `adapter_enabled=True`：执行话轮 A/B prompt，并插入 profile soft token；
- `adapter_enabled=False`：直接调用原始 Qwen，不插 token，也不经过 adapter。

单元测试已经验证：关闭 adapter 时包装器与直接调用基础模型使用同一条前向路径；反向传播时只有 adapter 收到梯度，Qwen 参数没有梯度。相关测试共 15 项通过。

## 4. 文件位置

- 音频-only 训练程序：`code/scripts/run_qwen_shared_binary_multitask_adapter.py`
- 音频-only 结果：`artifacts/qwen-audio-only-ab-baseline/paper-aligned-v1/summary.json`
- Soft-prompt adapter：`code/src/profile_turntaking/qwen_soft_prompt_adapter.py`
- 真实 Qwen 冒烟程序：`code/scripts/smoke_qwen_soft_prompt_adapter.py`
- Soft-prompt 训练入口：`code/scripts/run_qwen_soft_prompt_adapter_pilot.py`
- 单元测试：`code/tests/test_qwen_soft_prompt_adapter.py`

### 4.1 本机真实 Qwen 验证

真实 Qwen2.5-Omni-3B 已完成一条样本的架构验证，结果在 `artifacts/qwen-soft-prompt-adapter/smoke-one/report.json`。

- 输入为 30 秒因果音频、匹配的因果转写和 turn-change 自然语言 A/B 问题；
- `A`/`B` 对应 Qwen 词表 token 32/33；
- profile 生成 4 个 2048 维 soft token；
- adapter 收到梯度，全部 Qwen 参数未收到梯度；
- 一步训练损失为 0.4807。

这项测试只证明真实结构可前向、可反向、可隔离基础模型；它不是 profile 效果结果，不能报告为 Accuracy。

### 4.2 端到端训练 pilot

最小 pilot 已运行并保存 adapter checkpoint：`artifacts/qwen-soft-prompt-adapter/pilot-1-per-class/`。它使用 8 条训练、8 条验证和 8 条测试记录（四个任务中每类各一条）。

| 条件 | validation Accuracy | test Accuracy |
|---|---:|---:|
| hidden | 50.0% | 37.5% |
| given | 50.0% | 50.0% |
| shuffled | 50.0% | 50.0% |

这只是一个完整性验证：训练、保存、加载所需的输入以及 hidden/given/shuffled 对照都能真实运行。每组只有 8 条，绝不能当作 profile 结论。

## 5. 下一步

1. 扩大 soft-prompt adapter 的训练与评测样本，仍固定 train/validation/test split。
2. 在相同样本上比较 hidden / given / shuffled；三组只能改变 profile。
3. 最终同时报告四项普通 Accuracy、50/50 平衡 Accuracy，以及 given-hidden、given-shuffled 差值。
