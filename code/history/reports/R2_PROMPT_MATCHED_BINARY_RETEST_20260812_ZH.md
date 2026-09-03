# R2 embedding 与 Qwen prompt 的同样二分问题重测

日期：2026-08-12

## 1. 这次到底测了什么

这次没有重新训练模型，也没有换成新的二分分类头。

做的是：加载已经训练好的 R2 checkpoint，把它的五分类概率 `C / BC / T / I / NA` 重新解释成和 Qwen prompt 实验相同的四个 A/B 问题。

四个问题是：

1. `NA` vs 非 `NA`；
2. `C` vs `BC/T/I`；
3. `BC` vs `T/I`；
4. `T` vs `I`。

原始 0.5 阈值版本会在 `NA` 上塌陷，所以这次新增了 output-gated 阈值校准。阈值只用 R2 原验证集和预测分布门禁选择，不用这两组 50 条的 gold label 调参。

最终阈值：

| 分支 | 阈值 |
|---|---:|
| silence / NA | 0.35 |
| listener onset / C | 0.45 |
| brief response / BC | 0.50 |
| yield / T | 0.50 |

## 2. 输入是否一致

一致。两组各 50 条，直接使用之前 Qwen prompt 的同一批样本：

- `prompt_seed137`：50 条，每类 10 条；
- `prompt_seed237`：50 条，每类 10 条。

每条样本输入仍然是：截止预测点以前的音频、截止预测点以前的转写和说话状态、profile。

`hidden / given / shuffled` 三个条件里，只改变 profile，其余音频、转写、样本编号和预测点不变。

## 3. 输出是否还塌陷

不塌陷。output-gated 版本通过了自动门禁：

- 每个样本组、每个 profile 条件下，五类 `C / BC / T / I / NA` 都至少出现 2 次；
- 最大预测类别占比不超过 65%；
- 四个 A/B 分支都同时出现 A 和 B；
- 验证集同样通过门禁。

原始 0.5 版本没有通过门禁，因为它几乎不预测 `NA`。

## 4. 结果

主指标是 Macro-F1。数值越高越好。

| 方法 | 样本组 | hidden | given | shuffled | given-hidden | given-shuffled |
|---|---|---:|---:|---:|---:|---:|
| Qwen prompt |    | 0.1935 | 0.1737 | 0.2626 | -0.0199 | -0.0889 |
| Qwen prompt | prompt_seed237 | 0.1410 | 0.2235 | 0.2355 | +0.0825 | -0.0120 |
| R2 embedding output-gated | prompt_seed137 | 0.2201 | 0.2824 | 0.2575 | +0.0623 | +0.0249 |
| R2 embedding output-gated | prompt_seed237 | 0.2635 | 0.2658 | 0.2196 | +0.0024 | +0.0462 |

R2 output-gated 的输出分布：

| 样本组 | 条件 | C | BC | T | I | NA |
|---|---|---:|---:|---:|---:|---:|
| prompt_seed137 | hidden | 6 | 10 | 2 | 24 | 8 |
| prompt_seed137 | given | 9 | 10 | 2 | 23 | 6 |
| prompt_seed137 | shuffled | 6 | 9 | 3 | 24 | 8 |
| prompt_seed237 | hidden | 3 | 22 | 6 | 16 | 3 |
| prompt_seed237 | given | 7 | 16 | 6 | 18 | 3 |
| prompt_seed237 | shuffled | 2 | 22 | 4 | 18 | 4 |

直观解释：

- Qwen prompt：两组结果不稳定，一个样本组 correct profile 变差，另一个样本组 correct profile 比 hidden 好但低于 shuffled。
- R2 embedding output-gated：两组里 correct profile 都高于 hidden，也都高于 shuffled；同时输出分布不再塌陷。

## 5. 仍然要注意的限制

这次仍然不能当成最终论文证据。原因是 prompt 匹配的 50+50 样本与旧 R2 训练集有重叠：

- `prompt_seed137` 的 50 条里，有 19 条和旧 R2 训练样本完全重叠；
- `prompt_seed237` 的 50 条里，有 20 条和旧 R2 训练样本完全重叠；
- 四个会话 `SBC005 / SBC010 / SBC041 / SBC047` 都出现在旧 R2 训练会话中。

所以这次最准确的说法是：在本机诊断中，R2 embedding 经过验证集阈值校准后，输出合理且 correct profile 在两组 50 条上都优于 hidden/shuffled；它支持继续做 embedding 路线，但还需要一个无重叠测试集来作为正式证据。

## 6. embedding 和 prompt 的区别

prompt 方式是：把 profile 原文直接写进给 Qwen 的问题里，让 Qwen 在一次推理中同时理解音频、转写、profile 和 A/B 选项。

embedding 方式是：先用文本编码器把 profile 变成一串固定数字，再把这串数字作为单独的 profile 分支送进我们训练的小模型。

这里用的是冻结的 `sentence-transformers/all-MiniLM-L6-v2`。它只负责把自然语言 profile 转成 384 维向量，不参与训练。训练发生在后面的融合模型里。

可以这样理解：

- prompt：profile 是一段话，模型每次临时读这段话，然后决定怎么用；
- embedding：profile 先被压缩成向量，训练时模型反复学习“这个向量和话轮标签之间有什么关系”。

## 7. 为什么 embedding 可能放大 profile 的作用

prompt 里的 profile 容易被大模型当成背景说明，尤其当音频和转写更显眼时，profile 可能只轻微影响答案。

embedding 版本把 profile 单独拿出来做成一个分支，并且有一个可训练的 gate。这个 gate 的作用很简单：模型可以学会在某些样本上多看 profile，在另一些样本上少看 profile。

所以 embedding 的优势不是“语义一定更强”，而是它给了模型一个可训练的入口，让 profile 不只是提示词里的一段文字，而是能在训练中被反复调整使用的信号。

## 8. 文件位置

- R2 二分评测代码：`code/src/profile_turntaking/semantic_profile_binary_eval.py`
- R2 二分评测入口：`code/scripts/evaluate_existing_r2_binary.py`
- R2 二分结果：`artifacts/semantic-profile-embedding/minilm-r2-existing-checkpoint-prompt-matched-binary/summary.json`
- 对比表：`artifacts/semantic-profile-embedding/minilm-r2-existing-checkpoint-prompt-matched-binary/prompt_vs_r2_comparison.csv`
- 本报告：`code/reports/R2_PROMPT_MATCHED_BINARY_RETEST_20260812_ZH.md`
