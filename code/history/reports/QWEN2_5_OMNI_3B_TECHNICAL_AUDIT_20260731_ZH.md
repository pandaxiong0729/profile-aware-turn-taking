# Qwen2.5-Omni-3B 话轮预测技术排查

## 1. 先说明两篇相关研究

我之前说的“角色、关系、人格与话轮速度有关”，指的是：

- [Modeling Turn-Taking Speed and Speaker Characteristics](https://aclanthology.org/2025.sigdial-1.2/)（Onishi 等，SIGDIAL 2025）。它分析角色、关系和人格与话轮转换速度的关系，并用 Gamma 分布拟合速度分布。它不是我们的细粒度五分类模型。
- [Prompt-Guided Turn-Taking Prediction](https://aclanthology.org/2025.sigdial-1.9/)（Inoue 等，SIGDIAL 2025）。它把 “faster”“calmer” 等合成文字指令的 embedding 加进 VAP 模型，用来控制话轮时机。它没有使用我们这种真实、结构化、会随对话更新的个人 profile。

这两篇是后来检索到的 SIGDIAL 2025 论文，不是最初提供的三篇 PDF，所以此前没有在我们的材料中出现。

`hidden / given / shuffled` 只是检验模型是否真正使用 profile 的对照方法，不是论文卖点。方法本身应当是“稳定个人信息＋动态关系和情境”的因果 Profile Adapter。

## 2. 这次排查要回答什么

这次不比较 profile 效果，只回答一个更基础的问题：

> 本地 Qwen2.5-Omni-3B 是否确实收到音频，并且能否根据不同音频稳定地区分话轮结果？

当前五分类标签按项目约定直接作为正确参考，不再增加其他数据检查步骤。

## 3. 模型与输入

- 模型：Qwen2.5-Omni-3B，Q8_0 量化。
- 推理：llama.cpp 本地服务。
- 样本：8 条，每个二分类任务各有一条正例和一条反例。
- 共运行：80 次请求，全部成功返回。

正式未来预测请求只包含：

1. 截止预测点的音频；
2. 与该音频时间一致的部分转写；
3. `hidden profile`；
4. 当前任务的问题和允许输出的答案。

参考答案与预测点之后的信息不会发给模型。

## 4. 实际检查内容

| 检查 | 目的 |
|---|---|
| ASR 转写 | 确认音频是否真正传进模型 |
| 当前未来预测提示 | 测试当前输入方式 |
| 音频放在文字前面 | 排除多模态输入顺序问题 |
| 论文风格提示 | 排除提示词写法问题 |
| 静音替换 | 检查答案是否依赖音频 |
| 错配音频替换 | 检查换一段相反音频是否改变答案 |
| 已发生事件识别 | 降低任务难度，检查模型能否识别已经发生的事件 |
| 交换两个答案的书写顺序 | 检查模型是否只偏向答案位置 |

运行前检查已通过：8 条样本类别平衡；音频和转写文件校验一致；转写没有越过预测点；请求中没有参考答案；输出格式固定。

## 5. 结果

| 结果 | 数值 | 说明 |
|---|---:|---|
| ASR 有输出 | 8/8 | 音频确实传到了模型 |
| ASR 平均词重合 F1 | 0.64 | 7 条能较好听出内容，1 条异常 |
| 当前未来预测 | 4/8 | 每个任务都固定选择同一个答案 |
| 论文风格提示 | 4/8 | 没有改善 |
| 音频放在前面 | 3/8 | 没有解决塌缩 |
| 静音替换后答案改变 | 0/8 | 音频被替换，但正确的部分转写仍然保留 |
| 错配音频后答案改变 | 0/8 | 音频被替换，但正确的部分转写仍然保留 |
| 已发生事件识别 | 4/8 | 降低任务难度后仍按任务固定回答 |
| 交换答案顺序后保持原语义 | 2/8 | 6 条跟着答案书写顺序改变 |

最清楚的结论是：

1. 音频输入链路是通的，因为模型可以转写音频。
2. 静音和错配实验保留了正确转写，所以答案不变只说明这8条中音频没有提供超出文字的额外变化，不能推出模型无法判断。
3. 二分类输出明显受到答案顺序影响，所以不能只使用这种二选一提示。
4. 可以继续做 profile 对照，但应保留音频＋部分转写，并换成不容易受答案顺序影响的决策式五分类提示。

这不是 profile 方法的问题。它只说明原来的生成式二选一提示不稳定，不能把该提示的结果当成最终结论。

## 6. 下一步

保持同一个 Qwen2.5-Omni-3B，固定音频、部分转写、预测点和解码参数，只改变 profile，运行 `hidden / given / shuffled`。该实验已经使用决策式提示完成两批50条，结果见 `QWEN2_5_OMNI_3B_PROFILE_PILOT_20260731_ZH.md`。

## 7. 文件位置

- 可视化页面：`artifacts/omni-technical-audit/qwen2.5-omni-3b-q8/audit8-20260731/review.html`
- 指标：`artifacts/omni-technical-audit/qwen2.5-omni-3b-q8/audit8-20260731/metrics.json`
- 每次请求：`artifacts/omni-technical-audit/qwen2.5-omni-3b-q8/audit8-20260731/requests.jsonl`
- 每次输出：`artifacts/omni-technical-audit/qwen2.5-omni-3b-q8/audit8-20260731/responses.jsonl`
- 输入检查：`artifacts/omni-technical-audit/qwen2.5-omni-3b-q8/audit8-20260731/input_audit.json`
- 主程序：`code/src/profile_turntaking/omni_technical_audit.py`
- 命令入口：`code/scripts/run_omni_technical_audit.py`
- 自动测试：`code/tests/test_omni_technical_audit.py`
