# 自然语言 Profile 语义向量实验报告（R2）

日期：2026-08-12

## 一句话结论

这套实验已经可以完整运行，输入配对正确，模型没有输出塌缩，也确实使用了音频。主配置在测试集上出现了很小的正向差值：正确 profile 的 Macro-F1 为 **0.4072**，无 profile 为 **0.4037**，错误 profile 为 **0.4048**。但是验证集上错误 profile 反而略高，因此目前只能说“测试集出现正向趋势”，**还不能作为论文中 profile 有效的确定证据**。

## 1. 这里的 Transformers 到底是什么

这里没有训练 Qwen，也没有让大语言模型生成答案。

使用的是 Hugging Face 生态中的预训练句向量模型：

`sentence-transformers/all-MiniLM-L6-v2`

它的工作只有一步：把原来给 Qwen 的同一段四行自然语言 profile 转成一个 384 维语义向量。例如：

```text
speaker_00 has age group 25-34, gender male, social role comm./comp, and background ...
speaker_01 has age group 35-44, gender female, social role actress/fi, and background ...
Their relationship is romantic partners.
The conversation situation is casual social conversation.
```

MiniLM 在本实验中完全冻结，不更新参数。真正训练的是后面很小的音频支路、转写支路、profile 投影层、融合门和五分类器。

因此这里的“embedding”可以直白理解为：**先让一个已经学过英语语义的模型把 profile 压缩成一串数字，再训练一个小模型判断这串数字能否帮助话轮预测。**

## 2. 每条数据输入和输出是什么

每条样本的预测边界记为 `t`，模型只能看见 `t` 以前的信息。

输入包括：

1. 截止 `t` 的 5.9 秒单声道因果音频；
2. 与这段音频严格对应、截止 `t` 的部分转写；
3. 截止 `t` 的说话人活动状态；
4. 一段固定模板的自然语言 profile，或者隐藏/打乱后的对照 profile。

输出是预测 `t+100 ms` 开始、随后 500 ms 内发生的五类事件之一：

`C / BC / T / I / NA`

标签单独保存在 `reference_labels.jsonl`，不会进入模型输入。

### 三个公平对照

| 条件 | 模型看到的内容 |
| --- | --- |
| hidden | 音频和转写不变，profile 支路输入全零 |
| given | 音频和转写不变，输入该会话的正确 profile |
| shuffled | 音频和转写不变，输入另一个会话的错误 profile |

同一个样本的三次输入中，样本编号、音频文件和 SHA-256、转写和 SHA-256、预测边界、任务定义都相同，只有 profile 改变。

## 3. 数据规模

数据按会话划分，训练、验证、测试之间没有会话重叠。

| 划分 | 会话数 | Profile 数 | 样本数 | 每类样本数 |
| --- | ---: | ---: | ---: | ---: |
| 训练集 | 10 | 10 | 1,500 | 300 |
| 验证集 | 3 | 3 | 250 | 50 |
| 测试集 | 3 | 3 | 250 | 50 |
| 合计 | 16 | 16 | 2,000 | — |

训练会话：`SBC005/006/009/010/024/041/044/045/047/060`。

验证会话：`SBC029/034/043`。

测试会话：`SBC007/017/058`。

重要限制：当前每段会话只有一份固定 profile。每个会话内部的五类样本又大致平衡，所以 profile 不能简单依靠“记住某个会话更常出现哪一类”得到提升。模型必须从很少的 10 个训练 profile 中学出可以迁移到新人物的交互规律，这对当前数据量很困难。

## 4. 模型实际怎么训练

- 音频：只使用预测边界以前的音频，提取 132 维多时间尺度声学与边界特征；
- 转写：冻结的 MiniLM 将因果转写编码为 384 维向量；
- Profile：同一个冻结 MiniLM 将完整四行 profile 编码为 384 维向量；
- 融合：三个分支投影到 128 维，profile 通过可训练 gate 加入音频—转写表示；
- 分类：输出五类概率；
- 损失：加权交叉熵；
- 训练时随机隐藏 50% profile，使同一个 checkpoint 同时支持 hidden 和 given；
- 随机种子：`13 / 37 / 71`，结果取三次平均。

主配置使用最简单的完整文本向量和加法门控，没有清洗、改写或重新分类 profile 字段。另行测试了交互融合、逐行编码和向量去中心化，但这些改动没有得到更稳定的结果。

## 5. 主结果

### 测试集，250 条，三次训练平均

| Profile 条件 | Macro-F1 | Balanced Accuracy | Log Loss | Brier Score |
| --- | ---: | ---: | ---: | ---: |
| hidden | 0.4037 ± 0.0167 | 0.4107 ± 0.0161 | 1.5541 | 0.7625 |
| given | **0.4072 ± 0.0178** | **0.4173 ± 0.0124** | **1.5460** | **0.7566** |
| shuffled | 0.4048 ± 0.0178 | 0.4160 ± 0.0142 | 1.5466 | 0.7569 |

测试集差值：

- `given - hidden` Macro-F1：`+0.0035`；
- `given - shuffled` Macro-F1：`+0.0024`。

三个随机种子的这两个 Macro-F1 差值方向都为正，但提升非常小。

### 验证集，250 条，三次训练平均

| Profile 条件 | Macro-F1 | Log Loss |
| --- | ---: | ---: |
| hidden | 0.4241 | 1.5355 |
| given | 0.4260 | 1.5288 |
| shuffled | **0.4285** | **1.5238** |

验证集上 `given` 没有超过 `shuffled`。因此代码中的结论门禁为 `false`，不会把测试集的小幅正差值自动写成“profile 已被证明有效”。

### 测试集每类 F1

| 条件 | C | BC | T | I | NA |
| --- | ---: | ---: | ---: | ---: | ---: |
| hidden | 0.6090 | 0.2423 | 0.3346 | 0.3785 | 0.4542 |
| given | 0.6350 | 0.2275 | 0.3276 | 0.3913 | 0.4547 |
| shuffled | 0.6246 | 0.2341 | 0.3290 | 0.3886 | 0.4476 |

正确 profile 的主要正变化出现在 `C` 和 `I`，但 `BC` 和 `T` 下降，说明当前 profile 利用方式还不稳定。

## 6. 输出是否塌缩、音频是否真的生效

不是塌缩结果：三个条件、三个随机种子的预测都覆盖至少三类，且没有任何单一类别占到 80%。例如 seed 13 的 given 输出分布为：`C=54, BC=89, T=28, I=26, NA=53`。

把音频特征置零后，hidden 条件下有 `63.2%–73.6%` 的测试样本改变预测。这说明模型明显依赖音频，不是只靠 profile 或类别先验输出。

把转写语义向量置零后，有 `12%–28%` 的样本改变预测，说明文本也有增量作用。

## 7. 尝试过哪些改进

| 版本 | 测试 given-hidden | 测试 given-shuffled | 结果 |
| --- | ---: | ---: | --- |
| 完整文本向量＋加法门控 | **+0.0035** | **+0.0024** | 测试方向为正，验证不稳定 |
| 完整文本向量＋交互融合 | -0.0063 | -0.0012 | 退化 |
| 去中心化＋加法门控 | -0.0146 | -0.0178 | 明显退化 |
| 去中心化＋交互融合 | +0.0016 | +0.0010 | 接近零且跨种子不稳定 |

这些结果排除了两个容易误判的原因：不是因为输出全变成一类，也不是因为 profile 向量完全没有进入模型。更可能的瓶颈是 profile 样本太少、每段会话 profile 固定、静态人口信息与未来 600 ms 内具体事件的关系弱。

## 8. 现在能汇报什么

可以汇报：

1. R2 自然语言语义 embedding 流水线已经实现并可复现；
2. hidden/given/shuffled 的配对输入审计全部通过；
3. 基线没有塌缩且对音频敏感；
4. 测试集存在很小的正向趋势；
5. 验证集没有重复该趋势，所以暂时不能宣称 profile 已经带来可靠提升。

不能汇报成结论：

> “正确 profile 已稳定优于错误 profile。”

当前数据不支持这句话。

## 9. 下一步最有价值的工作

当前 R2 已完成。下一步不应直接堆更复杂的 R4，而应先增加真正与当前互动状态有关、并且能在同一会话内变化的 profile 信息，例如动态提炼的 relationship state、当前共同任务、熟悉程度或最近一段时间的 backchannel/打断习惯。这样才能检验“同一段音频下，动态 profile 改变是否带来可泛化的预测改进”。

如果只做下一项快速实验，建议实现 R3：把年龄、性别、关系、场景分别做显式 field embedding，同时保留当前 R2 作为对照。R3 参数更少，也更适合只有 16 个 profile 的小数据条件。

## 10. 文件和复现命令

核心代码：

- `code/src/profile_turntaking/semantic_profile_experiment.py`
- `code/scripts/run_semantic_profile_embedding.py`
- `code/tests/test_semantic_profile_experiment.py`

主结果：

- `artifacts/semantic-profile-embedding/minilm-additive-raw-v2/summary.json`
- `artifacts/semantic-profile-embedding/minilm-additive-raw-v2/predictions.jsonl`
- `artifacts/semantic-profile-embedding/comparison.csv`

从仓库根目录复现：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\code[semantic-profile,dev]"

.\.venv\Scripts\python.exe code\scripts\run_semantic_profile_embedding.py train `
  --data-dir data\processed\sbcsae_semantic_profile_v1 `
  --cache-tag semantic `
  --output-dir artifacts\semantic-profile-embedding\minilm-additive-raw-v2 `
  --profile-fusion additive `
  --profile-preprocessing raw `
  --seeds 13 37 71 `
  --device cpu
```

本机该训练约十秒完成，因为 MiniLM 向量已经缓存。第一次重新建立全部向量缓存会更慢，但不需要每次训练重复编码。
