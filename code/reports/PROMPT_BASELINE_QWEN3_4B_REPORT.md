# Profile Prompt 低成本验证实验报告：Qwen3-4B

更新时间：2026-07-14

## 1. 实验问题

本实验不训练或微调任何模型，只回答一个前置问题：在相同的因果历史转写上，把正确的说话人 profile 用固定自然语言模板提供给现成大模型，是否比不提供 profile 更有利于预测下一 40 ms 的五分类话轮事件。

五类为：`C / BC / T / I / NA`。三种条件使用同一个模型、同一批 sample IDs、同一 system prompt 和确定性解码：

| 条件 | Profile 输入 |
| --- | --- |
| `hidden` | `Profile information is unavailable.` |
| `given` | 当前会话的正确结构化 profile，经固定模板转成英文自然语言 |
| `shuffled` | 另一测试会话的完整 profile，作为错误 profile 负控制 |

这是文本 Prompt baseline。模型没有接收 WAV，因而看不到音量、停顿、语调和重叠声学信息。

## 2. 模型、硬件与运行环境

| 项目 | 设置 |
| --- | --- |
| 模型 | `Qwen/Qwen3-4B-GGUF` |
| 量化 | `Q4_K_M`，4-bit |
| 模型文件 | `Qwen3-4B-Q4_K_M.gguf`，2,497,280,256 bytes |
| SHA-256 | `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`，与官方 metadata 一致 |
| 推理运行时 | llama.cpp `b9987`，Windows CUDA 12.4 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8,151 MiB |
| CPU / RAM | Intel Core Ultra 9 285H；31.4 GiB RAM |
| 上下文 | 8,192 tokens；历史转写最多 6,000 characters |
| 解码 | `temperature=0`；reasoning off；最多生成 16 tokens |

模型加载约 20 秒。运行期间观察到约 3.74 GiB 显存占用、约 2.7–2.9 GiB server RAM、约 63°C GPU 温度；模型停止后显存恢复为空闲。当前电脑可以稳定运行该 4B 量化模型。

模型来源：<https://huggingface.co/Qwen/Qwen3-4B-GGUF>；运行时来源：<https://github.com/ggml-org/llama.cpp/releases/tag/b9987>。

## 3. 数据与防泄漏

输入来自 `data/processed/sbcsae_mvp/manifest.jsonl`。Prompt 仅通过白名单读取：

- `prediction_time_s`；
- `transcript_prefix`，只包含预测时刻之前已经结束的转写；
- 当前实验条件对应的 profile。

`requests.jsonl` 与 `gold.jsonl` 分离。真实发送给模型的 1,500 个请求中没有 `target`、`training_target`、`next_40ms_label`、`annotation_only_not_model_input` 或 `label_evidence`，泄漏扫描匹配数为 0。

Prompt 在验证集上固定，正式测试后没有修改模板或解析规则。

### 验证集 pilot

- `val` split；
- 每类 20 条，共 100 个样本；
- 每个样本运行 `hidden/given/shuffled`，共 300 次请求；
- seed = 13。

### 正式测试

- `test` split；
- 每类 100 条，共 500 个样本；
- 500 条来自 3 个 speaker-disjoint 测试会话；
- 每个样本运行三种 profile 条件，共 1,500 次请求；
- seed = 13；
- 1,500/1,500 输出均成功解析，三条件共同有效样本为 500。

正式测试是类别平衡诊断集，不代表真实线上类别先验。标签仍是 SBCSAE `.trn` 时间戳生成的弱标签，不是 frame-accurate 人工 gold。

## 4. 验证集结果

| 条件 | Macro-F1 | Accuracy |
| --- | ---: | ---: |
| hidden | 0.1822 | 0.23 |
| given | 0.2210 | 0.25 |
| shuffled | 0.1895 | 0.22 |

验证集上 `given-hidden` Macro-F1 为 `+0.0388`，正确 profile 修复 7 条 hidden 错误，同时破坏 5 条 hidden 正确预测。这个小样本只呈现轻微正向趋势。

## 5. 正式测试主要结果

| 条件 | Macro-F1 | Balanced Accuracy | Accuracy |
| --- | ---: | ---: | ---: |
| hidden | **0.2371** | **0.246** | **0.246** |
| given | 0.1957 | 0.218 | 0.218 |
| shuffled | 0.1793 | 0.208 | 0.208 |
| given − hidden | **−0.0415** | −0.028 | −0.028 |
| given − shuffled | +0.0163 | +0.010 | +0.010 |

正式测试与验证集的增益方向相反。正确 profile 比无 profile 的 Macro-F1 低 0.0415，accuracy 低 2.8 个百分点；正确 profile 略好于错误 profile，但优势很小。

### 每类 F1

| 类别 | hidden | given | shuffled | given − hidden |
| --- | ---: | ---: | ---: | ---: |
| C | 0.0896 | 0.1579 | 0.1220 | +0.0683 |
| BC | 0.2326 | 0.2493 | 0.2522 | +0.0167 |
| T | 0.2321 | 0.2155 | 0.2304 | −0.0166 |
| I | 0.3556 | 0.3164 | 0.2921 | −0.0392 |
| NA | **0.2759** | **0.0392** | **0.0000** | **−0.2366** |

正确 profile 对 `C` 和 `BC` 有少量改善，但损害 `T`、`I`，尤其明显损害 `NA`。

### 成对变化

`given` 相比 `hidden`：

- 113/500 条预测发生变化；
- 修复 hidden 错误 25 条；
- 破坏 hidden 正确预测 39 条；
- 两者都正确 84 条；
- 两者都错误 352 条；
- accuracy 的 exact McNemar p = 0.1034。

`given` 相比 `shuffled`：

- 66/500 条预测发生变化；
- given 修复 shuffled 错误 14 条；
- given 破坏 shuffled 正确预测 9 条；
- accuracy 的 exact McNemar p = 0.4049。

因此，现有样本没有提供可靠的统计证据证明正确 profile 提升了该文本模型。

### 预测类别偏差

| 条件 | C | BC | T | I | NA |
| --- | ---: | ---: | ---: | ---: | ---: |
| hidden | 34 | 201 | 124 | 125 | 16 |
| given | 52 | 237 | 132 | 77 | **2** |
| shuffled | 64 | 241 | 117 | 78 | **0** |

目标集中每类实际各有 100 条，但模型严重偏向 `BC`，并且加入任何具体 profile 后几乎不再预测 `NA`。这说明自然语言 profile 影响了决策，但影响更像类别先验偏置，而不是稳定地提供了下一 40 ms 所需的信息。

### 按测试会话的 Accuracy

| 会话 | 样本 | hidden | given | shuffled | given − hidden |
| --- | ---: | ---: | ---: | ---: | ---: |
| SBC007 | 134 | 0.2836 | 0.2985 | 0.2985 | +0.0149 |
| SBC017 | 125 | 0.2320 | 0.2160 | 0.2000 | −0.0160 |
| SBC058 | 241 | 0.2324 | 0.1743 | 0.1618 | −0.0581 |

profile 效果随会话变化，且最大测试会话上的负效应主导了总体结果。

## 6. 推理效率

| 指标 | 结果 |
| --- | ---: |
| 请求 | 1,500 |
| 有效输出 | 1,500（100%） |
| 总墙钟时间 | 338.2 秒（5 分 38 秒） |
| 平均延迟 | 223.5 ms |
| 中位延迟 | 166.0 ms |
| P95 | 451.4 ms |
| P99 | 523.6 ms |
| 最大延迟 | 1,359.2 ms |
| 平均吞吐 | 约 4.44 requests/s |

首轮小试的第一次请求需要约 22.8 秒完成 CUDA 图编译与缓存预热；预热后的延迟才代表批量实验速度。

## 7. 结论

这次低成本验证得到的是一个明确的负/不确定结果：

1. 自然语言 profile 确实会改变 Qwen3-4B 的话轮分类决策；
2. 这种变化没有在验证集和正式测试间稳定泛化；
3. 正式测试中，正确 profile 反而低于无 profile，且对 `NA` 的损害最大；
4. 正确 profile 虽略高于 shuffled profile，但配对差异不足以支持“模型理解并有效利用了 profile”的结论；
5. 文本模型缺少声学状态，难以完成 40 ms 级的 `C/I/NA` 判断，不能替代正式音频模型。

论文或汇报中可以把它写为“text-only zero-shot profile prompting baseline”，但不能写成 profile 已被证明有效。

## 8. 下一步建议

1. 保留本结果作为低成本外部模型基线，不再根据 test 调 Prompt。
2. 正式主实验继续使用同一音频 checkpoint 的 `hidden/given/shuffled` profile adapter 比较。
3. 如果继续做 Prompt 实验，只在 val 上增加完全因果的声学状态摘要，或改用可以直接接收预测时刻前音频的多模态模型；修改后必须使用新的最终 test protocol。
4. 优先人工复核 BC、I、NA 和时间边界，否则更大文本模型也无法解决弱标签与声学缺失问题。

## 9. 本地产物

正式测试目录：`artifacts/prompt-baseline/local-qwen3-4b/test-100-per-class/`

- `requests.jsonl`：不含答案的真实模型请求；
- `gold.jsonl`：本地评分标签；
- `responses.jsonl`：逐请求输出、解析结果和延迟；
- `predictions.csv`：500 条成对预测；
- `metrics.json`：完整三条件指标与混淆矩阵；
- `profile_comparison.csv`：主要对比表；
- `paired_changes.json`：given/hidden 的成对变化；
- `response_validity.json`：输出有效性；
- `server.stderr.log`：模型加载和 llama.cpp timing 日志。

这些逐样本产物含 SBCSAE 转写或标识，保留在 Git 忽略的 `artifacts/`，不上传 GitHub。仓库只应发布本聚合报告、代码和无版权示例。

## 10. 复现实验命令

在仓库根目录启动本地服务：

```powershell
.\models\llama.cpp-b9987\llama-server.exe `
  -m .\models\Qwen3-4B-Q4_K_M.gguf `
  --host 127.0.0.1 --port 8081 `
  -c 8192 -ngl all -np 1 `
  --no-webui --reasoning off --reasoning-budget 0
```

另开一个 PowerShell，生成固定测试请求：

```powershell
.\.venv\Scripts\python.exe code\scripts\run_prompt_baseline.py prepare `
  --manifest data\processed\sbcsae_mvp\manifest.jsonl `
  --output-dir artifacts\prompt-baseline\local-qwen3-4b\test-100-per-class `
  --split test --max-per-class 100 --seed 13
```

调用本地接口并评分：

```powershell
$env:PROMPT_API_KEY = "local-no-auth"
.\.venv\Scripts\python.exe code\scripts\run_prompt_baseline.py run `
  --run-dir artifacts\prompt-baseline\local-qwen3-4b\test-100-per-class `
  --endpoint http://127.0.0.1:8081/v1/chat/completions `
  --model Qwen3-4B-Q4_K_M.gguf

.\.venv\Scripts\python.exe code\scripts\run_prompt_baseline.py score `
  --run-dir artifacts\prompt-baseline\local-qwen3-4b\test-100-per-class
```
