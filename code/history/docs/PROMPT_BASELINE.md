# 现成大模型 Profile Prompt 低成本验证

## 目的

这个实验不训练、微调或保存任何模型参数。它只回答一个前置问题：在相同的历史转写上，把正确 profile 写进自然语言 prompt，是否比不提供 profile 更容易预测五分类话轮事件。

三种条件使用同一个现成模型、同一个 system prompt 和同一批样本：

| 条件 | 输入中的 profile | 作用 |
| --- | --- | --- |
| `hidden` | `Profile information is unavailable.` | 无 profile 基线 |
| `given` | 当前会话的正确 profile | 测量正确 profile 的增量 |
| `shuffled` | 另一会话的完整 profile | 错误 profile 负控制 |

输出始终是 `C / BC / T / I / NA` 中的一类。

## 重要边界

这是文本 Prompt baseline。模型会看到预测时刻之前已经完成的历史转写和可选 profile，但不会听到音频，因此看不到音量、语调、停顿和重叠。低分不能证明 profile 无效，高分也不能替代正式音频模型实验。

请求文件与答案文件物理分离：

- `requests.jsonl`：真正发送给 API 的 messages，不含 target；
- `gold.jsonl`：本地评分用答案，绝不发送；
- `responses.jsonl`：API 原始响应和解析后的预测。

代码通过字段白名单构造 prompt，不会把 `training_target`、`annotation_only_not_model_input`、`label_evidence` 或预测时刻之后的文字放入请求。

## 推荐测试时间线

### 1. 免费检查 Prompt

在仓库的 `code/` 目录安装项目后，先从验证集每类抽 2 条，只生成文件而不调用 API：

```bash
python scripts/run_prompt_baseline.py prepare \
  --manifest ../data/processed/sbcsae_mvp/manifest.jsonl \
  --output-dir ../artifacts/prompt-baseline/val-inspection \
  --split val \
  --max-per-class 2 \
  --seed 13
```

人工检查 `requests.jsonl`：历史文本是否截止正确时间、profile 是否可读、请求中是否没有答案。

### 2. 验证集小试

固定 Prompt 模板后，在验证集每类抽 20 条，共 100 个样本。三个条件会产生 300 次 API 请求：

```bash
python scripts/run_prompt_baseline.py prepare \
  --manifest ../data/processed/sbcsae_mvp/manifest.jsonl \
  --output-dir ../artifacts/prompt-baseline/val-20 \
  --split val \
  --max-per-class 20 \
  --seed 13
```

设置所用平台的密钥。脚本兼容提供 `/chat/completions` 接口的平台：

```bash
export PROMPT_API_KEY='在云平台的密钥管理中设置，不要写进代码或日志'
```

先只发送 6 个请求，验证接口格式和输出解析；这恰好覆盖前两个样本的三个 profile 条件：

```bash
python scripts/run_prompt_baseline.py run \
  --run-dir ../artifacts/prompt-baseline/val-20 \
  --endpoint '<API_BASE>/chat/completions' \
  --model '<MODEL_NAME>' \
  --limit 6
```

确认 `responses.jsonl` 正常后，去掉 `--limit` 继续运行。脚本会跳过已经成功写入的 `request_id`，中断后可以直接续跑：

```bash
python scripts/run_prompt_baseline.py run \
  --run-dir ../artifacts/prompt-baseline/val-20 \
  --endpoint '<API_BASE>/chat/completions' \
  --model '<MODEL_NAME>'
```

所有条件完成后评分：

```bash
python scripts/run_prompt_baseline.py score \
  --run-dir ../artifacts/prompt-baseline/val-20
```

不要根据 test 结果反复修改 Prompt。只能在 val 小试中确定模板、模型和解析规则。

### 3. 固定设置后跑测试集

建议先每类 100 条，即最多 500 个样本、1,500 次请求。测试集的 `T` 只有 125 条，因此不建议把每类上限设得更高来制造不对称规模。

```bash
python scripts/run_prompt_baseline.py prepare \
  --manifest ../data/processed/sbcsae_mvp/manifest.jsonl \
  --output-dir ../artifacts/prompt-baseline/test-100 \
  --split test \
  --max-per-class 100 \
  --seed 13
```

随后执行相同的 `run` 和 `score` 命令。

## 输出如何阅读

| 文件 | 含义 |
| --- | --- |
| `run_config.json` | 样本数、请求数、seed 和限制声明 |
| `requests.jsonl` | 不含答案的真实 API 请求 |
| `gold.jsonl` | 本地 target，不发送给模型 |
| `responses.jsonl` | 逐请求原始输出、预测、延迟和错误 |
| `metrics.json` | 三种条件的 Macro-F1、Balanced Accuracy、每类指标和混淆矩阵 |
| `profile_comparison.csv` | `hidden/given/shuffled` 以及 `given-hidden` 差值 |
| `predictions.json` / `predictions.csv` | 同一样本三条件的成对预测，CSV 方便老师直接查看 |
| `paired_changes.json` | profile 修正了多少无 profile 错误，又破坏了多少原本正确预测 |
| `response_validity.json` | 无效输出和三条件共同有效的样本量 |

支持“profile 可能有效”的最小证据应同时满足：

1. `given` 的 Macro-F1 高于 `hidden`；
2. `given` 高于 `shuffled`，否则可能只是额外文本带来的变化；
3. `given_fixes_hidden_error` 多于 `given_breaks_hidden_correct`；
4. 增益不是只来自大量类别，BC、T、I 的 per-class F1 也要单独看；
5. API 无效输出比例足够低，三条件评分使用完全相同的有效 sample IDs。

验证集 20 条/类只能判断流程和大致趋势，不能写成论文结论。正式结果仍需要完整音频模型、多个随机种子和人工核验标签。
