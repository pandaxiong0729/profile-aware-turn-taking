# Smoke 与质量验证报告

日期：2026-07-13
环境：Windows、Python 3.12、CPU PyTorch

## 最终验证

```powershell
python -m pytest tests
python -m compileall -q src scripts
profile-turntaking audit-preprocessed <完整数据参数>
profile-turntaking smoke --bundled-fixture --max-per-class 32 --work-dir <临时目录>
```

| 检查 | 结果 |
| --- | --- |
| 单元/集成测试 | 20 passed |
| Python bytecode compilation | passed |
| 全量数据跨产物审计 | 19/19 passed，0 error，0 warning failure |
| Bundled fixture prepare | passed |
| CPU train + checkpoint reload | passed |
| 同 checkpoint `hidden/given/shuffled` | passed |

## Bundled fixture 覆盖

| Label | Prepared samples |
| --- | ---: |
| `C` | 32 |
| `BC` | 32 |
| `T` | 10 |
| `I` | 11 |
| `NA` | 32 |

共 117 条样本，拆为 train 81、val 19、test 17；五个标签代码路径均被覆盖。训练只产生一个 checkpoint，评测对同一批 17 个 test sample IDs 依次运行三种 profile 条件并成功写出 `metrics.json`、`predictions.json` 和 `profile_comparison.csv`。

fixture 只有一段合成会话，所以 `shuffled` 会安全退化为全 `unknown`，不能作为 shuffled 负控制的科学结果。真实 SBCSAE test 有 3 段独立会话，代码会按会话整体轮换到另一段会话的 profile。

## 不能当作论文结果的原因

Smoke run 使用一段自行编写的合成对话、合成音频、3 秒上下文、弱标签和样本级分层拆分，只证明数据准备、训练、checkpoint 重载和 paired profile 评测链路可运行。所有 smoke Macro-F1 都不得引用为研究结果。

全量 SBCSAE 数据已经准备好，但在完成 VAD/overlap refinement、人工稀有类抽查和多随机种子训练之前，也不应把弱标签开发结果写成最终结论。
