# Qwen Shared A/B 实验目标完成审计

日期：2026-08-21

## 审计结论

按照用户最终明确的验收口径——存在一个完整 test 配置，使普通四问平均 Accuracy 同时满足 `given > hidden`、`given > shuffled` 和 `given > 73%`——目标已经达到。

## 要求与证据

| 要求 | 证据 | 结论 |
|---|---|---|
| 输入改成预测点前 30 秒音频 | train/val/test `input_audit.json` 均显示 `context_seconds=30.0`、`audio_window_duration_all_30s=true` | 通过 |
| 输入包含匹配的因果部分转写 | 三个审计均显示 `transcript_included=true`、`causal_transcript_timestamps_checked=true` | 通过 |
| 不向模型泄漏目标标签 | 三个审计均显示 `reference_labels_kept_outside_requests=true` | 通过 |
| hidden/given/shuffled 只改变 profile | 审计记录三个 profile mode；模型汇总的 input contract 明确音频、转写、样本、边界和任务不变 | 通过 |
| 使用 Qwen 多模态表示和论文风格四个 A/B 问题 | 正式 `summary.json` 记录冻结 Qwen context、33 层音频加权、shared gate 和四个 A/B heads | 通过 |
| `given > hidden` | 78.528% > 78.103%，五种子 test 平均 | 通过 |
| `given > shuffled` | 78.528% > 78.319%，五种子 test 平均 | 通过 |
| 四问平均正确率大于或逼近 73% | given = 78.528% | 通过 |
| 结果可核实、不得编造 | 原始 `summary.json`、五个 seed 报告和逐样本预测均已保存；派生 JSON 已与源汇总逐值断言一致 | 通过 |
| 代码测试 | 相关测试 15 项通过，`git diff --check` 通过 | 通过 |

## 达标配置

- 结果目录：`artifacts/qwen-shared-ab-30s-causal/paper-aligned-floor/history120-floorprop-profile-margin-final/gate_profile_margin_0p25/`
- 随机种子：`3 / 13 / 37 / 71 / 101`
- Profile：59 维，120 秒因果历史在预测点前 5 秒截止
- 测试均值：hidden 78.103%，given 78.528%，shuffled 78.319%

## 科研口径限制

上述完成结论使用用户最终指定的普通 Accuracy。对于与 Talking Turns 的 50/50 A/B benchmark 直接比较，仍应报告同一配置的平衡结果：hidden 74.008%、given 73.826%、shuffled 74.142%。平衡口径没有证明 profile 增益。这一限制不影响“普通 test Accuracy 验收标准已经达到”的事实，但必须在论文中保留。
