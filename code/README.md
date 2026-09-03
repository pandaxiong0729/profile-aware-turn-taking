# 当前代码

## 推荐入口

| 文件 | 作用 | 是否直接运行 |
| --- | --- | --- |
| `scripts/inspect_experiment_data.py` | 查看任意一条真实样本、输入、标签和两种模型预测 | 是 |
| `scripts/run_main_experiment.py` | 主实验的检查、结果查看、重新训练统一入口 | 是 |
| `scripts/run_qwen_shared_binary_multitask_adapter.py` | 主模型结构、loss、训练和评测实现 | 由上一行调用 |
| `scripts/run_espnet_talking_turns_baseline.py` | 官方 Talking Turns checkpoint 的运行与评分 | 是 |
| `scripts/audit_fiveclass_event_labels.py` | 检查事件标签、split、五类覆盖和来源一致性 | 由主入口调用 |
| `scripts/build_dynamic_behavior_profile_views.py` | 从历史对话重建 59 维 profile 特征 | 需要重建 profile 时运行 |
| `scripts/build_collaboration_manifest.py` | 生成交接必需文件的大小与 SHA-256 | 交接前运行 |
| `scripts/prepare_cloud_handoff.py` | 生成逐文件清单或云端压缩包 | 交接前运行 |

## 公共函数

| 文件 | 作用 |
| --- | --- |
| `src/profile_turntaking/audio.py` | WAV 读取、重采样和稳健单双声道混合 |
| `src/profile_turntaking/event_annotation.py` | IPU 与 C/BC/T/I/NA 事件规则实现 |
| `src/profile_turntaking/paper_aligned_floor_targets.py` | interruption 后是否取得话轮的目标构造 |
| `src/profile_turntaking/constants.py` | 标签及 backchannel 词表 |
| `src/profile_turntaking/schemas.py` | 数据结构 |
| `src/profile_turntaking/utils.py` | JSONL 和哈希等通用函数 |
| `src/profile_turntaking/__init__.py` | Python 包入口 |

## 当前测试

- `tests/test_qwen_shared_binary_adapter.py`：验证四任务目标、平衡抽样、profile margin、数值标准化和 adapter 前向结构。
- `tests/test_espnet_talking_turns_metrics.py`：验证五类概率名称与 ROC-AUC 的对应关系。

## 安装与运行

在仓库根目录：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\code[dev]"

# 先检查，再训练
.\.venv\Scripts\python.exe code\scripts\run_main_experiment.py check
.\.venv\Scripts\python.exe code\scripts\run_main_experiment.py train --device cuda

# 当前代码测试
.\.venv\Scripts\python.exe -m pytest code\tests
```

重新训练主 adapter 不需要加载 Qwen 权重，因为 Qwen 特征已经缓存。

## history

`code/history/` 保存之前的 prompt 试验、五分类试验、soft-prompt 探索、诊断脚本、旧测试、旧报告和旧配置。它们不被当前入口导入。确认论文不再需要后，可以整体删除；目前先保留。
