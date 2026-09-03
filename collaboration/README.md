# 合作者入口

按下面顺序操作即可：

```powershell
# 1. 安装当前主实验依赖
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\code[dev]"

# 2. 看一条真实数据，以及我们和 Talking Turns 的输入区别
.\.venv\Scripts\python.exe code\scripts\inspect_experiment_data.py --split test --index 0

# 3. 检查数据并查看保存结果
.\.venv\Scripts\python.exe code\scripts\run_main_experiment.py check
.\.venv\Scripts\python.exe code\scripts\run_main_experiment.py show-results

# 4. 从缓存重新训练主 adapter
.\.venv\Scripts\python.exe code\scripts\run_main_experiment.py train --device cuda

# 5. 重算 Talking Turns 保存结果
.\.venv\Scripts\python.exe code\scripts\run_espnet_talking_turns_baseline.py --score-only --device cpu
```

完整路径、环境和上传说明见 [CLOUD_HANDOFF_COMPLETE_GUIDE_ZH.md](CLOUD_HANDOFF_COMPLETE_GUIDE_ZH.md)。每个最小文件的用途见 [PROJECT_FULL_FILE_INVENTORY.csv](PROJECT_FULL_FILE_INVENTORY.csv)。当前实验数字见 [../code/reports/CURRENT_EXPERIMENT_REPORT.md](../code/reports/CURRENT_EXPERIMENT_REPORT.md)。
