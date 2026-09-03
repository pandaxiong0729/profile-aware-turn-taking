# Qwen-Omni hidden-state profile adapter 本机实验说明

这个实验不是让 Qwen 直接用 prompt 输出标签，而是把 Qwen2.5-Omni Thinker 当成冻结的多模态编码器。

每条样本仍然遵守同一个输入契约：

1. 预测点 `t` 以前的因果音频；
2. 与音频匹配、只到 `t` 以前的因果转写；
3. profile 条件：`hidden / given / shuffled`。

区别是：Qwen 先把“音频 + 转写 + 任务问题”编码成一个 hidden vector；profile 也用 Qwen 文本路径编码成另一个 vector；最后只训练一个小的融合分类头，输出五类 `C / BC / T / I / NA`。

## 一、输出文件

实验完成后应得到：

- `summary.json`：主结果、配置、样本数、profile 效果门禁；
- `predictions.jsonl`：逐样本预测和五类概率；
- `profile_comparison.csv`：hidden/given/shuffled 的 Macro-F1 对比；
- `seed-*.train.json`：每个随机种子的训练历史；
- `seed-*.metrics.json`：每个随机种子的详细评测；
- `*.qwen-hidden.meta.json`：Qwen hidden 缓存审计。

## 二、本机环境

需要：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\code[qwen-hidden,dev]"
.\.venv\Scripts\python.exe -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

如果 CUDA Torch 安装成功，应看到：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

输出里 `torch.cuda.is_available()` 应为 `True`。

## 三、先跑 10 条 smoke

先只缓存每个 split 的前 10 条，用来确认 Qwen 能加载、能返回 hidden states、输入格式没有错。

```powershell
.\.venv\Scripts\python.exe code\scripts\run_qwen_hidden_profile_experiment.py full `
  --data-dir data\processed\sbcsae_semantic_profile_v1 `
  --cache-dir artifacts\qwen-hidden-profile\smoke10\cache `
  --output-dir artifacts\qwen-hidden-profile\smoke10\run `
  --limit 10 `
  --torch-dtype float16 `
  --device-map auto `
  --offload-folder artifacts\qwen-hidden-profile\offload `
  --seeds 13 `
  --epochs 8 `
  --patience 3 `
  --device cpu
```

这一步只用于验证流水线，不能当 profile 效果结论。

## 四、正式本机实验

smoke 通过后，去掉 `--limit`，跑完整的 train/val/test：

```powershell
.\.venv\Scripts\python.exe code\scripts\run_qwen_hidden_profile_experiment.py full `
  --data-dir data\processed\sbcsae_semantic_profile_v1 `
  --cache-dir artifacts\qwen-hidden-profile\full\cache `
  --output-dir artifacts\qwen-hidden-profile\full\run `
  --torch-dtype float16 `
  --device-map auto `
  --offload-folder artifacts\qwen-hidden-profile\offload `
  --seeds 13 37 71 `
  --epochs 40 `
  --patience 8 `
  --device cpu
```

## 五、怎样看结果

主要看 `summary.json` 里的：

- `aggregate.hidden.macro_f1_mean`
- `aggregate.given.macro_f1_mean`
- `aggregate.shuffled.macro_f1_mean`
- `aggregate.given_minus_hidden_macro_f1.mean`
- `aggregate.given_minus_shuffled_macro_f1.mean`
- `interpretation_gate.all_modes_noncollapsed`

如果：

```text
given > hidden
given > shuffled
三组都不塌陷
```

才可以说本机实验支持 profile 有帮助。

如果只在测试集高、验证集不高，只能说是趋势，不能写成确定结论。
