# 模型说明

## Qwen2.5-Omni-3B

目录：`models/qwen2.5-omni-3b-local/`，约 11.16 GiB。

它用于从原始音频和因果转写重新提取 Qwen context `[2048]`，以及音频前端/编码器的 33 组末端时间表示 `[33,1280]`。

当前重新训练 adapter 时直接读取 `artifacts/main_experiment/qwen_feature_cache/`，因此不需要加载或上传这 11.16 GiB 权重。主要权重是三份 `model-0000x-of-00003.safetensors`；其余 JSON、词表和 tokenizer 文件用于恢复模型结构与分词。

## Talking Turns 官方模型

目录：`models/talking_turns/`，约 1.21 GiB。

- `checkpoint/`：Hugging Face 发布的 `espnet/Turn_taking_prediction_SWBD`，包含 `config.yaml` 和 `valid.loss.ave.pth`；
- `espnet_source/`：与 checkpoint 匹配的 ESPnet 源码版本 `cea64ab`。

```powershell
# 不重新推理，只验证并重算已经保存的结果
.\.venv\Scripts\python.exe code\scripts\run_espnet_talking_turns_baseline.py --score-only --device cpu

# 重新对全部 SBCSAE test 音频推理
.\.venv\Scripts\python.exe code\scripts\run_espnet_talking_turns_baseline.py --device cuda
```

这个模型只接受音频，不接受转写和 profile。

## `history/`

旧 GGUF、llama.cpp、重复 Hugging Face cache、Ultravox 和下载残片已经移到这里。它们不参与当前两条实验。该目录约占模型空间的大部分，确认不再追溯旧 prompt 试验后可以删除，也不应上传 GitHub 或云端主训练目录。
