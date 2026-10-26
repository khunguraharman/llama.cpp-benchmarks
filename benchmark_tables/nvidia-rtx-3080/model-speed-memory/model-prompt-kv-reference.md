# Model prompt length and FP16 KV-cache reference

Weights: **Q8_0**. Keys and values: **FP16 (2 bytes per element)**.

Average prompt length is the arithmetic mean of `prompt_tokens` over successful requests in each model-speed benchmark JSON. Each run has concurrency 1 and one output token per request.

| Model | Requests | Average input tokens | Hidden width | Layers | Query heads | KV heads | Head dimension | KV KiB/token | Average-prompt KV MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama 3.2 3B | 300 | 855.23 | 3,072 | 28 | 24 | 8 | 128 | 112 | 93.54 |
| Qwen3 4B | 300 | 835.74 | 2,560 | 36 | 32 | 8 | 128 | 144 | 117.53 |
| Phi 3.5 mini | 300 | 1142.46 | 3,072 | 32 | 32 | 32 | 96 | 384 | 428.42 |

## Calculation

`KV bytes/token = 2 (key + value) × layers × KV heads × head dimension × 2 (FP16 bytes)`

`Average-prompt KV MiB = average input tokens × KV bytes/token / 1,048,576`

These are theoretical logical KV payload sizes for one sequence containing the average input prompt, across all layers. They exclude output tokens, allocation padding, metadata, temporary buffers, model weights, and duplicate or retained cached sequences. They do not specify whether the cache resides in RAM or VRAM, or the total preallocated cache capacity. Q8_0 weight quantization does not change the FP16 KV calculation.

## Sources

- **Llama 3.2 3B**: [benchmark JSON](../../../results/nvidia-rtx-3080/model-speed-bench-results/Llama-3.2-3B-Instruct-Q8_0_throughput_1k_low_entropy_osl1.json); [architecture configuration](https://facebookresearch.github.io/fairseq2/stable/_modules/fairseq2/models/llama/config.html).
- **Qwen3 4B**: [benchmark JSON](../../../results/nvidia-rtx-3080/model-speed-bench-results/Qwen3-4B-Q8_0_throughput_1k_low_entropy_osl1.json); [architecture configuration](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json).
- **Phi 3.5 mini**: [benchmark JSON](../../../results/nvidia-rtx-3080/model-speed-bench-results/Phi-3.5-mini-instruct-Q8_0_throughput_1k_low_entropy_osl1.json); [architecture configuration](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/raw/main/config.json).

Architecture dimensions refer to the standard named model configurations; FP16 KV precision is supplied by the benchmark owner.
