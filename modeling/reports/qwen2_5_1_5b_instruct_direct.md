# Qwen2.5-1.5B-Instruct Direct Classification

This branch tests the same prompt-only vLLM classifier with:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

No model training is required. The prompt and inference code are the same
workflow as the 0.5B branch; vLLM downloads the 1.5B model files on first use
and runs local inference afterward.

Recommended smoke test:

```bash
python modeling/scripts/predict_llm_instruct.py \
  --eval-path data_generation/data/raw/deepseek_generated_3000.csv \
  --limit 20 \
  --batch-size 16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.60 \
  --timing
```

Recommended full evaluation:

```bash
python modeling/scripts/predict_llm_instruct.py \
  --eval-path data_generation/data/raw/deepseek_generated_3000.csv \
  --batch-size 16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.60 \
  --timing
```
