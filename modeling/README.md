# Modeling

This folder contains the local LLM inference pipeline for the response-mode classifier.

The active branch classifies each utterance directly with a lightweight local instruct model through vLLM instead of training a TF-IDF model, embedding cache, or neural classifier head. The archived baseline scripts are still kept for comparison.

Default local model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The prompt is stored in:

```text
modeling/prompts/llm_direct_classifier_system.md
```

## What vLLM Does Here

vLLM is the runtime that serves the downloaded instruct model for inference. In
this project it replaces direct `transformers` generation. The script now uses:

```python
from vllm import LLM, SamplingParams
```

vLLM handles model loading, GPU memory management, prompt batching, and fast
token generation. The model is still `Qwen/Qwen2.5-0.5B-Instruct`; vLLM is the
engine used to run it.

## Labels

Runtime labels:

- `text`
- `motion prompt`

The existing generated datasets still use the older labels:

- `chat` maps to `text`
- `motion_query` maps to `motion prompt`

## Local LLM Inference

Install the full stack from the repository root on a Linux/CUDA deployment
machine:

```bash
pip install -r requirements.txt
```

For only the direct vLLM classifier:

```bash
pip install -r modeling/requirements-vllm.txt
```

vLLM is intended for GPU server deployment. If running from Windows, use a Linux
server or WSL2/CUDA environment that supports vLLM.

### Runtime Commands

Single-input classification:

```bash
python modeling/scripts/predict_llm_instruct.py --text "Can you do a short dance?"
```

Function: loads `Qwen/Qwen2.5-0.5B-Instruct` with vLLM, classifies the text
passed through `--text`, prints the label, raw LLM output, and latency, then
exits.

Interactive terminal classification:

```bash
python modeling/scripts/predict_llm_instruct.py
```

Function: loads the model once, then repeatedly accepts terminal input after the
`>` prompt and prints a label for each utterance. Submit an empty line or press
Ctrl+C to exit.

Interactive timing-test classification:

```bash
python modeling/scripts/predict_llm_instruct.py --timing
```

Function: keeps the same interactive input/output flow, but also prints
`request_elapsed_seconds` for each utterance. When the program exits, it prints
`average_request_elapsed_seconds`, the average end-to-end time from receiving
input to completing the LLM classification.

Default yes/no mode:

```bash
python modeling/scripts/predict_llm_instruct.py --text "Point to the exit."
```

Function: asks the model whether concrete physical action is required and maps
`NO` to `text` and `YES` to `motion prompt`. This is the default method and
matches the earlier prompt version that reached 0.84 accuracy on the generated
200-row test split.

Optional generated-label mode:

```bash
python modeling/scripts/predict_llm_instruct.py --method generate --text "Point to the exit."
```

Function: asks the model to emit `text` or `motion prompt` directly.

The classifier is few-shot, not zero-shot. Each request includes the system
prompt plus curated user/assistant examples covering ordinary text questions,
capability questions, short motion commands, gesture requests, and negated
motion requests. This version intentionally restores the fuller prompt structure
from the previous 0.84-accuracy run, so the default `--max-model-len` is 8192.

Full generated-dataset evaluation:

```bash
python modeling/scripts/predict_llm_instruct.py \
  --eval-path data_generation/data/raw/deepseek_generated_3000.csv
```

Function: runs the prompt-only vLLM classifier on all 3000 generated examples,
prints progress, computes accuracy/confusion matrix/latency metrics, and writes
outputs under `modeling/artifacts/llm_instruct/`.

Quick 20-row smoke evaluation:

```bash
python modeling/scripts/predict_llm_instruct.py \
  --eval-path data_generation/data/raw/deepseek_generated_3000.csv \
  --limit 20
```

Function: verifies the inference path on 20 test examples without running the
full 3000-row evaluation.

Evaluation with explicit vLLM options:

```bash
python modeling/scripts/predict_llm_instruct.py \
  --eval-path data_generation/data/raw/deepseek_generated_3000.csv \
  --batch-size 16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

Function: runs evaluation while controlling batch size, tensor parallelism, and
GPU memory utilization. The default `--max-model-len` is 8192 because the
restored fuller few-shot prompt is longer than the compact prompt variants.

The first run downloads the model from Hugging Face and later runs reuse the
local cache.

Outputs are written under:

```text
modeling/artifacts/llm_instruct/
```

This directory is git-ignored because prediction artifacts can be regenerated.

An earlier 200-row test run is summarized in:

```text
modeling/reports/qwen2_5_0_5b_instruct_direct.md
```

## Data

Default vLLM evaluation input:

```text
data_generation/data/raw/deepseek_generated_3000.csv
```

Expected columns:

```text
id,utterance,label
```

Legacy dataset labels:

- `chat`
- `motion_query`

The stratified split files under `modeling/data/splits/` are retained for the
archived TF-IDF and embedding-head baselines. The vLLM classifier does not train
on those splits.

## Archived Baselines

Previous scripts used:

1. Stratified train/validation/test split
2. TF-IDF + logistic regression baseline
3. A prepared PyTorch MLP head for cached text embeddings

Recommended baseline run:

From the repository root:

```bash
python modeling/scripts/create_splits.py
python modeling/scripts/train_tfidf_logreg.py
```

To log metrics to Weights & Biases:

```bash
wandb login
python modeling/scripts/train_tfidf_logreg.py --use-wandb
```

Default W&B target:

```text
entity: chengzy2023-shanghaitech-university
project: mode-classifier
```

For server setup and GPU selection on a shared machine, see `modeling/server_setup.md`.

If you use the Anaconda Python installed on the local Windows machine:

```powershell
& "D:\anaconda3\python.exe" modeling/scripts/create_splits.py
& "D:\anaconda3\python.exe" modeling/scripts/train_tfidf_logreg.py
```

The first baseline run is summarized in `modeling/reports/tfidf_logreg_baseline.md`.

## Try The Baseline

After training `train_tfidf_logreg.py`, classify one utterance:

```bash
python modeling/scripts/predict_tfidf_logreg.py --text "Can you do a short dance?"
```

Or start an interactive prompt:

```bash
python modeling/scripts/predict_tfidf_logreg.py
```

## Why Start With TF-IDF

The classifier is a semantic binary task, but many important cues are lexical: `point`, `bring`, `turn`, `follow`, `show me`, `demonstrate`, `stay still`, and so on. A simple baseline helps reveal whether larger embeddings are adding real value.

## Embedding + MLP Plan

Use an embedding model to create cached vectors:

```text
utterance -> embedding model -> cached .npz vectors -> MLP head -> label
```

The MLP script expects cached embeddings instead of calling a model directly. This keeps expensive embedding generation separate from classifier training.

Recommended first embedding model:

```text
Qwen/Qwen3-Embedding-0.6B
```

Create the embedding cache:

```bash
python modeling/scripts/cache_text_embeddings.py \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --output modeling/data/embeddings/qwen3_0_6b_deepseek_3000.npz
```

If embeddings were written but metadata failed, regenerate metadata only:

```bash
python modeling/scripts/cache_text_embeddings.py --model-name Qwen/Qwen3-Embedding-0.6B --output modeling/data/embeddings/qwen3_0_6b_deepseek_3000.npz --metadata-only
```

Train the MLP head:

```bash
python modeling/scripts/train_embedding_mlp.py \
  --embedding-cache modeling/data/embeddings/qwen3_0_6b_deepseek_3000.npz \
  --use-wandb \
  --wandb-run-name qwen3-0.6b-mlp
```

This MLP script logs batch loss every 10 optimizer steps by default, plus train/validation metrics after each epoch. Change the frequency with `--log-every-n-steps`.

If W&B online sync is unstable, use offline mode and sync later:

```bash
python modeling/scripts/train_embedding_mlp.py --embedding-cache modeling/data/embeddings/qwen3_0_6b_deepseek_3000.npz --use-wandb --wandb-mode offline --wandb-run-name qwen3-0.6b-mlp
```

Try the embedding model:

```bash
python modeling/scripts/predict_embedding_mlp.py \
  --model-dir modeling/artifacts/embedding_mlp \
  --text "Can you do a short dance?"
```
