# Server Setup

This project is expected to run on a shared GPU server. Always check current GPU usage before starting an embedding or training job.

## 1. Get The Repo

On the server:

```bash
git clone git@github.com:GeorgeCzy/mode-classifier.git
cd mode-classifier
```

If the repo already exists:

```bash
cd mode-classifier
git pull
```

## 2. Create Environment

Fast pip setup for the full modeling stack:

```bash
pip install -r requirements.txt
```

For the current local LLM classifier only, install the minimal vLLM stack:

```bash
pip install -r modeling/requirements-vllm.txt
```

vLLM is meant for Linux GPU deployment. On Windows, use a compatible Linux
server or WSL2/CUDA setup rather than a plain Windows Python environment.

For Weights & Biases tracking:

```bash
wandb login
```

Conda is recommended when CUDA/PyTorch versions matter. Create the base GPU environment first:

```bash
conda env create -f modeling/environment.yml
conda activate mode-classifier
```

Then install the remaining Python packages. If PyPI is slow from the server, use a mirror and longer timeout:

```bash
pip install -r modeling/requirements-server-extra.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120
```

If the mirror has issues, fall back to official PyPI:

```bash
pip install -r modeling/requirements-server-extra.txt --timeout 120
```

For the core modeling scripts without W&B or embedding extras:

```bash
pip install -r modeling/requirements.txt
```

## 3. Check GPU Availability

Use either command:

```bash
nvidia-smi
```

or:

```bash
gpustat
```

Prefer an idle GPU with low memory usage and low utilization.

You can also run the helper:

```bash
python modeling/scripts/check_gpu.py
```

## 4. Select A GPU

Use `CUDA_VISIBLE_DEVICES` to bind the process to one GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Then verify the selected GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For one-off commands:

```bash
CUDA_VISIBLE_DEVICES=0 python modeling/scripts/train_tfidf_logreg.py
```

The TF-IDF baseline does not need GPU. The current vLLM direct classifier,
embedding generation, and embedding-head training are GPU-oriented.

## 5. Run vLLM Direct Classifier

This branch defaults to:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

vLLM downloads the Hugging Face model files on the first run, including config,
tokenizer files, and safetensors weights. Later runs reuse the local Hugging
Face/vLLM cache.

Single utterance:

```bash
python modeling/scripts/predict_llm_instruct.py --text "Point to the exit."
```

Evaluate the full generated dataset with the prompt-only vLLM classifier:

```bash
python modeling/scripts/predict_llm_instruct.py \
  --eval-path data_generation/data/raw/deepseek_generated_3000.csv \
  --batch-size 16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

For multi-GPU tensor parallel inference:

```bash
CUDA_VISIBLE_DEVICES=0,1 python modeling/scripts/predict_llm_instruct.py \
  --eval-path data_generation/data/raw/deepseek_generated_3000.csv \
  --tensor-parallel-size 2
```

## 6. Run Baseline

```bash
python modeling/scripts/create_splits.py
python modeling/scripts/train_tfidf_logreg.py --use-wandb
```

Artifacts are written to:

```text
modeling/artifacts/
```

This directory is intentionally ignored by Git.

## 7. Embedding + MLP Direction

The next planned server workflow is:

```text
1. Choose an idle GPU with nvidia-smi or gpustat
2. export CUDA_VISIBLE_DEVICES=<gpu_id>
3. Generate cached embeddings with a Qwen embedding model
4. Train modeling/scripts/train_embedding_mlp.py on the cached .npz
```

Start with a smaller embedding model if GPU memory is unclear. A practical order is:

```text
Qwen3-Embedding-0.6B -> Qwen3-Embedding-4B -> Qwen3-Embedding-8B
```

Do not launch long jobs on a busy shared GPU.

Example:

```bash
export CUDA_VISIBLE_DEVICES=1
python modeling/scripts/cache_text_embeddings.py \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --output modeling/data/embeddings/qwen3_0_6b_deepseek_3000.npz

python modeling/scripts/train_embedding_mlp.py \
  --embedding-cache modeling/data/embeddings/qwen3_0_6b_deepseek_3000.npz \
  --use-wandb \
  --wandb-run-name qwen3-0.6b-mlp
```

The MLP script logs `train/batch_loss` every 10 optimizer steps by default. Use `--log-every-n-steps 1` for denser logging or a larger value for quieter runs.

If W&B has network issues, use offline mode:

```bash
python modeling/scripts/train_embedding_mlp.py --embedding-cache modeling/data/embeddings/qwen3_0_6b_deepseek_3000.npz --use-wandb --wandb-mode offline --wandb-run-name qwen3-0.6b-mlp
```
