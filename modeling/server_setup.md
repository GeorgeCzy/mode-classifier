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

For Weights & Biases tracking:

```bash
wandb login
```

Conda is recommended when CUDA/PyTorch versions matter:

```bash
conda env create -f modeling/environment.yml
conda activate mode-classifier
```

If the server already has a working PyTorch environment:

```bash
pip install -r requirements.txt
```

For the lightweight TF-IDF baseline only:

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

Then verify PyTorch sees only that device:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For one-off commands:

```bash
CUDA_VISIBLE_DEVICES=0 python modeling/scripts/train_tfidf_logreg.py
```

The TF-IDF baseline does not need GPU, but embedding generation and embedding-head training may benefit from one.

## 5. Run Baseline

```bash
python modeling/scripts/create_splits.py
python modeling/scripts/train_tfidf_logreg.py --use-wandb
```

Artifacts are written to:

```text
modeling/artifacts/
```

This directory is intentionally ignored by Git.

## 6. Embedding + MLP Direction

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
