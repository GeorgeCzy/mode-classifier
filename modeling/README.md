# Modeling

This folder contains the first modeling pipeline for the response-mode classifier.

The current goal is to build a reliable baseline before using a large embedding model. The first version uses:

1. Stratified train/validation/test split
2. TF-IDF + logistic regression baseline
3. A prepared PyTorch MLP head for cached text embeddings

## Data

Default input:

```text
data_generation/data/raw/deepseek_generated_2000.csv
```

Expected columns:

```text
id,utterance,label
```

Labels:

- `chat`
- `motion_query`

## Recommended First Run

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
  --output modeling/data/embeddings/qwen3_0_6b_deepseek_2000.npz
```

If embeddings were written but metadata failed, regenerate metadata only:

```bash
python modeling/scripts/cache_text_embeddings.py --model-name Qwen/Qwen3-Embedding-0.6B --output modeling/data/embeddings/qwen3_0_6b_deepseek_2000.npz --metadata-only
```

Train the MLP head:

```bash
python modeling/scripts/train_embedding_mlp.py \
  --embedding-cache modeling/data/embeddings/qwen3_0_6b_deepseek_2000.npz \
  --use-wandb \
  --wandb-run-name qwen3-0.6b-mlp
```

This MLP script logs batch loss every 10 optimizer steps by default, plus train/validation metrics after each epoch. Change the frequency with `--log-every-n-steps`.

If W&B online sync is unstable, use offline mode and sync later:

```bash
python modeling/scripts/train_embedding_mlp.py --embedding-cache modeling/data/embeddings/qwen3_0_6b_deepseek_2000.npz --use-wandb --wandb-mode offline --wandb-run-name qwen3-0.6b-mlp
```

Try the embedding model:

```bash
python modeling/scripts/predict_embedding_mlp.py \
  --model-dir modeling/artifacts/embedding_mlp \
  --text "Can you do a short dance?"
```
