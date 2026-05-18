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

For server setup and GPU selection on a shared machine, see `modeling/server_setup.md`.

If you use the Anaconda Python installed on the local Windows machine:

```powershell
& "D:\anaconda3\python.exe" modeling/scripts/create_splits.py
& "D:\anaconda3\python.exe" modeling/scripts/train_tfidf_logreg.py
```

The first baseline run is summarized in `modeling/reports/tfidf_logreg_baseline.md`.

## Why Start With TF-IDF

The classifier is a semantic binary task, but many important cues are lexical: `point`, `bring`, `turn`, `follow`, `show me`, `demonstrate`, `stay still`, and so on. A simple baseline helps reveal whether larger embeddings are adding real value.

## Embedding + MLP Plan

After baseline evaluation, use an embedding model to create cached vectors:

```text
utterance -> embedding model -> cached .npz vectors -> MLP head -> label
```

The MLP script expects cached embeddings instead of calling a model directly. This keeps expensive embedding generation separate from classifier training.
