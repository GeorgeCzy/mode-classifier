# Mode Classifier

This repository is for building a classifier that decides whether a human utterance in a human-robot conversation should receive a primarily verbal response (`chat`) or a physical-action response (`motion_query`).

The first milestone is broad seed data generation. See `data_generation/` for dataset files, generation scripts, and progress notes.

The second milestone is classifier modeling. See `modeling/` for train/validation/test splits, baseline training, and the embedding-head scaffold.

For server setup, install the full modeling stack with:

```bash
pip install -r requirements.txt
```

Training scripts can log to Weights & Biases with:

```bash
python modeling/scripts/train_tfidf_logreg.py --use-wandb
```
