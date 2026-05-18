# Data Generation

This folder contains the seed data generation workflow for the response-mode classifier.

## Label Definitions

- `chat`: The human utterance can be answered primarily with speech or text. The robot may use small natural body language, but no explicit physical action is required.
- `motion_query`: The human utterance asks for, implies, or requires a clear robot action. This includes gestures, locomotion, object manipulation, demonstration, imitation, physical guidance, stopping, holding position, or answering nonverbally through motion.

## Edge Rules

- Questions about robot movement abilities are `chat` when the user only asks for information, such as "What gestures can you perform?"
- Requests like "show me", "demonstrate", "point to", "bring me", "turn toward", "follow me", or "answer with a gesture" are `motion_query`.
- If the user explicitly says not to move and asks for a verbal explanation, the label is `chat`.
- Safety or control commands that change the robot's physical behavior, such as "stop moving" or "back away", are `motion_query`.

## Current Seed Dataset

- `data/raw/seed_500.jsonl`: JSON Lines format for model training pipelines.
- `data/raw/seed_500.csv`: CSV format for quick inspection.
- Total examples: 500
- Distribution: 250 `chat`, 250 `motion_query`
- Language: English
- Fields: `id`, `utterance`, `label`
- Splits: Not assigned yet. A later step can create train/validation/test splits.

## Generation Method

The current seed dataset is generated from curated scenario groups inside `generate_seed_dataset.py`. The script does not call an LLM API. It stores hand-authored English utterances in broad categories, shuffles them with a fixed random seed, validates the label balance and uniqueness, then writes CSV and JSONL files.

## Regeneration

Run this from the repository root:

```powershell
python data_generation/generate_seed_dataset.py
```
