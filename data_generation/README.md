# Data Generation

This folder contains the data generation workflow for the response-mode classifier.

## Label Definitions

- `chat`: The human utterance can be answered primarily with speech or text. The robot may use small natural body language, but no explicit physical action is required.
- `motion_query`: The human utterance asks for, implies, or requires a clear robot action. This includes gestures, locomotion, object manipulation, demonstration, imitation, physical guidance, stopping, holding position, or answering nonverbally through motion.

## Edge Rules

- Questions about robot movement abilities are `chat` when the user only asks for information, such as "What gestures can you perform?"
- Requests like "show me", "demonstrate", "point to", "bring me", "turn toward", "follow me", or "answer with a gesture" are `motion_query`.
- If the user explicitly says not to move and asks for a verbal explanation, the label is `chat`.
- Safety or control commands that change the robot's physical behavior, such as "stop moving" or "back away", are `motion_query`.

## Current Seed Dataset

- `data/reference/manual_seed_500.csv`: Curated reference examples used inside LLM prompts.
- `data/reference/manual_seed_500.jsonl`: JSONL copy of the curated reference examples.
- Reference total examples: 500
- Reference distribution: 250 `chat`, 250 `motion_query`
- Exported training fields: `id`, `utterance`, `label`
- Splits: Not assigned yet. A later step can create train/validation/test splits.

## Generation Method

The current generation pipeline uses the DeepSeek chat completions API. The curated 500 examples are no longer hard-coded in a generator script; they are kept as prompt reference data so the LLM can learn the expected style and label boundaries.

The generator writes new candidates to `data/raw/` in both CSV and JSONL formats. The exported training files contain only `id`, `utterance`, and `label`.

Current generated dataset:

- `data/raw/deepseek_generated_2000.csv`
- `data/raw/deepseek_generated_2000.jsonl`
- Total examples: 2000
- Distribution: 1000 `chat`, 1000 `motion_query`
- Exact overlap with reference examples: 0

Component generated datasets:

- `data/raw/deepseek_generated_500.csv`
- `data/raw/deepseek_generated_extra_1500.csv`

The script reads the API key from one of these environment variables:

- `DEEPSEEK-mode-classifier-apikey`
- `DEEPSEEK_MODE_CLASSIFIER_APIKEY`
- `DEEPSEEK_API_KEY`

On Windows, the script also checks persisted User and Machine environment variables if the key was not inherited by the current process.

## Regeneration

Run this from the repository root:

```powershell
python data_generation/scripts/generate_with_deepseek.py --total 1500 --output-stem deepseek_generated_extra_1500 --id-prefix deepseek-extra --exclude-path data_generation/data/raw/deepseek_generated_500.csv
python data_generation/scripts/merge_datasets.py --input data_generation/data/raw/deepseek_generated_500.csv --input data_generation/data/raw/deepseek_generated_extra_1500.csv --output-stem deepseek_generated_2000 --id-prefix deepseek2000 --shuffle
python data_generation/scripts/validate_dataset.py data_generation/data/raw/deepseek_generated_2000.csv --expect-total 2000 --require-balanced
```

The default model is `deepseek-v4-flash`, using the OpenAI-compatible DeepSeek endpoint at `https://api.deepseek.com/chat/completions`.
