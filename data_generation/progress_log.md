# Progress Log

## 2026-05-18

- Initialized the local Git repository structure for the mode classifier project.
- Created a dedicated `data_generation/` workspace for seed dataset generation.
- Added label definitions and edge-case rules for `chat` vs `motion_query`.
- Added a reproducible seed dataset generator.
- Generated `seed_500.jsonl` and `seed_500.csv` with 500 English human-robot utterances.
- Validated the seed dataset: 250 `chat`, 250 `motion_query`, 500 unique utterances.
- Simplified raw dataset exports to the training fields only: `id`, `utterance`, `label`.
- Documented that the current generator uses curated examples rather than an LLM API.
- Replaced the rigid hard-coded seed generator with a DeepSeek API-based generation pipeline.
- Moved the curated 500 examples to `data/reference/` for use as prompt reference data.
- Added prompt templates and validation scripts for generated CSV/JSONL outputs.
- Added Windows persisted environment variable lookup for the DeepSeek API key.
- Generated `deepseek_generated_500.csv` and `deepseek_generated_500.jsonl` with the DeepSeek API.
- Validated the generated dataset: 500 unique utterances, 250 `chat`, 250 `motion_query`, and no exact overlap with reference examples.
