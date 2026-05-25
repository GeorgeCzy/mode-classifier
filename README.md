# Mode Classifier

This repository is for classifying whether a human utterance in a human-robot conversation should receive a primarily verbal/text response (`text`) or a physical-action response (`motion prompt`).

The first milestone is broad seed data generation. See `data_generation/` for dataset files, generation scripts, and progress notes.

The current branch uses a locally deployed lightweight instruct LLM with vLLM for direct classification. It no longer requires training a neural classifier head. See `modeling/` for the prompt, inference script, train/validation/test splits, and evaluation commands.

For server setup, install the full modeling stack with:

```bash
pip install -r requirements.txt
```

Run one local vLLM classification with Qwen2.5-0.5B-Instruct:

```bash
python modeling/scripts/predict_llm_instruct.py --text "Can you do a short dance?"
```

Evaluate accuracy and latency on the existing generated test split:

```bash
python modeling/scripts/predict_llm_instruct.py --eval-path modeling/data/splits/test.csv
```

The default model is:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

vLLM downloads the Hugging Face model the first time it runs and reuses the local cache after that. It is the inference engine used to load the model, batch prompts, manage GPU memory, and generate the labels efficiently on a deployment server.
