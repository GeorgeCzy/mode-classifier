# Qwen2.5-0.5B-Instruct Direct Classification

Run date: 2026-05-23

Model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Prompt:

```text
modeling/prompts/llm_direct_classifier_system.md
```

Inference method:

```text
Local vLLM offline inference.
Default method: yes/no decision over whether the robot must perform a concrete
physical action, mapped to runtime labels:
NO  -> text
YES -> motion prompt
```

Evaluation command:

```bash
python modeling/scripts/predict_llm_instruct.py --eval-path modeling/data/splits/test.csv
```

On a Linux/CUDA deployment machine:

```bash
python modeling/scripts/predict_llm_instruct.py --eval-path modeling/data/splits/test.csv --batch-size 16
```

Dataset:

```text
modeling/data/splits/test.csv
rows: 200
legacy labels: chat, motion_query
runtime labels: text, motion prompt
```

Historical results from the pre-vLLM local run:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8450 |
| Correct / total | 169 / 200 |
| Model load time | 5.50 s |
| Mean latency / sample | 0.2107 s |
| Median latency / sample | 0.2106 s |
| P95 latency / sample | 0.2164 s |
| Min latency / sample | 0.2040 s |
| Max latency / sample | 0.2288 s |

Confusion matrix:

| gold \ pred | text | motion prompt |
| --- | ---: | ---: |
| text | 88 | 12 |
| motion prompt | 19 | 81 |

Observed errors:

| ID | Gold | Predicted | Utterance |
| --- | --- | --- | --- |
| deepseek2000-01883 | text | motion prompt | Can you translate 'hello' into Spanish? |
| deepseek2000-01361 | text | motion prompt | Can you tell me about your capabilities? |
| deepseek2000-01657 | motion prompt | text | Press the button. |
| deepseek2000-01295 | motion prompt | text | Come here. |
| deepseek2000-00525 | motion prompt | text | Salute me formally. |
| deepseek2000-00260 | motion prompt | text | Signal me when it's safe. |
| deepseek2000-01091 | motion prompt | text | Arrange the cushions neatly. |
| deepseek2000-01474 | motion prompt | text | Press the button on the microwave. |
| deepseek2000-00689 | motion prompt | text | Answer by shaking your head no. |
| deepseek2000-00475 | motion prompt | text | Stay right there. |
| deepseek2000-01693 | motion prompt | text | Take a nap. |
| deepseek2000-01745 | motion prompt | text | Look at the screen. |

Notes:

- The current implementation uses vLLM as the inference backend. The numbers
  above were measured before the vLLM migration with the same model and prompt;
  rerun the command above on a vLLM-capable Linux/CUDA machine to refresh
  latency numbers for the new backend.
- Direct 0.5B LLM inference is simple and fast enough for a lightweight local
  baseline, but it is less accurate than the previous trained TF-IDF baseline on
  this generated test split.
- Most false negatives are short imperative action commands. Most false
  positives are capability, translation, or planning questions phrased with
  "Can you ...".
