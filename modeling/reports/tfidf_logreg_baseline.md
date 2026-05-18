# TF-IDF Logistic Regression Baseline

Run date: 2026-05-18

Input dataset:

```text
data_generation/data/raw/deepseek_generated_500.csv
```

Split:

| Split | Rows | chat | motion_query |
| --- | ---: | ---: | ---: |
| Train | 400 | 200 | 200 |
| Validation | 50 | 25 | 25 |
| Test | 50 | 25 | 25 |

Model:

```text
TfidfVectorizer(1-2 grams, max_features=5000)
+ LogisticRegression(class_weight="balanced")
```

Results:

| Split | Accuracy |
| --- | ---: |
| Train | 0.9975 |
| Validation | 0.9800 |
| Test | 0.9800 |

Validation confusion matrix:

| gold \ pred | chat | motion_query |
| --- | ---: | ---: |
| chat | 24 | 1 |
| motion_query | 0 | 25 |

Test confusion matrix:

| gold \ pred | chat | motion_query |
| --- | ---: | ---: |
| chat | 25 | 0 |
| motion_query | 1 | 24 |

Observed errors:

| Split | ID | Gold | Predicted | Utterance |
| --- | --- | --- | --- | --- |
| Validation | deepseek-00204 | chat | motion_query | Please just talk to me in words. |
| Test | deepseek-00418 | motion_query | chat | Can you guide me through this maze? |

Notes:

- The baseline is already strong, likely because many motion queries contain clear lexical cues.
- The observed errors are useful hard cases for later evaluation and prompt refinement.
- The next model should evaluate whether embeddings improve boundary cases rather than only headline accuracy.

