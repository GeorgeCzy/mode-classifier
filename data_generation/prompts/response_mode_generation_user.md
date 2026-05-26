Generate exactly {batch_size} new examples for the response-mode classifier.

Required label counts in this batch:

- `chat`: {chat_count}
- `motion_query`: {motion_count}

The counts above are strict. Return exactly {chat_count} examples labeled `chat`
and exactly {motion_count} examples labeled `motion_query`. If a requested count
is 0, do not return any examples with that label.

Coverage goals for this batch:

- Everyday small talk and factual questions
- Explanations, recommendations, planning, emotional support, and troubleshooting
- Robot capability questions that should remain `chat`
- Explicit movement requests, gestures, demonstrations, pointing, gaze/orientation, locomotion, fetching, object manipulation, safety stop/stay commands, and nonverbal answers
- Boundary cases that test the rule: if the utterance is ambiguous or could be handled either verbally or physically, label it `chat`; use `motion_query` only for clearly requested robot action

Reference examples:

{reference_examples}

Output schema:

{{
  "examples": [
    {{
      "utterance": "English human utterance",
      "label": "chat"
    }}
  ]
}}

Constraints:

- Use only the labels `chat` and `motion_query`.
- The total number of examples must be exactly {batch_size}.
- Each utterance must be one sentence or one short user turn.
- Do not include IDs; the script will assign IDs.
- Do not include scenario names, explanations, comments, or metadata.
- Do not repeat reference examples verbatim.
- Do not repeat examples within the batch.
