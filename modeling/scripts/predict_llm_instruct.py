"""Classify robot-directed utterances with a local instruct LLM."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT_PATH = ROOT / "modeling" / "prompts" / "llm_direct_classifier_system.md"
DEFAULT_EVAL_PATH = ROOT / "modeling" / "data" / "splits" / "test.csv"
DEFAULT_OUTPUT_DIR = ROOT / "modeling" / "artifacts" / "llm_instruct"
VALID_LABELS = ("text", "motion prompt")
TEXT_CUES = (
    "define, explain, recommend, translate, compare, tell me, what is, how do I, advice"
)
MOTION_CUES = (
    "point, place, bring, carry, spin, wave, gesture, use your hand, demonstrate, "
    "follow, turn, move, stop"
)
LABEL_MAP = {
    "chat": "text",
    "text": "text",
    "motion_query": "motion prompt",
    "motion prompt": "motion prompt",
    "motion_prompt": "motion prompt",
}
FEW_SHOT_EXAMPLES = (
    ("What is the capital of France?", "text"),
    ("How do I manage stress effectively?", "text"),
    ("Point to the exit.", "motion prompt"),
    ("Place the package on the table.", "motion prompt"),
    ("Can you translate hello into Spanish?", "text"),
    ("What is the difference between TCP and UDP?", "text"),
    ("Can you do a short dance?", "motion prompt"),
    ("Spin around slowly.", "motion prompt"),
    ("Can you define empathy?", "text"),
    ("Show me how you would greet someone.", "motion prompt"),
    ("Use your hand to gesture come here.", "motion prompt"),
    ("Can you recommend a book for learning Python?", "text"),
    ("Please do not move, just explain the answer.", "text"),
)


@dataclass
class Prediction:
    label: str
    raw_output: str
    latency_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct response-mode classification with a local instruct LLM."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--text", default=None, help="Single utterance to classify.")
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=None,
        help="CSV file with id, utterance, label columns. If set, run evaluation.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional eval row limit.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--method",
        choices=["yesno", "generate"],
        default="yesno",
        help=(
            "yesno asks whether concrete physical action is needed and maps the "
            "answer to labels; generate asks the LLM to emit a label."
        ),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use an already downloaded Hugging Face model.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to the vLLM engine.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the system prompt and exit.",
    )
    return parser.parse_args()


def read_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_vllm_engine(args: argparse.Namespace):
    try:
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError(
            "vLLM is not installed. On a deployment machine, run "
            "`pip install -r requirements.txt` or "
            "`pip install -r modeling/requirements-vllm.txt`."
        ) from error

    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    llm_kwargs: dict[str, Any] = {
        "model": args.model_name,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
    }
    if args.cache_dir:
        llm_kwargs["download_dir"] = str(args.cache_dir)
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop=["\n"],
    )
    tokenizer = llm.get_tokenizer()
    return llm, sampling_params, tokenizer


def label_prompt(utterance: str) -> str:
    return (
        "Classify this human utterance addressed to a humanoid robot.\n"
        "All inputs are text strings. The question is whether the robot should "
        "answer verbally or physically act.\n"
        f"Text cues: {TEXT_CUES}.\n"
        f"Motion cues: {MOTION_CUES}.\n"
        "Return only one label: text or motion prompt.\n\n"
        f"Utterance: {utterance}\n"
        "Label:"
    )


def yesno_prompt(utterance: str) -> str:
    return (
        "Classify this human utterance addressed to a humanoid robot.\n"
        "Does the utterance ask the robot to perform or change a concrete physical action?\n"
        "Answer YES for motion prompt. Answer NO for text.\n"
        f"Text cues: {TEXT_CUES}.\n"
        f"Motion cues: {MOTION_CUES}.\n"
        "Return only YES or NO.\n\n"
        f"Utterance: {utterance}\n"
        "Answer:"
    )


def example_answer(label: str, *, yesno_mode: bool) -> str:
    if yesno_mode:
        return "YES" if label == "motion prompt" else "NO"
    return label


def user_prompt_for_mode(
    utterance: str,
    *,
    yesno_mode: bool,
) -> str:
    if yesno_mode:
        return yesno_prompt(utterance)
    return label_prompt(utterance)


def build_messages(
    system_prompt: str,
    utterance: str,
    *,
    yesno_mode: bool = False,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    for example_text, label in FEW_SHOT_EXAMPLES:
        messages.append(
            {
                "role": "user",
                "content": user_prompt_for_mode(
                    example_text,
                    yesno_mode=yesno_mode,
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": example_answer(
                    label,
                    yesno_mode=yesno_mode,
                ),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": user_prompt_for_mode(
                utterance,
                yesno_mode=yesno_mode,
            ),
        }
    )
    return messages


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("-", "_")
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in LABEL_MAP:
        return LABEL_MAP[normalized]
    raise ValueError(f"Unsupported label: {label!r}")


def parse_generated_label(raw_output: str) -> str:
    cleaned = raw_output.strip().lower()
    cleaned = cleaned.replace("`", "").replace('"', "").replace("'", "")
    cleaned = re.sub(r"[^a-z_\s-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if "motion prompt" in cleaned or "motion_prompt" in cleaned:
        return "motion prompt"
    if cleaned.startswith("motion") or cleaned == "movement":
        return "motion prompt"
    if cleaned.startswith("text") or cleaned == "chat":
        return "text"
    raise ValueError(f"Could not parse model output as a label: {raw_output!r}")


def parse_generated_yesno(raw_output: str) -> str:
    cleaned = raw_output.strip().lower()
    cleaned = cleaned.replace("`", "").replace('"', "").replace("'", "")
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.startswith("yes"):
        return "motion prompt"
    if cleaned.startswith("no"):
        return "text"
    return parse_generated_label(raw_output)


def render_chat_prompt(
    tokenizer,
    system_prompt: str,
    utterance: str,
    *,
    yesno_mode: bool = False,
) -> str:
    messages = build_messages(
        system_prompt,
        utterance,
        yesno_mode=yesno_mode,
    )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_prompts(llm, sampling_params, prompts: list[str]) -> tuple[list[str], float]:
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    latency = time.perf_counter() - start
    raw_outputs = [output.outputs[0].text.strip() for output in outputs]
    per_prompt_latency = latency / len(prompts) if prompts else 0.0
    return raw_outputs, per_prompt_latency


def classify(
    *,
    tokenizer,
    llm,
    sampling_params,
    system_prompt: str,
    utterance: str,
    method: str,
) -> Prediction:
    predictions = classify_many(
        tokenizer=tokenizer,
        llm=llm,
        sampling_params=sampling_params,
        system_prompt=system_prompt,
        utterances=[utterance],
        method=method,
    )
    return predictions[0]


def classify_many(
    *,
    tokenizer,
    llm,
    sampling_params,
    system_prompt: str,
    utterances: list[str],
    method: str,
) -> list[Prediction]:
    yesno_mode = method == "yesno"
    prompts = [
        render_chat_prompt(
            tokenizer,
            system_prompt,
            utterance,
            yesno_mode=yesno_mode,
        )
        for utterance in utterances
    ]
    raw_outputs, per_prompt_latency = generate_prompts(llm, sampling_params, prompts)
    predictions = []
    for raw_output in raw_outputs:
        label = parse_generated_yesno(raw_output) if yesno_mode else parse_generated_label(raw_output)
        predictions.append(
            Prediction(
                label=label,
                raw_output=raw_output,
                latency_seconds=per_prompt_latency,
            )
        )
    return predictions


def read_eval_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"id", "utterance", "label"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return rows[:limit] if limit else rows


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, round((len(sorted_values) - 1) * percent))
    return sorted_values[index]


def build_metrics(results: list[dict[str, Any]], *, model_name: str, load_seconds: float) -> dict[str, Any]:
    latencies = [float(row["latency_seconds"]) for row in results]
    correct = sum(1 for row in results if row["correct"])
    total = len(results)
    confusion = {
        gold: {pred: 0 for pred in VALID_LABELS}
        for gold in VALID_LABELS
    }
    for row in results:
        confusion[row["label"]][row["prediction"]] += 1
    return {
        "model_name": model_name,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "load_seconds": load_seconds,
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "median": statistics.median(latencies) if latencies else 0.0,
            "p95": percentile(latencies, 0.95),
            "min": min(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
        "confusion_matrix": {
            "labels": list(VALID_LABELS),
            "matrix": [
                [confusion[gold][pred] for pred in VALID_LABELS]
                for gold in VALID_LABELS
            ],
        },
    }


def write_eval_outputs(
    *,
    output_csv: Path,
    metrics_path: Path,
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "utterance",
                "label",
                "prediction",
                "correct",
                "latency_seconds",
                "raw_output",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def run_eval(args: argparse.Namespace, tokenizer, llm, sampling_params, system_prompt: str, load_seconds: float) -> None:
    eval_path = args.eval_path or DEFAULT_EVAL_PATH
    rows = read_eval_rows(eval_path, args.limit)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.warmup > 0 and rows:
        for row in rows[: args.warmup]:
            classify(
                tokenizer=tokenizer,
                llm=llm,
                sampling_params=sampling_params,
                system_prompt=system_prompt,
                utterance=row["utterance"],
                method=args.method,
            )

    results: list[dict[str, Any]] = []
    for start_index in range(0, len(rows), args.batch_size):
        chunk = rows[start_index : start_index + args.batch_size]
        predictions = classify_many(
            tokenizer=tokenizer,
            llm=llm,
            sampling_params=sampling_params,
            system_prompt=system_prompt,
            utterances=[row["utterance"] for row in chunk],
            method=args.method,
        )
        for offset, (row, prediction) in enumerate(zip(chunk, predictions), start=1):
            index = start_index + offset
            label = normalize_label(row["label"])
            result = {
                "id": row["id"],
                "utterance": row["utterance"],
                "label": label,
                "prediction": prediction.label,
                "correct": prediction.label == label,
                "latency_seconds": f"{prediction.latency_seconds:.6f}",
                "raw_output": prediction.raw_output,
            }
            results.append(result)
            print(
                f"[{index:04d}/{len(rows):04d}] "
                f"gold={label} pred={prediction.label} "
                f"latency={prediction.latency_seconds:.3f}s "
                f"text={row['utterance']}"
            )

    metrics = build_metrics(results, model_name=args.model_name, load_seconds=load_seconds)
    output_csv = args.output_csv or args.output_dir / "test_predictions.csv"
    metrics_path = args.output_dir / "metrics.json"
    write_eval_outputs(
        output_csv=output_csv,
        metrics_path=metrics_path,
        results=results,
        metrics=metrics,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=True))
    print(f"Wrote predictions to {output_csv}")
    print(f"Wrote metrics to {metrics_path}")


def run_interactive(args: argparse.Namespace, tokenizer, llm, sampling_params, system_prompt: str) -> None:
    print("Enter a human utterance. Press Ctrl+C or submit an empty line to exit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        prediction = classify(
            tokenizer=tokenizer,
            llm=llm,
            sampling_params=sampling_params,
            system_prompt=system_prompt,
            utterance=text,
            method=args.method,
        )
        print(f"label: {prediction.label}")
        print(f"raw_output: {prediction.raw_output}")
        print(f"latency_seconds: {prediction.latency_seconds:.4f}")


def main() -> None:
    args = parse_args()
    system_prompt = read_prompt(args.prompt_path)
    if args.print_prompt:
        print(system_prompt)
        return

    load_start = time.perf_counter()
    llm, sampling_params, tokenizer = load_vllm_engine(args)
    load_seconds = time.perf_counter() - load_start
    print(f"Loaded {args.model_name} with vLLM in {load_seconds:.2f}s")

    if args.eval_path is not None or args.limit is not None:
        run_eval(args, tokenizer, llm, sampling_params, system_prompt, load_seconds)
        return
    if args.text:
        prediction = classify(
            tokenizer=tokenizer,
            llm=llm,
            sampling_params=sampling_params,
            system_prompt=system_prompt,
            utterance=args.text,
            method=args.method,
        )
        print(f"text: {args.text}")
        print(f"label: {prediction.label}")
        print(f"raw_output: {prediction.raw_output}")
        print(f"latency_seconds: {prediction.latency_seconds:.4f}")
    else:
        run_interactive(args, tokenizer, llm, sampling_params, system_prompt)


if __name__ == "__main__":
    main()
