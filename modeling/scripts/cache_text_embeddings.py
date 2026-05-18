"""Cache text embeddings for the response-mode dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data_generation" / "data" / "raw" / "deepseek_generated_2000.csv"
DEFAULT_OUTPUT = ROOT / "modeling" / "data" / "embeddings" / "qwen3_0_6b_deepseek_2000.npz"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TEXT_PREFIX = "Represent this human-robot utterance for response mode classification: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached text embeddings.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--text-prefix", default=DEFAULT_TEXT_PREFIX)
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for index, row in enumerate(rows, start=1):
        if set(row) != {"id", "utterance", "label"}:
            raise ValueError(f"Row {index} has unexpected fields: {sorted(row)}")
    return rows


def main() -> None:
    args = parse_args()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is not installed. Run `pip install -r requirements.txt`."
        ) from error

    rows = read_rows(args.input)
    ids = np.array([row["id"] for row in rows], dtype=str)
    labels = np.array([row["label"] for row in rows], dtype=str)
    texts = [
        f"{args.text_prefix}{row['utterance']}" if args.text_prefix else row["utterance"]
        for row in rows
    ]

    model = SentenceTransformer(args.model_name, device=args.device)
    model.max_seq_length = args.max_seq_length
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=not args.no_normalize,
    ).astype("float32")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=ids,
        labels=labels,
        embeddings=embeddings,
    )

    metadata = {
        "input": str(args.input.relative_to(ROOT)),
        "output": str(args.output.relative_to(ROOT)),
        "model_name": args.model_name,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "text_prefix": args.text_prefix,
        "normalize_embeddings": not args.no_normalize,
        "rows": len(rows),
        "embedding_dim": int(embeddings.shape[1]),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote embeddings: {args.output}")
    print(f"Wrote metadata: {args.output.with_suffix('.json')}")
    print(f"shape: {embeddings.shape}")


if __name__ == "__main__":
    main()

