"""Train a small MLP head on cached text embeddings.

Expected embedding cache format: an .npz file with arrays:

- ids: string sample ids
- embeddings: float matrix with shape [num_examples, embedding_dim]

The labels and split membership are read from the CSV split files created by
create_splits.py. This keeps expensive embedding generation separate from MLP
training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = ROOT / "modeling" / "data" / "splits"
DEFAULT_OUTPUT_DIR = ROOT / "modeling" / "artifacts" / "embedding_mlp"
LABEL_TO_INDEX = {"chat": 0, "motion_query": 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP on cached embeddings.")
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260518)
    return parser.parse_args()


def read_split(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_embedding_cache(path: Path) -> tuple[dict[str, int], np.ndarray]:
    cache = np.load(path, allow_pickle=False)
    ids = cache["ids"].astype(str).tolist()
    embeddings = cache["embeddings"].astype("float32")
    return {sample_id: index for index, sample_id in enumerate(ids)}, embeddings


def split_to_tensors(
    rows: list[dict[str, str]],
    id_to_index: dict[str, int],
    embeddings: np.ndarray,
) -> TensorDataset:
    indices = [id_to_index[row["id"]] for row in rows]
    labels = [LABEL_TO_INDEX[row["label"]] for row in rows]
    x = torch.from_numpy(embeddings[indices])
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)


def evaluate(model: nn.Module, dataset: TensorDataset, device: torch.device) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    correct = 0
    total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.numel()
            loss_total += float(loss.item()) * labels.numel()
    return {"loss": loss_total / total, "accuracy": correct / total}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id_to_index, embeddings = load_embedding_cache(args.embedding_cache)
    train_rows = read_split(args.split_dir / "train.csv")
    val_rows = read_split(args.split_dir / "val.csv")
    test_rows = read_split(args.split_dir / "test.csv")

    train_dataset = split_to_tensors(train_rows, id_to_index, embeddings)
    val_dataset = split_to_tensors(val_rows, id_to_index, embeddings)
    test_dataset = split_to_tensors(test_rows, id_to_index, embeddings)

    model = MLPHead(
        input_dim=embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    best_val_accuracy = -1.0
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, train_dataset, device)
        val_metrics = evaluate(model, val_dataset, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        print(
            f"epoch={epoch} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_dataset, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_val_accuracy": best_val_accuracy,
                "test": test_metrics,
                "history": history,
                "labels": INDEX_TO_LABEL,
                "embedding_dim": int(embeddings.shape[1]),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Wrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()

