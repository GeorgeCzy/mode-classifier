"""Optional Weights & Biases helpers for training scripts."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any


DEFAULT_WANDB_ENTITY = "chengzy2023-shanghaitech-university"
DEFAULT_WANDB_PROJECT = "mode-classifier"


def add_wandb_args(parser: ArgumentParser) -> None:
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name", default=None)


def init_wandb(args: Namespace, *, config: dict[str, Any]):
    if not args.use_wandb:
        return None

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "wandb is not installed. Run `pip install -r requirements.txt` first."
        ) from error

    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config,
    )
