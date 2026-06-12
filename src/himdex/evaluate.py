from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .data import CSVTextDataset, build_image_folder_dataset
from .model import HimdexConfig, HimdexForImageClassification, HimdexForTextClassification
from .tokenizer import ByteTokenizer


def _validate_label_mapping(dataset_mapping: dict[str, int], checkpoint_mapping: dict[str, int]) -> None:
    if dataset_mapping != checkpoint_mapping:
        raise ValueError(
            "Dataset labels do not match checkpoint labels. "
            f"dataset={dataset_mapping} checkpoint={checkpoint_mapping}"
        )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_path: str | Path,
    task: str | None = None,
    batch_size: int = 64,
    validation_ratio: float = 0.1,
    seed: int = 42,
    device: str | None = None,
    text_column: str = "text",
    label_column: str = "label",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    resolved_task = task or checkpoint.get("task")
    if resolved_task not in {"text-classification", "image-classification"}:
        raise ValueError("task must be text-classification or image-classification")

    config = HimdexConfig(**checkpoint["config"])
    checkpoint_labels = checkpoint["label_to_id"]
    if resolved_task == "text-classification":
        dataset = CSVTextDataset(
            data_path,
            tokenizer=ByteTokenizer(),
            max_length=config.max_text_length,
            text_column=text_column,
            label_column=label_column,
        )
        label_to_id = dataset.label_to_id
        model: nn.Module = HimdexForTextClassification(config, num_labels=len(label_to_id))
    else:
        dataset = build_image_folder_dataset(data_path, image_size=config.image_size)
        label_to_id = dataset.class_to_idx
        model = HimdexForImageClassification(config, num_labels=len(label_to_id))

    _validate_label_mapping(label_to_id, checkpoint_labels)
    model.load_state_dict(checkpoint["model_state"])

    validation_size = max(1, int(len(dataset) * validation_ratio))
    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    _, validation_dataset = random_split(
        dataset, [train_size, validation_size], generator=generator
    )
    loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=0)

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(torch_device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            if resolved_task == "text-classification":
                inputs = (
                    batch["input_ids"].to(torch_device),
                    batch["attention_mask"].to(torch_device),
                )
                labels = batch["label"].to(torch_device)
            else:
                images, labels = batch
                inputs = (images.to(torch_device),)
                labels = labels.to(torch_device)

            with torch.autocast(device_type=torch_device.type, enabled=torch_device.type == "cuda"):
                logits = model(*inputs)
                loss = loss_fn(logits, labels)

            batch_count = labels.size(0)
            total_loss += loss.item() * batch_count
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += batch_count

    return {
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "task": resolved_task,
        "validation_ratio": validation_ratio,
        "seed": seed,
        "validation_samples": total,
        "validation_loss": total_loss / max(total, 1),
        "validation_accuracy": correct / max(total, 1),
        "device": str(torch_device),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Himdex classification checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", choices=["text-classification", "image-classification"], default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        task=args.task,
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        device=args.device,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    payload = json.dumps(metrics, indent=2)
    print(payload)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
