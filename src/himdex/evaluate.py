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


def _classification_report(
    labels: list[int],
    predictions: list[int],
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    class_count = len(id_to_label)
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for label, prediction in zip(labels, predictions):
        confusion[label][prediction] += 1

    per_class = []
    f1_scores = []
    for class_id in range(class_count):
        true_positive = confusion[class_id][class_id]
        false_positive = sum(confusion[row][class_id] for row in range(class_count)) - true_positive
        false_negative = sum(confusion[class_id]) - true_positive
        support = sum(confusion[class_id])
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_scores.append(f1)
        per_class.append(
            {
                "label": id_to_label[class_id],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    return {
        "labels": [id_to_label[index] for index in range(class_count)],
        "confusion_matrix": confusion,
        "per_class": per_class,
        "macro_f1": sum(f1_scores) / max(class_count, 1),
    }


def _sample_preview(dataset: Any, index: int, task: str, text_column: str) -> dict[str, Any]:
    if task == "text-classification":
        row = dataset.rows[index]
        text = row[text_column]
        return {
            "text": text[:500],
        }
    path, _ = dataset.samples[index]
    return {"path": path}


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
    return_predictions: bool = False,
    prediction_limit: int = 100,
    include_correct_predictions: bool = False,
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
    id_to_label = {index: label for label, index in label_to_id.items()}
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
    all_labels: list[int] = []
    all_predictions: list[int] = []
    exported_predictions: list[dict[str, Any]] = []
    validation_indices = list(validation_dataset.indices)
    cursor = 0

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
            probabilities = torch.softmax(logits.float(), dim=-1)
            predictions = logits.argmax(dim=-1)
            total_loss += loss.item() * batch_count
            correct += (predictions == labels).sum().item()
            total += batch_count
            batch_labels = labels.detach().cpu().tolist()
            batch_predictions = predictions.detach().cpu().tolist()
            all_labels.extend(batch_labels)
            all_predictions.extend(batch_predictions)

            if return_predictions and len(exported_predictions) < prediction_limit:
                confidences = probabilities.max(dim=-1).values.detach().cpu().tolist()
                for offset, (label_id, prediction_id, confidence) in enumerate(
                    zip(batch_labels, batch_predictions, confidences)
                ):
                    is_correct = label_id == prediction_id
                    if is_correct and not include_correct_predictions:
                        continue
                    dataset_index = validation_indices[cursor + offset]
                    exported_predictions.append(
                        {
                            "dataset_index": dataset_index,
                            "label": id_to_label[label_id],
                            "prediction": id_to_label[prediction_id],
                            "confidence": confidence,
                            "correct": is_correct,
                            **_sample_preview(dataset, dataset_index, resolved_task, text_column),
                        }
                    )
                    if len(exported_predictions) >= prediction_limit:
                        break
            cursor += batch_count

    metrics = {
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
    metrics.update(_classification_report(all_labels, all_predictions, id_to_label))
    if return_predictions:
        metrics["predictions"] = exported_predictions
    return metrics


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
    parser.add_argument("--output-predictions", default="")
    parser.add_argument("--prediction-limit", type=int, default=100)
    parser.add_argument("--include-correct-predictions", action="store_true")
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
        return_predictions=bool(args.output_predictions),
        prediction_limit=args.prediction_limit,
        include_correct_predictions=args.include_correct_predictions,
    )
    predictions = metrics.pop("predictions", None)
    payload = json.dumps(metrics, indent=2)
    print(payload)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    if args.output_predictions:
        output_path = Path(args.output_predictions)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(predictions or [], indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
