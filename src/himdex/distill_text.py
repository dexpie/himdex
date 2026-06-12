from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .benchmark_tfidf import himdex_split_indices
from .hybrid_model import _require_hybrid_dependencies
from .inference import HimdexEncoder
from .model import HimdexConfig, HimdexForTextClassification
from .tokenizer import ByteTokenizer


class DistillationTextDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: ByteTokenizer,
        max_length: int,
        label_order: list[str],
        teacher_scores: torch.Tensor,
        text_column: str = "text",
        label_column: str = "label",
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        self.label_to_id = {label: index for index, label in enumerate(label_order)}
        self.rows: list[dict[str, str]] = []
        self.teacher_scores = teacher_scores.float()

        with self.path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError("CSV file has no header")
            for column in (text_column, label_column):
                if column not in reader.fieldnames:
                    raise ValueError(f"Missing column: {column}")
            for row in reader:
                if row[label_column] not in self.label_to_id:
                    raise ValueError(f"Teacher is missing label: {row[label_column]}")
                self.rows.append(row)

        if len(self.rows) != len(self.teacher_scores):
            raise ValueError("Teacher scores must match dataset size")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer.encode(row[self.text_column], self.max_length)
        label = self.label_to_id[row[self.label_column]]
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "teacher_scores": self.teacher_scores[index],
        }


@dataclass(frozen=True)
class DistillationMetrics:
    validation_loss: float
    validation_accuracy: float
    teacher_accuracy: float


def load_backbone(model: nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint["model_state"]
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in source.items()
        if key.startswith("backbone.")
    }
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected backbone keys: {unexpected}")
    print(f"loaded backbone {checkpoint_path}; missing={len(missing)}")


def read_texts_and_labels(
    path: str | Path,
    text_column: str,
    label_column: str,
) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header")
        for column in (text_column, label_column):
            if column not in reader.fieldnames:
                raise ValueError(f"Missing column: {column}")
        for row in reader:
            texts.append(row[text_column])
            labels.append(row[label_column])
    return texts, labels


def hybrid_decision_scores(
    teacher_path: str | Path,
    texts: list[str],
    checkpoint_path: str | Path | None,
    batch_size: int,
    device: str | None,
) -> tuple[torch.Tensor, list[str]]:
    joblib, csr_matrix, hstack, _, _ = _require_hybrid_dependencies()
    artifact = joblib.load(teacher_path)
    if artifact.get("format") != "himdex-hybrid-text-classifier-v1":
        raise ValueError("Unsupported Himdex hybrid model artifact")

    checkpoint = checkpoint_path or artifact["checkpoint"]
    encoder = HimdexEncoder(checkpoint, device=device)
    embeddings = encoder.encode_text(texts, batch_size=batch_size).numpy()
    embeddings *= float(artifact["embedding_weight"])
    tfidf_features = artifact["tfidf"].transform(texts)
    features = hstack([tfidf_features, csr_matrix(embeddings)], format="csr")
    scores = artifact["classifier"].decision_function(features)
    classes = [str(label) for label in artifact["classifier"].classes_.tolist()]
    return torch.tensor(scores, dtype=torch.float32), classes


def run_epoch(
    model: HimdexForTextClassification,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    alpha: float,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    scaler = torch.amp.GradScaler("cuda", enabled=training and device.type == "cuda")

    progress = tqdm(loader, desc="distill" if training else "validate", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        teacher_scores = batch["teacher_scores"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(input_ids, attention_mask)
                hard_loss = F.cross_entropy(logits, labels)
                teacher_probs = F.softmax(teacher_scores / temperature, dim=-1)
                student_log_probs = F.log_softmax(logits / temperature, dim=-1)
                soft_loss = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean",
                ) * (temperature * temperature)
                loss = (1.0 - alpha) * hard_loss + alpha * soft_loss
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += batch_size
        progress.set_postfix(loss=total_loss / total, acc=correct / total)

    return total_loss / max(total, 1), correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill a Himdex hybrid teacher into a pure text classifier.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--backbone-from", required=True)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--output-dir", default="runs/himdex_text_distilled")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--teacher-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--text-pooling", choices=["cls", "mean", "cls-mean"], default="cls-mean")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    tokenizer = ByteTokenizer()
    device = torch.device(args.device)

    texts, labels = read_texts_and_labels(args.data, args.text_column, args.label_column)
    teacher_scores, teacher_classes = hybrid_decision_scores(
        args.teacher,
        texts,
        args.backbone_from,
        args.teacher_batch_size,
        args.device,
    )
    student_checkpoint = (
        torch.load(args.resume_from, map_location="cpu") if args.resume_from else None
    )
    if student_checkpoint is not None and "label_to_id" in student_checkpoint:
        label_to_id = student_checkpoint["label_to_id"]
        label_order = [
            label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])
        ]
        class_to_teacher_index = {
            label: index for index, label in enumerate(teacher_classes)
        }
        teacher_scores = teacher_scores[
            :, [class_to_teacher_index[label] for label in label_order]
        ]
    else:
        label_order = teacher_classes
        label_to_id = {label: index for index, label in enumerate(label_order)}

    teacher_predictions = teacher_scores.argmax(dim=-1)
    label_ids = torch.tensor([label_to_id[label] for label in labels])
    teacher_accuracy = float((teacher_predictions == label_ids).float().mean().item())
    print(f"teacher_accuracy={teacher_accuracy:.4f}")

    checkpoint = student_checkpoint or torch.load(args.backbone_from, map_location="cpu")
    config = HimdexConfig(**checkpoint["config"])
    if not args.resume_from:
        config.text_pooling = args.text_pooling
    dataset = DistillationTextDataset(
        args.data,
        tokenizer,
        config.max_text_length,
        label_order,
        teacher_scores,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    train_indices, validation_indices = himdex_split_indices(
        len(dataset), args.validation_ratio, args.seed
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=args.batch_size,
        num_workers=0,
    )

    model = HimdexForTextClassification(config, num_labels=len(label_order))
    if student_checkpoint is not None:
        model.load_state_dict(student_checkpoint["model_state"])
        print(f"resumed {args.resume_from}")
    else:
        load_backbone(model, args.backbone_from)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = DistillationMetrics(
        validation_loss=float("inf"),
        validation_accuracy=-1.0,
        teacher_accuracy=teacher_accuracy,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            device,
            args.temperature,
            args.alpha,
            optimizer,
            args.grad_clip,
        )
        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            device,
            args.temperature,
            args.alpha,
        )
        scheduler.step()
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_accuracy={train_accuracy:.4f} "
            f"validation_loss={validation_loss:.4f} validation_accuracy={validation_accuracy:.4f}"
        )
        if validation_accuracy > best.validation_accuracy:
            best = DistillationMetrics(
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                teacher_accuracy=teacher_accuracy,
            )
            torch.save(
                {
                    "task": "text-classification",
                    "training": "hybrid-distillation",
                    "config": config.to_dict(),
                    "label_to_id": label_to_id,
                    "teacher": args.teacher,
                    "resumed_from": args.resume_from or None,
                    "teacher_accuracy": teacher_accuracy,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "temperature": args.temperature,
                    "alpha": args.alpha,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "model_state": model.state_dict(),
                },
                output_dir / "himdex.pt",
            )

    print(
        f"saved {output_dir / 'himdex.pt'}; "
        f"best_validation_accuracy={best.validation_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
