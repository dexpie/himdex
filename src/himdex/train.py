from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .data import CSVTextDataset, build_image_folder_dataset
from .model import HimdexConfig, HimdexForImageClassification, HimdexForTextClassification
from .tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Himdex on text or image classification.")
    parser.add_argument("--task", choices=["text-classification", "image-classification"], required=True)
    parser.add_argument("--data", required=True, help="CSV path for text or ImageFolder path for images.")
    parser.add_argument("--output-dir", default="runs/himdex")
    parser.add_argument("--backbone-from", default="", help="Optional masked-text or classification checkpoint.")
    parser.add_argument("--resume-from", default="", help="Optional classification checkpoint to continue.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--text-pooling", choices=["cls", "mean", "cls-mean"], default="cls")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_backbone(model: nn.Module, checkpoint_path: str) -> None:
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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    scaler = torch.amp.GradScaler("cuda", enabled=training and device.type == "cuda")

    progress = tqdm(loader, desc="train" if training else "validate", leave=False)
    for batch in progress:
        if task == "text-classification":
            inputs = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            labels = batch["label"].to(device)
        else:
            images, labels = batch
            inputs = (images.to(device),)
            labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(*inputs)
                loss = loss_fn(logits, labels)
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


def main() -> None:
    args = parse_args()
    if args.backbone_from and args.resume_from:
        raise ValueError("Use either --backbone-from or --resume-from, not both.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = ByteTokenizer()

    checkpoint_path = args.resume_from or args.backbone_from
    pretrained = torch.load(checkpoint_path, map_location="cpu") if checkpoint_path else None
    if pretrained is not None:
        config = HimdexConfig(**pretrained["config"])
    else:
        config = HimdexConfig(
            vocab_size=tokenizer.vocab_size,
            image_size=args.image_size,
            patch_size=args.patch_size,
            max_text_length=args.max_text_length,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    if args.task == "text-classification" and not args.resume_from:
        config.text_pooling = args.text_pooling

    if args.task == "text-classification":
        dataset = CSVTextDataset(
            args.data,
            tokenizer=tokenizer,
            max_length=config.max_text_length,
            text_column=args.text_column,
            label_column=args.label_column,
        )
        label_to_id = dataset.label_to_id
        model = HimdexForTextClassification(config, num_labels=len(label_to_id))
    else:
        dataset = build_image_folder_dataset(args.data, image_size=config.image_size)
        label_to_id = dataset.class_to_idx
        model = HimdexForImageClassification(config, num_labels=len(label_to_id))

    if args.resume_from:
        model.load_state_dict(pretrained["model_state"])
        print(f"resumed {args.resume_from}")
    elif args.backbone_from:
        load_backbone(model, args.backbone_from)

    validation_size = max(1, int(len(dataset) * args.validation_ratio))
    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, validation_dataset = random_split(
        dataset, [train_size, validation_size], generator=generator
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=0)

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, device, args.task, optimizer, args.grad_clip
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, device, args.task
        )
        scheduler.step()
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_accuracy={train_accuracy:.4f} "
            f"validation_loss={validation_loss:.4f} validation_accuracy={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "task": args.task,
                    "config": config.to_dict(),
                    "label_to_id": label_to_id,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "model_state": model.state_dict(),
                },
                output_dir / "himdex.pt",
            )

    print(f"saved {output_dir / 'himdex.pt'}; best_validation_accuracy={best_accuracy:.4f}")


if __name__ == "__main__":
    main()
