from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import CSVTextDataset, build_image_folder_dataset
from .model import HimdexConfig, HimdexForImageClassification, HimdexForTextClassification
from .tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Himdex on text or image classification.")
    parser.add_argument("--task", choices=["text-classification", "image-classification"], required=True)
    parser.add_argument("--data", required=True, help="CSV path for text or ImageFolder path for images.")
    parser.add_argument("--output-dir", default="runs/himdex")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
) -> tuple[float, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)

        if task == "text-classification":
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += batch_size
        progress.set_postfix(loss=total_loss / total, acc=correct / total)

    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = ByteTokenizer()
    config = HimdexConfig(
        vocab_size=tokenizer.vocab_size,
        image_size=args.image_size,
        patch_size=args.patch_size,
        max_text_length=args.max_text_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )

    label_to_id: dict[str, int]
    if args.task == "text-classification":
        dataset = CSVTextDataset(
            args.data,
            tokenizer=tokenizer,
            max_length=args.max_text_length,
            text_column=args.text_column,
            label_column=args.label_column,
        )
        label_to_id = dataset.label_to_id
        model = HimdexForTextClassification(config, num_labels=len(label_to_id))
    else:
        dataset = build_image_folder_dataset(args.data, image_size=args.image_size)
        label_to_id = dataset.class_to_idx
        model = HimdexForImageClassification(config, num_labels=len(label_to_id))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, loader, optimizer, device, args.task)
        print(f"epoch={epoch} loss={loss:.4f} accuracy={acc:.4f}")

    checkpoint = {
        "task": args.task,
        "config": config.to_dict(),
        "label_to_id": label_to_id,
        "model_state": model.state_dict(),
    }
    checkpoint_path = output_dir / "himdex.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"saved {checkpoint_path}")


if __name__ == "__main__":
    main()
