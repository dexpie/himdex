from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .model import HimdexConfig, HimdexForMaskedTextModeling
from .tokenizer import ByteTokenizer


class MixedTextCorpus(Dataset):
    def __init__(self, paths: list[Path], tokenizer: ByteTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts: list[str] = []
        for path in paths:
            self._load_path(path)
        if not self.texts:
            raise ValueError("No text found for pretraining.")

    def _load_path(self, path: Path) -> None:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in {".txt", ".csv"}:
                    self._load_path(child)
            return

        if path.suffix.lower() == ".txt":
            with path.open("r", encoding="utf-8", errors="ignore") as file:
                self.texts.extend(line.strip() for line in file if line.strip())
            return

        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="", errors="ignore") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames is None:
                    return
                text_column = "text" if "text" in reader.fieldnames else reader.fieldnames[0]
                self.texts.extend(row[text_column].strip() for row in reader if row.get(text_column, "").strip())

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer.encode(self.texts[index], self.max_length)
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
        }


def mask_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: ByteTokenizer,
    mask_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = input_ids.clone()
    can_mask = attention_mask.bool()
    can_mask &= input_ids.ne(tokenizer.pad_token_id)
    can_mask &= input_ids.ne(tokenizer.cls_token_id)

    mask = torch.rand(input_ids.shape, device=input_ids.device).lt(mask_probability) & can_mask
    for row in range(mask.size(0)):
        if not mask[row].any():
            candidates = can_mask[row].nonzero(as_tuple=False).flatten()
            if candidates.numel() > 0:
                choice = candidates[torch.randint(candidates.numel(), (1,), device=input_ids.device)]
                mask[row, choice] = True

    labels[~mask] = -100
    masked_input = input_ids.clone()
    masked_input[mask] = tokenizer.mask_token_id
    return masked_input, labels


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    tokenizer: ByteTokenizer,
    device: torch.device,
    mask_probability: float,
    max_batches: int,
) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    correct = 0
    masked_tokens = 0
    batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            masked_input, labels = mask_batch(
                input_ids, attention_mask, tokenizer, mask_probability
            )
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(masked_input, attention_mask)
                loss = loss_fn(logits.flatten(0, 1), labels.flatten())

            predictions = logits.argmax(dim=-1)
            active = labels.ne(-100)
            correct += (predictions[active] == labels[active]).sum().item()
            masked_tokens += active.sum().item()
            total_loss += loss.item()
            batches += 1
            if max_batches and batches >= max_batches:
                break

    return total_loss / max(batches, 1), correct / max(masked_tokens, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain Himdex with masked byte prediction.")
    parser.add_argument("--data", required=True, help="File or directory containing .txt/.csv text data.")
    parser.add_argument("--output-dir", default="runs/himdex_text_pretrain")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0, help="Maximum optimizer steps for this run.")
    parser.add_argument("--resume-from", default="", help="Optional checkpoint to continue from.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = ByteTokenizer()
    checkpoint = torch.load(args.resume_from, map_location="cpu") if args.resume_from else None
    if checkpoint is not None:
        config = HimdexConfig(**checkpoint["config"])
    else:
        config = HimdexConfig(
            vocab_size=tokenizer.vocab_size,
            max_text_length=args.max_text_length,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )

    corpus = MixedTextCorpus([Path(args.data)], tokenizer, config.max_text_length)
    validation_size = max(1, int(len(corpus) * args.validation_ratio))
    train_size = len(corpus) - validation_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, validation_dataset = random_split(
        corpus, [train_size, validation_size], generator=generator
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=0)

    device = torch.device(args.device)
    model = HimdexForMaskedTextModeling(config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        print(f"resumed {args.resume_from}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if checkpoint is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    steps_this_run = args.max_steps or args.epochs * len(train_loader)

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return max(step, 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(steps_this_run - args.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    previous_steps = int(checkpoint.get("optimizer_steps", 0)) if checkpoint else 0
    run_steps = 0
    total_loss = 0.0

    model.train()
    for epoch in range(1, args.epochs + 1):
        progress = tqdm(train_loader, desc=f"pretrain epoch {epoch}", leave=False)
        for batch in progress:
            if args.max_steps and run_steps >= args.max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            masked_input, labels = mask_batch(
                input_ids, attention_mask, tokenizer, args.mask_probability
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(masked_input, attention_mask)
                loss = loss_fn(logits.flatten(0, 1), labels.flatten())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            run_steps += 1
            total_loss += loss.item()
            progress.set_postfix(
                loss=total_loss / run_steps,
                lr=optimizer.param_groups[0]["lr"],
                step=previous_steps + run_steps,
            )

        if args.max_steps and run_steps >= args.max_steps:
            break

    validation_loss, masked_accuracy = evaluate(
        model,
        validation_loader,
        tokenizer,
        device,
        args.mask_probability,
        args.eval_batches,
    )
    train_loss = total_loss / max(run_steps, 1)
    total_steps = previous_steps + run_steps
    print(
        f"steps={total_steps} train_loss={train_loss:.4f} "
        f"validation_loss={validation_loss:.4f} masked_accuracy={masked_accuracy:.4f}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "himdex_text_pretrain.pt"
    temporary_path = checkpoint_path.with_suffix(".tmp")
    torch.save(
        {
            "task": "masked-text-modeling",
            "config": config.to_dict(),
            "dataset_size": len(corpus),
            "optimizer_steps": total_steps,
            "run_steps": run_steps,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "masked_accuracy": masked_accuracy,
            "resumed_from": args.resume_from or None,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)
    print(f"saved {checkpoint_path}")


if __name__ == "__main__":
    main()
