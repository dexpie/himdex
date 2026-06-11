from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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
    can_mask = attention_mask.clone()
    can_mask &= input_ids.ne(tokenizer.pad_token_id)
    can_mask &= input_ids.ne(tokenizer.cls_token_id)

    random_values = torch.rand(input_ids.shape, device=input_ids.device)
    mask = random_values.lt(mask_probability) & can_mask
    labels[~mask] = -100

    masked_input = input_ids.clone()
    masked_input[mask] = tokenizer.mask_token_id
    return masked_input, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain Himdex on mixed text with masked byte prediction.")
    parser.add_argument("--data", required=True, help="File or directory containing .txt/.csv text data.")
    parser.add_argument("--output-dir", default="runs/himdex_text_pretrain")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop early after this many optimizer steps.")
    parser.add_argument("--resume-from", default="", help="Optional Himdex masked-text checkpoint to continue from.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = ByteTokenizer()
    checkpoint = None
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location="cpu")
        config = HimdexConfig(**checkpoint["config"])
    else:
        config = HimdexConfig(
            vocab_size=tokenizer.vocab_size,
            max_text_length=args.max_text_length,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )

    data_path = Path(args.data)
    dataset = MixedTextCorpus([data_path], tokenizer, config.max_text_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device(args.device)
    model = HimdexForMaskedTextModeling(config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        print(f"resumed {args.resume_from}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    optimizer_steps = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        progress = tqdm(loader, desc=f"pretrain epoch {epoch}", leave=False)
        for batch in progress:
            if args.max_steps and optimizer_steps >= args.max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            masked_input, labels = mask_batch(
                input_ids,
                attention_mask,
                tokenizer=tokenizer,
                mask_probability=args.mask_probability,
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(masked_input, attention_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            optimizer_steps += 1
            progress.set_postfix(loss=total_loss / total_batches)

        print(f"epoch={epoch} loss={total_loss / max(total_batches, 1):.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "himdex_text_pretrain.pt"
    torch.save(
        {
            "task": "masked-text-modeling",
            "config": config.to_dict(),
            "dataset_size": len(dataset),
            "optimizer_steps": optimizer_steps,
            "resumed_from": args.resume_from or None,
            "model_state": model.state_dict(),
        },
        checkpoint_path,
    )
    print(f"saved {checkpoint_path}")


if __name__ == "__main__":
    main()
