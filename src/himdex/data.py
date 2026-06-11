from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer


class CSVTextDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: ByteTokenizer,
        max_length: int,
        text_column: str = "text",
        label_column: str = "label",
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        self.rows: list[dict[str, str]] = []
        self.label_to_id: dict[str, int] = {}

        with self.path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError("CSV file has no header")
            if text_column not in reader.fieldnames:
                raise ValueError(f"Missing text column: {text_column}")
            if label_column not in reader.fieldnames:
                raise ValueError(f"Missing label column: {label_column}")

            for row in reader:
                label = row[label_column]
                if label not in self.label_to_id:
                    self.label_to_id[label] = len(self.label_to_id)
                self.rows.append(row)

        if not self.rows:
            raise ValueError("CSV dataset is empty")

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
        }


def build_image_folder_dataset(path: str | Path, image_size: int) -> Any:
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return datasets.ImageFolder(root=str(path), transform=transform)
