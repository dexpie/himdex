from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .model import HimdexBackbone, HimdexConfig
from .tokenizer import ByteTokenizer


class HimdexEncoder:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.config = HimdexConfig(**checkpoint["config"])
        self.tokenizer = ByteTokenizer()
        self.backbone = HimdexBackbone(self.config)
        backbone_state = {
            key.removeprefix("backbone."): value
            for key, value in checkpoint["model_state"].items()
            if key.startswith("backbone.")
        }
        self.backbone.load_state_dict(backbone_state)
        self.backbone.to(self.device).eval()

    @torch.inference_mode()
    def encode_text(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 128,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if not texts:
            return torch.empty((0, self.config.hidden_size))

        batches: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            encoded = [
                self.tokenizer.encode(text, self.config.max_text_length)
                for text in texts[start : start + batch_size]
            ]
            input_ids = torch.stack([item.input_ids for item in encoded]).to(self.device)
            attention_mask = torch.stack([item.attention_mask for item in encoded]).to(self.device)
            embeddings = self.backbone.forward_text(input_ids, attention_mask)[:, 0]
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            batches.append(embeddings.cpu())
        return torch.cat(batches, dim=0)

    @torch.inference_mode()
    def encode_images(
        self,
        paths: list[str | Path],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if not paths:
            return torch.empty((0, self.config.hidden_size))

        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        batches: list[torch.Tensor] = []
        for start in range(0, len(paths), batch_size):
            images = torch.stack(
                [
                    transform(Image.open(path).convert("RGB"))
                    for path in paths[start : start + batch_size]
                ]
            ).to(self.device)
            embeddings = self.backbone.forward_image(images)[:, 0]
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            batches.append(embeddings.cpu())
        return torch.cat(batches, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Himdex embeddings.")
    parser.add_argument("--checkpoint", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", action="append", help="Text to encode; repeat for batches.")
    group.add_argument("--image", action="append", help="Image path to encode; repeat for batches.")
    parser.add_argument("--output", help="Optional .pt output path.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder = HimdexEncoder(args.checkpoint, device=args.device)
    embeddings = (
        encoder.encode_text(args.text, batch_size=args.batch_size)
        if args.text
        else encoder.encode_images(args.image, batch_size=args.batch_size)
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, output_path)
        print(f"saved {output_path} shape={tuple(embeddings.shape)}")
        return
    print(json.dumps({"shape": list(embeddings.shape), "embeddings": embeddings.tolist()}))


if __name__ == "__main__":
    main()
