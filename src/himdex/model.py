from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class HimdexConfig:
    vocab_size: int = 259
    image_size: int = 224
    patch_size: int = 16
    max_text_length: int = 256
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class PatchEmbed(nn.Module):
    def __init__(self, config: HimdexConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(images)
        return tokens.flatten(2).transpose(1, 2)


class HimdexBackbone(nn.Module):
    """Shared token encoder for text bytes and image patches."""

    text_modality_id = 0
    image_modality_id = 1

    def __init__(self, config: HimdexConfig) -> None:
        super().__init__()
        self.config = config
        self.text_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.patch_embedding = PatchEmbed(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.modality_embedding = nn.Embedding(2, config.hidden_size)

        max_image_tokens = self.patch_embedding.num_patches + 1
        self.max_positions = max(config.max_text_length, max_image_tokens)
        self.position_embedding = nn.Embedding(self.max_positions, config.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * config.mlp_ratio,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.modality_embedding.weight, std=0.02)

    def forward_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        tokens = self.text_embedding(input_ids)
        return self._encode_tokens(
            tokens=tokens,
            modality_id=self.text_modality_id,
            attention_mask=attention_mask,
        )

    def forward_image(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedding(images)
        return self._encode_tokens(
            tokens=tokens,
            modality_id=self.image_modality_id,
            attention_mask=None,
        )

    def _encode_tokens(
        self,
        tokens: torch.Tensor,
        modality_id: int,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = tokens.shape
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens[:, 1:] if modality_id == self.text_modality_id else tokens], dim=1)

        seq_len = tokens.size(1)
        if seq_len > self.max_positions:
            raise ValueError(f"Sequence length {seq_len} exceeds {self.max_positions}")

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        modality = torch.full((batch_size, seq_len), modality_id, device=tokens.device)
        tokens = tokens + self.position_embedding(positions) + self.modality_embedding(modality)
        tokens = self.dropout(tokens)

        padding_mask = None
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()
            padding_mask = ~attention_mask
            padding_mask[:, 0] = False

        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return self.final_norm(encoded)


class HimdexForTextClassification(nn.Module):
    def __init__(self, config: HimdexConfig, num_labels: int) -> None:
        super().__init__()
        self.backbone = HimdexBackbone(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.backbone.forward_text(input_ids, attention_mask)
        return self.classifier(encoded[:, 0])


class HimdexForMaskedTextModeling(nn.Module):
    def __init__(self, config: HimdexConfig) -> None:
        super().__init__()
        self.backbone = HimdexBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.backbone.forward_text(input_ids, attention_mask)
        return self.lm_head(encoded)


class HimdexForImageClassification(nn.Module):
    def __init__(self, config: HimdexConfig, num_labels: int) -> None:
        super().__init__()
        self.backbone = HimdexBackbone(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoded = self.backbone.forward_image(images)
        return self.classifier(encoded[:, 0])
