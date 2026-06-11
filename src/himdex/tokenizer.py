from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EncodedText:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class ByteTokenizer:
    """Tiny UTF-8 byte tokenizer with no training step.

    This is intentionally simple: every dataset and language can be encoded
    immediately, which is useful while Himdex is still young.
    """

    pad_token_id = 0
    cls_token_id = 1
    unk_token_id = 2
    mask_token_id = 2
    byte_offset = 3
    vocab_size = 259

    def encode(self, text: str, max_length: int) -> EncodedText:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")

        raw = text.encode("utf-8", errors="replace")
        ids = [self.cls_token_id]
        ids.extend(byte + self.byte_offset for byte in raw[: max_length - 1])

        attention = [1] * len(ids)
        pad_count = max_length - len(ids)
        if pad_count > 0:
            ids.extend([self.pad_token_id] * pad_count)
            attention.extend([0] * pad_count)

        return EncodedText(
            input_ids=torch.tensor(ids, dtype=torch.long),
            attention_mask=torch.tensor(attention, dtype=torch.bool),
        )

    def decode(self, input_ids: list[int] | torch.Tensor) -> str:
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.detach().cpu().tolist()

        bytes_out = bytearray()
        for token_id in input_ids:
            if token_id in (self.pad_token_id, self.cls_token_id):
                continue
            byte_value = token_id - self.byte_offset
            if 0 <= byte_value <= 255:
                bytes_out.append(byte_value)

        return bytes(bytes_out).decode("utf-8", errors="replace")
