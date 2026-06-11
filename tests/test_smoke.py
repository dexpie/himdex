from pathlib import Path

import torch

from himdex import (
    ByteTokenizer,
    HimdexEncoder,
    HimdexConfig,
    HimdexForImageClassification,
    HimdexForMaskedTextModeling,
    HimdexForTextClassification,
)


def test_text_forward_shape():
    tokenizer = ByteTokenizer()
    config = HimdexConfig(max_text_length=32, hidden_size=64, num_layers=1, num_heads=4)
    model = HimdexForTextClassification(config, num_labels=3)
    encoded = tokenizer.encode("halo himdex", max_length=32)

    logits = model(encoded.input_ids.unsqueeze(0), encoded.attention_mask.unsqueeze(0))

    assert logits.shape == (1, 3)


def test_image_forward_shape():
    config = HimdexConfig(image_size=32, patch_size=8, hidden_size=64, num_layers=1, num_heads=4)
    model = HimdexForImageClassification(config, num_labels=5)
    images = torch.randn(2, 3, 32, 32)

    logits = model(images)

    assert logits.shape == (2, 5)


def test_masked_text_forward_shape():
    tokenizer = ByteTokenizer()
    config = HimdexConfig(max_text_length=32, hidden_size=64, num_layers=1, num_heads=4)
    model = HimdexForMaskedTextModeling(config)
    encoded = tokenizer.encode("pretrain himdex", max_length=32)

    logits = model(encoded.input_ids.unsqueeze(0), encoded.attention_mask.unsqueeze(0))

    assert logits.shape == (1, 32, tokenizer.vocab_size)


def test_reference_checkpoint_embedding():
    checkpoint = Path(__file__).parents[1] / "checkpoints" / "himdex_text_compact.pt"
    encoder = HimdexEncoder(checkpoint, device="cpu")

    embeddings = encoder.encode_text(["himdex"])

    assert embeddings.shape == (1, 128)
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(1), atol=1e-5)
