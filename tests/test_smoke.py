from pathlib import Path

import torch

from himdex.benchmark_tfidf import himdex_split_indices, run_benchmark
from himdex.hybrid_model import predict_texts, train_hybrid_model

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


def test_text_cls_mean_pooling_forward_shape():
    tokenizer = ByteTokenizer()
    config = HimdexConfig(
        max_text_length=32,
        hidden_size=64,
        num_layers=1,
        num_heads=4,
        text_pooling="cls-mean",
    )
    model = HimdexForTextClassification(config, num_labels=3)
    encoded = tokenizer.encode("halo himdex pooling", max_length=32)

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


def test_tfidf_benchmark_runs_on_toy_data(tmp_path):
    data_path = tmp_path / "toy.csv"
    data_path.write_text(
        "text,label\n"
        "great product,positive\n"
        "excellent service,positive\n"
        "really good,positive\n"
        "bad product,negative\n"
        "awful service,negative\n"
        "really bad,negative\n",
        encoding="utf-8",
    )

    results = run_benchmark(
        data_path,
        ["word-unigram"],
        validation_ratio=0.33,
        max_features=100,
    )

    assert len(results) == 1
    assert 0.0 <= results[0].accuracy <= 1.0


def test_himdex_split_is_deterministic():
    first = himdex_split_indices(100, validation_ratio=0.1, seed=42)
    second = himdex_split_indices(100, validation_ratio=0.1, seed=42)

    assert first == second
    assert len(first[0]) == 90
    assert len(first[1]) == 10


def test_encoder_batches_text():
    checkpoint = Path(__file__).parents[1] / "checkpoints" / "himdex_text_compact.pt"
    encoder = HimdexEncoder(checkpoint, device="cpu")

    embeddings = encoder.encode_text(["a", "b", "c"], batch_size=2)

    assert embeddings.shape == (3, 128)


def test_hybrid_model_trains_and_predicts_on_toy_data(tmp_path):
    data_path = tmp_path / "toy.csv"
    model_path = tmp_path / "hybrid.joblib"
    checkpoint = Path(__file__).parents[1] / "checkpoints" / "himdex_text_compact.pt"
    data_path.write_text(
        "text,label\n"
        "space rocket launches,space\n"
        "orbit satellite mission,space\n"
        "team wins match,sports\n"
        "coach scores goal,sports\n"
        "new rocket orbit,space\n"
        "team match goal,sports\n",
        encoding="utf-8",
    )

    metrics = train_hybrid_model(
        data_path,
        checkpoint,
        model_path,
        validation_ratio=0.33,
        max_features=100,
        batch_size=2,
        device="cpu",
    )
    predictions = predict_texts(
        model_path,
        ["rocket mission", "team goal"],
        checkpoint_path=checkpoint,
        batch_size=2,
        device="cpu",
    )

    assert model_path.exists()
    assert 0.0 <= metrics.accuracy <= 1.0
    assert len(predictions) == 2
