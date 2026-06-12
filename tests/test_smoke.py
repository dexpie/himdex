from pathlib import Path

import torch

from himdex.benchmark_tfidf import himdex_split_indices, run_benchmark
from himdex.checkpoint_tools import average_state_dicts
from himdex.distill_text import DistillationTextDataset
from himdex.evaluate import evaluate_checkpoint
from himdex.hybrid_model import predict_texts, train_hybrid_model
from himdex.train import build_optimizer, freeze_backbone_parameters

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


def test_distillation_dataset_aligns_teacher_scores(tmp_path):
    data_path = tmp_path / "toy.csv"
    data_path.write_text(
        "text,label\n"
        "space rocket,space\n"
        "team goal,sports\n",
        encoding="utf-8",
    )
    tokenizer = ByteTokenizer()
    teacher_scores = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])

    dataset = DistillationTextDataset(
        data_path,
        tokenizer,
        max_length=16,
        label_order=["space", "sports"],
        teacher_scores=teacher_scores,
    )

    item = dataset[0]
    assert item["label"].item() == 0
    assert torch.equal(item["teacher_scores"], teacher_scores[0])


def test_average_state_dicts_blends_float_tensors():
    first = {
        "weight": torch.tensor([1.0, 3.0]),
        "step": torch.tensor(1),
    }
    second = {
        "weight": torch.tensor([3.0, 7.0]),
        "step": torch.tensor(2),
    }

    averaged = average_state_dicts(first, second, second_weight=0.25)

    assert torch.allclose(averaged["weight"], torch.tensor([1.5, 4.0]))
    assert averaged["step"].item() == 1


def test_evaluate_checkpoint_runs_on_toy_text_data(tmp_path):
    data_path = tmp_path / "toy.csv"
    data_path.write_text(
        "text,label\n"
        "space rocket,space\n"
        "orbit satellite,space\n"
        "team goal,sports\n"
        "coach match,sports\n",
        encoding="utf-8",
    )
    tokenizer = ByteTokenizer()
    config = HimdexConfig(max_text_length=16, hidden_size=32, num_layers=1, num_heads=4)
    model = HimdexForTextClassification(config, num_labels=2)
    checkpoint_path = tmp_path / "himdex.pt"
    torch.save(
        {
            "task": "text-classification",
            "config": config.to_dict(),
            "label_to_id": {"space": 0, "sports": 1},
            "model_state": model.state_dict(),
        },
        checkpoint_path,
    )

    metrics = evaluate_checkpoint(
        checkpoint_path,
        data_path,
        batch_size=2,
        validation_ratio=0.5,
        device="cpu",
    )

    assert metrics["validation_samples"] == 2
    assert 0.0 <= metrics["validation_accuracy"] <= 1.0


def test_freeze_backbone_parameters_keeps_head_trainable():
    config = HimdexConfig(max_text_length=16, hidden_size=32, num_layers=1, num_heads=4)
    model = HimdexForTextClassification(config, num_labels=2)

    trainable = freeze_backbone_parameters(model)

    assert trainable > 0
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_build_optimizer_supports_layerwise_learning_rates():
    config = HimdexConfig(max_text_length=16, hidden_size=32, num_layers=1, num_heads=4)
    model = HimdexForTextClassification(config, num_labels=2)

    optimizer = build_optimizer(
        model,
        lr=1e-4,
        weight_decay=0.01,
        backbone_lr=1e-5,
        head_lr=2e-4,
    )

    assert [group["name"] for group in optimizer.param_groups] == ["backbone", "head"]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 2e-4]
