from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from .benchmark_tfidf import build_pipeline, himdex_split_indices, load_csv
from .inference import HimdexEncoder


@dataclass(frozen=True)
class HybridModelMetrics:
    accuracy: float
    train_samples: int
    validation_samples: int
    features: int


def _require_hybrid_dependencies():
    try:
        import joblib
        from scipy.sparse import csr_matrix, hstack
        from sklearn.metrics import accuracy_score
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise RuntimeError('Install hybrid dependencies: pip install -e ".[benchmark]"') from exc
    return joblib, csr_matrix, hstack, accuracy_score, LinearSVC


def train_hybrid_model(
    data_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    validation_ratio: float = 0.1,
    seed: int = 42,
    max_features: int = 200_000,
    embedding_weight: float = 1.25,
    classifier_c: float = 1.0,
    batch_size: int = 128,
    text_column: str = "text",
    label_column: str = "label",
    device: str | None = None,
) -> HybridModelMetrics:
    joblib, csr_matrix, hstack, accuracy_score, LinearSVC = _require_hybrid_dependencies()

    texts, labels = load_csv(data_path, text_column, label_column)
    train_indices, validation_indices = himdex_split_indices(
        len(texts), validation_ratio, seed
    )
    train_texts = [texts[index] for index in train_indices]
    train_labels = [labels[index] for index in train_indices]
    validation_texts = [texts[index] for index in validation_indices]
    validation_labels = [labels[index] for index in validation_indices]

    tfidf = build_pipeline("word-char", max_features).named_steps["tfidf"]
    train_tfidf = tfidf.fit_transform(train_texts)
    validation_tfidf = tfidf.transform(validation_texts)

    encoder = HimdexEncoder(checkpoint_path, device=device)
    train_embeddings = encoder.encode_text(train_texts, batch_size=batch_size).numpy()
    validation_embeddings = encoder.encode_text(validation_texts, batch_size=batch_size).numpy()
    train_embeddings *= embedding_weight
    validation_embeddings *= embedding_weight

    train_features = hstack([train_tfidf, csr_matrix(train_embeddings)], format="csr")
    validation_features = hstack([validation_tfidf, csr_matrix(validation_embeddings)], format="csr")

    classifier = LinearSVC(C=classifier_c)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(validation_features)

    artifact = {
        "format": "himdex-hybrid-text-classifier-v1",
        "checkpoint": str(checkpoint_path),
        "tfidf": tfidf,
        "classifier": classifier,
        "embedding_weight": embedding_weight,
        "classifier_c": classifier_c,
        "max_features_per_vectorizer": max_features,
        "validation_ratio": validation_ratio,
        "seed": seed,
        "text_column": text_column,
        "label_column": label_column,
        "labels": sorted(set(labels)),
        "features": train_features.shape[1],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    return HybridModelMetrics(
        accuracy=float(accuracy_score(validation_labels, predictions)),
        train_samples=len(train_indices),
        validation_samples=len(validation_indices),
        features=train_features.shape[1],
    )


def predict_texts(
    model_path: str | Path,
    texts: list[str],
    checkpoint_path: str | Path | None = None,
    batch_size: int = 128,
    device: str | None = None,
) -> list[str]:
    joblib, csr_matrix, hstack, _, _ = _require_hybrid_dependencies()
    artifact = joblib.load(model_path)
    if artifact.get("format") != "himdex-hybrid-text-classifier-v1":
        raise ValueError("Unsupported Himdex hybrid model artifact")

    checkpoint = checkpoint_path or artifact["checkpoint"]
    encoder = HimdexEncoder(checkpoint, device=device)
    embeddings = encoder.encode_text(texts, batch_size=batch_size).numpy()
    embeddings *= float(artifact["embedding_weight"])
    tfidf_features = artifact["tfidf"].transform(texts)
    features = hstack([tfidf_features, csr_matrix(embeddings)], format="csr")
    return artifact["classifier"].predict(features).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run a Himdex hybrid text classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--output", default="runs/himdex_hybrid_text.joblib")
    train.add_argument("--validation-ratio", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-features", type=int, default=200_000)
    train.add_argument("--embedding-weight", type=float, default=1.25)
    train.add_argument("--classifier-c", type=float, default=1.0)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--text-column", default="text")
    train.add_argument("--label-column", default="label")
    train.add_argument("--device", default=None)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--model", required=True)
    predict.add_argument("--checkpoint", default=None)
    predict.add_argument("--text", action="append", required=True)
    predict.add_argument("--batch-size", type=int, default=128)
    predict.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        metrics = train_hybrid_model(
            args.data,
            args.checkpoint,
            args.output,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            max_features=args.max_features,
            embedding_weight=args.embedding_weight,
            classifier_c=args.classifier_c,
            batch_size=args.batch_size,
            text_column=args.text_column,
            label_column=args.label_column,
            device=args.device,
        )
        print(
            f"saved {args.output}; accuracy={metrics.accuracy:.4f} "
            f"features={metrics.features} train_samples={metrics.train_samples} "
            f"validation_samples={metrics.validation_samples}"
        )
        return

    predictions = predict_texts(
        args.model,
        args.text,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
    )
    for text, prediction in zip(args.text, predictions):
        print(f"{prediction}\t{text}")


if __name__ == "__main__":
    main()
