from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .benchmark_tfidf import build_pipeline, himdex_split_indices, load_csv
from .inference import HimdexEncoder


@dataclass(frozen=True)
class HybridResult:
    name: str
    checkpoint: str
    dataset: str
    accuracy: float
    train_seconds: float
    predict_seconds: float
    embedding_seconds: float
    features: int
    max_features_per_vectorizer: int
    embedding_weight: float
    validation_ratio: float
    seed: int
    train_samples: int
    validation_samples: int


def run_hybrid_benchmark(
    data_path: str | Path,
    checkpoint_path: str | Path,
    validation_ratio: float = 0.1,
    seed: int = 42,
    max_features: int = 200_000,
    embedding_weight: float = 1.0,
    batch_size: int = 128,
    text_column: str = "text",
    label_column: str = "label",
    device: str | None = None,
) -> HybridResult:
    try:
        from scipy.sparse import csr_matrix, hstack
        from sklearn.metrics import accuracy_score
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise RuntimeError('Install benchmark dependencies: pip install -e ".[benchmark]"') from exc

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
    started = time.perf_counter()
    train_embeddings = encoder.encode_text(train_texts, batch_size=batch_size).numpy()
    validation_embeddings = encoder.encode_text(validation_texts, batch_size=batch_size).numpy()
    embedding_seconds = time.perf_counter() - started
    train_embeddings *= embedding_weight
    validation_embeddings *= embedding_weight

    train_features = hstack([train_tfidf, csr_matrix(train_embeddings)], format="csr")
    validation_features = hstack([validation_tfidf, csr_matrix(validation_embeddings)], format="csr")

    classifier = LinearSVC(C=2.0)
    started = time.perf_counter()
    classifier.fit(train_features, train_labels)
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    predictions = classifier.predict(validation_features)
    predict_seconds = time.perf_counter() - started

    return HybridResult(
        name="word-char+himdex",
        checkpoint=str(checkpoint_path),
        dataset=str(data_path),
        accuracy=float(accuracy_score(validation_labels, predictions)),
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        embedding_seconds=embedding_seconds,
        features=train_features.shape[1],
        max_features_per_vectorizer=max_features,
        embedding_weight=embedding_weight,
        validation_ratio=validation_ratio,
        seed=seed,
        train_samples=len(train_indices),
        validation_samples=len(validation_indices),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TF-IDF plus Himdex embeddings.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--embedding-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="benchmarks/hybrid_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_hybrid_benchmark(
        args.data,
        args.checkpoint,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        max_features=args.max_features,
        embedding_weight=args.embedding_weight,
        batch_size=args.batch_size,
        text_column=args.text_column,
        label_column=args.label_column,
        device=args.device,
    )
    print(
        f"{result.name}: accuracy={result.accuracy:.4f} features={result.features} "
        f"embedding_seconds={result.embedding_seconds:.2f} train_seconds={result.train_seconds:.2f}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
