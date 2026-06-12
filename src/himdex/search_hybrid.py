from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .benchmark_tfidf import himdex_split_indices, load_csv
from .inference import HimdexEncoder


@dataclass(frozen=True)
class HybridSearchResult:
    accuracy: float
    word_ngram_range: tuple[int, int]
    char_ngram_range: tuple[int, int]
    max_features: int
    embedding_weight: float
    classifier_c: float
    features: int
    train_seconds: float


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def parse_ngram_ranges(value: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        start, end = item.split("-", maxsplit=1)
        parsed = (int(start), int(end))
        if parsed[0] <= 0 or parsed[1] < parsed[0]:
            raise ValueError(f"Invalid n-gram range: {item}")
        ranges.append(parsed)
    if not ranges:
        raise ValueError("Expected at least one n-gram range")
    return ranges


def _require_search_dependencies():
    try:
        import joblib
        from scipy.sparse import csr_matrix, hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import accuracy_score
        from sklearn.pipeline import FeatureUnion
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise RuntimeError('Install search dependencies: pip install -e ".[benchmark]"') from exc
    return joblib, csr_matrix, hstack, TfidfVectorizer, accuracy_score, FeatureUnion, LinearSVC


def build_word_char_vectorizer(
    word_ngram_range: tuple[int, int],
    char_ngram_range: tuple[int, int],
    max_features: int,
):
    _, _, _, TfidfVectorizer, _, FeatureUnion, _ = _require_search_dependencies()
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=word_ngram_range,
                    min_df=2,
                    max_df=0.98,
                    max_features=max_features,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=char_ngram_range,
                    min_df=2,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def _cache_matches(cache: dict[str, Any], metadata: dict[str, Any]) -> bool:
    return cache.get("metadata") == metadata


def load_or_create_embeddings(
    cache_path: str | Path,
    checkpoint_path: str | Path,
    data_path: str | Path,
    train_texts: list[str],
    validation_texts: list[str],
    validation_ratio: float,
    seed: int,
    text_column: str,
    label_column: str,
    batch_size: int,
    device: str | None,
):
    joblib, *_ = _require_search_dependencies()
    metadata = {
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "train_samples": len(train_texts),
        "validation_samples": len(validation_texts),
        "validation_ratio": validation_ratio,
        "seed": seed,
        "text_column": text_column,
        "label_column": label_column,
    }
    cache_path = Path(cache_path)
    if cache_path.exists():
        cache = joblib.load(cache_path)
        if _cache_matches(cache, metadata):
            return cache["train_embeddings"], cache["validation_embeddings"], True

    encoder = HimdexEncoder(checkpoint_path, device=device)
    train_embeddings = encoder.encode_text(train_texts, batch_size=batch_size).numpy()
    validation_embeddings = encoder.encode_text(validation_texts, batch_size=batch_size).numpy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "metadata": metadata,
            "train_embeddings": train_embeddings,
            "validation_embeddings": validation_embeddings,
        },
        cache_path,
    )
    return train_embeddings, validation_embeddings, False


def run_hybrid_search(
    data_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    output_model: str | Path | None = None,
    cache_path: str | Path = "work/himdex_hybrid_embeddings.joblib",
    validation_ratio: float = 0.1,
    seed: int = 42,
    max_features_list: list[int] | None = None,
    word_ngram_ranges: list[tuple[int, int]] | None = None,
    char_ngram_ranges: list[tuple[int, int]] | None = None,
    embedding_weights: list[float] | None = None,
    classifier_cs: list[float] | None = None,
    batch_size: int = 512,
    text_column: str = "text",
    label_column: str = "label",
    device: str | None = None,
) -> dict[str, Any]:
    joblib, csr_matrix, hstack, _, accuracy_score, _, LinearSVC = _require_search_dependencies()
    max_features_list = max_features_list or [200_000]
    word_ngram_ranges = word_ngram_ranges or [(1, 2)]
    char_ngram_ranges = char_ngram_ranges or [(3, 5)]
    embedding_weights = embedding_weights or [1.0, 1.25, 1.5]
    classifier_cs = classifier_cs or [0.75, 1.0, 1.25]

    texts, labels = load_csv(data_path, text_column, label_column)
    train_indices, validation_indices = himdex_split_indices(
        len(texts), validation_ratio, seed
    )
    train_texts = [texts[index] for index in train_indices]
    train_labels = [labels[index] for index in train_indices]
    validation_texts = [texts[index] for index in validation_indices]
    validation_labels = [labels[index] for index in validation_indices]

    embeddings_started = time.perf_counter()
    train_embeddings, validation_embeddings, cache_hit = load_or_create_embeddings(
        cache_path=cache_path,
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        train_texts=train_texts,
        validation_texts=validation_texts,
        validation_ratio=validation_ratio,
        seed=seed,
        text_column=text_column,
        label_column=label_column,
        batch_size=batch_size,
        device=device,
    )
    embedding_seconds = time.perf_counter() - embeddings_started

    results: list[HybridSearchResult] = []
    best: HybridSearchResult | None = None
    best_artifact: dict[str, Any] | None = None

    for word_range in word_ngram_ranges:
        for char_range in char_ngram_ranges:
            for max_features in max_features_list:
                tfidf = build_word_char_vectorizer(word_range, char_range, max_features)
                train_tfidf = tfidf.fit_transform(train_texts)
                validation_tfidf = tfidf.transform(validation_texts)
                for embedding_weight in embedding_weights:
                    train_features = hstack(
                        [train_tfidf, csr_matrix(train_embeddings * embedding_weight)],
                        format="csr",
                    )
                    validation_features = hstack(
                        [validation_tfidf, csr_matrix(validation_embeddings * embedding_weight)],
                        format="csr",
                    )
                    for classifier_c in classifier_cs:
                        started = time.perf_counter()
                        classifier = LinearSVC(C=classifier_c, max_iter=5000)
                        classifier.fit(train_features, train_labels)
                        predictions = classifier.predict(validation_features)
                        train_seconds = time.perf_counter() - started
                        result = HybridSearchResult(
                            accuracy=float(accuracy_score(validation_labels, predictions)),
                            word_ngram_range=word_range,
                            char_ngram_range=char_range,
                            max_features=max_features,
                            embedding_weight=embedding_weight,
                            classifier_c=classifier_c,
                            features=train_features.shape[1],
                            train_seconds=train_seconds,
                        )
                        results.append(result)
                        print(
                            "hybrid-search "
                            f"accuracy={result.accuracy:.4f} "
                            f"word={word_range} char={char_range} "
                            f"max_features={max_features} "
                            f"embedding_weight={embedding_weight} "
                            f"classifier_c={classifier_c} "
                            f"features={result.features}"
                        )
                        if best is None or result.accuracy > best.accuracy:
                            best = result
                            best_artifact = {
                                "format": "himdex-hybrid-text-classifier-v1",
                                "checkpoint": str(checkpoint_path),
                                "tfidf": tfidf,
                                "classifier": classifier,
                                "embedding_weight": embedding_weight,
                                "classifier_c": classifier_c,
                                "max_features_per_vectorizer": max_features,
                                "word_ngram_range": word_range,
                                "char_ngram_range": char_range,
                                "validation_ratio": validation_ratio,
                                "seed": seed,
                                "text_column": text_column,
                                "label_column": label_column,
                                "labels": sorted(set(labels)),
                                "features": train_features.shape[1],
                            }

    if best is None:
        raise ValueError("No hybrid search results were produced")

    payload = {
        "dataset": str(data_path),
        "checkpoint": str(checkpoint_path),
        "validation_ratio": validation_ratio,
        "seed": seed,
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "embedding_cache": str(cache_path),
        "embedding_cache_hit": cache_hit,
        "embedding_seconds": embedding_seconds,
        "best": asdict(best),
        "results": [asdict(result) for result in results],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if output_model:
        if best_artifact is None:
            raise ValueError("Missing best artifact")
        model_path = Path(output_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_artifact, model_path)
        payload["output_model"] = str(model_path)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"best accuracy={best.accuracy:.4f} embedding_weight={best.embedding_weight} "
        f"classifier_c={best.classifier_c} saved={output_path}"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search Himdex Hybrid text-classifier settings.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="benchmarks/hybrid_search.json")
    parser.add_argument("--output-model", default="")
    parser.add_argument("--cache", default="work/himdex_hybrid_embeddings.joblib")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features-list", default="200000")
    parser.add_argument("--word-ngram-ranges", default="1-2")
    parser.add_argument("--char-ngram-ranges", default="3-5")
    parser.add_argument("--embedding-weights", default="1.0,1.25,1.5")
    parser.add_argument("--classifier-cs", default="0.75,1.0,1.25")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_hybrid_search(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        output_model=args.output_model or None,
        cache_path=args.cache,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        max_features_list=parse_int_list(args.max_features_list),
        word_ngram_ranges=parse_ngram_ranges(args.word_ngram_ranges),
        char_ngram_ranges=parse_ngram_ranges(args.char_ngram_ranges),
        embedding_weights=parse_float_list(args.embedding_weights),
        classifier_cs=parse_float_list(args.classifier_cs),
        batch_size=args.batch_size,
        text_column=args.text_column,
        label_column=args.label_column,
        device=args.device,
    )


if __name__ == "__main__":
    main()
