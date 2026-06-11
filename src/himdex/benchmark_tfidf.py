from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    accuracy: float
    train_seconds: float
    predict_seconds: float
    features: int
    train_samples: int
    validation_samples: int


def load_csv(
    path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header")
        for column in (text_column, label_column):
            if column not in reader.fieldnames:
                raise ValueError(f"Missing column: {column}")
        for row in reader:
            text = row[text_column].strip()
            if text:
                texts.append(text)
                labels.append(row[label_column])
    if not texts:
        raise ValueError("CSV dataset is empty")
    return texts, labels


def himdex_split_indices(
    dataset_size: int,
    validation_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    validation_size = max(1, int(dataset_size * validation_ratio))
    train_size = dataset_size - validation_size
    permutation = torch.randperm(
        dataset_size, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    return permutation[:train_size], permutation[train_size:]


def build_pipeline(name: str, max_features: int):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise RuntimeError('Install benchmark dependencies: pip install -e ".[benchmark]"') from exc

    word_unigram = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.98,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    word_bigram = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_ngram = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )

    vectorizers = {
        "word-unigram": word_unigram,
        "word-unigram-bigram": word_bigram,
        "char-3-5": char_ngram,
        "word-char": FeatureUnion(
            [
                ("word", word_bigram),
                ("char", char_ngram),
            ]
        ),
    }
    if name not in vectorizers:
        raise ValueError(f"Unknown benchmark: {name}")
    return Pipeline(
        [
            ("tfidf", vectorizers[name]),
            ("classifier", LinearSVC(C=2.0)),
        ]
    )


def feature_count(pipeline) -> int:
    vectorizer = pipeline.named_steps["tfidf"]
    if hasattr(vectorizer, "transformer_list"):
        return sum(
            len(transformer.vocabulary_)
            for _, transformer in vectorizer.transformer_list
        )
    return len(vectorizer.vocabulary_)


def run_benchmark(
    data_path: str | Path,
    names: list[str],
    validation_ratio: float = 0.1,
    seed: int = 42,
    max_features: int = 200_000,
    text_column: str = "text",
    label_column: str = "label",
) -> list[BenchmarkResult]:
    from sklearn.metrics import accuracy_score

    texts, labels = load_csv(data_path, text_column, label_column)
    train_indices, validation_indices = himdex_split_indices(
        len(texts), validation_ratio, seed
    )
    train_texts = [texts[index] for index in train_indices]
    train_labels = [labels[index] for index in train_indices]
    validation_texts = [texts[index] for index in validation_indices]
    validation_labels = [labels[index] for index in validation_indices]

    results: list[BenchmarkResult] = []
    for name in names:
        pipeline = build_pipeline(name, max_features)
        started = time.perf_counter()
        pipeline.fit(train_texts, train_labels)
        train_seconds = time.perf_counter() - started

        started = time.perf_counter()
        predictions = pipeline.predict(validation_texts)
        predict_seconds = time.perf_counter() - started
        result = BenchmarkResult(
            name=name,
            accuracy=float(accuracy_score(validation_labels, predictions)),
            train_seconds=train_seconds,
            predict_seconds=predict_seconds,
            features=feature_count(pipeline),
            train_samples=len(train_indices),
            validation_samples=len(validation_indices),
        )
        results.append(result)
        print(
            f"{name}: accuracy={result.accuracy:.4f} features={result.features} "
            f"train_seconds={result.train_seconds:.2f} predict_seconds={result.predict_seconds:.2f}"
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TF-IDF baselines against Himdex splits.")
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--models",
        default="word-unigram,word-unigram-bigram,char-3-5,word-char",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output", default="benchmarks/tfidf_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    results = run_benchmark(
        args.data,
        names,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        max_features=args.max_features,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "dataset": str(args.data),
                "validation_ratio": args.validation_ratio,
                "seed": args.seed,
                "max_features_per_vectorizer": args.max_features,
                "results": [asdict(result) for result in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
