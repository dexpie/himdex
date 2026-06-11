from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    source: str
    modality: str
    task: str
    approx_bytes: int
    license_note: str
    description: str
    hf_path: str | None = None
    hf_name: str | None = None
    split: str = "train"
    text_column: str | None = None
    label_column: str | None = None
    image_column: str | None = None
    torchvision_name: str | None = None
    github_url: str | None = None
    kaggle_slug: str | None = None


GB = 1024**3
MB = 1024**2


DATASET_REGISTRY: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        id="github_tiny_shakespeare",
        source="github",
        modality="text",
        task="text-lm",
        approx_bytes=2 * MB,
        license_note="Small public educational corpus; verify upstream before redistribution.",
        description="Tiny Shakespeare plain-text corpus for quick language-model smoke runs.",
        github_url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    ),
    DatasetSpec(
        id="hf_ag_news",
        source="hf",
        modality="text",
        task="text-classification",
        approx_bytes=40 * MB,
        license_note="Use Hugging Face dataset card for exact license/provenance.",
        description="News topic classification dataset.",
        hf_path="ag_news",
        split="train",
        text_column="text",
        label_column="label",
    ),
    DatasetSpec(
        id="hf_imdb",
        source="hf",
        modality="text",
        task="text-classification",
        approx_bytes=90 * MB,
        license_note="Use Hugging Face dataset card for exact license/provenance.",
        description="Movie review sentiment classification dataset.",
        hf_path="imdb",
        split="train",
        text_column="text",
        label_column="label",
    ),
    DatasetSpec(
        id="hf_yelp_polarity",
        source="hf",
        modality="text",
        task="text-classification",
        approx_bytes=250 * MB,
        license_note="Use Hugging Face dataset card for exact license/provenance.",
        description="Large review polarity classification dataset.",
        hf_path="yelp_polarity",
        split="train",
        text_column="text",
        label_column="label",
    ),
    DatasetSpec(
        id="hf_wikitext_2",
        source="hf",
        modality="text",
        task="text-lm",
        approx_bytes=15 * MB,
        license_note="Use Hugging Face dataset card for exact license/provenance.",
        description="Clean Wikipedia text for language modeling experiments.",
        hf_path="wikitext",
        hf_name="wikitext-2-raw-v1",
        split="train",
        text_column="text",
    ),
    DatasetSpec(
        id="hf_beans",
        source="hf",
        modality="image",
        task="image-classification",
        approx_bytes=200 * MB,
        license_note="Use Hugging Face dataset card for exact license/provenance.",
        description="Small plant disease image classification dataset.",
        hf_path="beans",
        split="train",
        image_column="image",
        label_column="labels",
    ),
    DatasetSpec(
        id="hf_cifar10",
        source="hf",
        modality="image",
        task="image-classification",
        approx_bytes=180 * MB,
        license_note="Use Hugging Face dataset card for exact license/provenance.",
        description="10-class natural image classification dataset via Hugging Face streaming.",
        hf_path="cifar10",
        split="train",
        image_column="img",
        label_column="label",
    ),
    DatasetSpec(
        id="torchvision_mnist",
        source="torchvision",
        modality="image",
        task="image-classification",
        approx_bytes=20 * MB,
        license_note="Classic benchmark; verify upstream license before redistribution.",
        description="Handwritten digit classification dataset.",
        torchvision_name="MNIST",
    ),
    DatasetSpec(
        id="torchvision_fashion_mnist",
        source="torchvision",
        modality="image",
        task="image-classification",
        approx_bytes=35 * MB,
        license_note="MIT license according to original Fashion-MNIST project.",
        description="Clothing item classification dataset.",
        torchvision_name="FashionMNIST",
    ),
    DatasetSpec(
        id="torchvision_cifar10",
        source="torchvision",
        modality="image",
        task="image-classification",
        approx_bytes=180 * MB,
        license_note="Classic benchmark; verify upstream license before redistribution.",
        description="10-class natural image classification dataset.",
        torchvision_name="CIFAR10",
    ),
    DatasetSpec(
        id="kaggle_sms_spam",
        source="kaggle",
        modality="text",
        task="text-classification",
        approx_bytes=5 * MB,
        license_note="Kaggle dataset terms apply; requires Kaggle account token.",
        description="SMS spam classification dataset.",
        kaggle_slug="uciml/sms-spam-collection-dataset",
    ),
)


def registry_by_id() -> dict[str, DatasetSpec]:
    return {spec.id: spec for spec in DATASET_REGISTRY}


def default_dataset_ids() -> list[str]:
    return [
        "github_tiny_shakespeare",
        "hf_ag_news",
        "hf_imdb",
        "hf_wikitext_2",
        "hf_beans",
        "hf_cifar10",
        "torchvision_mnist",
        "torchvision_fashion_mnist",
    ]
