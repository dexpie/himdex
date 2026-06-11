from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .dataset_registry import DATASET_REGISTRY, DatasetSpec, default_starter_ids, registry_by_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare curated starter datasets for Himdex.")
    parser.add_argument("--root", default="data/himdex_starter", help="Output dataset root.")
    parser.add_argument("--budget-gb", type=float, default=5.0, help="Hard local storage budget.")
    parser.add_argument(
        "--include-sources",
        default="github,hf,torchvision",
        help="Comma-separated sources: github,hf,torchvision,kaggle.",
    )
    parser.add_argument(
        "--dataset-ids",
        default=",".join(default_starter_ids()),
        help="Comma-separated registry ids. Use 'all' for every known registry item.",
    )
    parser.add_argument("--rows-per-text-dataset", type=int, default=50000)
    parser.add_argument("--images-per-image-dataset", type=int, default=15000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def ensure_budget(root: Path, budget_bytes: int, next_estimate: int) -> bool:
    return dir_size(root) + next_estimate <= budget_bytes


def write_manifest(root: Path, records: list[dict]) -> None:
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def selected_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    allowed_sources = {source.strip() for source in args.include_sources.split(",") if source.strip()}
    by_id = registry_by_id()
    if args.dataset_ids == "all":
        ids = [spec.id for spec in DATASET_REGISTRY]
    else:
        ids = [item.strip() for item in args.dataset_ids.split(",") if item.strip()]

    missing = [dataset_id for dataset_id in ids if dataset_id not in by_id]
    if missing:
        raise ValueError(f"Unknown dataset ids: {', '.join(missing)}")

    return [by_id[dataset_id] for dataset_id in ids if by_id[dataset_id].source in allowed_sources]


def prepare_github_text(spec: DatasetSpec, root: Path, budget_bytes: int, dry_run: bool) -> dict:
    assert spec.github_url is not None
    target_dir = root / "prepared" / "text_lm"
    target_path = target_dir / f"{spec.id}.txt"
    if target_path.exists() and target_path.stat().st_size > 0:
        return record_for(spec, target_path, "ready")
    if dry_run:
        return record_for(spec, target_path, "planned")

    if not ensure_budget(root, budget_bytes, spec.approx_bytes):
        return record_for(spec, target_path, "skipped_budget")

    target_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(spec.github_url, timeout=60) as response:
        data = response.read()
    if dir_size(root) + len(data) > budget_bytes:
        return record_for(spec, target_path, "skipped_budget")
    target_path.write_bytes(data)
    return record_for(spec, target_path, "ready")


def prepare_hf_text(
    spec: DatasetSpec,
    root: Path,
    budget_bytes: int,
    dry_run: bool,
    rows_per_dataset: int,
) -> dict:
    target_dir = root / "prepared" / ("text_classification" if spec.task == "text-classification" else "text_lm")
    suffix = ".csv" if spec.task == "text-classification" else ".txt"
    target_path = target_dir / f"{spec.id}{suffix}"
    if target_path.exists() and target_path.stat().st_size > 0:
        return record_for(spec, target_path, "ready")
    if dry_run:
        return record_for(spec, target_path, "planned")
    if not ensure_budget(root, budget_bytes, min(spec.approx_bytes, budget_bytes)):
        return record_for(spec, target_path, "skipped_budget")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -e .") from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        spec.hf_path,
        spec.hf_name,
        split=spec.split,
        streaming=True,
        trust_remote_code=False,
    )

    if spec.task == "text-classification":
        assert spec.text_column is not None and spec.label_column is not None
        with target_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["source", "text", "label"])
            writer.writeheader()
            rows_written = 0
            for row in tqdm(dataset, desc=spec.id):
                text = str(row.get(spec.text_column, "")).strip()
                if not text:
                    continue
                writer.writerow(
                    {
                        "source": spec.id,
                        "text": text.replace("\r\n", "\n"),
                        "label": str(row.get(spec.label_column, "")),
                    }
                )
                rows_written += 1
                if rows_written >= rows_per_dataset or dir_size(root) >= budget_bytes:
                    break
    else:
        assert spec.text_column is not None
        with target_path.open("w", encoding="utf-8") as file:
            rows_written = 0
            for row in tqdm(dataset, desc=spec.id):
                text = str(row.get(spec.text_column, "")).strip()
                if text:
                    file.write(text + "\n")
                    rows_written += 1
                if rows_written >= rows_per_dataset or dir_size(root) >= budget_bytes:
                    break

    return record_for(spec, target_path, "ready")


def prepare_hf_image(
    spec: DatasetSpec,
    root: Path,
    budget_bytes: int,
    dry_run: bool,
    images_per_dataset: int,
) -> dict:
    target_dir = root / "prepared" / "image_classification" / spec.id
    if target_dir.exists() and any(target_dir.rglob("*.*")):
        return record_for(spec, target_dir, "ready")
    if dry_run:
        return record_for(spec, target_dir, "planned")
    if not ensure_budget(root, budget_bytes, min(spec.approx_bytes, budget_bytes)):
        return record_for(spec, target_dir, "skipped_budget")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -e .") from exc

    assert spec.image_column is not None and spec.label_column is not None
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        spec.hf_path,
        spec.hf_name,
        split=spec.split,
        streaming=True,
        trust_remote_code=False,
    )

    count = 0
    for row in tqdm(dataset, desc=spec.id):
        label = str(row.get(spec.label_column, "unknown"))
        image = row.get(spec.image_column)
        if image is None:
            continue
        label_dir = target_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        image_path = label_dir / f"{count:08d}.jpg"
        image.convert("RGB").save(image_path, quality=90)
        count += 1
        if count >= images_per_dataset or dir_size(root) >= budget_bytes:
            break

    return record_for(spec, target_dir, "ready")


def prepare_torchvision_image(spec: DatasetSpec, root: Path, budget_bytes: int, dry_run: bool) -> dict:
    raw_dir = root / "raw" / "torchvision"
    target_dir = raw_dir / spec.id
    actual_dir = raw_dir / str(spec.torchvision_name)
    if actual_dir.exists() and any(actual_dir.rglob("*")):
        return record_for(spec, actual_dir, "ready")
    if dry_run:
        return record_for(spec, target_dir, "planned")
    if not ensure_budget(root, budget_bytes, spec.approx_bytes):
        return record_for(spec, target_dir, "skipped_budget")

    try:
        from torchvision import datasets
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -e .") from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    for empty_archive in raw_dir.glob("*.tar.gz"):
        if empty_archive.stat().st_size == 0:
            empty_archive.unlink()
    dataset_cls = getattr(datasets, spec.torchvision_name)
    dataset_cls(root=str(raw_dir), train=True, download=True)
    return record_for(spec, target_dir, "ready")


def prepare_kaggle(spec: DatasetSpec, root: Path, budget_bytes: int, dry_run: bool) -> dict:
    target_dir = root / "raw" / "kaggle" / spec.id
    if target_dir.exists() and any(target_dir.rglob("*")):
        return record_for(spec, target_dir, "ready")
    if dry_run:
        return record_for(spec, target_dir, "planned")
    if not kaggle_credentials_exist():
        return record_for(spec, target_dir, "skipped_missing_kaggle_token")
    if not ensure_budget(root, budget_bytes, spec.approx_bytes):
        return record_for(spec, target_dir, "skipped_budget")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -e .") from exc

    assert spec.kaggle_slug is not None
    target_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(spec.kaggle_slug, path=str(target_dir), unzip=True, quiet=False)
    return record_for(spec, target_dir, "ready")


def kaggle_credentials_exist() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def record_for(spec: DatasetSpec, path: Path, status: str) -> dict:
    return {
        "id": spec.id,
        "source": spec.source,
        "modality": spec.modality,
        "task": spec.task,
        "status": status,
        "path": str(path),
        "approx_bytes": spec.approx_bytes,
        "description": spec.description,
        "license_note": spec.license_note,
    }


def prepare_one(
    spec: DatasetSpec,
    root: Path,
    budget_bytes: int,
    dry_run: bool,
    rows_per_text_dataset: int,
    images_per_image_dataset: int,
) -> dict:
    if spec.source == "github":
        return prepare_github_text(spec, root, budget_bytes, dry_run)
    if spec.source == "hf" and spec.modality == "text":
        return prepare_hf_text(spec, root, budget_bytes, dry_run, rows_per_text_dataset)
    if spec.source == "hf" and spec.modality == "image":
        return prepare_hf_image(spec, root, budget_bytes, dry_run, images_per_image_dataset)
    if spec.source == "torchvision":
        return prepare_torchvision_image(spec, root, budget_bytes, dry_run)
    if spec.source == "kaggle":
        return prepare_kaggle(spec, root, budget_bytes, dry_run)
    return record_for(spec, root, "skipped_unsupported")


def print_plan(specs: Iterable[DatasetSpec], budget_bytes: int) -> None:
    print(f"Budget: {budget_bytes / (1024**3):.2f} GB")
    for spec in specs:
        print(
            f"- {spec.id}: {spec.source}/{spec.modality}/{spec.task}, "
            f"est {spec.approx_bytes / (1024**2):.1f} MB"
        )


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    budget_bytes = int(args.budget_gb * 1024**3)
    specs = selected_specs(args)

    print_plan(specs, budget_bytes)
    records: list[dict] = []
    for spec in specs:
        try:
            record = prepare_one(
                spec,
                root,
                budget_bytes,
                args.dry_run,
                args.rows_per_text_dataset,
                args.images_per_image_dataset,
            )
        except Exception as exc:
            record = record_for(spec, root, f"error: {exc}")
        records.append(record)
        print(f"{record['status']}: {spec.id}")
        if dir_size(root) >= budget_bytes:
            print("Budget reached; stopping.")
            break

    write_manifest(root, records)
    print(f"Manifest: {root / 'manifest.json'}")
    print(f"Used: {dir_size(root) / (1024**3):.3f} GB")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
