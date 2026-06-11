# Himdex Dataset Pack 5GB

Himdex includes a dataset preparation pipeline for building a local multimodal
training pack with a strict storage budget. The default budget is **5GB**.

## Strategy

Large public dataset platforms contain far more data than a local development
machine should download blindly. Himdex uses a controlled registry:

- curated public dataset entries with license/provenance notes;
- Hugging Face streaming where available;
- configurable row and image limits per dataset;
- a generated manifest for auditability;
- separate pretraining and fine-tuning workflows.

## Registry

Default registry entries:

- `github_tiny_shakespeare`: compact text corpus for language modeling checks.
- `hf_ag_news`: news topic classification.
- `hf_imdb`: movie review sentiment classification.
- `hf_wikitext_2`: Wikipedia-style language modeling text.
- `hf_beans`: plant disease image classification.
- `hf_cifar10`: natural image classification via Hugging Face streaming.
- `torchvision_mnist`: handwritten digit classification.
- `torchvision_fashion_mnist`: clothing image classification.
- `torchvision_cifar10`: natural image classification, available as an optional
  torchvision source.
- `kaggle_sms_spam`: SMS spam classification, optional with Kaggle credentials.

## Installation

```powershell
git clone https://github.com/dexpie/himdex.git
cd himdex
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Preview The Plan

```powershell
himdex-prepare-data --budget-gb 5 --dry-run
```

## Build The Dataset Pack

```powershell
himdex-prepare-data --root data\himdex_pack --budget-gb 5
```

Expected outputs:

```text
data/himdex_pack/manifest.json
data/himdex_pack/prepared/text_classification/*.csv
data/himdex_pack/prepared/text_lm/*.txt
data/himdex_pack/prepared/image_classification/
data/himdex_pack/raw/torchvision/
```

Increase local sampling while keeping the same storage budget:

```powershell
himdex-prepare-data --root data\himdex_pack --budget-gb 5 --rows-per-text-dataset 200000 --images-per-image-dataset 50000
```

## Kaggle

Kaggle datasets require account credentials.

Environment variable option:

```powershell
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_api_key"
himdex-prepare-data --include-sources github,hf,torchvision,kaggle --budget-gb 5
```

File option:

```text
C:\Users\<your-user>\.kaggle\kaggle.json
```

After credentials are configured, Kaggle registry entries can be included with
`--include-sources`.

## Text Pretraining

```powershell
himdex-pretrain-text --data data\himdex_pack\prepared --epochs 1 --batch-size 16 --max-text-length 256
```

Masked byte prediction works well for mixed text sources because it does not
depend on incompatible label spaces across datasets.

## Fine-Tuning

Text classification:

```powershell
himdex-train --task text-classification --data data\himdex_pack\prepared\text_classification\hf_ag_news.csv --epochs 3
```

Image classification:

```powershell
himdex-train --task image-classification --data data\images --epochs 3
```

Keep classification datasets task-specific unless their label spaces are
intentionally unified.
