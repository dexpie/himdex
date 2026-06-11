# Himdex

Efficient multimodal Transformer for text and image learning.

Himdex is a compact PyTorch model family built around a shared Transformer
backbone. Text is encoded as UTF-8 byte tokens, images are encoded as visual
patch tokens, and task-specific heads can be attached for classification,
masked text modeling, and future multimodal training.

## Highlights

- Shared Transformer backbone for text and image pipelines.
- Byte-level tokenizer with no separate tokenizer training step.
- Patch-based image encoder compatible with the same sequence backbone.
- Training commands for text classification, image classification, and masked
  text pretraining.
- Curated dataset preparation pipeline with a configurable local storage
  budget.
- Resumable masked-text pretraining from existing checkpoints.

## Architecture

```text
UTF-8 text -> ByteTokenizer -> token embedding \
                                                -> shared Transformer -> task head
Images     -> image patches  -> patch embedding /
```

The byte tokenizer makes Himdex easy to apply across languages and noisy text.
The patch encoder turns images into token sequences, letting the same backbone
support visual tasks.

## Installation

```powershell
git clone https://github.com/dexpie/himdex.git
cd himdex
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Text Classification

Prepare a CSV with `text` and `label` columns.

```csv
text,label
"aku suka produk ini",positif
"ini buruk sekali",negatif
```

Train:

```powershell
himdex-train --task text-classification --data data\reviews.csv --epochs 5 --batch-size 16
```

Custom columns:

```powershell
himdex-train --task text-classification --data data\reviews.csv --text-column kalimat --label-column kelas
```

Quick local run:

```powershell
himdex-train --task text-classification --data examples\text_toy.csv --epochs 1 --batch-size 2 --hidden-size 64 --num-layers 1 --max-text-length 32
```

## Image Classification

Use an `ImageFolder` layout:

```text
data/images/
  cat/
    a.jpg
    b.jpg
  dog/
    c.jpg
    d.jpg
```

Train:

```powershell
himdex-train --task image-classification --data data\images --epochs 5 --batch-size 16
```

Checkpoints are written to:

```text
runs/himdex/himdex.pt
```

## Dataset Pack

Himdex includes a dataset preparation command for public sources such as GitHub,
Hugging Face, torchvision, and optional Kaggle datasets.

```powershell
himdex-prepare-data --root data\himdex_pack --budget-gb 5
```

See [DATASETS.md](DATASETS.md) for registry details, Kaggle setup, and storage
budget controls.

## Masked Text Pretraining

Run masked byte prediction on prepared text data:

```powershell
himdex-pretrain-text --data data\himdex_pack\prepared --epochs 1 --batch-size 8 --max-text-length 128
```

Resume from an existing checkpoint:

```powershell
himdex-pretrain-text --data data\himdex_pack\prepared --resume-from checkpoints\himdex_text_base.pt --epochs 1 --batch-size 8 --max-steps 200
```

## Reference Checkpoint

This repository includes a compact reference checkpoint:

```text
checkpoints/himdex_text_base.pt
```

It is trained with masked byte prediction and can be used to test loading,
resume training, or build downstream experiments.

## Technical Direction

Himdex is designed to grow through focused, measurable improvements:

- stronger text pretraining objectives;
- masked image modeling and contrastive visual learning;
- text-image alignment for shared embedding spaces;
- efficient fine-tuning with LoRA, quantization, pruning, and distillation;
- automatic dataset inspection and task routing;
- evaluation across accuracy, latency, memory, robustness, and transfer.

## License

Himdex is released under the MIT License.
