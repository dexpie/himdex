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
- Built-in embedding API and CLI for downstream applications.

## Himdex Base v3

The current base checkpoint has 11.24M parameters and was continued to 6,000
masked-byte prediction steps from Base v2.

| Benchmark | Validation accuracy |
| --- | ---: |
| AG News pure Himdex best | 69.24% |
| AG News pure Himdex Base v3 distilled + polished | 67.60% |
| AG News pure Himdex Base v3 cls-mean | 66.98% |
| AG News pure Himdex Base v3 continued | 65.08% |
| AG News TF-IDF word+char baseline | 91.90% |
| AG News Himdex Hybrid packaged model | 92.00% |
| Beans image classification | 78.71% |

See [MODEL_CARD.md](MODEL_CARD.md) for configuration, training metrics,
limitations, and evaluation details.

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

Install dataset connectors when needed:

```powershell
pip install -e ".[data]"
```

Install benchmark and hybrid-model dependencies when needed:

```powershell
pip install -e ".[benchmark]"
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

Use a stronger pure neural pooling head:

```powershell
himdex-train --task text-classification --data data\reviews.csv --backbone-from checkpoints\himdex_text_base_v3.pt --text-pooling cls-mean
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
himdex-pretrain-text --data data\himdex_pack\prepared --resume-from checkpoints\himdex_text_base_v3.pt --epochs 1 --batch-size 32 --max-steps 2000
```

## Generate Embeddings

Python API:

```python
from himdex import HimdexEncoder

encoder = HimdexEncoder("checkpoints/himdex_text_base_v3.pt")
embeddings = encoder.encode_text(["Himdex is open source."])
print(embeddings.shape)
```

Command line:

```powershell
himdex-embed --checkpoint checkpoints\himdex_text_base_v3.pt --text "Himdex is open source" --output embeddings.pt
```

## Text Baselines

Run TF-IDF baselines on the same deterministic split used by `himdex-train`:

```powershell
pip install -e ".[benchmark]"
himdex-benchmark-tfidf --data data\himdex_pack\prepared\text_classification\hf_ag_news.csv
```

Run the hybrid benchmark that combines TF-IDF word+character features with
Himdex embeddings:

```powershell
himdex-benchmark-hybrid --data data\himdex_pack\prepared\text_classification\hf_ag_news.csv --checkpoint checkpoints\himdex_text_base_v3.pt
```

Train and run a persisted Himdex Hybrid text classifier:

```powershell
himdex-hybrid train --data data\himdex_pack\prepared\text_classification\hf_ag_news.csv --checkpoint checkpoints\himdex_text_base_v3.pt --output checkpoints\himdex_hybrid_ag_news_base_v3.joblib
himdex-hybrid predict --model checkpoints\himdex_hybrid_ag_news_base_v3.joblib --checkpoint checkpoints\himdex_text_base_v3.pt --text "NASA launches a new satellite"
```

Distill the hybrid teacher into a pure neural classifier:

```powershell
himdex-distill-text --data data\himdex_pack\prepared\text_classification\hf_ag_news.csv --teacher checkpoints\himdex_hybrid_ag_news_base_v3.joblib --backbone-from checkpoints\himdex_text_base_v3.pt --resume-from runs\himdex_ag_news_pure_cls_mean_continued\himdex.pt
```

## Reference Checkpoint

This repository includes a compact reference checkpoint:

```text
checkpoints/himdex_text_base_v3.pt
checkpoints/himdex_hybrid_ag_news_base_v3.joblib
```

It is trained with masked byte prediction and can be used to test loading,
resume training, or build downstream experiments.

## Technical Direction

Himdex is designed to grow through focused, measurable improvements:

- stronger text pretraining objectives;
- pure text distillation from the packaged Himdex Hybrid model;
- hybrid neural/sparse text models;
- subword and character-aware text encoders;
- masked image modeling and contrastive visual learning;
- text-image alignment for shared embedding spaces;
- efficient fine-tuning with LoRA, quantization, pruning, and distillation;
- automatic dataset inspection and task routing;
- evaluation across accuracy, latency, memory, robustness, and transfer.

## License

Himdex is released under the MIT License.
