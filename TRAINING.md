# Himdex Training Notes

## Himdex Base v2

The current reference checkpoint uses masked byte prediction on mixed text
prepared by `himdex-prepare-data`.

Latest continuation run:

```text
steps: 4,000
batch size: 32
sequence length: 256
hidden size: 384
layers: 6
heads: 6
train loss: 3.1802
validation loss: 3.1203
masked accuracy: 17.06%
published checkpoint: checkpoints/himdex_text_base_v2.pt
```

Recommended local command:

```powershell
himdex-pretrain-text --data data\himdex_pack\prepared --epochs 2 --batch-size 32 --hidden-size 384 --num-layers 6 --num-heads 6 --max-text-length 256 --max-steps 4000 --output-dir runs\himdex_text_base_v2
```

Continue from a checkpoint:

```powershell
himdex-pretrain-text --data data\himdex_pack\prepared --resume-from checkpoints\himdex_text_base_v2.pt --epochs 1 --batch-size 32 --max-steps 2000 --output-dir runs\himdex_text_base_v2_continued
```

For open-source releases, keep raw datasets out of git. Commit code, configs,
docs, and compact reference checkpoints only.

## Baseline Tracking

The current AG News target to beat is the word+character TF-IDF baseline:

```text
TF-IDF word+character: 91.90%
Himdex hybrid: 91.94%
Pure Himdex Base v2 classifier: 69.24%
```

The next pure-neural training target is to reduce this gap with stronger
pretraining, subword/character-aware text encoders, and distillation from the
hybrid classifier.
