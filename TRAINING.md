# Himdex Training Notes

## Himdex Base v3

The current reference checkpoint uses masked byte prediction on mixed text
prepared by `himdex-prepare-data`.

Latest continuation run:

```text
steps: 6,000
continued from: checkpoints/himdex_text_base_v2.pt
batch size: 16
sequence length: 256
hidden size: 384
layers: 6
heads: 6
train loss: 3.1412
validation loss: 3.1240
masked accuracy: 16.94%
published checkpoint: checkpoints/himdex_text_base_v3.pt
```

Fixed-seed evaluation on the same held-out split showed Base v3 improving
masked-byte validation loss from 3.1468 to 3.1322 compared with Base v2.

Recommended local command:

```powershell
himdex-pretrain-text --data data\himdex_pack\prepared --resume-from checkpoints\himdex_text_base_v2.pt --epochs 1 --batch-size 16 --max-steps 2000 --output-dir runs\himdex_text_base_v3
```

Continue from a checkpoint:

```powershell
himdex-pretrain-text --data data\himdex_pack\prepared --resume-from checkpoints\himdex_text_base_v3.pt --epochs 1 --batch-size 16 --max-steps 2000 --output-dir runs\himdex_text_base_v3_continued
```

Evaluate a classification checkpoint:

```powershell
himdex-evaluate --checkpoint checkpoints\himdex_ag_news_pure_avg_v2.pt --data data\himdex_starter\prepared\text_classification\hf_ag_news.csv --batch-size 64
```

Run a fast head-only polish pass:

```powershell
himdex-train --task text-classification --data data\himdex_starter\prepared\text_classification\hf_ag_news.csv --resume-from checkpoints\himdex_ag_news_pure_avg_v2.pt --freeze-backbone --epochs 1 --batch-size 64 --lr 1e-4
```

Run a layer-wise polish pass with a slower backbone learning rate:

```powershell
himdex-train --task text-classification --data data\himdex_starter\prepared\text_classification\hf_ag_news.csv --resume-from checkpoints\himdex_ag_news_pure_avg_v2.pt --epochs 1 --batch-size 64 --backbone-lr 2e-6 --head-lr 2e-5 --weight-decay 0.01
```

For open-source releases, keep raw datasets out of git. Commit code, configs,
docs, and compact reference checkpoints only.

## Baseline Tracking

The current AG News target to beat is the word+character TF-IDF baseline:

```text
TF-IDF word+character: 91.90%
Himdex hybrid with Base v3 embeddings: 92.00%
Pure Himdex averaged classifier: 69.66%
Pure Himdex averaged classifier v1: 69.58%
Pure Himdex previous best classifier: 69.24%
Pure Himdex Base v3 distilled + polished classifier: 67.60%
Pure Himdex Base v3 cls-mean classifier: 66.98%
Pure Himdex Base v3 continued classifier: 65.08%
```

Latest head-only polish test from `himdex_ag_news_pure_avg_v2.pt` trained only
1,540 parameters and reached 69.42% validation accuracy, so it was kept as an
experiment instead of replacing the 69.66% released checkpoint.

Latest layer-wise polish test from `himdex_ag_news_pure_avg_v2.pt` used
`backbone_lr=2e-6` and `head_lr=2e-5`, reaching 69.32% validation accuracy.
That confirms the released averaged checkpoint remains the stronger pure neural
artifact for now.

The next pure-neural training target is to reduce this gap with stronger
pretraining, subword/character-aware text encoders, better pooling, and
distillation from the hybrid classifier.
