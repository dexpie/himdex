# Changelog

## 0.14.0 - 2026-06-12

- Added `himdex-search-hybrid`, a cached grid-search CLI for Himdex Hybrid text
  classifiers.
- Added the searched AG News hybrid artifact
  `himdex_hybrid_ag_news_search_v1.joblib`.
- Improved the strongest packaged AG News artifact from 92.26% to 92.34%
  validation accuracy with `embedding_weight=1.3` and `classifier_c=0.9`.
- Added benchmark search output for the winning grid.

## 0.13.0 - 2026-06-12

- Tuned the packaged AG News Himdex Hybrid model by lowering the LinearSVC
  regularization strength to `C=1.0` and raising the embedding weight to `1.25`.
- Added `--classifier-c` to `himdex-hybrid train` and
  `himdex-benchmark-hybrid`.
- Added the packaged `himdex_hybrid_ag_news_base_v3_v2.joblib` artifact.
- Improved the strongest packaged AG News artifact from 92.00% to 92.26%
  validation accuracy on the deterministic Himdex split.

## 0.12.0 - 2026-06-12

- Added `--backbone-lr` and `--head-lr` to `himdex-train` for layer-wise
  fine-tuning.
- Added `--grad-accum-steps` so local runs can use a larger effective batch
  size without increasing VRAM.
- Stored optimizer learning rates, gradient accumulation steps, and effective
  batch size in classification checkpoints.
- Adjusted cosine LR scheduling to respect the smallest optimizer group LR.
- Added smoke coverage for layer-wise optimizer groups.
- Tested a layer-wise polish pass from `himdex_ag_news_pure_avg_v2.pt`; it
  reached 69.32%, so the 69.66% released checkpoint remains the pure neural
  best.

## 0.11.0 - 2026-06-12

- Added `--freeze-backbone` to `himdex-train` for fast head-only fine-tuning.
- Kept frozen backbones in eval mode during training so head-only experiments
  use stable representations.
- Saved `freeze_backbone` and `trainable_parameters` metadata in training
  checkpoints.
- Added smoke coverage for freezing the backbone while keeping the classifier
  trainable.
- Tested a head-only polish pass from `himdex_ag_news_pure_avg_v2.pt`; it
  trained only 1,540 parameters and reached 69.42%, so the 69.66% checkpoint
  remains the released pure neural best.

## 0.10.0 - 2026-06-12

- Added `himdex-evaluate`, a reproducible evaluation CLI for text and image
  classification checkpoints.
- Added a smoke test for checkpoint evaluation on toy text data.
- Recorded a generated AG News evaluation JSON for the current pure averaged
  checkpoint.
- Tested a low-learning-rate polish pass from `himdex_ag_news_pure_avg_v2.pt`;
  it did not beat the released 69.66% checkpoint, so no weaker checkpoint was
  promoted.

## 0.9.0 - 2026-06-12

- Added the refined pure averaged AG News checkpoint
  `himdex_ag_news_pure_avg_v2.pt`.
- Improved the pure Himdex AG News validation best from 69.58% to 69.66% by
  narrowing the interpolation search around the distilled checkpoint.
- Recorded the new benchmark and checkpoint metadata for reproducible local
  comparison.

## 0.8.0 - 2026-06-12

- Added `himdex-average-checkpoints` for averaging compatible Himdex model
  checkpoints.
- Added the pure averaged AG News checkpoint `himdex_ag_news_pure_avg_v1.pt`.
- Improved the pure Himdex AG News best from 69.24% to 69.58% validation
  accuracy by averaging the previous pure best with a distilled checkpoint.

## 0.7.0 - 2026-06-12

- Added `himdex-distill-text` for distilling a packaged Himdex Hybrid teacher
  into a pure neural text classifier.
- Added label-order handling for distillation resume runs so teacher logits are
  aligned with the student classifier head.
- Recorded the first pure neural distillation run. Distillation plus hard-label
  polish improved the Base v3 pure path to 67.60% validation accuracy, up from
  66.98%, while still trailing the 69.24% pure Himdex best.

## 0.6.0 - 2026-06-12

- Added pure neural text pooling modes for `himdex-train`: `cls`, `mean`, and
  `cls-mean`.
- Added a stronger MLP classifier head for mean-based text pooling.
- Recorded the AG News pure `cls-mean` experiment, which improved the Base v3
  pure classifier path to 66.98% validation accuracy while still trailing the
  69.24% pure Himdex best.

## 0.5.0 - 2026-06-12

- Added `himdex-hybrid`, a train/predict CLI for persisted Himdex hybrid text
  classifiers.
- Added the `himdex_hybrid_ag_news_base_v3.joblib` reference artifact.
- Promoted Himdex Hybrid AG News as the strongest packaged text model so far,
  reaching 92.00% validation accuracy and beating the 91.90% TF-IDF word+char
  baseline on the same split.

## 0.4.0 - 2026-06-11

- Added Himdex Base v3, continued from Base v2 to 6,000 masked-byte
  pretraining steps.
- Added Base v3 checkpoint metadata and AG News hybrid benchmark output.
- Improved the AG News hybrid benchmark from 91.94% to 92.00% validation
  accuracy when using Base v3 embeddings.
- Documented that the pure Base v3 AG News classifier improved to 65.08% after
  continued fine-tuning, but still trails the previous 69.24% pure-neural best.

## 0.3.0 - 2026-06-11

- Added TF-IDF and Himdex hybrid text benchmark commands.
- Added AG News sparse baseline results and a hybrid benchmark that reaches
  91.94% validation accuracy on the Himdex split.
- Added batched text/image embedding inference for safer local GPU usage.

## 0.2.1 - 2026-06-11

- Fixed PEP 639 license metadata compatibility with current setuptools.
- Updated GitHub Actions to Node 24-compatible action versions.
- Added package build validation to CI.

## 0.2.0 - 2026-06-11

- Added Himdex Base v2 with 11.24M parameters and a 384-dimensional embedding
  space.
- Added mixed-precision pretraining, validation metrics, gradient clipping,
  cosine learning-rate scheduling, atomic checkpoints, and optimizer resume.
- Added resumable text and image classification fine-tuning with best-checkpoint
  selection.
- Added `HimdexEncoder` and the `himdex-embed` command for text and image
  embeddings.
- Added a model card with AG News and Beans transfer benchmarks.
- Added GitHub Actions CI and optional dependency groups.
- Renamed public dataset paths and checkpoint variants for clearer packaging.

## 0.1.0 - 2026-06-11

- Initial open-source release with shared text/image Transformer backbone,
  dataset preparation, masked-text pretraining, and classification commands.
