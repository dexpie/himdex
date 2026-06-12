# Changelog

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
