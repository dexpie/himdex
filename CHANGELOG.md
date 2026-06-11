# Changelog

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
