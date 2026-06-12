# Himdex Base v3 Model Card

Himdex Base v3 is an 11.24M-parameter multimodal Transformer backbone for text
and image learning. It represents text as UTF-8 byte tokens and images as patch
tokens processed by a shared encoder.

## Configuration

| Field | Value |
| --- | ---: |
| Parameters | 11,241,475 |
| Hidden size | 384 |
| Transformer layers | 6 |
| Attention heads | 6 |
| Text sequence length | 256 bytes |
| Image resolution | 224 x 224 |
| Image patch size | 16 x 16 |
| Vocabulary | 259 byte-level tokens |

## Pretraining

Himdex Base v3 was continued from Base v2 with masked byte prediction on the
Himdex local dataset pipeline.

| Metric | Value |
| --- | ---: |
| Optimizer steps | 6,000 |
| Train loss | 3.1412 |
| Validation loss | 3.1240 |
| Masked-byte accuracy | 16.94% |

With a fixed evaluation seed on the same held-out split, Base v3 improved
masked-byte validation loss from 3.1468 to 3.1322 compared with Base v2.

## Transfer Benchmarks

Results use deterministic random validation splits created by the Himdex CLI.
They are development benchmarks, not official dataset leaderboard submissions.
Classification checkpoints can be re-evaluated with `himdex-evaluate`.

| Task | Dataset | Validation split | Best accuracy | Best loss |
| --- | --- | ---: | ---: | ---: |
| Text classification, pure averaged Himdex v3 | AG News | 10% | 69.68% | 0.7706 |
| Text classification, pure averaged Himdex v2 | AG News | 10% | 69.66% | 0.7709 |
| Text classification, pure averaged Himdex v1 | AG News | 10% | 69.58% | 0.7708 |
| Text classification, previous best pure Himdex | AG News | 10% | 69.24% | 0.7788 |
| Text classification, Base v3 distilled + polished | AG News | 10% | 67.60% | 0.8238 |
| Text classification, Base v3 cls-mean | AG News | 10% | 66.98% | 0.8305 |
| Text classification, Base v3 continued | AG News | 10% | 65.08% | 0.8646 |
| Image classification | Beans | 15% | 78.71% | 0.5591 |

The pure averaged Himdex v3 AG News report has macro F1 of 69.37%. Its weakest
class is label `2`, with 56.08% recall and 62.84% F1.

## Text Baseline Benchmarks

All AG News text benchmarks below use the same deterministic 45,000/5,000
train/validation split as the Himdex fine-tuning run.

| Model | Features | Validation accuracy |
| --- | ---: | ---: |
| TF-IDF word unigram | 27,548 | 90.56% |
| TF-IDF word unigram + bigram | 200,000 | 91.60% |
| TF-IDF character 3-5 gram | 169,009 | 91.46% |
| TF-IDF word + character | 369,009 | 91.90% |
| TF-IDF word + character + Himdex Base v2 embedding | 369,393 | 91.94% |
| TF-IDF word + character + Himdex Base v3 embedding searched | 369,393 | 92.34% |
| TF-IDF word + character + Himdex Base v3 embedding v2 | 369,393 | 92.26% |
| TF-IDF word + character + Himdex Base v3 embedding v1 | 369,393 | 92.00% |

The searched Himdex Hybrid AG News model is the strongest text artifact in this
repository so far. It beats the strongest sparse-only TF-IDF baseline on this
split, while also making the next research target clear: close the gap between
the pure neural model and the hybrid/classical baseline.

## Intended Use

- text and image representation learning;
- transfer-learning experiments;
- compact model research and education;
- testing multimodal training objectives;
- downstream classification after fine-tuning.

## Limitations

- Pretraining data and compute are currently modest relative to large
  foundation models.
- Text uses byte-level tokenization, which is robust across languages but less
  sequence-efficient than a trained subword tokenizer.
- Pure Himdex text classification currently trails strong sparse TF-IDF
  baselines on AG News.
- Text-image alignment has not yet been contrastively trained.
- Reported metrics come from local validation splits and should not be compared
  directly with official test-set leaderboards.

## Responsible Use

Evaluate Himdex on representative data before deployment. Dataset biases,
domain shift, and class imbalance can affect downstream behavior.
