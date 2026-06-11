# Himdex Base v2 Model Card

Himdex Base v2 is an 11.24M-parameter multimodal Transformer backbone for text
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

Himdex Base v2 was trained with masked byte prediction on the Himdex 5GB
dataset pipeline using a curated local subset of public text sources.

| Metric | Value |
| --- | ---: |
| Optimizer steps | 4,000 |
| Train loss | 3.1802 |
| Validation loss | 3.1203 |
| Masked-byte accuracy | 17.06% |

## Transfer Benchmarks

Results use deterministic random validation splits created by the Himdex CLI.
They are development benchmarks, not official dataset leaderboard submissions.

| Task | Dataset | Validation split | Best accuracy | Best loss |
| --- | --- | ---: | ---: | ---: |
| Text classification | AG News | 10% | 69.24% | 0.7788 |
| Image classification | Beans | 15% | 78.71% | 0.5591 |

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
- Text-image alignment has not yet been contrastively trained.
- Reported metrics come from local validation splits and should not be compared
  directly with official test-set leaderboards.

## Responsible Use

Evaluate Himdex on representative data before deployment. Dataset biases,
domain shift, and class imbalance can affect downstream behavior.
