# Benchmarks

Benchmarks use deterministic validation splits matched to the Himdex training
CLI so sparse baselines, hybrid models, and neural models can be compared on the
same examples.

## AG News

Source file used locally:

```text
data/himdex_starter/prepared/text_classification/hf_ag_news.csv
```

Split:

```text
seed: 42
train samples: 45,000
validation samples: 5,000
```

Results:

| Model | Accuracy |
| --- | ---: |
| TF-IDF word unigram | 90.56% |
| TF-IDF word unigram + bigram | 91.60% |
| TF-IDF character 3-5 gram | 91.46% |
| TF-IDF word + character | 91.90% |
| TF-IDF word + character + Himdex Base v2 embedding | 91.94% |

Reproduce:

```powershell
himdex-benchmark-tfidf --data data\himdex_starter\prepared\text_classification\hf_ag_news.csv --output benchmarks\ag_news_tfidf.json
himdex-benchmark-hybrid --data data\himdex_starter\prepared\text_classification\hf_ag_news.csv --checkpoint checkpoints\himdex_text_base_v2.pt --output benchmarks\ag_news_hybrid.json
```
