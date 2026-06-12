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
| TF-IDF word + character + Himdex Base v3 embedding searched | 92.34% |
| TF-IDF word + character + Himdex Base v3 embedding v2 | 92.26% |
| TF-IDF word + character + Himdex Base v3 embedding v1 | 92.00% |
| Himdex pure averaged classifier v2 | 69.66% |
| Himdex pure averaged classifier v1 | 69.58% |
| Himdex Base v3 distilled + polished classifier | 67.60% |
| Himdex Base v3 cls-mean classifier | 66.98% |
| Himdex Base v3 classifier continued | 65.08% |

Reproduce:

```powershell
himdex-benchmark-tfidf --data data\himdex_starter\prepared\text_classification\hf_ag_news.csv --output benchmarks\ag_news_tfidf.json
himdex-benchmark-hybrid --data data\himdex_starter\prepared\text_classification\hf_ag_news.csv --checkpoint checkpoints\himdex_text_base_v2.pt --output benchmarks\ag_news_hybrid.json
himdex-benchmark-hybrid --data data\himdex_starter\prepared\text_classification\hf_ag_news.csv --checkpoint checkpoints\himdex_text_base_v3.pt --output benchmarks\ag_news_hybrid_base_v3_v2.json
himdex-search-hybrid --data data\himdex_starter\prepared\text_classification\hf_ag_news.csv --checkpoint checkpoints\himdex_text_base_v3.pt --output benchmarks\ag_news_hybrid_search_v1.json --output-model checkpoints\himdex_hybrid_ag_news_search_v1.joblib --embedding-weights 1.0,1.1,1.2,1.25,1.3,1.4,1.5 --classifier-cs 0.8,0.9,1.0,1.1,1.2
himdex-hybrid train --data data\himdex_starter\prepared\text_classification\hf_ag_news.csv --checkpoint checkpoints\himdex_text_base_v3.pt --embedding-weight 1.3 --classifier-c 0.9 --output checkpoints\himdex_hybrid_ag_news_search_v1.joblib
himdex-evaluate --checkpoint checkpoints\himdex_ag_news_pure_avg_v2.pt --data data\himdex_starter\prepared\text_classification\hf_ag_news.csv --batch-size 64 --output benchmarks\ag_news_pure_avg_v2_eval.json
himdex-distill-text --data data\himdex_starter\prepared\text_classification\hf_ag_news.csv --teacher checkpoints\himdex_hybrid_ag_news_search_v1.joblib --backbone-from checkpoints\himdex_text_base_v3.pt --resume-from runs\himdex_ag_news_pure_cls_mean_continued\himdex.pt
himdex-average-checkpoints --first runs\himdex_ag_news_v2_continued\himdex.pt --second runs\himdex_ag_news_v2_best_distill_a01\himdex.pt --second-weight 0.74 --output checkpoints\himdex_ag_news_pure_avg_v2.pt
```
