# Himdex Checkpoints

Compact reference checkpoints can live here when they are useful for examples,
resume training, or downstream experiments.

- `himdex_text_base_v3.pt`: 11.24M-parameter primary reference checkpoint.
- `himdex_ag_news_pure_avg_v2.pt`: current pure neural AG News classifier
  checkpoint produced by refined averaging of the previous pure best with a
  distilled checkpoint.
- `himdex_ag_news_pure_avg_v1.pt`: previous pure neural AG News averaged
  checkpoint.
- `himdex_hybrid_ag_news_search_v1.joblib`: current strongest packaged AG News
  hybrid classifier found by `himdex-search-hybrid`.
- `himdex_hybrid_ag_news_base_v3_v2.joblib`: current packaged AG News hybrid
  classifier v2, tuned with `embedding_weight=1.25` and `classifier_c=1.0`.
- `himdex_hybrid_ag_news_base_v3.joblib`: packaged AG News hybrid classifier
  v1 that combines word/character TF-IDF features with Himdex Base v3
  embeddings.
- `himdex_text_base_v2.pt`: previous 11.24M-parameter reference checkpoint.
- `himdex_text_compact.pt`: 128-hidden-size checkpoint for lightweight tests.

Large checkpoints should be published with GitHub Releases, Hugging Face Hub, or
Git LFS instead of being committed directly to the repository.
