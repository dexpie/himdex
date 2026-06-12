# Himdex Checkpoints

Compact reference checkpoints can live here when they are useful for examples,
resume training, or downstream experiments.

- `himdex_text_base_v3.pt`: 11.24M-parameter primary reference checkpoint.
- `himdex_ag_news_pure_avg_v1.pt`: pure neural AG News classifier checkpoint
  produced by averaging the previous pure best with a distilled checkpoint.
- `himdex_hybrid_ag_news_base_v3.joblib`: packaged AG News hybrid classifier
  that combines word/character TF-IDF features with Himdex Base v3 embeddings.
- `himdex_text_base_v2.pt`: previous 11.24M-parameter reference checkpoint.
- `himdex_text_compact.pt`: 128-hidden-size checkpoint for lightweight tests.

Large checkpoints should be published with GitHub Releases, Hugging Face Hub, or
Git LFS instead of being committed directly to the repository.
