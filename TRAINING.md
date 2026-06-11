# Himdex Training Notes

## Masked Text Pretraining

The current reference checkpoint uses masked byte prediction on mixed text
prepared by `himdex-prepare-data`.

Latest continuation run:

```text
source checkpoint: checkpoints/himdex_text_base.pt
steps: 200
batch size: 8
sequence length: 128
hidden size: 128
layers: 2
heads: 4
loss: 3.3251
published checkpoint: checkpoints/himdex_text_base.pt
```

Recommended local command:

```powershell
himdex-pretrain-text --data data\himdex_pack\prepared --epochs 1 --batch-size 8 --hidden-size 128 --num-layers 2 --num-heads 4 --max-text-length 128 --max-steps 250 --output-dir runs\himdex_text_base
```

Continue from a checkpoint:

```powershell
himdex-pretrain-text --data data\himdex_pack\prepared --resume-from checkpoints\himdex_text_base.pt --epochs 1 --batch-size 8 --max-steps 200 --output-dir runs\himdex_text_base_v2
```

For open-source releases, keep raw datasets out of git. Commit code, configs,
docs, and compact reference checkpoints only.
