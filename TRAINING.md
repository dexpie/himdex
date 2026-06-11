# Himdex Training Notes

## Starter Text Pretraining

The current starter run uses masked byte prediction on mixed text prepared by
`himdex-prepare-data`.

Latest local continuation run:

```text
resume checkpoint: runs/himdex_text_starter/himdex_text_pretrain.pt
steps: 200
batch size: 8
sequence length: 128
hidden size: 128
layers: 2
heads: 4
loss: 3.3251
published demo checkpoint: checkpoints/himdex_text_starter.pt
```

Recommended local command:

```powershell
himdex-pretrain-text --data data\himdex_starter\prepared --epochs 1 --batch-size 8 --hidden-size 128 --num-layers 2 --num-heads 4 --max-text-length 128 --max-steps 250 --output-dir runs\himdex_text_starter
```

To continue from a previous checkpoint:

```powershell
himdex-pretrain-text --data data\himdex_starter\prepared --resume-from runs\himdex_text_starter\himdex_text_pretrain.pt --epochs 1 --batch-size 8 --max-steps 200 --output-dir runs\himdex_text_starter_v2
```

For open-source releases, keep raw datasets out of git. Commit code, configs,
docs, and small demo checkpoints only.
