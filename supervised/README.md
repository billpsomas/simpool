# Supervised experiments of SimPool

## Pre-trained models
You can download checkpoints, logs and configs for all supervised models, both official reproductions and SimPool.

:warning: **UNDER CONSTRUCTION** :warning:

<table>
  <tr>
    <th>Architecture</th>
    <th>Mode</th>
    <th>Gamma</th>
    <th>Epochs</th>
    <th>Accuracy</th>
    <th colspan="3">download</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>Official</td>
    <td>-</td>
    <td>100</td>
    <td>72.7</td>
    <td><a href="https://huggingface.co/billpsomas/vits_supervised_official_ep100/blob/main/vits_supervised_official_ep100.pth.tar">checkpoint</a></td>
    <td><a href="https://huggingface.co/billpsomas/vits_supervised_official_ep100/blob/main/summary.csv">logs</a></td>
    <td><a href="https://huggingface.co/billpsomas/vits_supervised_official_ep100/blob/main/args.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>SimPool</td>
    <td>-</td>
    <td>100</td>
    <td>74.3</td>
    <td><a href="https://huggingface.co/billpsomas/vits_supervised_simpool_no_gamma_ep100/blob/main/vits_supervised_simpool_no_gamma_ep100.pth.tar">checkpoint</a></td>
    <td><a href="https://huggingface.co/billpsomas/vits_supervised_simpool_no_gamma_ep100/blob/main/summary.csv">logs</a></td>
    <td><a href="https://huggingface.co/billpsomas/vits_supervised_simpool_no_gamma_ep100/blob/main/args.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>SimPool</td>
    <td>1.25</td>
    <td>100</td>
    <td>74.2</td>
    <td><a href="https://huggingface.co/billpsomas/vits_supervised_simpool_ep100/blob/main/vits_gem1.25_supervised_beta.pth.tar">checkpoint</a></td>
    <td><a href="https://huggingface.co/billpsomas/vits_supervised_simpool_ep100/blob/main/summary.csv">logs</a></td>
    <td><a href="https://huggingface.co/billpsomas/vits_supervised_simpool_ep100/blob/main/args.yaml">configs</a></td>
  </tr>
</table>

## Training
Having created the supervised environment and downloaded the ImageNet dataset, you are now ready to train! For our main experiments, we train ViT-S, ResNet-50 and ConvNeXt-S.

### ViT-S 

Train ViT-S with SimPool on ImageNet-1k for 100 epochs:

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 train.py --model vit_small_patch16_224 --gp simpool --gamma 1.25 \
--data-dir /path/to/imagenet/ --output /path/to/output/ --experiment vits_supervised_simpool --batch-size 74  --sched cosine \ 
--epochs 100 --subset -1 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .2 --model-ema --model-ema-decay 0.99996 \
--aa rand-m9-mstd0.5-inc1 --remode pixel --reprob 0.25 --lr 5e-4 --weight-decay .05 --drop 0.1 --drop-path .1 
```

> For ViT-S official ([CLS]) adjust `--gp token`. For ViT-S with GAP adjust `--gp avg`. For no $\gamma$ adjust `--gamma None`. :exclamation: 
> NOTE: Here we use 8 GPUs x 74 batch size per GPU = 592 global batch size.

### ResNet-50

Train ResNet-50 with SimPool on ImageNet-1k for 100 epochs:

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 train.py --model resnet50 --gp simpool --gamma 2.0 \
--data-dir /path/to/imagenet/ --output /path/to/output/ --experiment resnet50_supervised_simpool --batch-size 128 \ 
--epochs 100 --subset -1 --sched cosine --lr 0.4 --remode pixel --reprob 0.6 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 \
--resplit --split-bn --jsd --dist-bn reduce 
```

> For ResNet-50 official (GAP) adjust `--gp avg`. For no $\gamma$ adjust `--gamma None`. :exclamation: 
> NOTE: Here we use 8 GPUs x 128 batch size per GPU = 1024 global batch size.

### ConvNeXt-S

Train ConvNeXt-S with SimPool on ImageNet-1k for 100 epochs:

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 train.py --model convnext_small --gp simpool --gamma 2.0 \
--data-dir /path/to/imagenet/ --output /path/to/output/ --experiment convnexts_supervised_simpool --batch-size 128 \
--sched cosine --epochs 100 --subset -1 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema \
--model-ema-decay 0.9999 --aa rand-m9-mstd0.5-inc1 --remode pixel --reprob 0.25 --lr 1e-3 --weight-decay .05 --drop-path .4

```

> For ConvNeXt-S official (GAP) adjust `--gp avg`. For no $\gamma$ adjust `--gamma None`. :exclamation: 
> NOTE: Here we use 8 GPUs x 128 batch size per GPU = 1024 global batch size.

### Extra notes

- Use `--subset 260` to train on ImageNet-20\% dataset.
- When loading our weights using `--pretrained_weights`, take care of any inconsistencies in model keys!
- Default value of $\gamma$ is: 1.25 for transformers, 2.0 for convolutional networks. 
- In some cases, we observed that using no $\gamma$ facilitates the training, results in slightly better metrics, but also lowers the attention map quality.

## More training

:warning: **UNDER CONSTRUCTION** :warning: