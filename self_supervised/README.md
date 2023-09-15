# Self-supervised experiments of SimPool

## Pre-trained models
You can download checkpoints, logs and configs for all self-supervised models, both official reproductions and SimPool.

<table>
  <tr>
    <th>Architecture</th>
    <th>Mode</th>
    <th>Gamma</th>
    <th>Epochs</th>
    <th>k-NN</th>
    <th>Linear Probing</th>
    <th colspan="3">download</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>Official</td>
    <td>-</td>
    <td>100</td>
    <td>68.9</td>
    <td>71.5</td>
    <td><a href="https://huggingface.co/billpsomas/vits_dino_official_ep100/resolve/main/checkpoint.pth">checkpoint</a></td>
    <td><a href="https://huggingface.co/billpsomas/vits_dino_official_ep100/resolve/main/log.txt">logs</a></td>
    <td><a href="https://huggingface.co/billpsomas/vits_dino_official_ep100/resolve/main/configs.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>SimPool</td>
    <td>1.25</td>
    <td>100</td>
    <td>69.7</td>
    <td>72.8</td>
    <td><a href="https://huggingface.co/billpsomas/vits_dino_simpool_ep100/resolve/main/checkpoint.pth">checkpoint</a></td>
    <td><a href="https://huggingface.co/billpsomas/vits_dino_simpool_ep100/resolve/main/log.txt">logs</a></td>
    <td><a href="https://huggingface.co/billpsomas/vits_dino_simpool_ep100/resolve/main/configs.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>SimPool</td>
    <td>1.25</td>
    <td>300</td>
    <td>72.6</td>
    <td>75.0</td>
    <td><a href="https://huggingface.co/billpsomas/vits_dino_simpool_ep300/resolve/main/checkpoint.pth">checkpoint</a></td>
    <td><a href="https://huggingface.co/billpsomas/vits_dino_simpool_ep300/resolve/main/log.txt">logs</a></td>
    <td><a href="https://huggingface.co/billpsomas/vits_dino_simpool_ep300/resolve/main/configs.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>SimPool</td>
    <td>-</td>
    <td>100</td>
    <td>69.8</td>
    <td>72.6</td>
    <td><a href="https://huggingface.co/billpsomas/vits_dino_simpool_no_gamma_ep100/resolve/main/checkpoint.pth">checkpoint</a></td>
    <td><a href="https://huggingface.co/billpsomas/vits_dino_simpool_no_gamma_ep100/resolve/main/log.txt">logs</a></td>
    <td><a href="https://huggingface.co/billpsomas/vits_dino_simpool_no_gamma_ep100/resolve/main/configs.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>Official</td>
    <td>-</td>
    <td>100</td>
    <td>61.8</td>
    <td>63.8</td>
    <td><a href="https://huggingface.co/billpsomas/resnet50_dino_official_ep100/resolve/main/checkpoint.pth">checkpoint</a></td>
    <td><a href="https://huggingface.co/billpsomas/resnet50_dino_official_ep100/resolve/main/log.txt">logs</a></td>
    <td><a href="https://huggingface.co/billpsomas/resnet50_dino_official_ep100/resolve/main/configs.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>SimPool</td>
    <td>2.0</td>
    <td>100</td>
    <td>63.8</td>
    <td>64.4</td>
    <td><a href="https://huggingface.co/billpsomas/resnet50_dino_simpool_ep100/resolve/main/checkpoint.pth">checkpoint</a></td>
    <td><a href="https://huggingface.co/billpsomas/resnet50_dino_simpool_ep100/resolve/main/log.txt">logs</a></td>
    <td><a href="https://huggingface.co/billpsomas/resnet50_dino_simpool_ep100/resolve/main/configs.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>SimPool</td>
    <td>-</td>
    <td>100</td>
    <td>63.7</td>
    <td>64.2</td>
    <td><a href="https://huggingface.co/billpsomas/resnet50_dino_simpool_no_gamma_ep100/resolve/main/checkpoint.pth">checkpoint</a></td>
    <td><a href="https://huggingface.co/billpsomas/resnet50_dino_simpool_no_gamma_ep100/resolve/main/log.txt">logs</a></td>
    <td><a href="https://huggingface.co/billpsomas/resnet50_dino_simpool_no_gamma_ep100/resolve/main/configs.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ConvNeXt-S</td>
    <td>Official</td>
    <td>-</td>
    <td>100</td>
    <td>59.3</td>
    <td>63.9</td>
    <td><a href="https://huggingface.co/billpsomas/convnext_small_dino_official_ep100/resolve/main/checkpoint.pth">checkpoint</a></td>
    <td><a href="https://huggingface.co/billpsomas/convnext_small_dino_official_ep100/resolve/main/log.txt">logs</a></td>
    <td><a href="https://huggingface.co/billpsomas/convnext_small_dino_official_ep100/resolve/main/configs.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ConvNeXt-S</td>
    <td>SimPool</td>
    <td>2.0</td>
    <td>100</td>
    <td>68.7</td>
    <td>72.2</td>
    <td><a href="https://huggingface.co/billpsomas/convnext_small_dino_simpool_ep100/resolve/main/checkpoint.pth">checkpoint</a></td>
    <td><a href="https://huggingface.co/billpsomas/convnext_small_dino_simpool_ep100/resolve/main/log.txt">logs</a></td>
    <td><a href="https://huggingface.co/billpsomas/convnext_small_dino_simpool_ep100/resolve/main/configs.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ConvNeXt-S</td>
    <td>SimPool</td>
    <td>-</td>
    <td>100</td>
    <td>68.8</td>
    <td>72.2</td>
    <td><a href="https://huggingface.co/billpsomas/convnext_small_dino_simpool_no_gamma_ep100/resolve/main/checkpoint.pth">checkpoint</a></td>
    <td><a href="https://huggingface.co/billpsomas/convnext_small_dino_simpool_no_gamma_ep100/resolve/main/log.txt">logs</a></td>
    <td><a href="https://huggingface.co/billpsomas/convnext_small_dino_simpool_no_gamma_ep100/resolve/main/configs.yaml">configs</a></td>
  </tr>
</table>

## Training
Having created the self-supervised environment and downloaded the ImageNet dataset, you are now ready to train! We pre-train ResNet-50, ConvNeXt-S and ViT-S with [DINO](https://github.com/facebookresearch/dino).

### ResNet-50

Train ResNet-50 with SimPool on ImageNet-1k for 100 epochs:
<!---
data_path = /mnt/data/imagenet/
output_dir = /mnt/datalv/bill/logs/
-->

```bash
python3 -m torch.distributed.launch --nproc_per_node=16 main_dino.py --arch resnet50 --mode simpool \
--data_path /path/to/imagenet/ --output_dir /path/to/output/ --subset -1 --num_workers 10 --batch_size_per_gpu 90 \
--out_dim 60000 --use_bn_in_head True --teacher_temp 0.07 --warmup_teacher_temp_epochs 50 --use_fp16 False \
--weight_decay 0.000001 --weight_decay_end 0.000001 --clip_grad 0.0 --epochs 100 --lr 0.3 --min_lr 0.0048 \
--optimizer lars --global_crops_scale 0.14 1.0 --local_crops_number 6 --local_crops_scale 0.05 0.14
```

> For ResNet-50 official adjust `--mode official`. For no $\gamma$ adjust `--gamma None`. :exclamation: 
> NOTE: Here we use 16 GPUs x 90 batch size per GPU = 1280 global batch size.

Linear probing of ResNet-50 with SimPool on ImageNet-1k for 100 epochs:
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 eval_linear.py --batch_size_per_gpu 256 --arch resnet50 --mode simpool \
--pretrained_weights /path/to/checkpoint/ --data_path /path/to/imagenet/ --output_dir /path/to/output/ --epochs 100
```

> For ResNet-50 official adjust `--mode official`. :exclamation: NOTE: Here we use 4 GPUs x 256 batch size per GPU = 1028 global batch size.

## ConvNeXt-S

Train ConvNeXt-S with SimPool on ImageNet-1k for 100 epochs:

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch convnext_small --mode simpool \
--data_path /path/to/imagenet/ --output_dir /path/to/output/ --subset -1 --num_workers 10 --batch_size_per_gpu 60 \
--out_dim 65536 --use_bn_in_head False --weight_decay 0.04 --weight_decay_end 0.4 --clip_grad 0.3 --epochs 100 \
--min_lr 2e-6 --optimizer adamw --lr 0.001 --freeze_last_layer 3
```

> For ConvNeXt-S official adjust `--mode official`. For no $\gamma$ adjust `--gamma None`. :exclamation: 
> NOTE: Here we use 8 GPUs x 60 batch size per GPU = 480 global batch size.

Linear probing of ConvNeXt-S with SimPool on ImageNet-1k for 100 epochs:

```bash
 python3 -m torch.distributed.launch --nproc_per_node=4 eval_linear.py --batch_size_per_gpu 256 --n_last_blocks 1 \
--arch convnext_small --mode simpool --pretrained_weights /path/to/checkpoint/ --data_path /path/to/imagenet/ \
--output_dir /path/to/output/ --epochs 100
```

> For ConvNeXt-S official adjust `--mode official`. :exclamation: NOTE: Here we use 4 GPUs x 256 batch size per GPU = 1028 global batch size.

## ViT-S 

Train ViT-S with SimPool on ImageNet-1k for 100 epochs:

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch vit_small --mode simpool --gamma 1.25 \
--data_path /path/to/imagenet/ --output_dir /path/to/output/ --optimizer adamw --use_bn_in_head False --out_dim 65536 \
--subset -1 --batch_size_per_gpu 100 --local_crops_number 6 --epochs 100 --num_workers 10 --lr 0.0005 --min_lr 0.00001 \
--global_crops_scale 0.25 1.0 --local_crops_scale 0.05 0.25 --norm_last_layer False --warmup_teacher_temp_epochs 30 \
--weight_decay 0.04 --weight_decay_end 0.4
```

> For ViT-S  official adjust `--mode official`. For no $\gamma$ adjust `--gamma None`. :exclamation: 
> NOTE: Here we use 8 GPUs x 100 batch size per GPU = 800 global batch size.

Linear probing of ViT-S with SimPool on ImageNet-1k for 100 epochs:

```bash
 python3 -m torch.distributed.launch --nproc_per_node=4 eval_linear.py --batch_size_per_gpu 256 --n_last_blocks 1 \
--arch vit_small --mode simpool --pretrained_weights /path/to/checkpoint/ --data_path /path/to/imagenet/ \
--output_dir /path/to/output/ --epochs 100
```