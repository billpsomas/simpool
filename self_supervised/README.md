# Self-supervised experiments of SimPool

## ResNet-50 official

data_path = /mnt/data/imagenet/
output_dir = /mnt/datalv/bill/logs/

```
python3 -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch resnet50 --data_path /mnt/data/imagenet/ --output_dir /mnt/datalv/bill/logs/ --subset -1 --num_workers 10 --batch_size_per_gpu 2 --out_dim 60000 --use_bn_in_head True --teacher_temp 0.07 --warmup_teacher_temp_epochs 50 --use_fp16 False --weight_decay 0.000001 --weight_decay_end 0.000001 --clip_grad 0.0 --epochs 100 --lr 0.3 --min_lr 0.0048 --optimizer lars --global_crops_scale 0.14 1.0 --local_crops_number 6 --local_crops_scale 0.05 0.14
```

```
python3 -m torch.distributed.launch --nproc_per_node=16 main_dino.py --arch resnet50 --data_path /path/to/imagenet/ --output_dir /path/to/output/ --subset -1 --num_workers 10 --batch_size_per_gpu 90 --out_dim 60000 --use_bn_in_head True --teacher_temp 0.07 --warmup_teacher_temp_epochs 50 --use_fp16 False --weight_decay 0.000001 --weight_decay_end 0.000001 --clip_grad 0.0 --epochs 100 --lr 0.3 --min_lr 0.0048 --optimizer lars --global_crops_scale 0.14 1.0 --local_crops_number 6 --local_crops_scale 0.05 0.14
```

## ConvNeXt-S official

```
python3 -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch convnext_small --data_path /path/to/imagenet/ --output_dir /path/to/output/ --subset -1 --num_workers 10 --batch_size_per_gpu 60 --out_dim 65536 --use_bn_in_head False --weight_decay 0.04 --weight_decay_end 0.4 --clip_grad 0.3 --epochs 100 --min_lr 2e-6 --optimizer adamw --lr 0.001 --freeze_last_layer 3
```

## ViT-S official

```
python3 -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch vit_small --data_path /path/to/imagenet/ --output_dir /path/to/output/ --optimizer adamw --use_bn_in_head False --out_dim 65536 --subset -1 --batch_size_per_gpu 100 --local_crops_number 6 --epochs 100 --num_workers 10 --lr 0.0005 --min_lr 0.00001 --global_crops_scale 0.25 1.0 --local_crops_scale 0.05 0.25 --norm_last_layer False --warmup_teacher_temp_epochs 30 --weight_decay 0.04 --weight_decay_end 0.4
```

## ResNet-50 SimPool

```
python3 -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch resnet18 --mode simpool --data_path /mnt/data/imagenet/ --output_dir /mnt/datalv/bill/logs/temp/ --subset -1 --num_workers 10 --batch_size_per_gpu 2 --out_dim 60000 --use_bn_in_head True --teacher_temp 0.07 --warmup_teacher_temp_epochs 50 --use_fp16 False --weight_decay 0.000001 --weight_decay_end 0.000001 --clip_grad 0.0 --epochs 100 --lr 0.3 --min_lr 0.0048 --optimizer lars --global_crops_scale 0.14 1.0 --local_crops_number 6 --local_crops_scale 0.05 0.14
```

## ViT-S SimPool

```
python3 -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_small --mode simpool --data_path /mnt/data/imagenet/ --output_dir /mnt/datalv/bill/logs/temp/ --optimizer adamw --use_bn_in_head False --out_dim 65536 --subset -1 --batch_size_per_gpu 100 --local_crops_number 6 --epochs 100 --num_workers 10 --lr 0.0005 --min_lr 0.00001 --global_crops_scale 0.25 1.0 --local_crops_scale 0.05 0.25 --norm_last_layer False --warmup_teacher_temp_epochs 30 --weight_decay 0.04 --weight_decay_end 0.4
```

## ConvNeXt-S SimPool

```
python3 -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch convnext_small --mode official --data_path /mnt/data/imagenet/ --output_dir /mnt/datalv/bill/logs/temp/ --subset -1 --num_workers 10 --batch_size_per_gpu 2 --out_dim 65536 --use_bn_in_head False --weight_decay 0.04 --weight_decay_end 0.4 --clip_grad 0.3 --epochs 100 --min_lr 2e-6 --optimizer adamw --lr 0.001 --freeze_last_layer 3
```

