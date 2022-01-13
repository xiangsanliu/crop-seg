#!/bin/bash
#SBATCH --job-name=crop-seg
#SBATCH --time=0
#SBATCH --output=outputs/seg.out.%j
#SBATCH --error=errors/seg.err.%j
#SBATCH --partition=gpu
### 该作业需要1个节点
#SBATCH --nodes=1
### 该作业需要8个CPU
#SBATCH --ntasks=8
### 申请1块GPU卡
#SBATCH --gres=gpu:1


python main_epoch.py --config configs/segformer_b4_tianchi2_label_no_overlap.py \
    --n_classes 5 \
    --total_epochs 100 \
    --device cuda