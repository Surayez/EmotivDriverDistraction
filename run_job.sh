#!/bin/bash
#SBATCH --job-name=Emotiv266_ResNet
#SBATCH --account=hz18
#SBATCH --time=01:00:00
#SBATCH --gres-gpu:1

problem=Emotiv266
classifier='resnet_lstm'
epoch=1

python3 compare_models.py -p $problem -e $epoch -c $classifier