#!/bin/bash
#SBATCH --job-name=SelfA_ResNet
#SBATCH --account=hz18
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=m3f

problem=Emotiv266
model=SelfA_ResNet

cd ..
python3 compare_models.py -p Emotiv266 -c $model -i 5