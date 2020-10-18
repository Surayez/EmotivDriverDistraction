#!/bin/bash
#SBATCH --job-name=MHA_FCN
#SBATCH --account=hz18
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=m3f

problem=Emotiv266
model=MHA_FCN

cd ..
python3 compare_models.py -p Emotiv266 -c $model -i 5