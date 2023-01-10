#!/bin/bash
#SBATCH -J gn001_FovSOS_FSD
#SBATCH -p gpus
#SBATCH -n 1
#SBATCH -w gn001
#SBATCH --mem-per-cpu=32000
#SBATCH --mail-user=camilo.jara@ug.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --gres=gpu:1

ml fosscuda/2019b
ml CUDA/11.4.0
ml cuDNN/8.2.4.15

srun python -u test_against_others_nlhpc.py

