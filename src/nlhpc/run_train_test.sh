#!/bin/bash
#SBATCH -J gn002_FovSOS_FS
#SBATCH -p gpus
#SBATCH -n 1
#SBATCH -w gn002
#SBATCH --mem-per-cpu=32000
#SBATCH --mail-user=camilo.jara@ug.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --gres=gpu:1

ml fosscuda/2019b
ml CUDA/11.4.0
ml cuDNN/8.2.4.15

srun python -u train_test_multiple_times_nlhpc.py

