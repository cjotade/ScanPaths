#!/bin/bash
#SBATCH -J debug_train_test_FovSOS_FS
#SBATCH -p debug
#SBATCH -n 1
#SBATCH --mem-per-cpu=4800
#SBATCH --mail-user=camilo.jara@ug.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

ml fosscuda/2019b
ml CUDA/11.4.0
ml cuDNN/8.2.4.15

srun python -u train_test_multiple_times_nlhpc.py
