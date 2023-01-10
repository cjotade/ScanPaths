#!/bin/bash
#SBATCH -J process_features
#SBATCH -p slims
#SBATCH -n 1
#SBATCH --mem-per-cpu=2400
#SBATCH --mail-user=camilo.jara@ug.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -o process_features%j_%x.out
#SBATCH -e process_features%j_%x.err

ml fosscuda/2019b
ml CUDA/11.4.0
ml cuDNN/8.2.4.15

srun python process_features_nlhpc.py
