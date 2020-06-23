#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --ntasks=100
#SBATCH --time=00:40:00
#SBATCH --job-name="tensor"
#SBATCH --output=tensor.%j.o
#SBATCH --error=tensor.%j.e
srun -u -n 100 ./tensor.py --samples 100 --days 30
