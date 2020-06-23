#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --job-name="nested_case2"
#SBATCH --output=nested_case2.%j.o
#SBATCH --error=nested_case2.%j.e
python3 nested.py --cores 24 --nlive 300 --dlogz 0.1 --case 2
