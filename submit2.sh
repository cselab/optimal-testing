#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --nodes=20
#SBATCH --ntasks=240
#SBATCH --ntasks-per-node=12
#SBATCH --time=10:00:00
#SBATCH --job-name="case2"
#SBATCH --output=case2.%j.o
#SBATCH --error=case2.%j.e
srun -u -n 240 ./optimization_shared_canton_sequential.py --nSensors 4 --nMeasure 1 --path './samples_case2' --nProcs 240 --Ny 1500 --Ntheta 1500 --start_day 21
mv result* ./samples_case2
echo "ALL OK 2"
