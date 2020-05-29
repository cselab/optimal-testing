#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --nodes=20
#SBATCH --ntasks=240
#SBATCH --ntasks-per-node=12
#SBATCH --time=06:00:00
#SBATCH --job-name="case1"
#SBATCH --output=case1.%j.o
#SBATCH --error=case1.%j.e
srun -u -n 240 ./optimization_shared_canton_sequential.py --nSensors 4 --nMeasure 1 --path './samples_case1' --nProcs 240 --Ny 1500 --Ntheta 1500 --start_day -1
mv result* ./samples_case1
echo "ALL OK 1"
