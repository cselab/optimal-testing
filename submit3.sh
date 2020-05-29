#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --nodes=20
#SBATCH --ntasks=240
#SBATCH --ntasks-per-node=12
#SBATCH --time=06:00:00
#SBATCH --job-name="case3_2"
#SBATCH --output=case3_2.%j.o
#SBATCH --error=case3_2.%j.e
srun -u -n 240 ./optimization_shared_canton_sequential.py --nSensors 4 --nMeasure 1 --path './samples_case3_2' --nProcs 240 --Ny 1500 --Ntheta 1500 --start_day 84
mv result* ./samples_case3_2
echo "ALL OK 3"
