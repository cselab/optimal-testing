bsub -n 360 -W 24:00 -R "select[model==XeonGold_6150]fullnode" mpirun -n 360 ./run-sequential.py --Ntheta 800 --Ny 800 --case 3 --path "nested-sampling/case3" --nSensors 4
