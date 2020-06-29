bsub -n 360 -W 24:00 mpirun -n 360 ./run-sequential.py --Ntheta 800 --Ny 800 --case 1 --path "nested-sampling/case1" --nSensors 6
bsub -n 360 -W 24:00 mpirun -n 360 ./run-sequential.py --Ntheta 800 --Ny 800 --case 2 --path "nested-sampling/case2" --nSensors 6
