CORES=36
NLIVE=100
CASE=3
DLOGZ=0.1
SAMPLES=120
SENSORS=2
NY=120

python3 nested.py --nlive $NLIVE --case $CASE --dlogz $DLOGZ --cores $CORES
mpirun -n $CORES ./samples.py --case $CASE --samples $SAMPLES
mpirun -n $CORES ./run-sequential.py --nSurveys $SENSORS --path "case$CASE" --Ny $NY --Ntheta $SAMPLES --case $CASE

