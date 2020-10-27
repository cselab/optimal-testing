CORES=22
NLIVE=100
CASE=2
DLOGZ=0.1
SENSORS=2
NY=110
SAMPLES=110
CORES_SAMPLES=10

python3 nested.py --nlive $NLIVE --case $CASE --dlogz $DLOGZ --cores $CORES
mpirun -n $CORES_SAMPLES ./samples.py --case $CASE --samples $SAMPLES
mpirun -n $CORES ./run-sequential.py --nSurveys $SENSORS --path "case$CASE" --Ny $NY --Ntheta $SAMPLES --case $CASE

