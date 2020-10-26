CORES=4
NLIVE=50
CASE=2
DLOGZ=0.1
SAMPLES=10
SENSORS=2
NY=10

python3 nested.py --nlive $NLIVE --case $CASE --dlogz $DLOGZ --cores $CORES
mpirun -n $CORES ./samples.py --case $CASE --samples $SAMPLES
mpirun -n $CORES ./run-sequential.py --nSurveys $SENSORS --path "case$CASE" --Ny $NY --Ntheta $SAMPLES --case $CASE

