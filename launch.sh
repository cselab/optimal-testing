CORES=4
NLIVE=50
CASE=3
DLOGZ=0.1
SENSORS=4
NY=20
SAMPLES=20
CORES_SAMPLES=4

python3 nested.py --nlive $NLIVE --case $CASE --dlogz $DLOGZ --cores $CORES
mpirun -n $CORES_SAMPLES ./samples.py --case $CASE --samples $SAMPLES
mpirun -n $CORES ./run-sequential.py --nSurveys $SENSORS --path "case$CASE" --Ny $NY --Ntheta $SAMPLES --case $CASE
