CORES=1
NLIVE=50
CASE=4
DLOGZ=10000
SENSORS=4
NY=10
SAMPLES=10
CORES_SAMPLES=1

# python3 nested.py --nlive $NLIVE --case $CASE --dlogz $DLOGZ --cores $CORES
# mpirun -n $CORES_SAMPLES ./samples.py --case $CASE --samples $SAMPLES
mpirun -n $CORES ./run-sequential.py --nSurveys $SENSORS --path "case$CASE" --Ny $NY --Ntheta $SAMPLES --case $CASE

