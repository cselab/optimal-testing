# Optimal Testing Strategies for Identification of Asymptomatic COVID-19 Infections

This repository contains the code to compute optimal testing strategies to identify asymptomatic Infection.

It relies on three libraries:

1. **covid19**: provides the epidemiological model
2. **korali**:  provides the algorithm to compute the optimal testing strategy.
3. **dynesty**: nested sampling algorithm

## Installation

In the following we provide instruction to install these three libraries, assuming that you are in the root folder of this repository.

### Install the COVID19 library

1. `git clone git@github.com:cselab/covid19.git`
2. `cd covid19`
3. `git submodule update --init --recursive`
4. `mkdir build`
5. `cd build`
6. `cmake ..`
7. `make -j`

After compiling the **covid19** you have to add the directories containing covid19 and the subdirectory build to your pythonpath by running `export PYTHONPATH=/path/to/covid19:/path/to/covid19/build:$PYTHONPATH`

In order to test whether this was successful run the command `python3 -c "import libepidemics"`. If there is no error you are ready to use **covid19** to simulate epidemiological models.`

### Install the KORALI library

1. `git clone git@github.com:cselab/korali.git`
2. `cd korali`
3. `./install`

In order to test whether this was successful run the command ``python3 -c "import korali"`. If there is no error you are ready to use **korali** to solve many kinds of Bayesian problems. To get more information visit [Korali's homepage](https://www.cse-lab.ethz.ch/korali/).

### Install the DYNESTY library

1. `pip3 install dynesty`


## Run the Optimal Testing with nested sampling
1. Go the the nested-sampling directory. If you do not want to use any available data, go to step 5
   `cd nested-sampling`
2. Download data of reported infected people
   `python3 get-data.py`
3. Run the sampling. Here you need to specify whether you assume an uniform prior (Scenario I), an informed prior based on data before the first outbreak (Scenario II, case=2) or all data available (Scenario III,case=3)
   `python3 nested.py` --case 2 --cores 12
4. Wait for an eternity, especially for case=3 (at least 36 hours, no matter how many cores you use)
5. Evaluate the model at the samples you took from steps 2,3,4 (uniform=0) or at uniformly distributed samples (uniform=1)
   `python3 get-samples.py --uniform 0 --days 120 --samples 500`
6. Go back to the root directory, or login again if you fell asleep when nested sampling was running.
   `cd ..`
7. Run the optimal sensor placement
   `srun -u -n 240 ./sensor_placement.py --nSensors 3 --nMeasure 1 --path './nested-sampling' --nProcs 240 --Ny 500 --Ntheta 500 --start_day 84`


## Run the Optimal Testing with korali sampling
1. Download data of reported infected people
   `python3 get-data.py`
2. Run the sampling. Here you need to specify whether you assume an uniform prior (Scenario I), an informed prior based on data before the first outbreak (Scenario II, case=2) or all data available (Scenario III,case=3)
   `python3 run-sampling.py uniform/social-distancing/second-outbreak`
3. Evaluate the model using the samples
   `python3 run-model.py`
4. Run the optimal sensor placement
   `python3 run-optimal-sensor-placement.py`
5. Plot the results 
  `python3 plot-results.py`

## References

Data [openZH database](https://raw.githubusercontent.com/daenuprobst/covid19-cases-switzerland/master/covid19_cases_switzerland_openzh.csv). For the inference on the swiss data we replace nan's with 0.
