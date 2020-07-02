# Optimal Testing Strategies for Identification of Asymptomatic COVID-19 Infections

This repository contains the code to compute optimal testing strategies to identify asymptomatic infections for COVID-19 in Switzerland.

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
4. Nested-sampling will take a while.
5. Evaluate the model at the samples you took from steps 2,3,4 or at uniformly distributed samples
   `mpirun -n 16 ./samples_get.py --case X --samples 100`
   This will produce 100x100 samples. They will be stored in directory caseX, X=1,2,3 in multiple files.
6. Go back to the root directory, or login again if you fell asleep when nested sampling was running.
   `cd ..`
7. Run the optimal sensor placement
   `mpirun -n 128 ./run-sequential.py --nSensors 3 --path './nested-sampling/caseX' --Ny 100 --Ntheta 100 --case X`
    where X=1,2,3 

## Plot the results (part 1/2, nested sampling)
1. Go to the nested-sampling directory and run
   `python3 plot.py --case Y`
   where Y=2 or Y=3. This will produce the one-dimensional marginalized posteriors for the model and nuisance parameters as well as the fits for the reported cases in each canton.

## Plot the results (part 2/2, sensor placement)
1. Will be added soon.

## References
Data [openZH database](https://raw.githubusercontent.com/daenuprobst/covid19-cases-switzerland/master/covid19_cases_switzerland_openzh.csv). For the inference on the swiss data we replace nan's with 0.
