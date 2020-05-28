# Optimal Testing Strategies for Identification of Asymptomatic COVID-19 Infections

This repository contains the code to compute optimal testing strategies to identify asymptomatic Infection.

It relies on two libraries, **covid19**, providing the epidemiological models and **korali**, which provides the sampling algorithm to incorporate data to the prior used to determine the optimal testing strategy, as well as the algorithm to compute the optimal testing strategy.

## Installation

In the following we provide instruction to install these two libraries, assuming that you are in the root folder of this repository.

### Install the COVID19 library

1. `git clone git@github.com:cselab/covid19.git`
2. `cd covid19`
3. `git submodule update --init --recursive`
3. `git checkout OSP`
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

## Run the Optimal Testing

1. Download data of reported infected people
   `python3 get-data.py`
2. Run the sampling. Here you need to specify whether you assume an uniform prior (Scenario I), an informed prior based on data before the first outbreak (Scenario II) or all data available (Scenario III)
   `python3 run-sampling.py uniform/social-distancing/second-outbreak`
3. Evaluate the model using the samples
   `python3 run-model.py`
4. Run the optimal sensor placement
   `python3 run-optimal-sensor-placement.py`
5. Plot the results 
  `python3 plot-results.py`

## References

Data [openZH database](https://raw.githubusercontent.com/daenuprobst/covid19-cases-switzerland/master/covid19_cases_switzerland_openzh.csv). For the inference on the swiss data we replace nan's with 0.
