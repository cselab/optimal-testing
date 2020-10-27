# Optimal Allocation of Limited Test Resources for the Quantification of COVID-19 Infections

This repository contains the code to compute optimal testing allocations to identify asymptomatic infections for COVID-19 in Switzerland.
The employed epidemiological model is coded in C++ and uses python bindings. It relies on:

1. **Cmake**: minimum required version 3.12 
2. **boost**: https://www.boost.org/

In the following, we provide a detailed explanation on how to clone the directory, install the code and use it.

## Clone the directory

**https**: git clone --recurse-submodules https://github.com/cselab/optimal-testing

**ssh**: git clone --recurse-submodules git@github.com:cselab/optimal-testing.git

## Installation

First, install [Cmake](https://www.cmake.org/) and [Boost](https://www.boost.org/).
Second, install the [Dynesty](https://dynesty.readthedocs.io/en/latest/) library by using pip3:

`pip3 install dynesty`

Finally, compile the epidemiological model as follows:

1. `cd optimal-testing/covid19`
2. `mkdir -p build`
3. `cd build`
4. `cmake ..`
5. `make`

## Run the Optimal Test Allocation
The optimal test allocation is run via the script *launch.sh*. The script contains the following arguments

1. CORES   : maximum number of available cores (MPI processes)
2. CASE    : which case to run (1,2,3 or 4)
3. SENSORS : how many surveys to allocate (one survey corresponds to testing people in one canton)
4. SAMPLES : how many model parameter samples will be used for the Monte-Carlo approximation of the integral in the utility function
5. NY      : how many measurement samples will be used for the Monte-Carlo approximation of the integral in the utility function
6. NLIVE   : parameter used for nested-sampling for cases 2,3,4 (use 50 for quick results or around 500-1000 for more accurate sampling)
7. DLOGZ   : parameter used as termination criterion for nested-sampling (use 0.1 or smaller number)
8. CORES_SAMPLES   : how many cores will be used to evaluate the model using the model parameter samples. Must be less than or equal to CORES and SAMPLES must be divisible by this number.

This script will sample the model parameters first. The samples will be drawn uniformly for case 1 and with nested-sampling for the other cases.
Note that nested-sampling for cases 3 and 4 takes a while.
Then, the epidemiological model is evaluated using those samples.
Finally, the sequential optimization will be performed, to find the optimal test allocation.

## Plot the results (part 1/2, sampling)
The posterior distributions the arise from nested-sampling are plotted by running 
   ` python3 plot-sampling.py --case X`
where X=2,3,4. The plots are saved as .pdf files, in the directory caseX. This directory contains:
1. cantons.pdf : shows the fitted model for all cantons (reported infections' data plotted against model output)
2. posteriorX.pdf : shows the one-dimensional marginal posterior distributions of the model parameters, after they are updated with data for the reported infections
3. result.npy : contains the utility function evaluations
4. map.npy : contains the maximum a posteriori estimates of the model parameters, after the inference is completed.
5. samples_X.pickle : contains the samples of the model parameters after the inference
6. prediction_country.pdf : model prediction plotted against data for total reported cases in Switzerland
7. day=Y.npy : epidemiological model evaluation for all cantons on day Y
8. dispersion.npy : error model dispersion

## Plot the results (part 2/2, test allocation)
1. Run the plotting script `python3 plot-ots.py --case Y` where Y=1,2,3. This will create the plots for the different cases as shown in the publication. The figures are saved in figures/caseY. Plotting case Y=3 assumes that the optimal testing was run for case 3 and case 4.
2. The tables from the Supplementary Information are created using `python3 print-tables.py --case Y` for Y=1,2,3. The generated screen output can be copy pasted to generate the tables in latex.

## Compare optimal strategy with non-specific strategy
To do this comparison you must first:
1. Run the optimal testing for case 1, so that the utility function is computed and the optimal strategy is defined.
2. Run the optimal testing for case 2 and the *plot-sampling.py* script afterwards, so that the maximum a posteriori estimates for the model parameters are used to generate artificial survey measurements.

Then, in the directory 'comparison':
1. Execute `python3 comparison.py --surveys 2`
2. Plot the results by using `python3 plot.py`

## References
1. Data [openZH database](https://raw.githubusercontent.com/daenuprobst/covid19-cases-switzerland/master/covid19_cases_switzerland_openzh.csv). For the inference on the swiss data we do not use the NaN values.

## Brief description of available scripts
1. common.py : contains the dates used for test allocation for the four cases.
2. nested.py : does the nested-sampling. The prior distributions for the model parameters used for each case can be found here.
3. plot-ots.py: Plot the resulting optimal testing strategies
4. plot-sampling.py: plots the results from nested sampling
5. print-tables.py: prints the tables to be used in latex
6. run-sequential.py: performs the evaluation of the utility function following sequential optimisation
7. samples.py: evaluates the model at model parameter samples
8. seiin.py  : wrapper for the employed epidemiological model's C++ code
9. swiss_cantons.py : will download data used for cases 2,3,4 as well as cantons connections matrix for epidemiological model
