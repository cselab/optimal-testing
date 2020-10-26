# Optimal Allocation of Limited Test Resources for the Quantification of COVID-19 Infections

This repository contains the code to compute optimal testing allocations to identify asymptomatic infections for COVID-19 in Switzerland.
The employed epidemiological model is coded in C++ and uses python bindings. It relies on:

1. **Cmake**: minimum required version 3.12 
2. **boost**: 

In the following, we provide a detailed explanation on how to clone the directory, install the code and use it.

## Clone the directory

git clone --recurse-submodules https://github.com/cselab/optimal-testing

## Installation

First, install Cmake https://wwwcmake.org/ and boost https://www.boost.org/ .
Then compile the epidemiological model as follows:

1. `cd optimal-testing/covid19`
2. `mkdir -p build`
3. `cd build`
4. `cmake ..`
5. `make`

Second, install the dynesty library by using pip3:

1. `pip3 install dynesty`


## Run the Optimal Testing with nested sampling
The optimal testing allocation is run via the script launch.sh. The script takes the following arguments

1. CORES   : how many cores will be used
2. CASE    : which case to run (1,2,3 or 4)
3. SENSORS : how many surveys to allocate (one survey corresponds to testing people in one canton)
4. SAMPLES : how many model parameter samples will be used for the Monte-Carlo approximation of the integral in the utility function
5. NY      : how many measurement samples will be used for the Monte-Carlo approximation of the integral in the utility function
6. NLIVE   : parameter used for nested-sampling for cases 2,3,4 (use 50 for quick results or around 500-1000 for more accurate sampling)
7. DLOGZ   : parameter used as termination criterion for nested-sampling (use 0.1 or smaller number)

This script will sample the model parameters first. The samples will be drawn uniformly for case 1 and with nested-sampling for the other cases.
Then, the epidemiological model will be evaluated using those samples.
Finally, the sequential optimization will be applied, to find the optimal test allocation.

## Plot the results (part 1/2, sampling)
1. The posterior distributions the arise from nested-sampling are plotted by running 
   ` python3 plot_nested_sampling_results --case X`
   where X=2,3,4. The plots are saved as .pdf files, in the directory caseX.

## Plot the results (part 2/2, test allocation)
1. Go to the figures directory
3. To plot the illustration of the different cases, run `python3 plot-scenarios.py`.
2. Run the plotting script `python3 plot-ots.py --case Y` where Y=1,2,3. This will create the plots for the different cases as shown in the publication. The figures are saved in figures/caseY.
3. To create the tables from the Supplementary Information, run `python3 print-tables.py --case Y` for Y=1,2,3. The generated screen output can be copy pasted to generate the tables in latex.
4. The comparison of the effectiveness of the different strategies can be done `python3 plot-effectiveness.py`. The figures are saved in figures/effectiveness.


## References
1. Data [openZH database](https://raw.githubusercontent.com/daenuprobst/covid19-cases-switzerland/master/covid19_cases_switzerland_openzh.csv). For the inference on the swiss data we do not use the NaN values.
2. Nested sampling with the dynesty library [Dynesty](https://dynesty.readthedocs.io/en/latest/)
3. Korali library [Korali](https://github.com/cselab/korali)
