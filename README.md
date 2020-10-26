# Optimal Allocation of Limited Test Resources for the Quantification of COVID-19 Infections

This repository contains the code to compute optimal testing allocations to identify asymptomatic infections for COVID-19 in Switzerland.

It relies on two libraries:

1. **korali**:  provides the algorithm to compute the optimal testing strategy.
2. **dynesty**: nested sampling algorithm, to sample the epidemiological model and compute the Monte-Carlo integral for the expected utility

Additionally, the employed epidemiological model is coded in C++ and uses python bindings. It relies on:

1. **Cmake**: minimum required version 3.12 
2. **boost**: 

In the following, we provide a detailed explanation on how to clone the directory, install the code and use it.


## Clone the directory

`git clone --recurse submodules https://github.com/cselab/optimal-testing`


## Installation

First, install Cmake https://wwwcmake.org/ and boost https://www.boost.org/ .
Then compile the epidemiological model as follows:

1. `cd optimal-testing/covid19`
2. `mkdir -p build`
3. `cd build`
4. `cmake ..`
5. `make`

Second, install the Korali library [Korali's homepage](https://www.cse-lab.ethz.ch/korali/):

1. `git clone git@github.com:cselab/korali.git`
2. `cd korali`
3. `./install`

Third, install the dynesty library by using pip3:

1. `pip3 install dynesty`


## Run the Optimal Testing with nested sampling
1. Go the the nested-sampling directory. If you do not want to use any available data, go to step 5
   `cd nested-sampling`
2. Download data of reported infected people
   `python3 get-data.py`
3. Run the sampling. Here you need to specify whether you assume a uniform prior (Scenario I), an informed prior based on data before the first outbreak (Scenario II, case=2) or all data available (Scenario III,case=3)
   `python3 nested.py` --case 2 --cores 12
4. Nested-sampling will take a while.
5. Evaluate the model at the samples you took from steps 2,3,4 or at uniformly distributed samples
   `mpirun -n 16 ./samples.py --case X --samples 100`
   This will produce 100x100 samples. They will be stored in directory caseX, X=1,2,3 in multiple files.
6. Go back to the root directory.
   `cd ..`
7. Run the optimal test allocation
   `mpirun -n 128 ./run-sequential.py --nSensors 3 --path './nested-sampling/caseX' --Ny 100 --Ntheta 100 --case X`
    where X=1,2,3 

## Plot the results (part 1/2, nested sampling)
1. Go to the nested-sampling directory and run
   `python3 plot.py --case Y`
   where Y=2 or Y=3. This will produce the one-dimensional marginalized posteriors for the model and nuisance parameters as well as the fits for the reported cases in each canton.

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
