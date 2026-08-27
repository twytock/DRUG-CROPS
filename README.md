
## Repository Description
Source code for generating the figures in the paper "Repurposing Drug Combinations for the Treatment of Brain Cancers."

## System Requirements and Installation
The python code has been run with python 3.10 with the packages indicated in the `drug-crops.yml` file found at the top level of this repository. 

The required version of python and the dependent packages can be downloaded here: https://www.anaconda.com/distribution/. 
Create a new anaconda environment and install from the default channels with `conda env create -f drug-crops.yml`.
Activate the environment with `conda activate drug-crops`

## Repository Organization
This repository includes a jupyter notebook for reproducing the experimental analysis:

1. Documentation for [analyzing the cell viability data (Experimental Results)](https://github.com/twytock/DRUG-CROPS/blob/main/code/Experimental_Results/README.md).

2. The source code is stored in the [code](https://github.com/twytock/DRUG-CROPS/tree/main/code/Experimental_Results) folder.

3. The required input data is stored in the [data](https://github.com/twytock/DRUG-CROPS/tree/main/data/Experimental_Results) folder.

4. The outputs generated from running the code are stored in the [output](https://github.com/twytock/DRUG-CROPS/tree/main/output/Experimental_Results) folder.