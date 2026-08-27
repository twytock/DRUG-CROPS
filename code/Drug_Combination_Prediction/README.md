## Repository Description
Source code for performing the drug pair predictions in the paper "Repurposing Drug Combinations for the Treatment of Brain Cancers."

## System Requirements and Installation

### Drug combination prediction
The drug combination prediction algorithm uses CPLEX, which requires Python v. 3.6.
To perform the drug prediction algorithm, the packages indicated in the file `drug_comb_pred.yml` found in this directory must be installed using the command `conda env create -f drug_comb_pred.yml`. 
Before running the code, activate the environment with Activate the environment with `conda activate drug_comb_pred`.

## Drug Combination Prediction Documentation

The combination prediction is distributed among three scripts:

1. `process_perturbations.py` -- generates the training data for the nonadditive prediction method
2. `mrp_miqp_alt_obj_evk.py` -- performs the additive drug combination prediction
3. `fs_corrected_alt_obj_evk.py` -- performs the nonadditive drug combination prediction
4. `experimental_analysis.py` -- gathers the prediction results and averages them into a ranked list

### Running `process_perturbations.py`
Running the command `python code/Drug_Combination_Prediction/process_perturbations.py -h` will produce the following usage statement:

```
usage: process_perturbations.py [-h] [-V v]

Generate training data for the nonadditivity prediction algorithm from the
LINCS data.

optional arguments:
  -h, --help         show this help message and exit
  -V v, --version v  version of the expression files to use
```

The version for all the files is "\_1021" by default. 
The characters `<v>` are used to represent this version in filenames below.
Using a different file version will require renaming of the input files. 


#### Input file list
All input files are present in `./data/Drug_Combination_Prediction/`

1. `td_avg_expr_resp_proj<v>.pkl` -- time-course and dose averaged responses to perturbations, projected onto the eigengenes
2. `combined_data<v>.gctx` -- combined SRA & lincs gene expression data, projected onto the eigengenes
3. `lincs_cell_type_groups.pkl` -- pandas series with an index given by the cell line string and a value corresponding to the cell line  group
4. `cellinfo_beta.txt` -- a table of metadata regarding the LINCS cell lines
5. `LINCS_cell_lines_unique_v2.xlsx` -- excel table mapping the LINCS cell lines to their SRA counterparts.

#### Output file list
All output files are deposited in `./output/Drug_Combination_Prediction/`
1. `lincs_perturbation_info<v>.pkl` -- metadata for the perturbations, including initial and final genotype, treatment, cell line, and cell type
2. `lincs_indep_vars<v>.pkl` -- table of independent variables in the training data (stored as a pandas DataFrame)
3. `lincs_deviations<v>.pkl` -- table of dependent variables in the training data (stored as a pandas DataFrame)

#### Code function summary
To generate the training data, run `python code/Drug_Combination_Prediction/process_perturbations.py` at the top level of this respository. 
The script assembles the training data by sampling over perturbations that are shared between two or more cell lines in the LINCS data. 
If a perturbation is shared by more than 5 cell lines, than perturbations are sampled from among the available cell line pairs.

### Running `mrp_miqp_alt_obj_evk.py`
Running the command `python code/Drug_Combination_Prediction/mrp_miqp_alt_obj_evk.py -h` will produce the following usage statement:

```
usage: mrp_miqp_alt_obj_evk.py [-h] [-d D] [-n n] [-f f] [-V v] [-Z z] [-N N]
                               [-C c] [-T]

Select perturbations that are best for treating cancers using the additive method.

optional arguments:
  -h, --help            show this help message and exit
  -d D, --dataset D     Name of the dataset
  -n n, --num_feat n    Number of features used in the model
  -f f, --feature_file f
                        Name of pickle file storing the features
  -V v, --version v     version of files to use
  -Z z, --save-version z
                        version of files to use
  -N N, --num_pert N    Number of perturbations composing the combination
  -C c, --cost_factor c
                        Number to balance the contributions of the objective
                        function
```

The default version is "\_1021" and the default save version is "\_0122". 
The symbols `<v>` and `<z>` are used in place of the version and save version, respectively, when they appear in file names.
The symbol `<D>` is used in place of the selected dataset.
This version string will also be attached to the script's output files. 

#### To make predictions for one case only:
To generate the predictions from the Darmanis et al. dataset with cost factor ($\theta$) = 0, enter the command
`python -m scoop <nprocs> code/Drug_Combination_Prediction/mrp_miqp_alt_obj_evk.py -d Darmanis_et_al_2017 -C 0` 
at the base directory of the repository, where `<nprocs>` is the number of processors to use.

#### To make predictions for all cases considered in the paper:
The bash script for batch processing the prediction for all parameters and datasets is provided in `submit_mrp.sh`, 
which will run the scripts for all datasets. To obtain results for all 3 values of $\theta$ considered in the paper, run
```
bash submit_mrp.sh 0
bash submit_mrp.sh 0.5
bash submit_mrp.sh 1
```
and note that the script will allocate 8 processors x 4 threads to the task by default.
The default version output is "\_0122".

#### Input file list

Dataset-specific input files are present in `./data/Drug_Combination_Prediction/<D>/`
1. cats.pkl -- pandas Series mapping each experiment to a cell state
2. states.tsv -- tab-separated file with a state index, name, and description
3. pairs.tsv -- tab-separated file of the state index of ther unhealthy --> healthy state pairs
4. qncorr_data.pkl -- pandas DataFrame of the quantile normalized external data

More general input files are present in `./data/Drug_Combination_Prediction/`:
1. `combined_data<v>.gctx` -- combined SRA & lincs gene expression data, projected onto the eigengenes
2. `combined_perturbations%s.pkl` -- combined SRA & lincs perturbation responses
3. `alt_obj_kernel.pkl` -- matrix used for mapping the growth and stemness related genes to the eigengenes
4. `compoundinfo_beta.txt` -- drug compound information
5. `updated_casmapping_050422_data.xlsx` -- mapping between the LINCS identifiers and NH-HTAL drug identifiers.
6. `non_lincs_drugs_casmapping.xlsx` -- mapping between the non-LINCS identifiers and the NU-HTAL identifiers.


#### Output file list
All output file is deposited in `./output/Drug_Combination_Prediction/<D>/MIQP_MRP_ALT_OBJ_EVK/`.
1. `<cti_str>-<ctf_str>-<$\theta$>_<N><z>.pkl` -- the set of drug combination predictions to move from initial cell type <cti_str> toward target cell typ <ctf_str> for a given value of $\theta$ and number of drugs in each combination.
2. Subdirectories contain results for specific states.

#### Code function summary
The script iterates over up to 25 individual initial-target state pairs for each initial-target phenotype pair. 
The script uses MIQP to optimize the objective function, which is reduces to a quadratic programming problem in the linear case. 
This yields predictions of drug pairs for each initial-target state pair, which are stored in the files indicated in the "Output file list".

### Running `fs_corrected_alt_obj_evk.py`
Running the command `python code/Drug_Combination_Prediction/fs_corrected_alt_obj_evk.py -h`
will produce the following usage statement:
```
usage: fs_corrected_alt_obj_evk.py [-h] [-d D] [-n n] [-f f] [-V v] [-Z z]
                                   [-N N] [-C c] [-R] [-T] [-Q Q] [-q q]

Select perturbations that are best for treating cancers.

optional arguments:
  -h, --help            show this help message and exit
  -d D, --dataset D     Name of the dataset
  -n n, --num_feat n    Number of features used in the model
  -f f, --feature_file f
                        Name of pickle file storing the features
  -V v, --version v     version of files to use
  -Z z, --save-version z
                        version of files to save to
  -N N, --num_pert N    Number of perturbations composing the combination
  -C c, --cost_factor c
                        Number to balance the contributions of the objective
                        function
  -R, --no_recalc       Restart using previously calculated gsmf/gsmi pairs
```

#### To make predictions for one case only:
These parameters operate in the same way as the `mrp_corrected_alt_obj_evk.py` script described above.
To generate the predictions from the Darmanis et al. dataset with cost factor ($\theta$) = 0, enter the command
`python -m scoop <nprocs> code/Drug_Combination_Prediction/fs_corrected_alt_obj_evk.py -d Darmanis_et_al_2017 -C 0` 
at the base directory of the repository, where `<nprocs>` is the number of processors to use.

#### To make predictions for all cases considered in the paper:
The bash script for batch processing the prediction for all parameters and datasets is provided in `submit_mrp.sh`, 
which will run the scripts for all datasets. To obtain results for all 3 values of $\theta$ considered in the paper, run
```
submit_fs.sh 0
submit_fs.sh 0.5
submit_fs.sh 1
```
and note that the script will allocate 10 processors x 2 threads to the task by default. 
The default version output is "\_0122".

#### Input file list
The input files for this script include those used in `mrp_miqp_alt_obj_evk.py` above. There are two additional files present in 
`./output/Drug_Combination_Prediction/`:
1. `lincs_indep_vars<v>.pkl` -- table of independent variables in the training data (stored as a pandas DataFrame)
2. `lincs_deviations<v>.pkl` -- table of dependent variables in the training data (stored as a pandas DataFrame)

#### Output file list
All output file is deposited in `./output/Drug_Combination_Prediction/<D>/FS_ALT_OBJ_EVK/`.

1. `<cti_str>-<ctf_str>-<$\theta$>_<N>_corrected_states<z>.pkl` -- the set of transcriptional states for each drug combination prediction to move from initial cell type <cti_str> toward target cell typ <ctf_str> for a given value of $\theta$ and number of drugs in each combination.
2. `<cti_str>-<ctf_str>-<$\theta$>_<N>_corrected_stats<z>.pkl` -- the set of drug combination predictions to move from initial cell type <cti_str> toward target cell typ <ctf_str> for a given value of $\theta$ and number of drugs in each combination.
3. Subdirectories contain results for specific states.

#### Code function summary
The code operates similarly to `mrp_miqp_alt_obj_evk.py` as described above.
The main differences are:
1. The script iterates over up to 10 individual initial-target state pairs for each initial-target phenotype pair instead of 25.
2. The script only optimizes on drug at a time because the drug response matrix changes depending on the drug chosen.
3. The script requires the files `lincs_indep_vars<v>.pkl` and `lincs_deviations<v>.pkl` to train the nonadditive model.
4. The script returns multiple drugs for each initial state pair because the drug pairs identified are local optima, not global.
The script yields predictions of drug pairs for each initial-target state pair, which are stored in the files indicated in the "Output file list".


### Running `experimental_analysis.py`
Running the command `python code/Drug_Combination_Prediction/experimental_analysis.py -h`
will produce the following usage statement:
```
usage: experimental_analysis.py [-h] [-V v] [-Z z] [-N N] [-I]

Generate a ranked list of drug pairs from the combination predictions.

optional arguments:
  -h, --help            show this help message and exit
  -V v, --version v     version of the prediction files to use
  -Z z, --save-version z
                        extension to be added to the file of processed
                        predictions
  -N N, --num_pert N    Number of perturbations composing the combination
  -I, --isNadd          Process nonadditive predictions (default is additive)
```

To obtain the ranked list of drug predictions, enter the command
`python code/Drug_Combination_Prediction/experimental_analysis.py` 
at the base directory of the repository. 
By default the version of the files is "\_0122". 
The symbol `<D>` is used as a place holder for one of the six datasets,
and `<z>` and `<v>` are placeolders for the save version and input files version, respectively

#### Input file list

The input files are present in 
`./output/Drug_Combination_Prediction/<D>/FS_ALT_OBJ_EVK/` or 
`./output/Drug_Combination_Prediction/<D>/MIQP_MRP_ALT_OBJ_EVK/`
, where `<D>` 
is one of the six datasets, for the nonadditive and additive predictions,respectively.

The file `compoundinfo_beta.txt` in `./data/Drug_Combination_Prediction` is used to 
facilitate the mapping between LINCS identifiers and drug names.
In addtion the `<D>/cats.pkl` file, described under the script 
`mrp_miqp_alt_obj_evk.py` above, is provided to map cell state identifiers to cell types.

#### Output file list

The following output files are deposited in 
`./output/Drug_Combination_Prediction/`
1. `<D>_obj_gsm_dp_d<z><v>.pkl` -- a dictionary of the selection stats for all drug pairs in a given dataset
2. `gse_obj_gsm_dp_d<z><v>.pkl` -- a dictionary of dictionsaries containing the selection stats for all drug pairs in all datasets; the first level dictionary is keyed by the dataset, and the values are the dictionaries saved in `<D>_obj_gsm_dp_d<z><v>.pkl`.
3. `ranked_drugs<z><v>.pkl` -- a table of the ranked drugs, saved as a pickle, that contains the predictions for each dataset in a column.
4. `nonadditive_predictions<z>.pkl` -- a table of the nonadditive predictions 
5. `additive_predictions<z>.pkl` -- a table of the additive predictions

Together, files `nonadditive_predictions<z>.pkl` and `additive_predictions<z>.pkl` contain the predictions included in the Supplementary Dataset S3.

#### Code function summary
The script gathers the individual drug predictions for each initial-target pair of states and averages them to generate a list of predictions. 

### Initial and target data from glioblastoma publications

The following datasets have the same structure of their input files:

1. Darmanis _et al._ (2017) -- DIR_NAME: Darmanis_et_al_2017
2. Neftel _et al._ (2019) -- DIR_NAME: Neftel_et_al_2019
3. Couturier _et al._ (2020) -- DIR_NAME: Couturier_et_al_2020
4. Richards _et al._ (2021) -- DIR_NAME: Richards_et_al_2021
5. Chinese Glioblastoma Genome Atlas (CGGA, 2014-2015) -- DIR_NAME: CGGA

The dataset-specific input files are 

1. `states.tsv` -- a tab delimited file describing the cell states in the dataset,
2. `pairs.tsv` -- a tab delimited file matching the state indices in `states.tsv` into initial-target pairs,
3. `cats.pkl` -- a table matching each expression profile to one of the categories in `states.tsv`,
4. `qncorr_data.pkl` -- a table of quantile-normalized expression data for the 37 selected eigengenes,

which are provided in the directory
`data/Drug_Combination_Prediction/<DIR_NAME>/`,
where `<DIR_NAME>` is indicated in the list above. The `states.tsv` and `pairs.tsv` files correspond to Supplementary Dataset S7.

The Cancer Genome Atlas, Ivy Glioblastoma Atlas, and Allen Brain Atlas data are combined with the transcriptional data collected from the SRA.
The `states.tsv`, `pairs.tsv`, and `cats.pkl` are 
