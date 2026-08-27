# Experimental Results Supporting Data
This folder contains the input data needed to analyze the experimental results.

## File descriptions

1. `2021-12-10-CTG-CH157-Lee.xlsx`:\t Spreadsheet containing the viability data for the dose matrix experiment

2. `README.md`:\t this file

3. `DatasetS3_combined_predictions.xlsx`:\t Spreadsheet containing the drug pairs predicted by DRUG-CROPS for both the nonadditive and additive versions of the method. This file is also provided in the Supplementary Material and `comb_pred_list_210522.pkl`. Predictions are broken down by which version of the objective function that is used: `0.00` -- the first term (cell-behavior term) only, `1.57` -- the second term (cell-growth term) only, and `0.79` -- both terms.

4. `DatasetS4_plate_schematic.xlsx`:\t  Spreadsheet containing the mapping of the compounds to the specific wells on the plates for our viability assays. This file is also provided in the Supplementary Material.

5. `DatasetS8_combined_drug_information.xlsx`:\t  Spreadsheet containing the metadata of the drugs (including their mechanism of action, regulatory status, and what conditions they are designed to treat). The metadata source is indicated by the sheet name. This file is also provided in the Supplementary Material.

6. Drug-name-to-well mappings (pickle files):\t The following files contain Python pickles of dictionaries that map drug paris of their specific categories to plate wells with the total volume of drug transferred: `antisel_well_d.pkl` (anti-selected pairs), `ld_inds_d.pkl` (low-dosage pairs), `noncanc_alpha_d.pkl` (noncancer drug pairs, assayed on plate alpha), `noncanc_beta_d.pkl` (noncancer drugs, assayed individually on plate beta), `nonsel_inds_d.pkl` (randomly selected drug pairs, outside the set of predicted drugs), `sel_inds_singles.pkl` (top-selected drugs, assayed individually), `sel_inds_d.pkl` (top-selected pairs).

6. `compoundinfo_beta.txt.gz`:\t Tab-separated file of the compound information provided by the LINCS dataset.

7. `drug2moa.txt`:\t Text file mapping drugs to their mechanisms of action determined from the metadata.

8. `non_lincs_drugs_casmapping.xlsx` and `updated_casmapping_050422_data.xlsx`:\t Spreadsheets mapping the CAS numbers of the drugs to their compound names. 

9. `transfer_file.xlsx`:\t Spreadsheet used to specify the concentrations of each drug.

10. `viability_data.csv`:\t CSV file of the cell viability after treatment with the prescribed drug pairs

