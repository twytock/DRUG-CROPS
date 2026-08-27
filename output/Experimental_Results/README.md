# Experimental Results Output
This folder contains the figures and tables that come out of the analysis of the experimental results.

## File Descriptions

1. `README.md`:\t This file

### Figures

#### Main text figures

##### Fig. 3
Plots of the bars indicating predicted drug pairs by method in Fig. 3C:

1. `fig3c_first_bar.svg`					

2. `fig3c_second_bar.svg`

3. `fig3c_third_bar.svg`	

Plots of the bars indicating the predicted individual drugs by method in Fig. 3D:

4. `fig3d_first_bar_single_drug.svg`

5. `fig3d_second_bar_single_drug.svg`

6. `fig3d_third_bar_single_drug.svg`

##### Fig. 4

1. `fig4a_drug_selection_counts.svg`:\t Panel associated with the drug selection counts.

2. `fig4bc_gross_comparisons.svg`:\t Plots of the viability of drugs and the synergy ratio.

##### Fig. 5

1. `fig5a_additive_nonadditive_overlap.svg`:\t Venn diagram of the drug pair predictions selected by the additive method only, the nonadditive method only, and both methods.

2. `fig5b_additive_nonadditive_overlap_exp.svg`:\t Venn diagram of the assayed drug pairs selected by the additive method only, the nonadditive method only, and both methods.

3. `fig5cd_example_LN-229_0.79_trend.svg`:\t Panels describing the AUROC analysis.

4. `fig5e_ranked_viability_auc_singlemethod.svg`:\t Boxplots of the viability broken down by each method.

5. `fig5f_low_dosage_comparison_singlemethod.svg`:\t Panel describing the expected-to-observed synergy ratio for the nonadditive and additive methods.

#### Supplementary figures

##### Fig. S1

1. `figS1ab_example_LN-229_0.79_synergy_trend_col.svg`:\t Scatter plot of the Viability difference (observed - expected) vs. the rank of the drug pair and ROC curve of the trends in scatter plot for each method.

2. `figS1c_ranked_auc_synergy_singlemethod.svg`:\t Box plots of the Synergy ROC-AUC for each cell line. Each element of the box plot corresponds to a drug pair.

3. `figS1d_singlemethod_viability_per_uM.svg`:\t Efficacy of the drugs expressed as the amount of viability reduction per micromole of drug.

##### Fig. S2

1. `figS2a_moa_counts.svg` and `figS2a_inset_drug_group_counts_inset.svg`:\t counts of the drugs from each method, broken down by mechanism of action.

2. `figS2b_data.svg`:\t Viability in each cell line for each drug, where the drug pairs are grouped by their mechanism of action.

3. `figS2c_stat_test_boxplot.svg`:\t Box plot of the coefficient of determination comparing randomized drug mechanisms of action versus the actual values.

##### Fig. S4

1. `figS4_dose_response_CH157_digitoxin_torin2.svg`

2. `figS4_dose_response_IOMM-Lee_digitoxin_torin2.svg`

#### Other figures

1. `low_dosage_comparison.svg`
2. `low_dosage_viability.svg`
3. `low_dosage_viability_singlemethod.svg`
4. `method_venn_diagrams.svg`
5. `ranked_auc_synergy.svg`

### Data files

#### Synergy statistics
Spreadsheets recording the $t$-statistics and their associated $P$-values for synergy distributions among the drugs.
1. `gross_synergy_pvals.xlsx`
2. `gross_synergy_stats.xlsx`
3. `low_dosage_statistics.xlsx`
4. `low_dosage_statistics_singlemethod.xlsx`
5. `ranked_synergy_auc_tstats.xlsx`
6. `ranked_synergy_auc_tstats_singlemethod.xlsx`
7. `ranked_viability_auc_tstats.xlsx`
8. `ranked_viability_auc_tstats_singlemethod.xlsx`

#### Viability statistics
1. `fig4_viability_stats.xlsx`:\t Viability data associated with the boxplots in Fig. 4B


