
"""analyze_toxicity_results.py
script to analyze the state-by-state analysis of all drug pairs"""

from glob import glob
from scipy.stats import percentileofscore
from collections import defaultdict
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import argparse ## need to set up the command line arguments

TOX = '../../output/Drug_Combination_Prediction'
IN = '../../data/Toxicity_Analysis'

met_l = ['alt_obj_kernel_0221']
cl_abbr_l = ['43','CH','Io','LN']
cmp_info = pd.read_table(f'{IN}/compoundinfo_beta.txt',index_col=0)

parser = argparse.ArgumentParser(description='Analyze the toxicity results.')
parser.add_argument('-d','--dataset',type=str,dest='DS',
                    help='Name of the dataset')
# parser.add_argument('-s','--state',type=str,dest='STATE',
#                     help='Name of the cell state that the instances belong to')
# parser.add_argument('-g','--gsm',type=str,dest='GSM',
#                     help='Label of the instance')
parser.add_argument('-e','--extension',type=str,dest='EXT',
                    help='Extension of the file name (date, etc.)')
args=parser.parse_args()


def pred_rank_of_drugs(pred_ser,OPT_NM_2_COM):
    single_drug_rank = defaultdict(float)
    paired_drug_rank = {}
    for key_tup,val in pred_ser[~pred_ser.isna()].items(): ## be aware of the number of entries in the key list
        if len(key_tup)==3:
            (d1,d2,objval)=key_tup
        elif len(key_tup)==4:
            (objval,__,d1,d2) = key_tup
        elif len(key_tup)==5:
            (objval,__,__,d1,d2) = key_tup
        else:
            (objval,__,__,__,d1,d2) = key_tup
        single_drug_rank[(OPT_NM_2_COM.loc[d1],objval)]+=val
        single_drug_rank[(OPT_NM_2_COM.loc[d2],objval)]+=val
        if OPT_NM_2_COM.loc[d1]<=OPT_NM_2_COM.loc[d2]:
            paired_drug_rank[(OPT_NM_2_COM.loc[d1],OPT_NM_2_COM.loc[d2],objval)]=val
        else:
            paired_drug_rank[(OPT_NM_2_COM.loc[d2],OPT_NM_2_COM.loc[d1],objval)]=val
        
    single_drug_rank = pd.Series(dict(single_drug_rank))
    paired_drug_rank = pd.Series(paired_drug_rank)
    paired_drug_rank.name='sel_pair_rank'
    return single_drug_rank.unstack(-1), paired_drug_rank.unstack(-1)

def optimized_drug_2_names(pred_state_gsm):
    d1_un = pred_state_gsm.index.get_level_values(-2).unique()
    d2_un = pred_state_gsm.index.get_level_values(-1).unique()
    all_drgs = set(d1_un)|set(d2_un)
    tup_l = []
    for elt in all_drgs:
        if '---lincs' in elt:
            tup_l.append((elt,elt.split('---lincs')[0]))
        elif elt=='omiplasib':
            tup_l.append((elt,'gsk-2126458'))
        elif elt=='mpa':
            tup_l.append((elt,'progesterone'))
        else:
            tup_l.append((elt,elt))
    OPT_NM_2_COM = pd.Series(dict(tup_l))
    return OPT_NM_2_COM

def toxicity_drug_2_names(single_drugs):
    lc_drug_names = []
    for elt in single_drugs.index.get_level_values(-1):
        if '---' in elt:
            brd = elt.split('---')[0]
            cand_name = cmp_info.loc[brd].cmap_name
            cand_name = cand_name if isinstance(cand_name,str) else cand_name.unique()[-1]
        else:
            cand_name = elt
        lc_drug_names.append((elt,cand_name.lower()))
    TOX_NM_2_COM = pd.Series(dict(lc_drug_names))
    return TOX_NM_2_COM

def experimental_drug_2_names(mvr,TOX_NM_2_COM,single_drugs,pair_drugs):
    exp_drug_names = set(mvr.drug1_name.values.tolist())&set(mvr.drug2_name.values.tolist())
    exp_name_conversion_d = dict([(elt2,set([elt for elt in TOX_NM_2_COM.values if elt2 in elt])) for elt2 in exp_drug_names])
    nms2del = []
    for nm,val in exp_name_conversion_d.items():
        if len(val)==0:
            if nm=='omiplasib':
                exp_name_conversion_d[nm]='gsk-2126458'
            else:
                nms2del.append(nm)
        elif len(val)==1:
            exp_name_conversion_d[nm]=val.pop()
        else:
            if nm=='oxacillin':
                exp_name_conversion_d[nm]=nm
            else:
                print(nm,val)
    for elt in nms2del:
        del exp_name_conversion_d[elt]
    EXP_NM_2_COM = pd.Series(exp_name_conversion_d)
    exp_tox_dat = []
    for drg,row in mvr.iterrows():
        if isinstance(row.loc['drug2_name'],float):
            if not row.loc['drug1_name'] in EXP_NM_2_COM.index:
                continue
            drg = EXP_NM_2_COM.loc[row.loc['drug1_name']]
            if not drg in single_drugs.index:
                continue
            conc = row.loc['drug1_uM']
            row_d = {'drug1_name':drg,'drug2_name':np.nan,'drug1_uM':conc,'drug2_uM':np.nan}
            for cl,clvia in row.iloc[-4:].items():
                row_d[f'via_{cl}']=clvia
            col_data = single_drugs.loc[drg] if len(single_drugs.loc[drg].shape)==1 else single_drugs.loc[drg].mean()
            for colnm,colval in col_data.items():
                row_d[f'pred_tox_{colnm}'] = colval        
            ## need to get the associated toxicity prediction.
            row_ser = pd.Series(row_d)
        else:
            if not (row.loc['drug1_name'] in EXP_NM_2_COM.index and row.loc['drug2_name'] in EXP_NM_2_COM.index):
                continue
            drg1 = EXP_NM_2_COM.loc[row.loc['drug1_name']]
            drg2 = EXP_NM_2_COM.loc[row.loc['drug2_name']]
            if not (drg2 in pair_drugs.loc[drg1].index or drg1 in pair_drugs.loc[drg2].index):
                continue
            if drg1 in pair_drugs.loc[drg2].index:
                (drg1,drg2) = (drg2,drg1)
            conc1 = row.loc['drug1_uM']
            conc2 = row.loc['drug2_uM']
            row_d = {'drug1_name':drg1,'drug2_name':drg2,'drug1_uM':conc1,'drug2_uM':conc2}
            for cl,clvia in row.iloc[-4:].items():
                row_d[f'via_{cl}']=clvia
            for colnm,colval in pair_drugs.xs((drg1,drg2),level=[0,1]).mean().items():
                row_d[f'pred_tox_{colnm}'] = colval        
            ## need to get the associated toxicity prediction.
            row_ser = pd.Series(row_d)
        exp_tox_dat.append(row_ser)
    exp_tox_dat = pd.concat(exp_tox_dat,axis=1).T    
    return EXP_NM_2_COM, exp_tox_dat

def rename_drugs_in_tox(df,TOX_NM_2_COM):
    if df.index.nlevels < 2:
        inds = TOX_NM_2_COM.loc[df.index].values
    else:
        indsA = TOX_NM_2_COM.loc[df.index.get_level_values(0)].values
        indsB = TOX_NM_2_COM.loc[df.index.get_level_values(1)].values
        inds = pd.MultiIndex.from_tuples([(i1,i2) if i1 <=i2 else (i2,i1) for i1,i2 in zip(indsA,indsB)])
    vals = df.values
    return pd.DataFrame(vals,index=inds,columns=df.columns)

def main():
    mv_res = pd.read_pickle(f'{IN}/median_viability_results.pkl')
    nadd_pred_d = pd.read_pickle(f'{IN}/gse_obj_gsm_dp_d_0522_nonadd_v2.pkl')
    nadd_pred_ser = pd.Series(nadd_pred_d[args.DS])
    add_pred_d = pd.read_pickle(f'{IN}/gse_obj_gsm_dp_d_0522_linear_v2.pkl')
    add_pred_ser = pd.Series(add_pred_d[args.DS])
    for state_dir in glob(f'{TOX}/{args.DS}/*/'):
        STATE = state_dir.split('/')[2]
        pred_state_nadd = nadd_pred_ser.xs(STATE,level=1).unstack(0).groupby(level=[-2,-1]).sum()
        OPT_NM_2_COM_nadd = optimized_drug_2_names(pred_state_nadd)
        nz_ps_nadd = pred_state_nadd.stack()
        nz_ps_nadd = nz_ps_nadd[nz_ps_nadd>0]
        sng_drg_rnk_nadd, pair_drg_rnk_nadd = pred_rank_of_drugs(nz_ps_nadd,OPT_NM_2_COM_nadd)
        pred_state_add = add_pred_ser.xs(STATE,level=1).unstack(0).groupby(level=[-2,-1]).sum()
        OPT_NM_2_COM_add = optimized_drug_2_names(pred_state_add)
        nz_ps_add = pred_state_add.stack()
        nz_ps_add = nz_ps_add[nz_ps_add>0]
        sng_drg_rnk_add, pair_drg_rnk_add = pred_rank_of_drugs(nz_ps_add,OPT_NM_2_COM_add)
        single_drug_dtox_add = {}
        single_drug_dtox_nadd = {}
        pair_drug_dtox_add = {}
        pair_drug_dtox_nadd = {}
        for fn in glob(f'{state_dir}*__{args.EXT}.pkl'):
            if "analyze-result" in fn:
                continue
            GSM = fn.split('/')[-1].split(f'__{args.EXT}')[0]
            all_drug_tox_by_state = pd.read_pickle(fn)
            if all_drug_tox_by_state.index.get_level_values(0).unique().shape[0]==1:
                ## reduce the space occupied by the data
                all_drug_tox_by_state = all_drug_tox_by_state.droplevel([0,1,2])
                all_drug_tox_by_state.to_pickle(fn)
                ## level 0 : DS
                ## level 1 : STATE
                ## level 2 : GSM
            result_d = {}
            for aorna,anagrp in all_drug_tox_by_state.groupby(level=0):
                ## aorna is 'add' or 'nadd'
                ## anagrp is the predictions
                anagrp = anagrp.droplevel(0)
                single_drugsx = anagrp.xs('',level=-1)
                pair_drugsx = anagrp[anagrp.index.get_level_values(-1)!='']
                TOX_NM_2_COM = toxicity_drug_2_names(single_drugsx)
                single_drugs = rename_drugs_in_tox(single_drugsx.unstack(0),TOX_NM_2_COM)
                pair_drugs = rename_drugs_in_tox(pair_drugsx.unstack(0),TOX_NM_2_COM)
                ## mydf, mypair_df need to be made
                ## these contain the estimated toxicity...
                if aorna=='nadd':
                    pred_state_gsm = nadd_pred_ser.xs((STATE,GSM),level=[1,2])
                    for drg,dtox in single_drugs.alt_obj_kernel_0221.items():
                        single_drug_dtox_nadd[(drg,GSM)]=dtox
                    for (drg1,drg2),dtox in pair_drugs.alt_obj_kernel_0221.items():
                        pair_drug_dtox_nadd[(drg1,drg2,GSM)]=dtox
                else:
                    pred_state_gsm = add_pred_ser.xs(STATE,level=1)
                    for drg,dtox in single_drugs.alt_obj_kernel_0221.items():
                        single_drug_dtox_add[(drg,GSM)]=dtox
                    for (drg1,drg2),dtox in pair_drugs.alt_obj_kernel_0221.items():
                        pair_drug_dtox_add[(drg1,drg2,GSM)]=dtox
                
                OPT_NM_2_COM = optimized_drug_2_names(pred_state_gsm)
                sng_drg_rnk, pair_drg_rnk = pred_rank_of_drugs(pred_state_gsm,OPT_NM_2_COM)
                IS_pair = pair_drg_rnk.index.intersection(pair_drugs.index)
                ## get percentile of the toxicity
                ## 
                EXP_NM_2_COM, exp_tox_dat = experimental_drug_2_names(mv_res,TOX_NM_2_COM,single_drugs,pair_drugs)
                low_dose = exp_tox_dat[((exp_tox_dat.drug1_uM!=10)|(exp_tox_dat.drug2_uM!=10))]
                #low_dose_single = exp_tox_dat[(exp_tox_dat.drug2_name.isna())&((exp_tox_dat.drug1_uM!=10))]
                ten_uM_pairs = exp_tox_dat[(~exp_tox_dat.drug2_name.isna())&((exp_tox_dat.drug1_uM==10)&(exp_tox_dat.drug2_uM==10))]
                ten_uM_single = exp_tox_dat[(exp_tox_dat.drug2_name.isna())&(exp_tox_dat.drug1_uM==10)]
                
                via_data_by_class = {'low_dose':low_dose, 'ten_uM_pairs':ten_uM_pairs, 'ten_uM_single':ten_uM_single}
                for via_cl,via_data in via_data_by_class.items():
                    for met in met_l:
                        XXX = via_data.loc[:,f'pred_tox_{met}'].values.reshape(-1,1)
                        scrs = via_data.loc[:,f'pred_tox_{met}'].values*-1
                        med_via_data = via_data.iloc[:,4:8].median(axis=1)
                        yyy = (med_via_data<0.5).astype(float).values
                        clf = LogisticRegression(solver="liblinear", random_state=0, fit_intercept=True).fit(XXX, yyy)
                        rocauc = roc_auc_score(yyy,clf.decision_function(XXX))
                        result_d[(aorna,via_cl,met,'median','rocauc')]=rocauc
                        precision, recall, thresholds = precision_recall_curve(yyy,scrs)
                        result_d[(aorna,via_cl,met,'median','prauc')] = auc(recall,precision)
                        for cl_abbr in cl_abbr_l:
                            sel_via_data = via_data.loc[:,f'via_{cl_abbr}']
                            yyy = (sel_via_data<0.5).astype(float).values
                            clf = LogisticRegression(solver="liblinear", random_state=0, fit_intercept=True).fit(XXX, yyy)
                            rocauc = roc_auc_score(yyy,clf.decision_function(XXX))
                            result_d[(aorna,via_cl,met,cl_abbr,'rocauc')]=rocauc
                            precision, recall, thresholds = precision_recall_curve(yyy,scrs)
                            result_d[(aorna,via_cl,met,cl_abbr,'prauc')] = auc(recall,precision)
                    
                for colnm,col in pair_drugs.items():
                    avg_col=col.groupby(level=[0,1]).mean()
                    PT = percentileofscore(col,avg_col.loc[IS_pair])
                    ## calculate the average
                    avg_ptile = np.mean(PT)
                    ## calculate the rank-weighted average
                    rnkwt_ptile = np.sum(pair_drg_rnk.loc[IS_pair].sum(axis=1)*PT)/np.sum(pair_drg_rnk.loc[IS_pair].sum(axis=1))
                    ## calculate the correlation
                    CC = avg_col.loc[IS_pair].corr(pair_drg_rnk.loc[IS_pair].sum(axis=1),method='spearman')
                    result_d[(aorna,'all',colnm,'pred','PT_mean')] = avg_ptile/100
                    result_d[(aorna,'all',colnm,'pred','PT_wt')] = rnkwt_ptile/100
                    result_d[(aorna,'all',colnm,'pred','CC')] = CC
                    for objval,objcol in pair_drg_rnk.items():
                        IS_pair_oc = objcol[~objcol.isna()].index.intersection(pair_drugs.index)
                        objPT = percentileofscore(col,avg_col.loc[IS_pair_oc])
                        objavg_PT = np.mean(objPT)
                        objrnkwt_PT = np.sum(objcol.loc[IS_pair_oc].values*objPT)/np.sum(objcol.loc[IS_pair_oc])
                        objCC = avg_col.loc[IS_pair_oc].corr(objcol.loc[IS_pair_oc],method='spearman')
                        result_d[(aorna,objval,colnm,'pred','PT_mean')] = objavg_PT/100
                        result_d[(aorna,objval,colnm,'pred','PT_wt')] = objrnkwt_PT/100
                        result_d[(aorna,objval,colnm,'pred','CC')] = objCC
            
            result_ser = pd.Series(result_d)
            result_ser.to_pickle(f'{TOX}/{args.DS}/{STATE}/toxcorr-results__{GSM}__{args.EXT}.pkl')
        IS_pair_nadd = pair_drg_rnk_nadd.index.intersection(pair_drugs.index)
        IS_pair_add = pair_drg_rnk_add.index.intersection(pair_drugs.index)
        IS_single_nadd = sng_drg_rnk_nadd.index.intersection(single_drugs.index)
        IS_single_add = sng_drg_rnk_add.index.intersection(single_drugs.index)
        pair_drg_rnk_nadd.loc[IS_pair_nadd].to_pickle(f'{TOX}/{args.DS}/{STATE}__pred-freq-pair-nadd.pkl')
        pair_drg_rnk_add.loc[IS_pair_add].to_pickle(f'{TOX}/{args.DS}/{STATE}__pred-freq-pair-add.pkl')
        sng_drg_rnk_nadd.loc[IS_pair_nadd].to_pickle(f'{TOX}/{args.DS}/{STATE}__pred-freq-single-nadd.pkl')
        sng_drg_rnk_add.loc[IS_pair_add].to_pickle(f'{TOX}/{args.DS}/{STATE}__pred-freq-single-add.pkl')
        pd.Series(single_drug_dtox_add).to_pickle(f'{TOX}/{args.DS}/{STATE}__avg-dtox-single-add.pkl')
        pd.Series(single_drug_dtox_nadd).to_pickle(f'{TOX}/{args.DS}/{STATE}__avg-dtox-single-nadd.pkl')
        pd.Series(pair_drug_dtox_add).to_pickle(f'{TOX}/{args.DS}/{STATE}__avg-dtox-pair-add.pkl')
        pd.Series(pair_drug_dtox_nadd).to_pickle(f'{TOX}/{args.DS}/{STATE}__avg-dtox-pair-nadd.pkl')
            
    ## find the rank of the kernels of the (1) most frequently selected drugs and (2) the selected drug pairs
    ## need to do this for both the additive and nonadditive cases
    return

if __name__=='__main__':
    main()