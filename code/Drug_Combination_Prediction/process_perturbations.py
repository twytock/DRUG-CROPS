"""
process_perturbations.py

directory: ./code/Drug_Combination_Prediction/
Created by Thomas Wytock on 2023-04-28.
Copyright (c) 2023 Thomas Wytock. All rights reserved.
"""

import pandas as pd
import numpy as np
import cmapPy.pandasGEXpress.parse as parse
import cmapPy.pandasGEXpress.concat as concat
import os,os.path as osp
import pickle as P
import argparse
from itertools import combinations
    
DATA_PREFIX = '../../data/Drug_Combination_Prediction/'
OUTPUT_PREFIX = '../../output/Drug_Combination_Prediction/'

parser = argparse.ArgumentParser(description='Generate training data for the nonadditivity prediction algorithm from the LINCS data.')

parser.add_argument('-V','--version',metavar='v',type=str,nargs=1,
                    help='version of the expression files to use',
                    default=['_1021'])
args = parser.parse_args()
over = args.version[0]

def construct_deviations(nct=5,version='_1121'):
    '''ARGUMENTS:
    
    nct : if the number of cell types sharing a given perturbation 
             is larger than this number, nct*(nct-2)/2 combinations 
             will be sampled, instead of all possible combinations 
    version : version string to append to the output file names
    '''

    tavg_pert = pd.read_pickle('%s/td_avg_expr_resp_proj%s.pkl' % (DATA_PREFIX,over))
    gctoo_l = []
    for ii in range(4):
        gctoo = parse.parse('%s/combined_data%s_part%d.gctx' % (DATA_PREFIX,over,ii+1))
        gctoo_l.append(gctoo)
    combined_data = concat.vstack(gctoo_l)    
    lincs_data = combined_data.row_metadata_df[combined_data.row_metadata_df.database=='LINCS']
    lincsct_grp_ser = pd.read_pickle('%s/lincs_cell_type_groups.pkl' % DATA_PREFIX)
    lincs_ctrl_inds = lincs_data[lincs_data.treatment.isin(['nc-drug','nc-kd','nc-oe','nc-untrt'])].index
    lincs_ctrl_data_df = combined_data.data_df.loc[lincs_ctrl_inds]
    dashed_d = {'9':'CMAP-HSF-KRTAP5-9','5':'CMAP-HSF-NKX2-5',
    '115G20.2':'GS1-115G20.2'}
    s_d = {'+':'oe_lincs','-':'kd_lincs'}
    ## note --  need to record the metadata separately from the perturbations
    ##      --  in the same format as the SRA perturbations
    ## key == ct1,ct2,new_id
    pert_info_d = {}
    devs_d = {}
    state_d = {}
    ct_updates = {'A-673': 'A673', 'BT-20': 'BT20', 'CL-34': 'CL34', 
                  'COR-L23': 'CORL23', 'DV-90': 'DV90', 'EFO-27': 'EFO27', 
                  'H3': 'HUES3', 'HCC-15': 'HCC15', 'HEC-108': 'HEC108', 
                  'HEK293': 'HEK293T', 'HL-60': 'HL60', 'HT-29': 'HT29',  
                  'HepG2': 'HEPG2', 'JHUEM-2': 'JHUEM2', 'MCH058': 'MCH58',
                  'NCI-H1299': 'H1299','NCI-H1694': 'NCIH1694', 
                  'NCI-H1836': 'NCIH1836', 'NCI-H2073': 'NCIH2073', 
                  'NCI-H508': 'NCIH508', 'NCI-H596': 'NCIH596', 
                  'NCI-H716': 'NCIH716', 'RMG-I': 'RMGI', 'RMUG-S': 'RMUGS', 
                  'SH-SY5Y': 'SHSY5Y', 'SK-BR-3': 'SKBR3', 
                  'SK-MEL-1': 'SKMEL1', 'SK-MEL-28': 'SKMEL28', 
                  'SNG-M': 'SNGM', 'SNU-1040': 'SNU1040', 'SNU-C5': 'SNUC5', 
                  'THP-1': 'THP1', 'TYK-nu': 'TYKNU', 'U266B1': 'U266', 
                  'VCaP': 'VCAP', 'WSU-DLCL2': 'WSUDLCL2'}
    #update cell_line column
    #print('updating data')
    lincs_data = lincs_data.replace(ct_updates)
    lincsGBct = lincs_data.loc[lincs_ctrl_inds].groupby('cell_line')
    for (pert,sns),grp in tavg_pert.groupby(level=[1,2]):
        p2 = dashed_d[pert] if pert in dashed_d.keys() else pert
        new_id='%s---%s' % (p2,s_d[sns]) if sns in s_d.keys() else p2+'_lincs'
        sel_cts = grp.index.get_level_values(0).value_counts().index
        if sel_cts.shape[0]<2:
            continue
        GB = grp.groupby(level=0)
        all_combs = np.asarray(list(combinations(sel_cts,2)))
        if len(sel_cts)>nct:
            L = int(nct*(nct-1)/2)
            sel_inds = np.random.permutation(range(all_combs.shape[0]))[:L]
            all_combs = all_combs[sel_inds]
        for ct1,ct2 in all_combs:
            if lincsct_grp_ser.loc[ct1]==lincsct_grp_ser.loc[ct2]:
                continue
            ## x0 delt dev
            ## u  v'   v-v'
            ## u' v    v'-v
            ## u,v  == expression, perturbation in ct1
            ## u',v' == expression, perturbation in ct2
            cgrp1 = GB.get_group(ct1).mean()
            cgrp2 = GB.get_group(ct2).mean()
            ctrl_grp1= lincs_ctrl_data_df.loc[lincsGBct.get_group(ct1).index].mean()
            ctrl_grp2= lincs_ctrl_data_df.loc[lincsGBct.get_group(ct2).index].mean()
            ctd_pert = cgrp2-cgrp1 #(CT2+PERT-CT2)-(CT1+PERT-CT1)
            devs_d[(ct1,ct2,new_id)] = ctd_pert
            devs_d[(ct2,ct1,new_id)] = -1*ctd_pert
            state_d[(ct1,ct2,new_id)] = np.hstack([ctrl_grp1.values,cgrp2.values])
            state_d[(ct2,ct1,new_id)] = np.hstack([ctrl_grp2.values,cgrp1.values])
            ctt1 = map_lincs_ct2cl(ct1)
            ctt2 = map_lincs_ct2cl(ct2)
            ## collect the metadata
            info_d = {'bioproject':'LINCS', 'init_cell_type':ctt1,'init_cell_line':ct1,
                      'fin_cell_type':ctt2,'fin_cell_line':ct2}
            if sns=='dose_avg':
                info_d['sense']='none'
                info_d['init_treatment']=pert
                info_d['fin_treatment']=pert
                info_d['init_genotype']='NONE'
                info_d['fin_genotype']='NONE'
            else:
                if sns=='-':
                    sns_key='kd'
                elif sns=='+':
                    sns_key='oe'
                else:
                    sns_key=sns
                info_d['sense']=sns_key
                info_d['init_treatment']=sns_key
                info_d['fin_treatment']=sns_key
                info_d['init_genotype']=pert
                info_d['fin_genotype']=pert
            info_d2 = info_d.copy()
            info_d2['init_cell_type']=ctt2
            info_d2['init_cell_line']=ct2
            info_d2['fin_cell_type']=ctt1
            info_d2['fin_cell_line']=ct1
            pert_info_d[(ct1,ct2,new_id)]=info_d
            pert_info_d[(ct2,ct1,new_id)]=info_d2
    with open('%s/lincs_perturbation_info%s.pkl' % (OUTPUT_PREFIX,version),'wb') as fh:
        P.dump(pert_info_d,fh,-1)
    with open('%s/lincs_indep_vars%s.pkl' % (OUTPUT_PREFIX,version),'wb') as fh:
        P.dump(state_d,fh,-1)
    with open('%s/lincs_deviations%s.pkl' % (OUTPUT_PREFIX,version),'wb') as fh:
        P.dump(devs_d,fh,-1)
    return pert_info_d, state_d, devs_d

lincs_cl_info = pd.read_table('%s/cellinfo_beta.txt' % DATA_PREFIX,index_col=0)
lincs2sra_mapping = pd.read_excel('%s/LINCS_cell_lines_unique_v2.xlsx' % DATA_PREFIX ,engine='openpyxl',index_col=0)

def map_lincs_ct2cl(lincs_cl):
    if lincs_cl in lincs2sra_mapping.index:
        return lincs2sra_mapping.suggested_cell_type.loc[lincs_cl]
    else:
        return lincs_cl_info.cell_type.loc[lincs_cl]

if __name__ == '__main__':
    VER = ''
    pert_info,state_d,other_d = construct_deviations(version=VER)
    pert_info_df = pd.DataFrame(pert_info)
    pert_info_df.to_pickle('%s/lincs_perturbation_info%s.pkl' % (OUTPUT_PREFIX,VER))
    state_df = pd.DataFrame(state_d)
    state_df.to_pickle('%s/lincs_indep_vars%s.pkl' % (OUTPUT_PREFIX,VER))
    other_df = pd.DataFrame(other_d)
    other_df.to_pickle('%s/lincs_deviations%s.pkl' % (OUTPUT_PREFIX,VER))
    


    