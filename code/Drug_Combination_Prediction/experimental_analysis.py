"""
experimental_analysis.py

directory: ./code/Drug_Combination_Prediction/
Created by Thomas Wytock on 2023-04-29.
Copyright (c) 2023 Thomas Wytock. All rights reserved.

This code averages the drug combination pairs to obtain a ranked list.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from glob import glob
import sys,os,os.path as osp
import pickle as P
import argparse

DATA_PREFIX = '../../data/Drug_Combination_Prediction/'
OUTPUT_PREFIX = '../../output/Drug_Combination_Prediction/'

parser = argparse.ArgumentParser(description='Generate a ranked list of drug pairs from the combination predictions.')

parser.add_argument('-V','--version',metavar='v',type=str,nargs=1,
                    help='version of the prediction files to use',
                    default=['_0122'])
parser.add_argument('-Z','--save-version',metavar='z',type=str,nargs=1,
                    help='extension to be added to the file of processed predictions',
                    default=['_0122'])

parser.add_argument('-N','--num_pert',metavar='N',type=int,nargs=1,
                    help='Number of perturbations composing the combination',
                    default=[2])


parser.add_argument('-I','--isNadd',action='store_true',help='Process nonadditive predictions (default is additive)')

## add or nadd

args = parser.parse_args()
dext = args.version[0]
date = args.save_version[0]
NG = args.num_pert[0]

dataset_l = ['lincs','Darmanis_et_al_2017','Neftel_et_al_2019',
             'Couturier_et_al_2020','CGGA','Richards_et_al_2021']

def main():
    comp_info = pd.read_table('%s/compoundinfo_beta.txt' % DATA_PREFIX,index_col=0)
    brd2name = comp_info.cmap_name ## is the mapping
    brd2name = brd2name[~brd2name.index.duplicated()]
    non_lincs_map = {'fluorouracil':'5-fluorouracil','progesterone':'mpa',
                     'jib-04':'jib04','MK-5108':'mk-5108','estradiol':'e2',
                     'trametinib':'gsk1120212','torin-2':'torin 2','nvp-bgj398':'bgj398',
                     'gsk-2126458':'omiplasib','gsk-j4':'gskj4','mc1568':'mc 1568',
                     'zm-447439':'zm447439'}
    ranked_drugs = {}
    gse_obj_gsm_dp_d = {}
    dext_out = dext
    date_out = date
    if args.isNadd:
        dext_out += '_nonadd'
        date_out += '_nonadd'
    else:
        dext_out += '_linear'
        date_out += '_linear'
    for ds in dataset_l:
        OUT = '%s/FS_ALT_OBJ_EVK' % ds if args.isNadd else '%s/MIQP_MRP_ALT_OBJ_EVK' % ds
        indir= OUTPUT_PREFIX+'/'+ OUT
        print('--->',ds)
        if not ds=='lincs':
            #ds_states = pd.read_table('%s/%s/states.tsv' % (DATA_PREFIX,ds),index_col=0)
            #ds_pairs = pd.read_table('%s/%s/pairs.tsv' % (DATA_PREFIX,ds))
            cats = pd.read_pickle('%s/%s/cats.pkl' % (DATA_PREFIX,ds))
        else:
            #ds_states = pd.read_table('%s/lincs/states.tsv' % (DATA_PREFIX),index_col=0)
            #ds_pairs = pd.read_table('%s/lincs/pairs.tsv' % (DATA_PREFIX))
            cats = pd.read_pickle('%s/lincs/cats.pkl' % (DATA_PREFIX))
        tmp_d = {}
        obj_gsm_dp_d = {}
        indir_l = glob('%s/' % (indir))
        for indir2 in indir_l:
            print(indir2)
            for num in [0,0.79,1.57]:       
                if args.isNadd:
                    fn_l = glob('%s/*%.2f_%d_corrected_stats%s.pkl' % (indir2,num,NG,dext))
                else:
                    fn_l = glob('%s/*%.2f_%d%s.pkl' % (indir2,num,NG,dext))
                print(num,len(fn_l))
                if args.isNadd:
                    ## Nonadditive averaging scheme
                    if len(fn_l)<1:
                        continue                    
                    gsmp_d = {}
                    for fn in fn_l:
                        df = pd.read_pickle(fn)
                        if not isinstance(df.columns,pd.MultiIndex):
                            df.columns = pd.MultiIndex.from_tuples(df.columns.tolist())
                        for (gsmi,gsmf),grp in df.groupby([('GSMi',''),('GSMf','')]):
                            grp = grp.sort_values((2,'OBJ'))
                            try:
                                cati = cats.loc[gsmi]
                                catf = cats.loc[gsmf]
                            except KeyError:
                                continue
                            dfp = grp.set_index([(1,'anm'),(2,'anm')]) 
                            OBJ_vals = dfp.loc[:,(2,'OBJ')]
                            if (OBJ_vals<1.1*OBJ_vals.min()).sum()<50:
                                sel_obj = OBJ_vals[(OBJ_vals<1.1*OBJ_vals.min())]
                            else:
                                sel_obj = OBJ_vals.sort_values().iloc[:50]
                            ## higher is better!
                            rnk_stats = sel_obj.rank(pct=True,ascending=False)
                            rnk_stats_p = defaultdict(float)
                            for drg1_drg2,val in rnk_stats.iteritems():
                                drg_p = []
                                for zz,drg in enumerate(drg1_drg2):
                                    if drg.startswith('BRD'):
                                        dname = brd2name.loc[drg.split('---')[0]].lower()
                                        dname = non_lincs_map.get(dname,dname)
                                    else:
                                        dname = non_lincs_map.get(drg.lower(),drg.lower())
                                    drg_p.append(dname)                                
                                drg_p = tuple(drg_p)
                                rnk_stats_p[drg_p]+=val
                            rnk_stats = pd.Series(dict(rnk_stats_p)).rank(pct=True,ascending=True)
                            dfpp = {}
                            for drg1_drg2,val in dfp.loc[:,(2,'alpha')].iteritems():
                                drg_p = []
                                for zz,drg in enumerate(drg1_drg2):
                                    if drg.startswith('BRD'):
                                        dname = brd2name.loc[drg.split('---')[0]].lower()
                                        dname = non_lincs_map.get(dname,dname)
                                    else:
                                        dname = non_lincs_map.get(drg.lower(),drg.lower())
                                    drg_p.append(dname)                                
                                drg_p = tuple(drg_p)
                                dfpp[drg_p]=val    
                            ## Since we are SUMMING, elements which appear in both orders tend to
                            ## be preferred over elements that appear once. Higher sums are 
                            ## more significant!
                            sum_stats = {}
                            rev_ind = rnk_stats.index.swaplevel(0,1)
                            both_ord = rnk_stats.index.intersection(rev_ind)
                            for (p1,p2) in both_ord:
                                if p1>p2:
                                    continue
                                SUM = rnk_stats.loc[[(p1,p2),(p2,p1)]].sum()
                                sum_stats[tuple(sorted([p1,p2]))]=SUM
                            fwd_only = rnk_stats.index.difference(rev_ind)
                            for (p1,p2) in fwd_only:
                                numorrow = dfpp[(p1,p2)]
                                if isinstance(numorrow,float) and numorrow==0:
                                    sum_stats[(p1,'')] = rnk_stats.loc[(p1,p2)]
                                elif isinstance(numorrow,pd.Series) and numorrow.iloc[0]==0:
                                    sum_stats[(p1,'')] = rnk_stats.loc[(p1,p2)]
                                else:
                                    NUM = rnk_stats.loc[(p1,p2)]
                                    if isinstance(NUM,float):
                                        sum_stats[(p1,p2)]=NUM
                                    elif isinstance(NUM,pd.Series):
                                        sum_stats[(p1,p2)]=NUM.iloc[0]
                            rnk_stats2 = pd.Series(sum_stats).rank(pct=True,ascending=True).sort_values(ascending=False)
                            ## already have translated the drug names.
                            ## do certain pairs yield more accurate predictions?
                            #tpv_rnk,mpv_rnk = bin_by_moa_targ(rnk_stats2)
                        #moa_d[(cati,gsmi,catf,gsmf)] = mpv_rnk
                        #targ_d[(cati,gsmi,catf,gsmf)] = tpv_rnk
                    #num_avg = pd.DataFrame(gsmp_d).sum(axis=1).sort_values(ascending=False)
                    ## the following line groups by the cell type pairs first, then sums & replaces 0s with NaN
                    ctp_avg = pd.DataFrame(gsmp_d).groupby(level=[0,2],axis=1).sum().replace(0,np.nan)
                    ## the following line ranks the drug pairs, sums over the cell type pairs & replaces 0s with NaN
                    num_avg = ctp_avg.rank(pct=True,ascending=True).sum(axis=1).replace(0,np.nan).sort_values()
                    for (cati,gsmi,catf,gsmf),stat in gsmp_d.items():
                        for (da,db),val in stat.iteritems():
                            obj_gsm_dp_d[('%.2f' %num,cati,gsmi,catf,gsmf,da,db)] = val
                    tmp_d['%.2f' % num]=num_avg
                else:
                    ## additive averaging scheme
                    if len(fn_l)<1:
                        continue
                    rnk_stats_d = {}
                    for fn in fn_l:
                        for (gsmi,gsmf),GRP in pd.read_pickle(fn).groupby(['GSM_i','GSM_f']):
                            ## GRP has the information regarding the data
                            CTI = GRP.CTI.unique()[0]
                            CTF = GRP.CTF.unique()[0]
                            dfp = GRP.set_index(['pert_1','pert_2'])
                            OBJ_vals = dfp.OBJ 
                            if (OBJ_vals<1.1*OBJ_vals.min()).sum()<50:
                                sel_obj = OBJ_vals[(OBJ_vals<1.1*OBJ_vals.min())]
                            else:
                                sel_obj = OBJ_vals.sort_values().iloc[:50]
                            
                            ## higher is better!
                            rnk_stats = sel_obj.rank(pct=True,ascending=False)
                            rnk_stats_p = defaultdict(float)
                            for drg1_drg2,val in rnk_stats.iteritems():
                                drg_p = []
                                for zz,drg in enumerate(drg1_drg2):
                                    if drg.startswith('BRD'):
                                        dname = brd2name.loc[drg.split('---')[0]].lower()
                                        dname = non_lincs_map.get(dname,dname)
                                    else:
                                        dname = non_lincs_map.get(drg.lower(),drg.lower())
                                    drg_p.append(dname)                                
                                drg_p = tuple(sorted(drg_p)+[CTI,CTF])
                                rnk_stats_p[drg_p]+=val
                            rnk_stats = pd.Series(dict(rnk_stats_p)).rank(pct=True,ascending=True)
                            rnk_stats_d[(gsmi,gsmf)] = rnk_stats
                    rnk_stats_df = pd.DataFrame(rnk_stats_d).T
                    rnk_stats2 = rnk_stats_df.sum(axis=0).rank(pct=True,ascending=True)
                    ## do certain pairs yield more accurate predictions?
                    #tpv_rnk,mpv_rnk = bin_by_moa_targ(rnk_stats2)
                    num_avg = rnk_stats2.unstack([-2,-1])
                    tmp_d['%.2f' % num]=num_avg.T
                    for (cati,catf),stat in num_avg.iteritems():
                        for (da,db),val in stat.iteritems():
                            obj_gsm_dp_d[('%.2f' %num,cati,catf,da,db)] = val
        
        if len(tmp_d)<1:
            continue
        if args.isNadd:
            tmp_df = pd.DataFrame(tmp_d)
            if tmp_df.empty:
                continue
            tmp_df.index = pd.MultiIndex.from_tuples(tmp_df.index.tolist())
            ranked_drugs[ds] = tmp_df.groupby(level=0,axis=1).sum().stack()
        else:
            tmp_df = pd.concat(tmp_d).T
            ranked_drugs[ds] =tmp_df.stack([0,1,2]).groupby(level=[0,1,2]).sum()
        with open('%s/%s_obj_gsm_dp_d%s%s.pkl' % (OUTPUT_PREFIX,ds,date,dext_out),'wb') as fh:
            P.dump(obj_gsm_dp_d,fh,-1)
        gse_obj_gsm_dp_d[ds]=obj_gsm_dp_d
    with open('%s/gse_obj_gsm_dp_d%s%s.pkl' % (OUTPUT_PREFIX,date,dext_out),'wb') as fh:
        P.dump(gse_obj_gsm_dp_d,fh,-1)
    with open('%s/ranked_drugs%s%s.pkl' % (OUTPUT_PREFIX,date,dext_out),'wb') as fh:
        P.dump(ranked_drugs,fh,-1)

    my_d = {}
    for ds,ser in ranked_drugs.items():
        for (a,b,c),val in ser.iteritems():
            my_d[(ds,a,b,c)]=val
    my_ser = pd.Series(my_d)
    obj_d = {}
    for hp in ['0.00','0.79','1.57']:
        if hp in my_ser.index.get_level_values(3):
            sel_data = my_ser.xs(hp,level=3).unstack([1,2])
            sel_rk = sel_data.rank(axis=1,pct=True).sum(axis=0)
            trim_rk = sel_rk.sort_values(ascending=False).head(100)
            obj_d[hp]=trim_rk
    ## account for different orders
    OP2 = pd.DataFrame(obj_d)
    new_ind_l = []
    nonunique_l = []
    for ind1,ind2 in OP2.index:
        nind_l =[]
        for elt in [ind1,ind2]:
            if elt.endswith('---lincs'):
                try:
                    tmp = brd2name.loc[elt.split('---')[0]]
                    if isinstance(tmp,pd.Series):
                        if tmp.unique().shape[0]>1:
                            nonunique_l.append(elt)
                        tmp = tmp.unique()[0]
                except KeyError:
                    tmp=elt
            else:
                tmp = elt
            nind_l.append(tmp)
        new_ind_l.append(tuple(nind_l))
    OP2.index = pd.MultiIndex.from_tuples(new_ind_l)
    if args.isNadd:
        OP2.to_pickle('%s/nonadditive_predictions%s.pkl' % (OUTPUT_PREFIX,date_out))
    else:
        OP2.to_pickle('%s/additive_predictions%s.pkl' % (OUTPUT_PREFIX,date_out))
        
    return OP2
    
      
if __name__ == '__main__':
    main()
