"""
multi_pert_analysis.py

wd: computational_github_code/code/Computational_Cross_Validation/

Created by Thomas Wytock on 2023-04-23.
Copyright (c) 2023 Thomas Wytock. All rights reserved.

Performs leave-one-BioProject-out cross validation.
"""

import sys, os, os.path as osp
import pandas as pd
import numpy as np
import argparse
from scipy.stats import linregress
import pickle as P
import networkx as nx
import cmapPy.pandasGEXpress.parse as parse
import cmapPy.pandasGEXpress.GCToo as GCToo
import cmapPy.pandasGEXpress.write_gctx as wg
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import LeaveOneOut
from collections import defaultdict
from scoop import futures ## needed for parallelization


kind = 'LOPO'

parser = argparse.ArgumentParser(description='Perform the leave-one-bioproject-out cross validation and calculate summary statistics.')

parser.add_argument('-V','--version',metavar='v',type=str,nargs=1,
                    help='version of files to use',
                    default=['_0122'])
args = parser.parse_args()
ver = args.version[0]

DATA_PREFIX = '../../data/Computational_Cross_Validation'
OUTPUT_PREFIX = '../../output/Computational_Cross_Validation'


def weights(df):
    if isinstance(df,pd.DataFrame):
        mu = df.mean(axis=0)
        sig = df.std(axis=0,ddof=1)
        WT = pd.Series(np.zeros(sig.shape),index=sig.index)
        Ainds = np.where(sig>0)[0]
        WT.iloc[Ainds] = 1/sig.iloc[Ainds]
        ## if contacts are absent, then exclude from weights
        Binds = np.where((sig==0) & (mu==0))[0]
        WT.iloc[Binds] = sig.iloc[Binds]
        Cinds = np.where((sig==0) & (mu!=0))[0]
        M = WT.max()
        if M > 0:
            WT.iloc[Cinds] = 2*M
        else:
            ## if all weights are zero, then weight all directions equally
            WT.iloc[:] = 1
        WT = WT.div(np.linalg.norm(WT.astype(np.float64)))
    else:
        mu = df.mean()
        sig = df.std(ddof=1)
        if sig>0:
            WT = 1/sig
        else:
            WT = 1/np.sqrt(df.shape[0])
    return WT

def get_weighted_data(G,u,data):
    """Uses linear weights to account for time-course and dose-response curves"""
    cols = G.nodes[u]['columns']
    if data.index.isin(cols).sum()==0:
        return -1
    cols = list(set(cols)&set(data.index.tolist()))
    dat = data.loc[cols].T
    cw_tc_kw ='weights_%s_TC' % u[2]
    cw_dr_kw ='weights_%s_DR' % u[2]
    if cw_tc_kw in G.nodes[u].keys() and \
       cw_dr_kw in G.nodes[u].keys():
        pass
    else:
        cw_tc_kw = [k for k in G.nodes[u].keys() if k.endswith('TC')][0]
        cw_dr_kw = [k for k in G.nodes[u].keys() if k.endswith('DR')][0]
    WT = G.nodes[u][cw_tc_kw]*G.nodes[u][cw_dr_kw]
    WT = WT.loc[cols]
    return dat,WT/WT.sum()


def calc_dep_indep(drug_data_fn,dev_data_fn,ver='_0122'): ## formerly: apply_KNN2data
    dev2ord = pd.read_pickle(dev_data_fn)
    if len(dev2ord.index.levels)>2:
        dev2ord = dev2ord[dev2ord.index.get_level_values(2).isna()]
        dev2ord.index =dev2ord.index.droplevel(2)        
    delta_gctx = parse.parse('%s/drug_delta_selcorr_df%s.gctx' % (OUTPUT_PREFIX,ver))
    delta_df = delta_gctx.data_df
    pert_df = delta_gctx.row_metadata_df
    
    nd_data_head = ['cell_type','cell_line','treatment','genotype']
    drug_data_all = parse.parse(drug_data_fn)
    drug_data = drug_data_all.data_df
    
    
    col_meta_fn = '%s/SRX_metadata%s.csv.gz' % (DATA_PREFIX,ver)
    col_meta = pd.read_csv(col_meta_fn,compression='gzip',index_col=0,header=0) 
    graph_d_fn = '%s/project_graph_d%s.pkl' % (OUTPUT_PREFIX,ver)
    graph_d = pd.read_pickle(graph_d_fn)
    comb_train_dat = {}
    comb_train_op = {}
    ii = 0
    di_tup_bpj_d ={}
    for di_tup,dev in dev2ord.iterrows():
        #(di1,di2)
        for di in di_tup[:-1]:
            bpj = pert_df.loc[di,'bioproject']
            di_tup_bpj_d[di_tup]=bpj
            pgraph = graph_d[bpj]
            spert_df = pert_df[pert_df.bioproject==bpj]
            if bpj!='ENCODE':
                sel_inds = col_meta[col_meta.bioproject==bpj].index
            else:
                sel_inds = col_meta[col_meta.database==bpj].index
            sel_cols = drug_data.loc[sel_inds]
            if len(sel_cols)<1:
                continue
            pcols = ['fin_%s' % (elt) for elt in nd_data_head]
            end = tuple(spert_df.loc[di,pcols].values)
            end_dat,end_WT = get_weighted_data(pgraph,end,sel_cols)
            end_vec = end_dat.dot(end_WT) # /end_WT.sum() end_WT.sum()==1 now.
        ## need to concat end_vec and delta
        sec_pert = delta_df.loc[di_tup[-1]]
        comb_vec = {}
        for k,v in end_vec.iteritems():
            comb_vec[k+'_s']=v
        for k,v in sec_pert.iteritems():
            comb_vec[k+'_p']=v
        comb_train_dat[di_tup] = comb_vec
        comb_train_op[di_tup] = dev
#         if ii > 10:
#             break
#         ii+=1
    indep_var = pd.DataFrame(comb_train_dat).T
    dep_var = pd.DataFrame(comb_train_op).T
    di_tup_bpj_ser = pd.Series(di_tup_bpj_d)
    di_tup_bpj_ser.to_pickle('%s/dev2bioproject%s.pkl' % (OUTPUT_PREFIX,ver))
    indep_var.to_pickle('%s/comb_corr_indep%s.pkl' % (OUTPUT_PREFIX,ver))
    dep_var.to_pickle('%s/comb_corr_dep%s.pkl' % (OUTPUT_PREFIX,ver))
    return

def LOPO(drug_data_fn):
    di_tup_bpj_ser = pd.read_pickle('%s/dev2bioproject%s.pkl' % (OUTPUT_PREFIX,ver))
    indep_var = pd.read_pickle('%s/comb_corr_indep%s.pkl' % (OUTPUT_PREFIX,ver))
    dep_var = pd.read_pickle('%s/comb_corr_dep%s.pkl' % (OUTPUT_PREFIX,ver))
    nd_data_head = ['cell_type','cell_line','treatment','genotype']
    drug_data = parse.parse(drug_data_fn).data_df
    N_eg = drug_data.shape[1]
    WT = weights(drug_data)
    pert_fn = '%s/drug_delta_selcorr_df%s.gctx' % (OUTPUT_PREFIX,ver)
    delta_gctx = parse.parse(pert_fn)
    pert_df = delta_gctx.row_metadata_df
    delta_df = delta_gctx.data_df
    wt_vec = {}
    for k,v in WT.iteritems():
        wt_vec[k+'_s']=v
    for k,v in WT.iteritems():
        wt_vec[k+'_p']=v
    WT2 = pd.Series(wt_vec)/np.sqrt(2)
    op_l = []
    tt_ind_d = defaultdict(dict)
    graph_d_fn = '%s/project_graph_d%s.pkl' % (OUTPUT_PREFIX,ver)
    graph_d = pd.read_pickle(graph_d_fn)
#     if osp.exists('%s/predictions_lopo%s.pkl' % (OUTPUT_PREFIX,ver)):
#         pdi_tup_est_df = pd.read_pickle('%s/predictions_lopo%s.pkl' % (OUTPUT_PREFIX,ver))
#         pdi_tup_lin_df = pd.read_pickle('%s/predictions_lin%s.pkl' % (OUTPUT_PREFIX,ver))
#         pdi_tup_stats_df = pd.read_pickle('%s/lopo_stats%s.pkl' % (OUTPUT_PREFIX,ver))
#         pdi_tup_states_df = pd.read_pickle('%s/lopo_metadata%s.pkl' % (OUTPUT_PREFIX,ver))
#     else:
    pdi_tup_dev_ser = pd.Series(dtype=float)
    pdi_tup_est_df = pd.DataFrame(dtype=float)
    pdi_tup_lin_df = pd.DataFrame(dtype=float)
    pdi_tup_stats_df = pd.DataFrame(dtype=float)
    pdi_tup_states_df = pd.DataFrame(dtype=object)
        
    comp_bpjs = di_tup_bpj_ser.loc[pdi_tup_est_df.index].unique()

    for bpj,sel_df in di_tup_bpj_ser.to_frame().groupby(0):
        if bpj in comp_bpjs:
            continue
        tt_ind_d[bpj]['train'] = di_tup_bpj_ser.index.difference(sel_df.index)
        tt_ind_d[bpj]['test'] = sel_df.index
    tt_ind_d = dict(tt_ind_d)
    test_op = {}
    pcols = ['fin_%s' % (elt) for elt in nd_data_head]
    di_tup_states_d = {}
    di_tup_est_d = {}
    di_tup_dev_d = {}
    di_tup_lin_d = {}
    di_tup_stats_d = defaultdict(dict)
    for bpj,tr_te_ind_d in tt_ind_d.items():
        print(bpj)
        spert_df = pert_df[pert_df.bioproject==bpj]
        pgraph = graph_d[bpj]
        tr_ind = tr_te_ind_d['train']
        te_ind = tr_te_ind_d['test']
        XVALS = indep_var.loc[tr_ind]*WT2
        XVT = indep_var.loc[te_ind]*WT2
        YVALS = dep_var.loc[tr_ind]*WT
        YVT = dep_var.loc[te_ind]*WT
        clf = KNeighborsRegressor(9,weights='distance',n_jobs=4)
        clf.fit(XVALS,YVALS)
        EST = clf.predict(XVT)
        est_df = pd.DataFrame(EST,index=te_ind,columns=dep_var.columns)
        st = XVT.iloc[:,:N_eg]
        st.columns = [col.split('_s')[0] for col in st.columns]
        prt =XVT.iloc[:,(N_eg):]
        prt.columns = [col.split('_p')[0] for col in prt.columns]
        lin_df = st+prt
        #SCR = np.sum(np.square(est_df-YVT),axis=1)
        r2_devs = 1 - np.sum(np.square(est_df-YVT),axis=1)/ np.sum(np.square(YVT),axis=1)
        for ditup,lin_row in (indep_var.loc[te_ind].iloc[:,N_eg:]*WT.values).iterrows():
            di_tup_lin_d[ditup]=lin_row
        for ditup,est_row in est_df.iterrows():
            di_tup_est_d[ditup]=est_row
        for ditup,r2_dev in r2_devs.iteritems():
            di_tup_dev_d[ditup]=r2_dev
        for (p1i,p2i) in te_ind:
            end = tuple(spert_df.loc[p1i,pcols].values)
            end_cols =drug_data.index.intersection(pgraph.nodes[end]['columns'])
            xend = tuple(spert_df.loc[p2i,pcols].values)
            xend_cols = drug_data.index.intersection(pgraph.nodes[xend]['columns'])
            S1 = set([elt for elt in pgraph.successors(end)])
            S2 = set([elt for elt in pgraph.successors(xend)])
            T1 = set([elt for elt in pgraph.predecessors(end)])
            T2 = set([elt for elt in pgraph.predecessors(xend)])
            try:
                conds = tuple(list(S1&S2)[0])
                iconds = tuple(list(T1&T2)[0])
                fconds = tuple(list(S1&S2)[0])
            except IndexError:
                continue
            di_tup_states_d[(p1i,p2i,'init')]=dict(zip(nd_data_head,iconds))
            di_tup_states_d[(p1i,p2i,'fin')]=dict(zip(nd_data_head,fconds))
            di_tup_states_d[(p1i,p2i,'pert1')]=dict(zip(nd_data_head,end))
            di_tup_states_d[(p1i,p2i,'pert2')]=dict(zip(nd_data_head,xend))
            ctrl_cols = drug_data.index.intersection(pgraph.nodes[iconds]['columns'])
            fin_cols = drug_data.index.intersection(pgraph.nodes[conds]['columns'])
            init_mean = drug_data.loc[ctrl_cols].mean()
            init_std = drug_data.loc[ctrl_cols].std()
            end_mean = drug_data.loc[end_cols].mean()
            end_std = drug_data.loc[end_cols].std()
            xend_mean = drug_data.loc[xend_cols].mean()
            xend_std = drug_data.loc[xend_cols].std()
            fin_mean = drug_data.loc[fin_cols].mean()
            fin_std = drug_data.loc[fin_cols].std()
            ## could sample mean/sigma via bootstrapping
            if all(np.r_[len(ctrl_cols)>1,len(end_cols)>1,len(xend_cols)>1,len(fin_cols)>1]):
                lin_param_l = []
                est_param_l = []
                mag_act_l = []
                mag_lin_l = []
                mag_est_l = []
                dev_est_l = []
                dev_lin_l = []
                dev_param_l = []
                ## bootstrap
                for jj in range(100):
                    iinit = init_mean+np.random.randn(init_std.shape[0])*init_std
                    iend = end_mean+np.random.randn(end_std.shape[0])*end_std
                    ixend = xend_mean+np.random.randn(xend_std.shape[0])*xend_std
                    ifin = fin_mean+np.random.randn(fin_std.shape[0])*fin_std
                    linp = (iend+ixend-iinit)*WT
                    e1 = np.c_[np.r_[iend,ixend-iinit],np.r_[ixend,iend-iinit]].T*WT2.values
                    estp = linp + np.mean(clf.predict(e1),axis=0)
                    actp = ifin*WT
                    lin_param_l.append(linregress(linp,actp))
                    est_param_l.append(linregress(estp,actp))
                    dev_est = np.linalg.norm(estp-actp)
                    dev_lin = np.linalg.norm(linp-actp)
                    dev_param_l.append(1-dev_est**2/dev_lin**2)
                    mag_act_l.append(np.linalg.norm(actp))
                    mag_lin_l.append(np.linalg.norm(linp))
                    mag_act_l.append(np.linalg.norm(actp))
                    mag_est_l.append(np.linalg.norm(estp))
                    dev_est_l.append(dev_est)
                    dev_lin_l.append(dev_lin)
                lin_param_arr = np.c_[lin_param_l]
                lin_param_bs = lin_param_arr.mean(axis=0)
                lin_param_zv = lin_param_bs/lin_param_arr.std(axis=0)
                est_param_arr = np.c_[est_param_l]
                est_param_bs = est_param_arr.mean(axis=0)
                est_param_zv = est_param_bs/est_param_arr.std(axis=0)
                dev_param_arr = np.r_[dev_param_l]
                dev_param_bs = dev_param_arr.mean(axis=0)
                dev_param_CIlb = np.percentile(dev_param_arr,2.5)
                dev_param_med = np.percentile(dev_param_arr,50)
                dev_param_CIub = np.percentile(dev_param_arr,97.5)
                di_tup_stats_d[(p1i,p2i)]['m_est'] = est_param_bs[0]
                di_tup_stats_d[(p1i,p2i)]['b_est'] = est_param_bs[1]
                di_tup_stats_d[(p1i,p2i)]['r_est'] = np.mean(est_param_arr[:,2]**2)
                di_tup_stats_d[(p1i,p2i)]['p_est'] = np.median(est_param_arr[:,3])
                di_tup_stats_d[(p1i,p2i)]['m_est_zv'] = est_param_zv[0]
                di_tup_stats_d[(p1i,p2i)]['b_est_zv'] = est_param_zv[1]
                di_tup_stats_d[(p1i,p2i)]['R_dev_avg'] = dev_param_bs 
                di_tup_stats_d[(p1i,p2i)]['R_dev_med'] = dev_param_med
                di_tup_stats_d[(p1i,p2i)]['R_dev_CIlb'] = dev_param_CIlb 
                di_tup_stats_d[(p1i,p2i)]['R_dev_CIub'] = dev_param_CIub                
                di_tup_stats_d[(p1i,p2i)]['m_lin'] = lin_param_bs[0]
                di_tup_stats_d[(p1i,p2i)]['b_lin'] = lin_param_bs[1]
                di_tup_stats_d[(p1i,p2i)]['r_lin'] = np.mean(lin_param_arr[:,2]**2)
                di_tup_stats_d[(p1i,p2i)]['p_lin'] = np.median(lin_param_arr[:,3])
                di_tup_stats_d[(p1i,p2i)]['mag_act'] = np.mean(mag_act_l)
                di_tup_stats_d[(p1i,p2i)]['mag_lin'] = np.mean(mag_lin_l)
                di_tup_stats_d[(p1i,p2i)]['mag_est'] = np.mean(mag_est_l)
                di_tup_stats_d[(p1i,p2i)]['dev_est'] = np.mean(dev_est_l)
                di_tup_stats_d[(p1i,p2i)]['dev_lin'] = np.mean(dev_lin_l)
                di_tup_stats_d[(p1i,p2i)]['mag_act_sig'] = np.std(mag_act_l)
                di_tup_stats_d[(p1i,p2i)]['mag_lin_sig'] = np.std(mag_lin_l)
                di_tup_stats_d[(p1i,p2i)]['mag_est_sig'] = np.std(mag_est_l)
                di_tup_stats_d[(p1i,p2i)]['dev_est_sig'] = np.std(dev_est_l)
                di_tup_stats_d[(p1i,p2i)]['dev_lin_sig'] = np.std(dev_lin_l)
            else:
                iinit = init_mean
                iend = end_mean
                ixend = xend_mean
                ifin = fin_mean
                linp = (iend+ixend-iinit)*WT
                estp = linp + est_df.loc[(p1i,p2i)]
                actp = ifin*WT
                mag_act = np.linalg.norm(actp)
                mag_lin = np.linalg.norm(linp)
                mag_est = np.linalg.norm(estp)
                dev_est = np.linalg.norm(estp-actp)
                dev_lin = np.linalg.norm(linp-actp)
                m_lin,b_lin,r_lin,p_lin,std_lin = linregress(linp,actp)
                m_est,b_est,r_est,p_est,std_est = linregress(estp,actp)
                di_tup_stats_d[(p1i,p2i)]['m_est'] = m_est
                di_tup_stats_d[(p1i,p2i)]['b_est'] = b_est
                di_tup_stats_d[(p1i,p2i)]['r_est'] = r_est**2
                di_tup_stats_d[(p1i,p2i)]['p_est'] = p_est
                di_tup_stats_d[(p1i,p2i)]['m_est_zv'] = np.nan
                di_tup_stats_d[(p1i,p2i)]['b_est_zv'] = np.nan
                di_tup_stats_d[(p1i,p2i)]['R_dev_avg'] = 1-dev_est**2/dev_lin**2
                di_tup_stats_d[(p1i,p2i)]['R_dev_med'] = np.nan
                di_tup_stats_d[(p1i,p2i)]['R_dev_CIlb'] = np.nan
                di_tup_stats_d[(p1i,p2i)]['R_dev_CIub'] = np.nan
                di_tup_stats_d[(p1i,p2i)]['m_lin'] = m_lin
                di_tup_stats_d[(p1i,p2i)]['b_lin'] = b_lin
                di_tup_stats_d[(p1i,p2i)]['r_lin'] = r_lin**2
                di_tup_stats_d[(p1i,p2i)]['p_lin'] = p_lin
                di_tup_stats_d[(p1i,p2i)]['m_lin_zv'] = np.nan
                di_tup_stats_d[(p1i,p2i)]['b_lin_zv'] = np.nan
                di_tup_stats_d[(p1i,p2i)]['mag_act'] = mag_act
                di_tup_stats_d[(p1i,p2i)]['mag_lin'] = mag_lin
                di_tup_stats_d[(p1i,p2i)]['mag_est'] = mag_est
                di_tup_stats_d[(p1i,p2i)]['dev_est'] = dev_est
                di_tup_stats_d[(p1i,p2i)]['dev_lin'] = dev_lin
                di_tup_stats_d[(p1i,p2i)]['mag_act_sig'] = np.nan
                di_tup_stats_d[(p1i,p2i)]['mag_lin_sig'] = np.nan
                di_tup_stats_d[(p1i,p2i)]['mag_est_sig'] = np.nan
                di_tup_stats_d[(p1i,p2i)]['dev_est_sig'] = np.nan
                di_tup_stats_d[(p1i,p2i)]['dev_lin_sig'] = np.nan                
    di_tup_dev_ser = pd.Series(di_tup_dev_d)
    di_tup_est_df = pd.DataFrame(di_tup_est_d).T
    di_tup_lin_df = pd.DataFrame(di_tup_lin_d).T
    di_tup_stats_df = pd.DataFrame(dict(di_tup_stats_d)).T
    di_tup_states_df = pd.DataFrame(di_tup_states_d).T
    pd.concat([pdi_tup_dev_ser,di_tup_dev_ser]).to_pickle('%s/r2_devs_lopo%s.pkl' % (OUTPUT_PREFIX,ver))
    pd.concat([pdi_tup_est_df,di_tup_est_df]).to_pickle('%s/predictions_lopo%s.pkl' % (OUTPUT_PREFIX,ver))
    pd.concat([pdi_tup_lin_df,di_tup_lin_df]).to_pickle('%s/predictions_lin%s.pkl' % (OUTPUT_PREFIX,ver))
    pd.concat([pdi_tup_stats_df,di_tup_stats_df]).to_pickle('%s/lopo_stats%s.pkl' % (OUTPUT_PREFIX,ver))
    pd.concat([pdi_tup_states_df,di_tup_states_df]).to_pickle('%s/lopo_states%s.pkl' % (OUTPUT_PREFIX,ver))
    return


if __name__ == '__main__':
    calc_dep_indep('%s/drug_bc_corr_data%s.gctx' % (DATA_PREFIX,ver),
                   '%s/deviations_2ord_corr%s.pkl' % (OUTPUT_PREFIX,ver))        
    LOPO('%s/drug_bc_corr_data%s.gctx' % (DATA_PREFIX,ver))
