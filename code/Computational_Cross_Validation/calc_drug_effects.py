"""
calc_drug_effects.py

wd: computational_github_code/code/Computational_Cross_Validation/

Created by Thomas Wytock on 2023-04-23.
Copyright (c) 2023 Thomas Wytock. All rights reserved.

Calculates the responses to drug perturbations and creates training data.
"""

import sys, os, os.path as osp
import pandas as pd
import numpy as np
import pickle as P
import argparse
import cmapPy.pandasGEXpress.parse as parse
import cmapPy.pandasGEXpress.GCToo as GCToo
import cmapPy.pandasGEXpress.write_gctx as wg
import networkx as nx


parser = argparse.ArgumentParser(description='Generate training data from the cell-state-perturbation network for each BioProject.')

parser.add_argument('-V','--version',metavar='v',type=str,nargs=1,
                    help='version of files to use',
                    default=['_0122'])
args = parser.parse_args()
ver = args.version[0]

DATA_PREFIX = '../../data/Computational_Cross_Validation'
OUTPUT_PREFIX = '../../output/Computational_Cross_Validation'
def calc_deltas(sel_prj=[],version=''):
    """calculate the transcriptional effects of cellular perturbations
    This function saves the transcriptional responses as a .gctx file named drug_bc_corr_data.gctx
    ARGUMENTS:
        sel_prj : list of selected BioProjects. If empty, do all BioProjects
        version : version of the datasets to use
    RETURNS: 0
    """
    #pert_op_fn = 'pert_info_df%s.pkl' % version
    drug_data_fn = '%s/drug_bc_corr_data%s.gctx' % (DATA_PREFIX,version)
    gctx_op_fn = '%s/drug_delta_selcorr_df%s.gctx' % (OUTPUT_PREFIX,version)
    if osp.exists(gctx_op_fn):
        return
    drug_data_all = parse.parse(drug_data_fn)
    drug_data = drug_data_all.data_df
    col_meta_fn = '%s/SRX_metadata%s.csv.gz' % (DATA_PREFIX,version)
    col_meta = pd.read_csv(col_meta_fn,compression='gzip',index_col=0,header=0) 

    graph_d_fn = '%s/project_graph_d%s.pkl' % (OUTPUT_PREFIX,version)
    graph_d = pd.read_pickle(graph_d_fn)
    nd_data_head = ['cell_type','cell_line','treatment','genotype']
    pert_l = []
    delta_l = []
    all_inds = set([])
    for prjno,pgraph in graph_d.items():
        print(prjno)
        if len(sel_prj)>0:
            if not prjno in sel_prj:
                continue
        if prjno in ['PRJNA258013','PRJNA321560','PRJNA349164']:
            ## unclear how to assign perturbations & controls for these projects
            continue
        if prjno != 'ENCODE':
            sel_inds = col_meta[col_meta.bioproject==prjno].index
        else:
            sel_inds = col_meta[col_meta.database==prjno].index
        if sel_inds.shape[0]<1:
            ## skip if a project is missing expression data
            continue
        sel_cols = drug_data.loc[sel_inds]
        for u,v in pgraph.edges():
            pert_data_d = {'bioproject':prjno}
            sns = pgraph.edges[u,v].get('sense','none')
            pert_data_d['sense']=sns
            for head,val in zip(nd_data_head,u):
                pert_data_d['init_%s' % head]=val
            for head,val in zip(nd_data_head,v):
                pert_data_d['fin_%s' % head]=val
            pert_l.append(pert_data_d.copy())
            
            ctrl_dat,ctrl_WT = get_weighted_data(pgraph,u,sel_cols)
            all_inds|=set(ctrl_dat.columns.unique())
            test_dat,test_WT = get_weighted_data(pgraph,v,sel_cols)
            all_inds|=set(test_dat.columns.unique())
            delta_mu = test_dat.dot(test_WT) - ctrl_dat.dot(ctrl_WT)
            delta=delta_mu
            delta_l.append(delta)
    delta_df = pd.DataFrame(delta_l)
    pert_df = pd.DataFrame(pert_l)
    pert_df.index = pert_df.index.astype(str)
    delta_df.index = delta_df.index.astype(str)
    ISECT=pert_df.index.intersection(delta_df.index)
    cdd = pd.Series(dict([(did,ii) for ii, did in enumerate(delta_df.columns)])).to_frame()
    cdd.columns = ['order']
    deltaGCT = GCToo.GCToo(data_df=delta_df,row_metadata_df=pert_df.loc[ISECT],col_metadata_df=cdd)
    wg.write(deltaGCT,gctx_op_fn)
    return 0


def get_weighted_data(G,u,data):
    """Uses linear weights to account for dose-response and time-course data."""
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

def match_pert_col(pdf,lbls,vals,isCtrl=False):
    '''Find the matching perturbation information'''
    prefix = 'init' if isCtrl else 'fin'
    c_l = []
    for lbl,val in zip(lbls,vals):
        c_l.append(pdf.loc[:,'%s_%s' % (prefix,lbl)]==val)
    return c_l

def select_pert_col(c_l):
    '''Find the overlapping perturbation information'''
    c0 = c_l[0]
    for elt in c_l[1:]:
        c0 = c0&elt
    return c0

def findPaths(G,u,n,excludeSet = None):
    '''Find paths in G originating from u of length n, recursively'''
    if excludeSet == None:
        excludeSet = set([u])
    else:
        excludeSet.add(u)
    if n==0:
        return [[u]]
    paths = [[u]+path for neighbor in G.neighbors(u) if neighbor not in excludeSet for path in findPaths(G,neighbor,n-1,excludeSet)]
    excludeSet.remove(u)
    return paths

def obtain_training_data(sel_prj=[],ReCalc=False,version=''):
    """Generate the training data from the graph and gene expression data"""
    drug_data_fn = '%s/drug_bc_corr_data%s.gctx' % (DATA_PREFIX,version)
    delta_fn = '%s/drug_delta_selcorr_df%s.gctx' % (OUTPUT_PREFIX,version)
    drug_data_all = parse.parse(drug_data_fn)
    drug_data = drug_data_all.data_df
    col_meta_fn = '%s/SRX_metadata%s.csv.gz' % (DATA_PREFIX,version)
    col_meta = pd.read_csv(col_meta_fn,compression='gzip',index_col=0,header=0) 
    
    ext = 'corr%s' % version
    nd_data_head = ['cell_type','cell_line','treatment','genotype']
    drug_data_all = parse.parse(drug_data_fn)
    drug_data = drug_data_all.data_df
    delta_gctx = parse.parse(delta_fn)
    delta_df = delta_gctx.data_df
    pert_df = delta_gctx.row_metadata_df#pd.read_pickle('pert_info_df.pkl')
    graph_d_fn = '%s/project_graph_d%s.pkl' % (OUTPUT_PREFIX,version)
    graph_d = pd.read_pickle(graph_d_fn)
    deviations_d = {}
#     if not osp.exists('deviations_by_prj'):
#         os.mkdir('deviations_by_prj')
    for prjno,pgraph in sorted(graph_d.items(),key=lambda it: it[0]):
#         if len(sel_prj)>0 or not ReCalc:
#             if not prjno in sel_prj:
#                 exist_fn = 'deviations_by_prj/%s_%s.pkl' % (prjno,ext)
#                 if osp.exists(exist_fn):
#                     prj_deviations_df = pd.read_pickle(exist_fn)
#                     deviations_d.update(dict(prj_deviations_df.iteritems()))
#                 continue
        print(prjno)
        prj_deviations_d = {}
        spert_df = pert_df[pert_df.bioproject==prjno]
        if prjno != "ENCODE":
            sel_inds = col_meta[col_meta.bioproject==prjno].index
        else:
            sel_inds = col_meta[col_meta.database==prjno].index
        sel_cols = drug_data.loc[sel_inds]
        if prjno in ['PRJNA258013','PRJNA321560','PRJNA349164']:
            ## these projects are skipped because it is unclear how the perturbations should be interpreted as a graph of single perturbations
            continue
        elif sel_cols.shape[0]<1:
            ## skip any projects that do not have indices in the dataset
            continue
        for nd in pgraph.nodes():
            paths2_l = findPaths(pgraph,nd,2)
            for paths2 in paths2_l:
                try:
                    if pgraph.edges[paths2[:2]]['class']=='cell_type' and pgraph.edges[paths2[1:]]['class']=='cell_type':
                        continue
                except KeyError:
                    pass
                strt,end = paths2[0],paths2[-1]
                strt_dat,strt_WT = get_weighted_data(pgraph,strt,sel_cols)
                end_dat,end_WT = get_weighted_data(pgraph,end,sel_cols)
                delta_tot = end_dat.dot(end_WT) - strt_dat.dot(strt_WT)
                edgs = [(paths2[0],paths2[1])]
                for p2 in nx.all_simple_paths(pgraph,strt,end,2):
                    if p2[1]==paths2[1]:
                        ## must be a different path
                        continue
                    edgs.append((p2[0],p2[1]))
                if len(edgs) < 2:
                    continue
                elif len(edgs) > 2:
                    print(paths2)
                delta_l = []
                ind_l = []
                for u,v in edgs:
                    c_l1 = match_pert_col(spert_df,nd_data_head,u,True)
                    c_l2 = match_pert_col(spert_df,nd_data_head,v,False)
                    c0 = select_pert_col(c_l1+c_l2)
                    ind = spert_df[c0].index[0]
                    delta_l.append(delta_df.loc[ind])
                    ind_l.append(ind)
                deviations = delta_tot - pd.concat(delta_l,axis=1).sum(axis=1)
                prj_deviations_d[tuple(ind_l)]=deviations
        #prj_deviations_df = pd.DataFrame(prj_deviations_d)
        #prj_deviations_df.to_pickle('deviations_by_prj/%s_%s.pkl' % (prjno,ext))
        deviations_d.update(prj_deviations_d)
    deviations_df = pd.DataFrame(deviations_d).T
    if hasattr(deviations_df.index,'levels') and len(deviations_df.index.levels)>2:
        deviations_df = deviations_df[deviations_df.index.get_level_values(2).isna()]
        deviations_df.index =deviations_df.index.droplevel(2)
    deviations_df.to_pickle('%s/deviations_2ord_%s.pkl' % (OUTPUT_PREFIX,ext))
    return

if __name__ == '__main__':
    calc_deltas(sel_prj=[],version=ver)
    obtain_training_data(sel_prj=[],ReCalc=True,version=ver)
    