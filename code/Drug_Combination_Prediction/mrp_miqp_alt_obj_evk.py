'''
mrp_miqp_alt_obj_evk.py

directory: ./code/Drug_Combination_Prediction/
Created by Thomas Wytock on 2023-04-26.
Copyright (c) 2023 Thomas Wytock. All rights reserved.

This code uses MIQP to identify drug pairs (additive method).
'''
## 
## should be the py36 environment so that cplex works

import sys, os, os.path as osp
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import argparse
import numpy as np
import pandas as pd
import time
from scoop import futures
from sklearn.neighbors import KNeighborsClassifier as KNC
from functools import partial
from scipy.stats import percentileofscore as POS
import cplex
from operator import itemgetter as ig
from glob import glob
import cmapPy.pandasGEXpress.parse as parse
import cmapPy.pandasGEXpress.concat as concat

## parse some command line arguments
parser = argparse.ArgumentParser(description='Select perturbations that are best for treating cancers using the additive method.')
parser.add_argument('-d','--dataset',metavar='D',type=str,nargs=1,help='Name of the dataset',
                    choices=['lincs','Darmanis_et_al_2017','Neftel_et_al_2019',
                             'Couturier_et_al_2020','CGGA','Richards_et_al_2021'], default=['lincs'])
parser.add_argument('-n','--num_feat',metavar='n',type=int,nargs=1,
                    help='Number of features used in the model',
                    default=[37])
parser.add_argument('-f','--feature_file',metavar='f',type=str,nargs=1,
                    help='Name of pickle file storing the features',
                    default=['drug_feat_altobj_l'])
parser.add_argument('-V','--version',metavar='v',type=str,nargs=1,
                    help='version of files to use',
                    default=['_1121'])
parser.add_argument('-Z','--save-version',metavar='z',type=str,nargs=1,
                    help='version of files to use',
                    default=['_0122'])
parser.add_argument('-N','--num_pert',metavar='N',type=int,nargs=1,
                    help='Number of perturbations composing the combination',
                    default=[2])
parser.add_argument('-C','--cost_factor',metavar='c',type=float,nargs=1,
                    help='Number to balance the contributions of the objective function',
                    default=[0.5])
args = parser.parse_args()
dis = args.dataset[0]
NG = args.num_pert[0]
costF = args.cost_factor[0]*np.pi/2.
ver = args.version[0]
ver = '' if not ver.startswith('_') else ver
over = '_1021' if ver!='' else ''
ver2 = args.save_version[0]
DATA_PREFIX = '../../data/Drug_Combination_Prediction'
OUTPUT_PREFIX = '../../output/Drug_Combination_Prediction'
try:
    if dis != 'lincs':
        newCAT_map = pd.read_pickle('%s/%s/cats.pkl' % (DATA_PREFIX,dis))
        oldCAT = pd.read_pickle('%s/lincs/cats.pkl' % (DATA_PREFIX))
        PAIRS = pd.read_table('%s/%s/pairs.tsv' % (DATA_PREFIX,dis))
        newSTATES = pd.read_table('%s/%s/states.tsv' % (DATA_PREFIX,dis), index_col=0)
        oldSTATES = pd.read_table('%s/lincs/states.tsv' % (DATA_PREFIX))
        oldSTATES.loc[:,'state_number']+=newSTATES.index.max()+1
        oldCAT2 = pd.Series(dict(zip(oldCAT.index,oldSTATES.state_name.loc[oldCAT.values].values)))
        oldSTATES = oldSTATES.set_index('state_number')
        rrdr = np.hstack([newSTATES.state_name.values,oldSTATES.state_name.values])
        CAT_map = pd.concat([newCAT_map,oldCAT2])
        STATES = pd.concat([newSTATES,oldSTATES])
        qn_df = pd.read_pickle('%s/%s/qncorr_data.pkl' % (DATA_PREFIX,dis))
    else:
        CAT_map = pd.read_pickle('%s/%s/cats.pkl' % (DATA_PREFIX,dis))
        PAIRS =  pd.read_table('%s/%s/pairs.tsv' % (DATA_PREFIX,dis))
        STATES =  pd.read_table('%s/%s/states.tsv' % (DATA_PREFIX,dis),index_col=0)  
except FileNotFoundError:
    print('[ERROR] %s/%s/cats.pkl was not found.' % (DATA_PREFIX,dis))
    raise
#combined_data.pkl
try:
    feat_l = pd.read_pickle('%s/%s.pkl' % (DATA_PREFIX,args.feature_file[0]))
except FileNotFoundError:
    print('[ERROR] %s/%s.pkl was not found.' % (DATA_PREFIX,args.feature_file[0]))
    raise

if np.abs(args.num_feat[0])>len(feat_l):
    print('[WARNING] The number of features requested is larger than the number of selected features.\nUsing all available features.')
if args.num_feat[0]<0:
    feat_l = feat_l[args.num_feat[0]:]
elif args.num_feat[0]>0:
    feat_l = feat_l[:args.num_feat[0]]

OUT = 'MIQP_MRP_ALT_OBJ_EVK'
if not osp.exists('%s/%s/%s' % (OUTPUT_PREFIX,dis,OUT)):
    os.makedirs('%s/%s/%s' % (OUTPUT_PREFIX,dis,OUT),exist_ok=True)

n_neighbors=9

gctoo_l = []
for ii in range(4):
    gctoo = parse.parse('%s/combined_data%s_part%d.gctx' % (DATA_PREFIX,over,ii+1))
    gctoo_l.append(gctoo)
drug_data_trim = concat.vstack(gctoo_l)    

def weights(df):
    WT = df.std(axis=0,ddof=1).apply(lambda i: 1/i)
    if np.any(np.isinf(WT)):
        M = np.amax(WT[~np.isinf(WT)])
        WT[np.isinf(WT)]=2*M
    WT = WT.div(np.linalg.norm(WT.astype(np.float64)))
    return WT

if dis=='lincs':
    drug_data_trim = drug_data_trim.loc[CAT_map.index]
    WT = weights(drug_data_trim)
    scaled_states =drug_data_trim.multiply(WT)
else:
    WT = weights(drug_data_trim)
    drug_data_comb = pd.concat([qn_df,drug_data_trim],axis=0)
    drug_data_trim = drug_data_comb
    scaled_states =drug_data_comb.multiply(WT)
    scaled_states = scaled_states.loc[CAT_map.index]
    drug_data_trim = drug_data_trim.loc[CAT_map.index]
    
delta_fn = '%s/combined_perturbations%s.pkl' % (DATA_PREFIX,ver)
delta_trim = pd.read_pickle(delta_fn).T
KERN = pd.read_pickle('%s/alt_obj_kernel.pkl' % DATA_PREFIX)

## filter out any uncharacterized compounds
cmp_info = pd.read_table('%s/compoundinfo_beta.txt' % DATA_PREFIX)
nuid_data_table = pd.read_excel('%s/updated_casmapping_050422_data.xlsx' % DATA_PREFIX,engine='openpyxl',index_col=0)
charac = cmp_info[cmp_info.pert_id.isin(nuid_data_table.BRD.unique())|
                  cmp_info.cmap_name.isin(nuid_data_table.LINCS_NAME.unique())].pert_id.unique()

nonlincs_nuid_table = pd.read_excel('%s/non_lincs_drugs_casmapping.xlsx' % DATA_PREFIX,engine='openpyxl',index_col=0)
linds = [ii.lower() for ii in nonlincs_nuid_table.index]

sel_l = []
for elt in delta_trim.columns:
    if elt.endswith('lincs'):
        if elt.split('---')[0] in charac:
            sel_l.append(True)
        elif 'oe' in elt:
            sel_l.append(True)
        elif 'kd' in elt:
            sel_l.append(True)
        elif 'mut' in elt:
            sel_l.append(True)
        else:
            sel_l.append(False)
    else:
        if elt.lower() in linds:
            sel_l.append(True)
        elif 'oe' in elt:
            sel_l.append(True)
        elif 'kd' in elt:
            sel_l.append(True)
        elif 'mut' in elt:
            sel_l.append(True)
        else:
            sel_l.append(False)
delta_trim = delta_trim.T[sel_l].T


drug_cols =[col for col in delta_trim.columns 
         if not col.split('---')[-1].endswith(('oe','kd','mut','oe_lincs','kd_lincs','mut_lincs')) 
         and not col.endswith(('starvation','stimulation','Sulindac','differentiation','+','-'))]
delta_trim = delta_trim.loc[:,drug_cols]


delta_data = delta_trim.loc[feat_l] #delta_trim.data_df.loc[feat_l]
ctGB = CAT_map.to_frame().groupby(0)
CAT = pd.Categorical(CAT_map)

clf_ct = KNC(n_neighbors, weights='distance',n_jobs=4)
clf_ct.fit(scaled_states.values,CAT_map.loc[scaled_states.index].values)

def cplex_optimize_miqp(Q,cvec,**kwargs):
    '''
    interface to perform constrained optimization with CPLEX

    Parameters
    ----------
    Q : 2-D, square, numpy array
        Quadratic component of the objective function
    cvec : numpy array
        Linear component of the objective function; same length as Q
    kwargs : dict of optional arguments
        lb : lower bounds for the variables
        ub : upper bounds for the variables
        
    Returns
    -------
    x : list
        optimal values of the control inputs
    obj : float
        objective function at optimum
    lpstat : int
        status of the cplex solver

    '''
    Q = 0.5*(Q+Q.T)
    Q = np.atleast_2d(Q) if not isinstance(Q,np.ndarray) else Q
    numvars = Q.shape[0];
    Q = Q + np.eye(numvars)*1e-12
    cvec = cvec[:,0]
    qmat = []
    for ii in range(2*numvars):
        if ii < numvars:
            qmat.append([['x%d'%(xx+1) for xx in range(numvars)],Q[ii,:].tolist()])
        else:
            qmat.append([['z%d' % (ii-numvars+1) ],[1e-12]])
    c = cplex.Cplex()
    out = c.set_results_stream(None)
    out = c.set_log_stream(None)
    out = c.set_error_stream(None)
    out = c.set_warning_stream(None)
    c.parameters.threads.set(2)
    c.parameters.optimalitytarget.set(c.parameters.optimalitytarget.values.optimal_global)
    varnames = ['x%d' % (ii+1) for ii in range(numvars)]
    indvarnames = ['z%d' % (ii+1) for ii in range(numvars)]
    if 'lb' in kwargs.keys() and 'ub' in kwargs.keys():
        LB = kwargs['lb']
        LB = LB if isinstance(LB,np.ndarray) else np.ones(numvars)*LB
        LB = LB.tolist()
        UB = kwargs['ub']
        UB = UB if isinstance(UB,np.ndarray) else np.ones(numvars)*UB
        UB = UB.tolist()
        c.variables.add(obj=cvec.tolist(),ub=UB,lb=LB,names=varnames)
    elif 'lb' in kwargs.keys():
        LB = kwargs['lb']
        LB = LB if isinstance(LB,np.array) else np.ones(numvars)*LB
        LB = LB.tolist()
        c.variables.add(obj=cvec.tolist(),lb=LB,names=varnames)
    elif 'ub' in kwargs.keys():
        UB = kwargs['ub']
        UB = UB if isinstance(UB,np.array) else np.ones(numvars)*UB
        UB = UB.tolist()
        c.variables.add(obj=cvec.tolist(),ub=UB,names=varnames)
    else:
        c.variables.add(obj=cvec.tolist(),names=varnames)
    #Nsol = kwargs['Nsol'] if 'Nsol' in kwargs.keys() else 10
    c.variables.add(types=[c.variables.type.binary]*numvars ,names=indvarnames)
    for ii in range(numvars):
        c.indicator_constraints.add(indvar='z%d' % (ii+1), 
                                    complemented=1,rhs=0,sense='E',
                                    lin_expr=cplex.SparsePair(ind=['x%d' % (ii+1)],val=[1]),name='ind%d' % (ii+1))
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=indvarnames,val=np.ones(numvars))],
                             rhs=[kwargs['NG']],senses=['E'],names=['lconst'])
    c.objective.set_sense(c.objective.sense.minimize)
    c.objective.set_quadratic(qmat)
#     x_l = []
#     obj_l = []
#     stat_l = []
#     for ii in range(Nsol):
#         c.solve()
#         try:
#             x = np.asarray(c.solution.get_values()); x_l.append(x)
#         except cplex.exceptions.errors.CplexSolverError:
#             print('No more solutions. Stopping.')
#             break            
#         lpstat = c.solution.get_status(); stat_l.append(lpstat)
#         obj = c.solution.get_objective_value(); obj_l.append(obj)
#         siv_nms = c.variables.get_names(list(np.where(x>0)[0][-1*kwargs['NG']:].astype(int)))
#         c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=siv_nms,val=np.ones(len(siv_nms)))],
#                                  rhs=[kwargs['NG']-0.5],senses=['L'],names=['econst%d' % ii])
#     else:
    c.solve()
    x = np.asarray(c.solution.get_values())
    lpstat = c.solution.get_status()
    obj = c.solution.get_objective_value()
    return x,obj,lpstat

def n_it(ng,N,n,t=0.05):
    prd = 1
    for ii in range(ng):
        prd *= (n-ii)/(N-ii)
    return np.ceil(np.log(t)/np.log(1-prd)).astype(int)

def miqp_avg(S_T,ng,ctf,STAT_IND,ub_ser=pd.Series(),PS=500,PARAM=None):
    S,T,gsm_fn = S_T
    (gsmi,gsmf),dat_ser,inTarg = miqp_mrp(T,S,ng,ctf)
    stats = dat_ser.iloc[:STAT_IND]
    alser = dat_ser.iloc[STAT_IND:]
    for jj,(gn,val) in enumerate(alser.iteritems()):
        stats['pert_%d' % (jj+1)] = gn
        stats['al_%d' % (jj+1)] = val
    stats['GSM_i']=gsmi
    stats['GSM_f']=gsmf
    stats.to_pickle(gsm_fn)
    return (gsmi,gsmf),stats

def miqp_mrp(T,S,ng,ctf,ub_ser=pd.Series(),PS=500,PARAM=None):
    ## global variable ch is used here
    gsmi,S = S
    gsmf,T = T
    if dis != 'lincs':
        ctigrp = np.where(CAT.categories==CAT_map.loc[gsmi])[0][0]
        ctfgrp = np.where(CAT.categories==ctf)[0][0]
    else:
        ctigrp = CAT_map.loc[gsmi]
        ctfgrp = CAT_map.loc[gsmf]
    ## still need to pass the correct matrix/vector to the CPLEX routine
    kwargs = {'NG':ng}
    kwargs['lb'] = 0
    kwargs['ub'] = 1
    ## need to add additional elements to the vector here
    wsep = (S-T).multiply(WT)
    SWT = S.multiply(WT)
    TWT = T.multiply(WT)
    DENOM = np.linalg.norm(SWT.dot(KERN).dot(SWT))
    if PARAM is None:
        PARAM = np.linalg.norm(wsep)/DENOM
        #PARAM=np.linalg.eigvalsh((delta_data@delta_data * WT).T*WT)[-1]/np.linalg.eigvalsh(KERN)[-1]
    C1sqr = np.cos(costF)**2
    C2sqr = (PARAM*np.sin(costF))**2
    kernel = (C1sqr*np.eye(KERN.shape[0])+C2sqr*KERN)
    wsep = wsep.to_frame()
    if PS > delta_data.shape[1]:
        N_IT=1
        PS = delta_data.shape[1]
    else:
        N_IT = n_it(ng,delta_data.shape[1],PS)
    OBJ = 1e10
    opt_sel_cols = np.zeros(PS)
    opt_al = np.zeros(PS)
    fixed_cols = []
    it_without_imp = 0
    for nn in range(N_IT):
        sel_cols2 = np.random.permutation(np.arange(delta_data.shape[1]))[:PS]
        if len(set(fixed_cols)-set(sel_cols2))>0:
            diff = list(set(fixed_cols)-set(sel_cols2))
            diff2 = list(set(sel_cols2)-set(fixed_cols))
            isect = list(set(fixed_cols)&set(sel_cols2))
            tmpl = diff2[len(diff)-len(isect):]+diff
            if len(tmpl)<PS:
                LL = list(set(np.arange(delta_data.shape[1]))-set(tmpl))
                tmpl += LL[:(PS-len(tmpl))]
            sel_cols = np.asarray(tmpl)
        else:
            sel_cols = sel_cols2
        rsel_deltas = delta_data.iloc[:,sel_cols]
        ## note the changes here -- need to triple check
        wDx = rsel_deltas.T.multiply(WT).T
        wQx = wDx.T@kernel@wDx
        wCx = (wsep.T@kernel+ C2sqr*TWT@KERN)@wDx
        al,obj,stat = cplex_optimize_miqp(wQx.values,wCx.T.loc[wQx.index].values,**kwargs)
        
        nzind = np.where(np.asarray(al[:len(al)//2])>1e-6)[0]
        if nzind.shape[0]>0:
            inds = np.where(np.asarray(al[:len(al)//2])>1e-6)[0]
            vals = [al[ind] for ind in inds]
        if obj<OBJ:
            it_without_imp = 0
            opt_al = np.array(al[:PS]) ## discard the indicator variables
            OBJ = obj
            opt_sel_cols = sel_cols
            ocnms = [elt for elt in opt_sel_cols[np.abs(opt_al)>1e-6]]
            if len(fixed_cols)<int(.05*PS)-len(set(ocnms)-set(fixed_cols)):
                fixed_cols.extend(ocnms)
            else:
                N = len(set(ocnms)-set(fixed_cols))+len(fixed_cols)-int(.05*PS)
                fixed_cols = fixed_cols[N:]+list(set(ocnms)-set(fixed_cols))
        else:
            it_without_imp+=1
            if it_without_imp>10:
                break
    wD1 = delta_data.T.multiply(WT).T            
    opt_ser= pd.Series(np.zeros(delta_data.shape[1]),index=delta_data.columns)
    for sci,ali in zip(opt_sel_cols[np.abs(opt_al)>1e-6],
                       opt_al[np.abs(opt_al)>1e-6]):
        opt_ser.iloc[sci]=ali
    al_df = opt_ser#.to_frame() #pd.DataFrame(al,index=wQ.index)
    
    R = SWT + wD1@al_df
    d0 = np.linalg.norm((S-T).multiply(WT))
    d1 = np.linalg.norm((S-T).multiply(WT)+wD1@al_df)
    pf = 1-d1/d0
    g0 = SWT.dot(KERN).dot(SWT)
    g1 = R.dot(KERN).dot(R)
    gpf = 1-g1/g0
    Cst = clf_ct.predict(R.to_frame().T.values)[0]
    if dis != 'lincs':
        C = np.where(CAT.categories==Cst)[0][0]
    else:
        C = Cst
        Cst = STATES.loc[Cst,'state_name']
    P,I,F = clf_ct.predict_proba(R.to_frame().T.values)[0,[C,ctigrp,ctfgrp]]
    OBJ = d1*d1*C1sqr + C2sqr*g1
    dat_ser=pd.Series([OBJ,d1,pf,g1,gpf,Cst,P,I,F],index=
                      ['OBJ','D_F','P_F','G_F','PG_F','KNN_GP',
                       'C_PRB','I_PRB','F_PRB'])
    nz_data = al_df[al_df>1e-6]
    dat_ser = dat_ser.append(nz_data)
    #dat_df = ptd.concat(tdat_ser_l,axis=1)
    return (gsmi,gsmf),dat_ser,dat_ser.KNN_GP == ctfgrp

def get_comp_gsms(d):
    fn_l = glob(d+'/*.pkl')
    df_l = []
    for fn in fn_l:
        df_l.append(pd.read_pickle(fn))
    return list(zip([fn.split('/')[-1].split('.')[0] for fn in fn_l],df_l))

def pandas_MIQP(cti,cti_df,ctf,ctf_df,comp_df=pd.DataFrame(),kind='AVG'):
    print("Starting cell type pair %s -> %s" % (cti,ctf))
    ctf_basin = STATES[STATES.state_name==ctf].index[0]
    ctistr = cti.replace(' - ','_')
    ctfstr = ctf.replace(' - ','_')
    cont = pd.Series(dict([((ii,ff),False) for ii in cti_df.index for ff in ctf_df.index]))
    DF_l = []
    STAT_LBLS = ['OBJ','D_F','P_F','G_F','PG_F','KNN_GP','C_PRB','I_PRB','F_PRB']
    STAT_IND = len(STAT_LBLS)


    if not osp.exists('%s/%s/%s/%s' % (OUTPUT_PREFIX,dis,OUT,ctistr)):
        os.makedirs('%s/%s/%s/%s' % (OUTPUT_PREFIX,dis,OUT,ctistr),exist_ok=True)
    if kind=='AVG':
        input_l = []
    for ELT in cti_df.iterrows():
        if kind=='MRP':
            gsm_fn = '%s/%s/%s/%s/%s-%s-%.2f_%d%s.pkl' % (OUTPUT_PREFIX,dis,OUT,ctistr,ELT[0],ctfstr,costF,NG,ver2)
    #         if osp.exists(gsm_fn):
    #             df = pd.read_pickle(gsm_fn)
    #             DF_l.append(df)
    #             continue
            part_miqp_mrp = partial(miqp_mrp,S=ELT,ng=NG,ctf=ctf)
            #isTarg = clf_ct.predict(ctf_df.multiply(WT))[0]==ctf_basin:
            SEL = cont.loc[[(ELT[0],fnm) for fnm in ctf_df.index]]
            gsm_res_df_l= list(futures.map(part_miqp_mrp,ctf_df[~SEL.values].iterrows()))
            stats_l = []
            for (gsmi,gsmf),ser,TF in gsm_res_df_l:
                stats = ser.iloc[:STAT_IND]
                alser = ser.iloc[STAT_IND:]
                for jj,(gn,val) in enumerate(alser.iteritems()):
                    stats['pert_%d' % (jj+1)] = gn
                    stats['al_%d' % (jj+1)] = val
                stats['GSM_i']=gsmi
                stats['GSM_f']=gsmf
                stats_l.append(stats)
                cont.loc[(gsmi,gsmf)]=TF
            df = pd.concat(stats_l,axis=1).T
            DF_l.append(df)
            df.to_pickle(gsm_fn)
        elif kind=='AVG':
            if ctf_df.shape[0]>1:
                avg_dists = select_ctf_target(ctf_df,ELT[1])
                sel_targ = ctf_df.iloc[np.argmin(avg_dists)]
                tname = sel_targ.name
            else:
                sel_targ=ctf_df.iloc[0]
                tname = sel_targ.name
            gsm_fn = '%s/%s/%s/%s/%s-%s-%.2f_%d%s.pkl' % (OUTPUT_PREFIX,dis,OUT,ctistr,ELT[0],ctfstr,costF,NG,ver2)
    #         if osp.exists(gsm_fn):
    #             df = pd.read_pickle(gsm_fn)
    #             DF_l.append(df)
    #             continue
            input_l.append((ELT,(tname,sel_targ),gsm_fn))                
    if kind=='AVG':
        part_miqp_avg = partial(miqp_avg,ng=NG,ctf=ctf,STAT_IND=STAT_IND)
            #isTarg = clf_ct.predict(ctf_df.multiply(WT))[0]==ctf_basin:
        gsm_res_df_l= list(futures.map(part_miqp_avg,input_l))
        stats_l = []
        for (gsmi,gsmf),ser in gsm_res_df_l:
            #stats = ser.iloc[:STAT_IND]
            #alser = ser.iloc[STAT_IND:]
            #for jj,(gn,val) in enumerate(alser.iteritems()):
            #    stats['pert_%d' % (jj+1)] = gn
            #    stats['al_%d' % (jj+1)] = val
            #stats['GSM_i']=gsmi
            #stats['GSM_f']=gsmf
            stats_l.append(ser)
            #cont.loc[(gsmi,gsmf)]=TF
        df = pd.concat(stats_l,axis=1).T
        DF_l.append(df)
    if comp_df.shape[0] > 1:
        DF_l.append(comp_df)
    DF = pd.concat(DF_l,axis=0)
    DF['CTI']=[cti for _ in range(DF.shape[0])]
    DF['CTF']=[ctf for _ in range(DF.shape[0])]
    ng_fn = '%s/%s/%s/%s-%s-%.2f_%d%s.pkl' % (OUTPUT_PREFIX,dis,OUT,ctistr,ctfstr,costF,NG,ver2)
    DF.to_pickle(ng_fn)
    return 0

def select_ctf_target(V,u):
    from scipy.spatial.distance import pdist
    DM = pdist((V-u),metric='cosine')
    cf = partial(condensed_formula,mm=V.shape[0])
    #RCS = np.cumsum(DM[cf(np.zeros(V.shape[0]-1,dtype=int),np.arange(1,V.shape[0],dtype=int))][::-1])
    rowsums = np.zeros(V.shape[0])
    for ii in range(V.shape[0]):
        A1 = np.c_[np.arange(ii),np.ones(ii)*ii]
        A2 = np.c_[np.ones(V.shape[0]-ii-1)*ii,np.arange(ii+1,V.shape[0])]
        ind_arr = np.r_[A1,A2].astype(int)
        rowsums[ii] = np.sum(DM[cf(ind_arr[:,0],ind_arr[:,1])])
    return rowsums/(V.shape[0]-1)

def condensed_formula(ii,jj,mm):
    return ii*mm-((ii+1)*(ii+2))//2 + jj
    
def MIQP_MRP(STHR=10):
    ### need to get this and forward selection corrected
    for _dum,row in PAIRS.iloc[::-1].iterrows():
        num_i,num_f = row.loc['disease_state'],row.loc['target_state']
        cti = STATES.loc[num_i,'state_name']
        ctf = STATES.loc[num_f,'state_name']
            
        if dis != 'lincs':
            targ_data = drug_data_trim.loc[ctGB.get_group(ctf).index]
            #MU = targ_data.median().to_frame().T
            init_data = drug_data_trim.loc[ctGB.get_group(cti).index]
        else:
            targ_data = drug_data_trim.loc[ctGB.get_group(num_f).index]
            #MU = targ_data.median().to_frame().T
            init_data = drug_data_trim.loc[ctGB.get_group(num_i).index]
        ctist = cti.replace(' - ','_')
        ctfst = ctf.replace(' - ','_')
        if cti == ctf: continue
        if init_data.shape[0]>STHR:
            rnd_inds = np.random.permutation(init_data.index)[:STHR]
            init_data = init_data.loc[rnd_inds]        
        z = pandas_MIQP(cti,init_data,ctf,targ_data,kind='AVG')
        print("%s calculated" % (cti))

def main():
    MIQP_MRP()

if __name__ == '__main__':
    main()
