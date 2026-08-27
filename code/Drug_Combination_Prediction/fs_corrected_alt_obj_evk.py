"""
fs_corrected_alt_obj_evk.py
directory: ./code/Drug_Combination_Prediction/
Created by Thomas Wytock on 2023-04-28.
Copyright (c) 2023 Thomas Wytock. All rights reserved.

This code performs forward selection to identify drug pairs (nonadditive method).
"""

## should be the py36 environment so that cplex works
import sys, os, os.path as osp
os.environ["OMP_NUM_THREADS"] = "2" 
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
import argparse
import numpy as np
import pandas as pd
import time
from scoop import futures
from sklearn.neighbors import KNeighborsClassifier as KNC, KNeighborsRegressor as KNR
from functools import partial
from scipy.stats import percentileofscore as POS
import cplex
from itertools import permutations
from glob import glob
import cmapPy.pandasGEXpress.parse as parse
import cmapPy.pandasGEXpress.concat as concat

## parse some command line arguments
parser = argparse.ArgumentParser(description='Select perturbations that are best for treating cancers.')
parser.add_argument('-d','--dataset',metavar='D',type=str,nargs=1,
                    help='Name of the dataset',
                    choices=['lincs','Darmanis_et_al_2017','Neftel_et_al_2019',
                             'Couturier_et_al_2020','CGGA','Richards_et_al_2021'],
                    default=['lincs'])
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
                    help='version of files to save to',
                    default=['_0122'])
parser.add_argument('-N','--num_pert',metavar='N',type=int,nargs=1,
                    help='Number of perturbations composing the combination',
                    default=[2])
parser.add_argument('-C','--cost_factor',metavar='c',type=float,nargs=1,
                    help='Number to balance the contributions of the objective function',
                    default=[0.5])
parser.add_argument('-R','--no_recalc',action="store_true",
                    help='Restart using previously calculated gsmf/gsmi pairs')

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
        PAIRS =  pd.read_table('%s/%s/pairs.tsv' % (DATA_PREFIX,dis))
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
    print('[ERROR] %s/cats.pkl was not found.' % dis)
    print('Possible files:')
    for fn in glob('*/cats.pkl'):
        print(fn)
    raise

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

OUT = 'FS_ALT_OBJ_EVK'
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

clf_ct = KNC(n_neighbors, weights='distance',n_jobs=1)
clf_ct.fit(scaled_states.values,CAT_map.loc[scaled_states.index].values)
## still need the CPLEX optimization, but always will be 1 at a time; 
## treat previous perts as fixed
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
        topN : how many solutions to keep
        
    Returns
    -------
    x : list of lists
        optimal values of the control inputs for each of the topN cases
    obj : float
        objective function at optimum for each of the topN cases
    lpstat : int
        status of the cplex solver for each of the topN cases

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
    c.parameters.threads.set(2)
    out = c.set_results_stream(None)
    out = c.set_log_stream(None)
    out = c.set_error_stream(None)
    out = c.set_warning_stream(None)
    c.parameters.optimalitytarget.set(c.parameters.optimalitytarget.values.optimal_global)
    varnames = ['x%d' % (ii+1) for ii in range(numvars)]
    indvarnames = ['z%d' % (ii+1) for ii in range(numvars)]
    topN = kwargs.get('topN',1)
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
    c.variables.add(types=[c.variables.type.binary]*numvars ,names=indvarnames)
    for ii in range(numvars):
        c.indicator_constraints.add(indvar='z%d' % (ii+1), 
                                    complemented=1,rhs=0,sense='E',
                                    lin_expr=cplex.SparsePair(ind=['x%d' % (ii+1)],val=[1]),name='ind%d' % (ii+1))
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=indvarnames,val=np.ones(numvars))],
                             rhs=[kwargs['NG']],senses=['E'],names=['lconst'])
    c.objective.set_sense(c.objective.sense.minimize)
    c.objective.set_quadratic(qmat)
    x_l = []; obj_l = []; stat_l = []
    for ii in range(topN):
        c.solve()
        lpstat = c.solution.get_status(); stat_l.append(lpstat)
        try:
            x = c.solution.get_values(); x_l.append(x)
            obj = c.solution.get_objective_value(); obj_l.append(obj)
            opt_ind = int(np.where(np.array(x)>1e-6)[0][0])
            c.variables.set_upper_bounds(opt_ind,0)
        except cplex.exceptions.errors.CplexSolverError:
            print('No more solutions. Stopping.')
            break
        ## sets the upper bound of the previous optimum to zero
    else:
        c.solve()
        lpstat = c.solution.get_status(); stat_l.append(lpstat)
        try:
            x = c.solution.get_values(); x_l.append(x)
            obj = c.solution.get_objective_value(); obj_l.append(obj)
        except cplex.exceptions.errors.CplexSolverError:
            print('No more solutions. Stopping.')
    return x_l,obj_l,stat_l

## code to make the prediction model
cols1 = [elt+'_s' for elt in feat_l]
cols2 = [elt+'_p' for elt in feat_l]
cols = cols1+cols2

## need to run process_perturbations.py to create these files
## function: construct_deviations 
##     note that only some of the deviations are used, 
#      so there is some random fluctuation depending on the deviations chosen.
indep_var = pd.read_pickle('%s/lincs_indep_vars%s.pkl' % (OUTPUT_PREFIX,'')).T
indep_var.columns = cols
dep_var = pd.read_pickle('%s/lincs_deviations%s.pkl' % (OUTPUT_PREFIX,'')).T
dep_var.columns=feat_l
wt_vec = {}
for k,v in WT.iteritems():
    wt_vec[k+'_s']=v
for k,v in WT.iteritems():
    wt_vec[k+'_p']=v
WT2 = pd.Series(wt_vec)/np.sqrt(2)
clf_nadd = KNR(9,weights='distance',n_jobs=1)
XVALS = indep_var*WT2
YVALS = dep_var*WT
clf_nadd.fit(XVALS,YVALS)


def fs_optimization(S,pdat_ser,D,T,NG,ub_ser=pd.Series(),PS=500,PARAM=None):
    ## difference: need to pass the delta matrices as D/DG because this is not globally stored
    STAT_LBLS = ['OBJ','D_F','P_F','G_F','PG_F','KNN_GP','C_PRB','I_PRB','F_PRB','alpha','anm']
    STAT_LBLNS = [(NG,lbl) for lbl in STAT_LBLS]
    gsmi,S = S
    gsmf,T = T
    if dis!='lincs':
        ctigrp = np.where(CAT.categories==CAT_map.loc[gsmi])[0][0]
        ctfgrp = np.where(CAT.categories==CAT_map.loc[gsmf])[0][0]
        #ctfgrp = STATES[STATES.state_name==CAT_map.loc[gsmf]].index[0]
        #ctigrp = STATES[STATES.state_name==CAT_map.loc[gsmi]].index[0]
    else:
        ctigrp = CAT_map.loc[gsmi]
        ctfgrp = CAT_map.loc[gsmf]
    
    #ctf = STATES.loc[ctfgrp,'state_name']
    ## need to edit this to account for genes
    ## still need to pass the correct matrix/vector to the CPLEX routine
    kwargs = {'NG':1} ## this is the number of new perturbations to be selected
    kwargs['topN']=10
    
    kwargs['lb'] = 0
    kwargs['ub'] = 1
    wsep = (S-T).multiply(WT)
    SWT = S.multiply(WT)
    TWT = T.multiply(WT)
    DENOM = np.linalg.norm(SWT.dot(KERN).dot(SWT))
    if PARAM is None:
        PARAM = np.linalg.norm(wsep)/DENOM
    C1sqr = np.cos(costF)**2
    C2sqr = (PARAM*np.sin(costF))**2
    kernel = (C1sqr*np.eye(KERN.shape[0])+C2sqr*KERN)
    ## number of perturbations to add is always 1
    allwDx=D.T.multiply(WT).T
    allwQx = allwDx.T@kernel@allwDx
    allwCx = (wsep.T@kernel + C2sqr*(TWT)@KERN)@allwDx
    selperts = np.where(np.diag(allwQx)+allwCx<0)[0]
    if selperts.shape[0]<1:
        ppdat_ser = pdat_ser.copy()
        for IND,((nng,nm),val) in enumerate(pdat_ser.iteritems()):
            if nm not in ['alpha','anm']:
                ppdat_ser.iloc[IND] = val
            elif nm=='alpha':
                ppdat_ser.iloc[IND] = 0
            elif nm=='anm':
                ppdat_ser.iloc[IND] = ''
        ppdat_ser.index = [(nng+1,nm) for nng,nm in pdat_ser.index]
        ppdat_ser = ppdat_ser.append(pdat_ser)
        return (gsmi,gsmf),[ppdat_ser],[S],[[]]
    selD = D.iloc[:,selperts]
    selD = selD.iloc[:,np.random.permutation(selD.shape[1])]
    N_IT = selD.shape[1]//PS + 1
    PS2 = selD.shape[1]//N_IT + 1
    OBJ = 1e10
    ## need to figure out which perturbations actually point in the desired direction
    cc_al_l = []
    cc_obj_l = []
    for nn in range(N_IT):
        if selD.shape[1]>PS2:
            LB = nn*PS2
            UB = min((nn+1)*PS2,selD.shape[1])
            if LB==UB:
                continue
            sel_cols = selD.columns[LB:UB]
            rsel_deltas = selD.loc[:,sel_cols]
        else:
            rsel_deltas = selD
            sel_cols = selD.columns
        wDx = rsel_deltas.T.multiply(WT).T
        wQx = wDx.T@kernel@wDx
        wCx = (wsep.to_frame().T@kernel + C2sqr*(TWT)@KERN)@wDx
        ## probably only need to consider wQx/wCx
        al_l,obj_l,stat_l = cplex_optimize_miqp(wQx.values,wCx.T.loc[wQx.index].values,**kwargs)
        ## not sure that we need to keep a single solution, we can keep topN
        for ii,al in enumerate(al_l):
            nzind = np.where(np.asarray(al[:len(al)//2])>1e-6)[0]
            if nzind.shape[0]>0:
                ind = np.where(np.asarray(al[:len(al)//2])>1e-6)[0][0]
                val = al[ind]
                cc_al_l.append((rsel_deltas.columns[ind],val))
                cc_obj_l.append(obj_l[ii])
    ## need a new strategy -- instead of going with the full series for each case,
    ## why not a series of the column names
    if len(cc_al_l)<1:
        d0 = np.linalg.norm(wsep)
        g0 = DENOM
        R = SWT.to_frame().T.values
        Cst = clf_ct.predict(R)[0]
        if dis != 'lincs':
            C = np.where(CAT.categories==Cst)[0][0]
        else:
            C = Cst
            Cst = STATES.loc[Cst,'state_name']
        P,I,F = clf_ct.predict_proba(R)[0,[C,ctigrp,ctfgrp]]
        dat_ser=pd.Series([OBJ,d0,0,g0,0,Cst,P,I,F,0,''],
                          index=STAT_LBLNS)
        return (gsmi,gsmf),[dat_ser],[S],[[]]
    dat_ser_l = []
    new_state_l = []
    d0 = np.linalg.norm(wsep)
    wD1 = selD.T.multiply(WT).T
    cnm_l = []
    for (cnm,val),obj in zip(cc_al_l,cc_obj_l):
        #pert_stats.append({'pert_nm':selD.columns[ind],'pert_al':al[ind],'pert_obj':obj})
        R = SWT + wD1.loc[:,cnm]*val
        d1 = np.linalg.norm(wsep+wD1.loc[:,cnm]*val)
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
        dat_ser=pd.Series([OBJ,d1,pf,g1,gpf,Cst,P,I,F,val,cnm],
                          index=STAT_LBLNS)
        dat_ser_l.append(pdat_ser.append(dat_ser))
        new_state_l.append(S+selD.loc[:,cnm]*val)
        cnm_l.append([cnm])
    #nz_data = al_df[al_df>1e-6]
    #dat_ser = dat_ser.append(nz_data)
    return (gsmi,gsmf),dat_ser_l,new_state_l,cnm_l

def get_corrected_deltas(deltas,S,pdat_ser):
    ## combine the data; check that arrays are oriented correctly
    my_d = {}
    Ls = [(ev+'_s',v) for ev,v in S.iteritems()]
    wt_l1 = [(ev+'_s',v) for ev,v in WT.iteritems()]
    wt_l2 = [(ev+'_p',v) for ev,v in WT.iteritems()]
    WT2 = pd.Series(dict(wt_l1+wt_l2))/np.sqrt(2)
    for dnm,dcol in deltas.iteritems():
        Lp = [(ev+'_p',v) for ev,v in dcol.iteritems()]
        my_d[dnm] = dict(Ls+Lp)
    merged_data = pd.DataFrame(my_d).T
    ## input into the model
    corrections = clf_nadd.predict(merged_data*WT2).T
    ## reconstruct the corrected_deltas
    corrected_deltas = deltas+(corrections.T/WT.loc[deltas.index].values).T
    anm_l = [(ii) for nm,ii in pdat_ser.index if ii.startswith('anm')]
    if len(anm_l)>0:
        sel_cols = [col for col in deltas.columns if col not in anm_l]
        corrected_deltas=corrected_deltas.loc[:,sel_cols]
    return corrected_deltas

def fs_wrap(stack,targ,ng,gsm_i,gsm_f):
    state,pdat_ser,onm_l = stack
    if onm_l!=[]:
        sel_cols = [col for col in delta_data.columns if col not in onm_l]
        sel_delta_data = delta_data.loc[:,sel_cols]
    else:
        sel_delta_data = delta_data
    cdeltas = get_corrected_deltas(sel_delta_data,state,pdat_ser)
    dum,dat_l,state_l,nm_l = fs_optimization((gsm_i,state),pdat_ser,cdeltas,(gsm_f,targ),ng)
    if onm_l!=[]:
        onm_l.extend(nm_l)
    else:
        onm_l = nm_l
    return list(zip(state_l,dat_l,onm_l))

def corrected_fs(gsm_i_gsm_f):
    gsm_i,gsm_f=gsm_i_gsm_f
    print("Starting experimental pair %s -> %s" % (gsm_i,gsm_f))
    ## need to alter the function call to conform to a new signature
    ## instead of taking in an initial/final cell type, it should take in an
    ## initial/final state then it should calculate the set of possible perts
    ## then it should find the best among that set
    cti = CAT_map.loc[gsm_i]
    ctf = CAT_map.loc[gsm_f]
    if dis != 'lincs':
        if not osp.exists('%s/%s/%s/%s/' % (OUTPUT_PREFIX,dis,OUT,cti)):
            os.makedirs('%s/%s/%s/%s/' % (OUTPUT_PREFIX,dis,OUT,cti),exist_ok=True)
        GFN1 = '%s/%s/%s/%s/%s-%s-%.2f_%d_corrected_states%s.pkl' % (OUTPUT_PREFIX,dis,OUT,cti,gsm_i,gsm_f,costF,NG,ver2)
    else:
        ctistr = STATES.loc[cti,'state_name']
        #ctfstr = STATES.loc[ctf,'state_name']
        if not osp.exists('%s/%s/%s/%s/' % (OUTPUT_PREFIX,dis,OUT,ctistr),exist_ok=True):
            os.makedirs('%s/%s/%s/%s/' % (OUTPUT_PREFIX,dis,OUT,ctistr))
        GFN1 = '%s/%s/%s/%s/%s-%s-%.2f_%d_corrected_states%s.pkl' % (OUTPUT_PREFIX,dis,OUT,ctistr,gsm_i,gsm_f,costF,NG,ver2)
    GFN2 = GFN1.replace('_states','_stats')
    if osp.exists(GFN1) and osp.exists(GFN2):
        return pd.read_pickle(GFN1),pd.read_pickle(GFN2) 
    init = drug_data_trim.loc[gsm_i]
    targ = drug_data_trim.loc[gsm_f]
    stack = [(init,pd.Series(),[])]
    for ng in range(1,NG+1):
        new_stack =[]
        ## need to parallelize -- possibly one level up
        if ng==1:
            state,pdat_ser,dummy = stack[0]
            cdeltas = get_corrected_deltas(delta_data,state,pdat_ser)
            dum,dat_l,state_l,nm_l = fs_optimization((gsm_i,state),pdat_ser,cdeltas,(gsm_f,targ),ng)
            new_stack.extend(list(zip(state_l,dat_l,nm_l)))
        else:
            pfs_wrap = partial(fs_wrap,ng=ng,targ=targ,gsm_i=gsm_i,gsm_f=gsm_f)
            new_stack_l = list(map(pfs_wrap,stack))
            for elt in new_stack_l:
                new_stack.extend(elt)
        stack=new_stack
    fin_states_d = {}
    dat_ser_l = []
    for state,dat_ser,nm_l in stack:
        nm_cols = [col for col in dat_ser.index if col[1]=='anm']
        drg_p = tuple(list(dat_ser.loc[nm_cols].values))
        fin_states_d[drg_p]=state
        dat_ser_l.append(dat_ser)
    if len(stack)==1 and dat_ser.empty:
        print("No possible drugs for pair %s --> %s" % (gsm_i,gsm_f))
        return pd.DataFrame(),pd.DataFrame()
    fin_states_df = pd.DataFrame(fin_states_d).T
    fin_states_df['GSMi'] = gsm_i
    fin_states_df['GSMf'] = gsm_f
    fin_states_df = fin_states_df.set_index(['GSMi','GSMf'],append=True) 
    dat_ser_df = pd.DataFrame(dat_ser_l)
    dat_ser_df[('GSMi','')] = gsm_i
    dat_ser_df[('GSMf','')] = gsm_f
    dat_ser_df.columns = pd.MultiIndex.from_tuples(dat_ser_df.columns)
    fin_states_df.to_pickle(GFN1)
    dat_ser_df.to_pickle(GFN2)
    return fin_states_df,dat_ser_df

def condensed_formula(ii,jj,mm):
    return ii*mm-((ii+1)*(ii+2))//2 + jj

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
    
def main(STHR=10):
    PAIRS2 = PAIRS
    for _dum,row in PAIRS2.iterrows():
        num_i,num_f = row.loc['disease_state'],row.loc['target_state']
        cti = STATES.loc[num_i,'state_name']
        ctf = STATES.loc[num_f,'state_name']
        print("Starting cell type pair %s -> %s" % (cti,ctf))
        if dis != 'lincs':
            targ_data = drug_data_trim.loc[ctGB.get_group(ctf).index]
            init_data = drug_data_trim.loc[ctGB.get_group(cti).index]
        else:
            targ_data = drug_data_trim.loc[ctGB.get_group(num_f).index]
            init_data = drug_data_trim.loc[ctGB.get_group(num_i).index]

        if init_data.shape[0]>STHR:
            rnd_inds = np.random.permutation(init_data.index)[:STHR]
            init_data = init_data.loc[rnd_inds]
        AFN1 ='%s/%s/%s/%s-%s-%.2f_%d_corrected_states%s.pkl' % (OUTPUT_PREFIX,dis,OUT,cti,ctf,costF,NG,ver2)
        AFN2 = AFN1.replace('_states','_stats')
        if osp.exists(AFN2):
            continue
        states_l = []
        stats_l = []
        epair_l = []
        for gsm_i,row in init_data.iterrows():
            if targ_data.shape[0]>1:
                avg_dists = select_ctf_target(targ_data,row)
                sel_targ = targ_data.iloc[np.argmin(avg_dists)]
            else:
                sel_targ=targ_data.iloc[0]
            gsm_f = sel_targ.name                        
            GFN1 = '%s/%s/%s/%s/%s-%s-%.2f_%d_corrected_states%s.pkl' % (OUTPUT_PREFIX,dis,OUT,cti,gsm_i,gsm_f,costF,NG,ver2)
            GFN2 = GFN1.replace('_states','_stats')
            if osp.exists(GFN1):
                states_df = pd.read_pickle(GFN1)
                stats_df = pd.read_pickle(GFN2)
                if '1.04023779e+04' in states_df.index:
                    nsdf = states_df.iloc[:,:-2].T
                    nsdf['GSMi']=gsm_i
                    nsdf['GSMf']=gsm_f
                    states_df = nsdf.set_index(['GSMi','GSMf'],append=True)
                    states_df.to_pickle(GFN1)
                states_l.append(states_df)
                if not isinstance(stats_df.columns,pd.MultiIndex):
                    stats_df.columns = pd.MultiIndex.from_tuples(stats_df.columns)
                    stats_df.to_pickle(GFN2)
                stats_l.append(stats_df)
                continue
            else:
                epair_l.append((gsm_i,gsm_f))
        if len(stats_l)<STHR and len(epair_l)>0:
            sel_epair_l = epair_l[:(STHR-len(stats_l))]
            op_l = list(futures.map(corrected_fs,sel_epair_l))
            for op in op_l:
                opstates,opstats = op
                if not opstates.empty:
                    states_l.append(opstates)
                    stats_l.append(opstats)
            #states_df,stats_df = corrected_fs(gsm_i,gsm_f)
            #states_df.to_pickle(GFN1)
            #stats_df.to_pickle(GFN2)
            #states_l.append(states_df)
            #stats_l.append(stats_df)
        all_states_df = pd.concat(states_l)
        all_stats_df = pd.concat(stats_l)
        all_states_df.to_pickle(AFN1)
        all_stats_df.to_pickle(AFN2)


if __name__ == '__main__':
    main()
