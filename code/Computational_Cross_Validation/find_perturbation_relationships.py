"""find_perturbation_relationships.py

Created by Thomas Wytock on 2023-04-23.
Copyright (c) 2023 Thomas Wytock. All rights reserved.

Constructs perturbation graphs of cell states from metadata tables.
""" 

import os,re,sys,os.path as osp
import pandas as pd
import numpy as np
import pickle as P
import argparse
import networkx as nx
from collections import defaultdict
from itertools import combinations as combs
# changed to reflect the addition of the glioblastoma radiation projects

parser = argparse.ArgumentParser(description='Generate a network where nodes are cell states and edges connect states related by a single perturbation.')

parser.add_argument('-V','--version',metavar='v',type=str,nargs=1,
                    help='version of files to use',
                    default=['_0122'])
args = parser.parse_args()
version = args.version[0]
DATA_PREFIX = '../../data/Computational_Cross_Validation'
OUTPUT_PREFIX = '../../output/Computational_Cross_Validation'

## load the SRA metadata
XXX = pd.read_csv('%s/combined_SRA_metadata-converted_genotype%s.csv' % (DATA_PREFIX,version),index_col=0)

## load the ENCODE metadata
YYY = pd.read_csv('%s/ENCODE_metadata.csv' % DATA_PREFIX,index_col=0,header=0)

#Generate a network where nodes are cell states and edges connect states related by a single perturbation.


prefix_d = {'p':1e-6,'n':1e-3,'u':1,'m':1000,'c':10000,'d':100000,'k':1e9}
#import re
re_molar = re.compile('([0-9.]+)\W*([cdknmpu])?(M)')
re_gl = re.compile('([0-9]+)\W*([cdknmpu])?(g/([cdknmpu])?L)')
re_molal = re.compile('([0-9]+)\W*([cdknmpu])?(g/([cdknmpu])?g)')
re_mass = re.compile('([0-9]+)\W*([cdknmpu])?(g)')
re_pct = re.compile('([0-9]+)\W*(%)')
re_dil = re.compile('1:([0-9]+)\W*dilution')

def extract_tcdr(annot_ser,isTC):
    '''extract_tcdr
    extracts a time-series of a given treatment from semi-colon separated lists.
    ----------------------------------------------------------------------
    ARGUMENTS
    annot_ser | pd.Series | index : SRX numbers, values : time/dosage
    isTC | Bool | True : Time Course; False : Dose Response
    ----------------------------------------------------------------------
    RETURNS
    t_ser/MM | pd.Series | index : SRX numbers, values: weights for treatment
    '''
    if isTC:
        t_ser = pd.Series(dict([(run,float(tpt[:-1])) for run,tpt in annot_ser.iteritems()]))
    else:
        my_d = {}
        for run,tpt in annot_ser.iteritems():
            for reobj in [re_molar,re_gl,re_molal,re_mass,re_pct,re_dil]:
                mobj = reobj.match(tpt)
                if mobj is not None:
                    if len(mobj.groups())<2:
                        quant = 1/float(mobj.groups()[0])
                        scale = 1e6
                    elif len(mobj.groups())<3:
                        quant,pct = mobj.groups()
                        scale=1e4
                    elif len(mobj.groups())<4:
                        quant,prefix,units = mobj.groups()
                        scale = prefix_d.get(prefix,1e6)
                    elif len(mobj.groups())<5:
                        quant,prefix,units,prefix2 = mobj.groups()
                        scale = 1e6*prefix_d.get(prefix,1e6)/prefix_d.get(prefix2,1e6)
                    my_d[run] = float(quant)*scale
                    break                    
            else:
                print('failed to match units')
                import pdb
                pdb.set_trace()
        t_ser = pd.Series(my_d)
    MM = t_ser.max()
    return t_ser/MM

def determine_tcdr(exp_rows,col):
    '''determine_tcdr
    determines whether there is a time course or dose response for a given treatment
    ----------------------------------------------------------------------
    ARGUMENTS
    exp_rows | pd.DataFrame | table of experimental info for a given treatment in a given project
    col | string | column of exp_rows to examine -- "treatment_length" or "treatment_concentration"
    ----------------------------------------------------------------------
    RETURNS
    info_d | dict | keys: attributes to categorize 
    trt_tps_d | dict (of dicts) | keys : treatments (keys : srx numbers, values : times/dosages)
    '''
    info_d = {}
    trt_tps_d = defaultdict(dict)
    isTC = col=='treatment_length'
    opkw = 'isTimeCourse' if isTC else 'isDoseResponse'
    ext = 'TC' if isTC else 'DR'
    if (exp_rows[col]!='NONE').sum()>0:
        if exp_rows[col].value_counts().shape[0]>1:
            if exp_rows[col].apply(lambda x: ';' in x).sum()<1:
                info_d[opkw]=True
                for ___,row in exp_rows.iterrows():
                    tmt = row.loc['treatment']
                    exp = row.loc['run']
                    tp = row.loc[col]
                    trt_tps_d[tmt][exp] = tp
            else:
                ## need to write customized code to match treatments with times
                for ___,row in exp_rows.iterrows():
                    tmt_l = row.loc['treatment'].split(';')
                    exp = row.loc['run']
                    tp_l = row.loc[col].split(';')
                    for tmt, tp in zip(tmt_l,tp_l):
                        trt_tps_d[tmt.strip()][exp] = tp.strip()
                trt_tps_d = dict(trt_tps_d)
                for trt,tps_d in trt_tps_d.items():
                    if len(tps_d.keys())>0:
                        info_d[opkw]=True
                        break
                else:
                    info_d[opkw]=False          
        else:
            for ___,row in exp_rows.iterrows():
                tmt = row.loc['treatment']
                exp = row.loc['run']
                tp = row.loc[col]
                trt_tps_d[tmt][exp] = tp

            info_d[opkw]=False
    else:
        for ___,row in exp_rows.iterrows():
            tmt = row.loc['treatment']
            exp = row.loc['run']
            tp = row.loc[col]
            trt_tps_d[tmt][exp] = tp
        info_d[opkw]=False
    trt_tps_d = dict(trt_tps_d)
    for trt,run_tps_d in trt_tps_d.items():
        if info_d[opkw]:
            run_tps_ser = pd.Series(run_tps_d)
            if 'Early_withdrawal' in run_tps_ser.values:
                run_tps_ser=run_tps_ser.replace('Early_withdrawal','0w')
            wts = extract_tcdr(run_tps_ser,isTC)
            info_d['weights_%s_%s' % (trt,ext)] = wts
        else:
            nkeys = len(run_tps_d.keys())
            wts = pd.Series(dict(zip(run_tps_d.keys(),np.ones(nkeys))))
            info_d['weights_%s_%s' % (trt,ext)] = wts        
    return info_d,trt_tps_d

def parse_treatments(ndtup,exp_rows):
    '''parse_treatments
    parses treatment conditions for a given bioproject
    ----------------------------------------------------------------------
    ARGUMENTS
    ndtup | tuple | (cell_type, cell_line, treatment, genotype) of the exp_rows
    exp_rows | pd.DataFrame | table of experimental info for a given treatment in a given project
    
    ----------------------------------------------------------------------
    RETURNS
    info_d | dict | keys: attributes to categorize 
    trt_tps_d | dict (of dicts) | keys : treatments (keys : srx numbers, values : times/dosages)
    '''
    ct,cl,trt,gt = ndtup
    info_d = {}
    ## information to extract
    ## 1. column ids
    ## 2. Time course/dose response
    ## 2a. if time course/dose response, need to get the weightings
    ## 3. more than one treatment?
    if 'run' in exp_rows.columns:
        info_d['columns'] = exp_rows.run.unique()
    else:
        info_d['columns'] = exp_rows.index.unique()
    info_d['cell_line']=cl
    info_d['cell_type']=ct
    a,trt_tps_d= determine_tcdr(exp_rows,'treatment_length')
    info_d.update(a)
    aa,trt_dsg_d = determine_tcdr(exp_rows,'treatment_concentration')
    info_d.update(aa)
    if ';' in trt:
        info_d['hasMulti'] = True
        if exp_rows.bioproject.unique()[0]=='PRJNA374861':
            trt_l = []
            for x in trt.split(';'):
                if x.strip().startswith('P'):
                    break
                else:
                    trt_l.append(x.strip())
            info_d['treatment']=trt_l
        else:
            info_d['treatment']=[x.strip() for x in trt.split(';')]
    else:
        info_d['hasMulti'] = False
        info_d['treatment']=[trt]
    if ';' in gt:
        #info_d['hasMulti'] = True
        info_d['genotype']=[x.strip() for x in gt.split(';')]
    else:
        #info_d['hasMulti'] = False
        info_d['genotype']=[gt]
    return info_d,trt_tps_d,trt_dsg_d

def match_pert_to_gn(trt,gt):
    ii=0
    matched_pairs = []
    for elt in trt:
        if elt.startswith(('kd','oe','mut')):
            matched_pairs.append((elt,gt[ii]))
            ii+=1
    return matched_pairs

def build_perturbation_network(Z,overwrite=True,sel_prjs=[],isENCODE=False):
    '''Constructs the network of single perturbations for all bioprojects'''
    if not isENCODE:
        prj_list = Z.bioproject.unique()
        Zgb = Z.groupby(['bioproject'])
    else:
        prj_list = ['ENCODE']
        Zgb = Z.groupby(['database'])
    prj_graph_d = defaultdict(nx.DiGraph)
    if osp.exists('%s/project_graph_d%s.pkl' % (OUTPUT_PREFIX,version)):
        prj_graph_d_old = pd.read_pickle('%s/project_graph_d%s.pkl' % (OUTPUT_PREFIX,version))
        prj_graph_d.update(prj_graph_d_old)
    
    for (prjno),prjgrp in Zgb:
        if prjno=='PRJNA590127':
            continue
        elif prjno not in sel_prjs and not len(sel_prjs)==0:
            continue
        print(prjno)
        ## not sure if we will need the columns at this point, but it may be easier if we have all in one place
        #
        # prj_graph_d[prjno].nodes[(ct,cl)]['columns']
        # cond_G.nodes[(ct,cl)]['cat'] = 'cell_type'
        # media = ctgrp.Medium.unique()
        if overwrite:
            prj_G = nx.DiGraph()
        else:
            prj_G = prj_graph_d[prjno]
        prjgrp_fill = prjgrp.fillna('NONE')
        tup_l = []
        metadata_d = {}
        for (ct,cl,trt,gt),cgrp in prjgrp_fill.groupby(['cell_type','cell_line','treatment','genotype']):
            ## create nodes and associated columns
            if 'run' not in cgrp.columns:
                cgrp = cgrp.reset_index()
                if 'cid' in cgrp.columns:
                    old_col = cgrp.columns.tolist()
                    new_col = [c if c !='cid' else 'run' for c in old_col]
                    cgrp.columns=new_col
            nd_data_d,trt_tps_d,trt_dsg_d = parse_treatments((ct,cl,trt,gt),cgrp)
            if prjno == 'PRJNA374861':
                ntrt = []
                for elt in trt.split('; '):
                    if elt.startswith('P'):
                        break
                    else:
                        ntrt.append(elt)
                trt = '; '.join(ntrt)
            tup_l.append((ct,cl,trt,gt))
            metadata_d[(ct,cl,trt,gt)] = nd_data_d
        prj_G.add_nodes_from(tup_l)    
        for nd in prj_G.nodes():
            for k,v in metadata_d[nd].items():
                prj_G.nodes[nd][k] = v
        edge_l = []
        edge_d = defaultdict(dict)
        for nd1,nd2 in combs(prj_G.nodes(),2):
            same_l = np.array([a==b for a,b in zip(nd1,nd2)])
            ## need to match 3 of 4
            ## cases (1) all but CL/CT
            if same_l[-2] and same_l[-1]:
                edge_l.extend([(nd1,nd2),(nd2,nd1)])
                edge_d[(nd1,nd2)]['class']='cell_type'
                edge_d[(nd1,nd2)]['name']='%s->%s' % (nd1[1],nd2[1])
                edge_d[(nd2,nd1)]['class']='cell_type'
                edge_d[(nd2,nd1)]['name']='%s->%s' % (nd2[1],nd1[1])
                continue
            elif sum(same_l[:2])==2:
                ## treatments are always drugs / nc-kd / resistant-other
                # strategy: identify 'control' for each column.
                # if "nc" in elt then its a control
                # alternately, if two factors are the same,
                # then can look for differences in others
                if prj_G.nodes[nd2]['hasMulti'] and prj_G.nodes[nd1]['hasMulti']:
                    ## both have multiple perturbations
                    treatAsComb = False
                    isect = set(prj_G.nodes[nd1]['treatment']) & set(prj_G.nodes[nd2]['treatment']) ## common treatment elements
                    if len(prj_G.nodes[nd2]['treatment']) == len(prj_G.nodes[nd1]['treatment']):
                        ## same length
                        if len(prj_G.nodes[nd2]['treatment'])-len(isect)==1:
                            #ii = 0
                            CONT = False
                            nd1_match = match_pert_to_gn(prj_G.nodes[nd1]['treatment'],prj_G.nodes[nd1]['genotype'])
                            nd2_match = match_pert_to_gn(prj_G.nodes[nd2]['treatment'],prj_G.nodes[nd2]['genotype'])
                            for trt in isect:
                                if trt.startswith(('kd','oe','mut')):
                                    sel_nd1_match = [elt for elt in nd1_match if elt[0]==trt]
                                    sel_nd2_match = [elt for elt in nd2_match if elt[0]==trt]
                                    #gt2 = prj_G.nodes[nd2]['genotype'][ii]
                                    #gt1 = prj_G.nodes[nd1]['genotype'][ii]
                                    if set(sel_nd2_match) != set(sel_nd1_match):
                                        CONT = True
                                        break
                                    
                            if CONT:
                                continue
                            diff2 = list(set(prj_G.nodes[nd2]['treatment'])-isect)[0]
                            diff1 = list(set(prj_G.nodes[nd1]['treatment'])-isect)[0]
                            ## if the term 'wt' is not in the intersection, it is NOT compared with others
                            if 'wt' in diff2 or 'wt' in diff1:
                                ## PRJNA224740
                                continue
                            ## look for 'sensitive','healthy','nc' in the differences
                            ## if 'nc' then we need to check that it matches
                            if diff1[:-1]==diff2[:-1] and diff1[-1] in ['-','+'] and diff2[-1] in ['-','+']:
                                ## receptor
                                if diff1.endswith('-'):
                                    cnd = nd1
                                    tnd = nd2
                                    diff = diff2
                                else:
                                    cnd = nd2
                                    tnd = nd1
                                    diff = diff1
                            else:
                                for e in ['sensitive','healthy','nc','androgen_dependent','normoxia','in_vitro','quiescent','uninvolved']:
                                    if e in diff1:
                                        if e == 'nc':
                                            ## the following condition is added to ensure that 
                                            ## the negative control matches the treatment in the other
                                            ## case, if the control is treatment specific. (PRJNA386460)
                                            if '-' in diff1:
                                                if not diff2.startswith(('kd','oe')) and diff1.endswith(('drug','resistant')):
                                                    ## this *could* leave to mismatches between drug/nondrug controls (PRJNA395363)
                                                    pass
                                                elif diff1.split('-')[1] == 'differentiation':
                                                    if 'differentiation' not in diff2:
                                                        continue
                                                elif diff1.split('-')[1] in ['LNA','CRISPRi','RNAi']:
                                                    if not diff2 == 'kd-%s' % (diff1.split('-')[1]):
                                                        continue
                                                elif diff1.split('-')[1] != diff2:
                                                    continue

                                        cnd = nd1
                                        tnd = nd2
                                        diff = diff2
                                        break
                                else:
                                    for e in ['sensitive','healthy','nc','androgen_dependent','normoxia','in_vitro','quiescent','uninvolved']:
                                        if e in diff2:
                                            if e == 'nc':
                                                ## the following condition is added to ensure that 
                                                ## the negative control matches the treatment in the other
                                                ## case, if the control is treatment specific. (PRJNA386460)
                                                if '-' in diff2:
                                                    if not diff1.startswith(('kd','oe')) and diff2.endswith(('drug','resistant')):
                                                        pass
                                                    elif diff2.split('-')[1] == 'differentiation':
                                                        if 'differentiation' not in diff1:
                                                            continue
                                                    elif diff2.split('-')[1] in ['LNA','CRISPRi','RNAi','transfection']:
                                                        if not diff1 == 'kd-%s' % (diff2.split('-')[1]):
                                                            continue
                                                    elif diff2.split('-')[1] != diff1:
                                                        continue
                                            cnd = nd2
                                            tnd = nd1
                                            diff = diff1
                                            break
                                    else:
                                        ## the difference is not in a control don't want 
                                        ## to link nodes
                                        continue
                        elif len(prj_G.nodes[nd2]['treatment'])==len(isect):
                            # e.g. 'kd; kd; nc' vs. 'nc; nc; kd'
                            ## need to check whether any in isect are kd/oe/mut
                            ii = 0
                            CONT = False ## flag variable for whether the genotypes are two different perturbations
                            for trt in isect:
                                if trt.startswith(('kd','oe','mut')):
                                    gt2 = prj_G.nodes[nd2]['genotype'][ii]
                                    gt1 = prj_G.nodes[nd1]['genotype'][ii]
                                    if gt1!=gt2:
                                        CONT = True
                                        break
                                    ii+=1
                            else:
                                import pdb
                                pdb.set_trace()
                                pass
                            if CONT:
                                continue
                            ## degenerate case probably won't reach here.
                        else:
                            continue
                            
                    elif abs(len(prj_G.nodes[nd2]['treatment']) - len(prj_G.nodes[nd1]['treatment'])) == 1:
                        if len(prj_G.nodes[nd2]['treatment'])>len(prj_G.nodes[nd1]['treatment']):
                            cnd = nd1
                            tnd = nd2
                        else:
                            cnd = nd2
                            tnd = nd1
                        diff = set(prj_G.nodes[tnd]['treatment'])-set(prj_G.nodes[cnd]['treatment'])
                        ## if the term 'wt' is not in the intersection, it is NOT compared with others
                        if 'wt' in diff:
                            ## PRJNA224740
                            continue
                        if len(diff)>1:
                            if isect | set(['nc-drug']) == set(prj_G.nodes[cnd]['treatment']):
                                for nd in prj_G.nodes():
                                    if nd[0]!=tnd[0] or nd[1]!=tnd[1]:
                                        continue
                                    if set(prj_G.nodes[nd]['treatment']) < (diff | isect):
                                        break
                                else:
                                    ## in this case there are NO subsets
                                    treatAsComb = True
                                if not treatAsComb:
                                    continue
                        elif len(diff)==1 and (set(prj_G.nodes[cnd]['treatment']) < set(prj_G.nodes[tnd]['treatment'])):
                            ## no updates to the diff/isect need to be made?
                            ## PRJNA495849 -- stop matching 'kd' if no other genetic perturbations are present
                            ## there either needs to be an 'nc-%s' % diff OR oe/mut/kd in isect
                            if len(diff & set(['kd','oe','mut']))>0:
                                if not (('nc-%s' % list(diff)[0]) in isect) and len(set(['kd','oe','mut']) & isect) < 1:
                                    continue
                            ## PRJNA495849 -- stop matching 'nc-*' without perturbations
                            if list(diff)[0].startswith('nc'):
                                continue
                        elif len(diff)==0:
                            ## in this case, the overlap is likely in kd/oe/mut
                            tVCS = pd.Categorical(prj_G.nodes[tnd]['treatment']).value_counts()
                            cVCS = pd.Categorical(prj_G.nodes[cnd]['treatment']).value_counts()
                            delta = tVCS!=cVCS
                            if sum(delta)==1:
                                trt = tVCS[delta].index[0]
                                isect -= set([trt])
                            else:
                                pass
                             
                        else:
                            continue
                    elif abs(len(prj_G.nodes[nd2]['treatment']) - len(prj_G.nodes[nd1]['treatment'])) > 1:
                        if len(prj_G.nodes[nd2]['treatment'])>len(prj_G.nodes[nd1]['treatment']):
                            cnd = nd1
                            tnd = nd2
                        else:
                            cnd = nd2
                            tnd = nd1
                        diff = set(prj_G.nodes[tnd]['treatment'])-set(prj_G.nodes[cnd]['treatment'])
                        ## can check -- if the number of genotype nodes are the same
                        ## added in response to PRJ495849
                        if prj_G.nodes[tnd]['genotype'] == prj_G.nodes[cnd]['genotype'] and 'nc-drug' in prj_G.nodes[cnd]['treatment']:
                            ## test for treatments that exist as a unit
                            ## PRJNA495849 -- 'TNFalpha; BV6; zVAD-fmk'
                            ## check whether any of the constituent perturbations appear by
                            ## iterating over the nodes with the same cell line cell type and examining the treatment
                            for nd in prj_G.nodes():
                                if nd[0]!=tnd[0] or nd[1]!=tnd[1]:
                                    continue
                                if set(prj_G.nodes[nd]['treatment']) < diff:
                                    break
                            else:
                                ## in this case there are NO subsets
                                treatAsComb = True
                            if not treatAsComb:
                                continue
 
                        else:
                            continue #?
                        
                    ## need to sort out whether kd/oe are the differences
                    ## at this point cnd/tnd/isect/diff are defined 
                    ## "test" nodes should have an additional treatment condition
                    ## this is in "diff"
                    ## three cases lead to edges : 
                    ##     1. len(diff)==1 and diff not in {'kd','oe','mut'} 
                    ##          Create edge (not gene perturbation)
                    ##     2. len(diff)==1 and diff in {'kd','oe','mut'} 
                    ##          Create edge (gene perturbation)
                    ##     3. len(diff)==0 and diff in {'kd','oe','mut'} 
                    ##          Create edge (gene perturbation)
                    tmp_d = {tnd:{},cnd:{}}
                    for nd in [cnd,tnd]:
                        ii=0
                        for ind, trt in enumerate(prj_G.nodes[nd]['treatment']):
                            if trt not in isect:
                                if not trt.startswith(('kd','oe','mut')):
                                    tmp_d[nd][ind] = trt
                                else:
                                    gt = prj_G.nodes[nd]['genotype'][ii]
                                    tmp_d[nd][ind] = (trt,gt)
                                    ii+=1
                    if len(set(tmp_d[tnd].values()) - set(tmp_d[cnd].values())) == 1:
                        diff = (set(tmp_d[tnd].values()) - set(tmp_d[cnd].values())).pop()
                        ind = list(tmp_d[tnd].values()).index(diff)
                        edge_l.append((cnd,tnd))
                        if isinstance(diff,tuple):
                            edge_d[(cnd,tnd)]['class'] = 'gene_pert'
                            edge_d[(cnd,tnd)]['sense'] = diff[0]
                            edge_d[(cnd,tnd)]['name'] = diff[1]
                        elif isinstance(diff,str):
                            if diff.endswith('+'):
                                edge_d[(cnd,tnd)]['class'] = 'receptor'
                                edge_d[(cnd,tnd)]['name'] = diff
                            else:
                                edge_d[(cnd,tnd)]['class'] = 'drug'
                                edge_d[(cnd,tnd)]['name'] = diff
                    elif treatAsComb:
                        edge_l.append((cnd,tnd))
                        edge_d[(cnd,tnd)]['class'] = 'drug'
                        edge_d[(cnd,tnd)]['name'] = '; '.join(list(diff))
                        
                elif prj_G.nodes[nd2]['hasMulti'] or prj_G.nodes[nd1]['hasMulti']:
                    ## only one has multi other is single
                    if prj_G.nodes[nd1]['hasMulti']:
                        cnd = nd2
                        tnd = nd1
                    else:
                        cnd = nd1
                        tnd = nd2
                    if prj_G.nodes[cnd]['genotype']==['NONE'] or prj_G.nodes[tnd]['genotype']==['NONE']:
                        if prj_G.nodes[cnd]['genotype']!=['NONE']:
                            continue
                        diff = set(prj_G.nodes[tnd]['treatment'])-set(prj_G.nodes[cnd]['treatment'])
                        if 'wt' in diff:
                            ## PRJNA224740
                            continue
                        if len(diff) == 1:
                            ## cnd could be (1) 'nc-kd', 'nc-drug', or drug ? other?
                            if prj_G.nodes[tnd]['genotype']!=['NONE']:
                                ## in this case there is at least 1 difference in genotype
                                ## AND one either one difference in treatment,
                                ##     OR a missing control ('nc-oe'/'nc-kd')
                                if not 'mut' in diff and not any(['nc-mut' in nd[2] for nd in prj_G.nodes()]):
                                    continue
                                
                            elif list(diff)[0].startswith('nc'):
                                ## exclude differences between control experiments
                                ## this is effectively the difference between a wild-type and control
                                continue
                            
                            edge_l.append((cnd,tnd))
                            if list(diff)[0].startswith(('kd','oe','mut')):
                                edge_d[(cnd,tnd)]['class']='gene_pert'
                                edge_d[(cnd,tnd)]['sense']=list(diff)[0]
                                edge_d[(cnd,tnd)]['name']=prj_G.nodes[cnd]['genotype'][0]
                            else:
                                edge_d[(cnd,tnd)]['class']='drug'
                                edge_d[(cnd,tnd)]['name']=list(diff)[0]
                                #edge_d[(cnd,tnd)]['name']=prj_G.nodes[cnd]['genotype'][0]
                        elif len(diff) > 0 and prj_G.nodes[cnd]['treatment'][0]=='nc-drug':
                            ## need a way to test if several treatments should be treated as a unit
                            ## perhaps have to hard-code this in?
                            ## PRJNA495849 -- 'TNFalpha; BV6; zVAD-fmk'
                            ## can check whether any of the constituent perturbations appear by
                            ## iterating over the nodes with the same cell line cell type and examining the treatment
                            for nd in prj_G.nodes():
                                if nd[0]!=tnd[0] or nd[1]!=tnd[1]:
                                    continue
                                if set(prj_G.nodes[nd]['treatment']) < set(prj_G.nodes[tnd]['treatment']):
                                    break
                            else:
                                ## in this case there are NO subsets
                                edge_l.append((cnd,tnd))
                                edge_d[(cnd,tnd)]['class']='drug'
                                edge_d[(cnd,tnd)]['name']='; '.join(list(diff))                                
                    else:
                        ## both have genotype need to count
                        tVCS = pd.Categorical(prj_G.nodes[tnd]['treatment']).value_counts()
                        cVCS = pd.Categorical(prj_G.nodes[cnd]['treatment']).value_counts()
                        if (tVCS.index.difference(cVCS.index).shape[0]) == 1:
                            ## need to CHECK COMMON TREATMENTS
                            isect = set(['kd','oe','mut']) & set(tVCS.index.intersection(cVCS.index))
                            if len(isect)>0:
                                ## if there are matching gene perturbations, need to check the genotypes
                                ## PRJNA224740
                                if not set(prj_G.nodes[cnd]['genotype']) < set(prj_G.nodes[tnd]['genotype']):
                                    continue
                            trt = tVCS[~tVCS.index.isin(cVCS.index)].index[0]
                            edge_l.append((cnd,tnd))
                            if trt.startswith(('kd','oe','mut')):
                                gtdiff = set(prj_G.nodes[tnd]['genotype']) - set(prj_G.nodes[cnd]['genotype'])
                                if len(gtdiff)==1:
                                    edge_d[(cnd,tnd)]['class'] = 'gene_pert'
                                    edge_d[(cnd,tnd)]['sense'] = trt
                                    edge_d[(cnd,tnd)]['name'] = list(gtdiff)[0]
                            else:
                                edge_d[(cnd,tnd)]['class'] = 'drug'
                                edge_d[(cnd,tnd)]['name'] = trt 
                        elif (tVCS.index.difference(cVCS.index).shape[0]) == 0:
                            ## since one is a singleton and there are no differences
                            ## this case is something like 'kd; kd' vs. 'kd' or 'oe; oe' vs. 'oe'
                            edge_d[(cnd,tnd)]['class'] = 'gene_pert'
                            edge_d[(cnd,tnd)]['sense'] = cVCS.index[0]
                            edge_d[(cnd,tnd)]['name'] = list(set(prj_G.nodes[tnd]['genotype']) - set(prj_G.nodes[cnd]['genotype']))[0]
                else:
                    ## both are single perts
                    #0. check whether trt == 'wt' for either -- continue if that is the case
                    if prj_G.nodes[nd1]['treatment'][0] == 'wt' or prj_G.nodes[nd2]['treatment'][0]=='wt':
                        continue
                    #1. check whether trt == 'nc' for either
                    isNC1 = prj_G.nodes[nd1]['treatment'][0].startswith('nc')
                    isNC2 = prj_G.nodes[nd2]['treatment'][0].startswith('nc')
                    #2. isNC1 XOR isNC2 generate an edge between the nc and the treatment
                    if sum([isNC1,isNC2]) == 1:
                        if isNC1:
                            cnd = nd1
                            tnd = nd2
                        else:
                            cnd = nd2
                            tnd = nd1
                        ## need to deal with experiments where there are multiple negative controls
                        if '-' in prj_G.nodes[cnd]['treatment'][0]:
                            elt = prj_G.nodes[cnd]['treatment'][0].split('-')[1]
                            if elt.startswith(('kd','oe','mut')) and elt!=prj_G.nodes[tnd]['treatment'][0]:
                                continue
                            elif elt=='drug' and prj_G.nodes[tnd]['treatment'][0].startswith(('kd','oe','mut')):
                                continue
                            elif elt in ['LNA','CRISPRi','RNAi','transfection']:
                                ## added in response to PRJEB18540
                                if prj_G.nodes[tnd]['treatment'][0] != ('kd-%s' % elt):
                                    continue
                                
                        edge_l.append((cnd,tnd))
                        if prj_G.nodes[tnd]['genotype'][0]!='NONE':
                            edge_d[(cnd,tnd)]['class']='gene_pert'
                            edge_d[(cnd,tnd)]['sense']=prj_G.nodes[tnd]['treatment'][0]
                            edge_d[(cnd,tnd)]['name']=prj_G.nodes[tnd]['genotype'][0]
                        else:
                            edge_d[(cnd,tnd)]['class']='drug'                         
                            edge_d[(cnd,tnd)]['name']=prj_G.nodes[tnd]['treatment'][0]
                    elif prj_G.nodes[nd1]['treatment'][0][:-1]==prj_G.nodes[nd2]['treatment'][0][:-1] \
                         and prj_G.nodes[nd1]['treatment'][0][-1] in ['-','+'] \
                         and prj_G.nodes[nd2]['treatment'][0][-1] in ['-','+']:
                        if diff1.endswith('-'):
                            cnd = nd1
                            tnd = nd2
                            diff = prj_G.nodes[nd2]['treatment'][0]
                        else:
                            cnd = nd2
                            tnd = nd1
                            diff = prj_G.nodes[nd1]['treatment'][0]
                        edge_d[(cnd,tnd)]['class']='receptor'                         
                        edge_d[(cnd,tnd)]['name']=diff
        prj_G.add_edges_from(edge_l)
        for u,v in prj_G.edges():
            for k,val in edge_d[(u,v)].items():
                prj_G.edges[u,v][k]=val
        prj_graph_d[prjno] = prj_G
    if overwrite:
        prj_graph_d = dict(prj_graph_d)
        with open('%s/project_graph_d%s.pkl' % (OUTPUT_PREFIX,version),'wb') as fh:
            P.dump(prj_graph_d,fh,-1)
    

def main():
    build_perturbation_network(XXX,True,[],False)
    ## NEED TO INCLUDE ENCODE 
    build_perturbation_network(YYY,True,['ENCODE'],True)


if __name__ == '__main__':
    main()
