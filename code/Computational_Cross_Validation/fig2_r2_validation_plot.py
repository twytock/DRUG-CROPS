"""fig2_r2_validation_plot.py

Created by Thomas Wytock on 2023-04-23.
Copyright (c) 2023 Thomas Wytock. All rights reserved.

Makes the hexagonal binned 2-D histogram and accompanying subplots in Fig. 2
"""
import pandas as pd
import numpy as np
import argparse
import sys,os,os.path as osp
import matplotlib.pyplot as plt
import cmapPy.pandasGEXpress.parse as parse
from matplotlib.colors import LogNorm
from matplotlib import cm
plt.rcParams['svg.fonttype']='none'
import cmapPy.pandasGEXpress.parse as parse
from matplotlib import cm

lbl_cm_d = {'LOO':plt.get_cmap('Reds'),'OTHER_iso':plt.get_cmap('Greens'),
           'ENCODE_iso':plt.get_cmap('Blues'),'PRJDB6952_iso':plt.get_cmap('Oranges')}
#KW = sys.argv[1]
KIND = 'lopo'

DATA_PREFIX = '../../data/Computational_Cross_Validation'
OUTPUT_PREFIX = '../../output/Computational_Cross_Validation'

parser = argparse.ArgumentParser(description='Generate Fig. 2 from the leave-one-bioproject-out cross validation summary statistics.')

parser.add_argument('-V','--version',metavar='v',type=str,nargs=1,
                    help='version of files to use',
                    default=['_0122'])
args = parser.parse_args()
version = args.version[0]

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005


rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure

def weights(df):
    WT = df.std(axis=0,ddof=1).apply(lambda i: 1/i)
    if np.any(np.isinf(WT)):
        M = np.amax(WT[~np.isinf(WT)])
        WT[np.isinf(WT)]=2*M
    WT = WT.div(np.linalg.norm(WT.astype(np.float64)))
    return WT

sel_data = parse.parse('%s/drug_bc_corr_data%s.gctx' % (DATA_PREFIX,version)).data_df

WT = weights(sel_data)
#ver='_0122'
d2o = pd.read_pickle('%s/deviations_2ord_corr%s.pkl' % (OUTPUT_PREFIX,version))

pert_fn = '%s/drug_delta_selcorr_df%s.gctx' % (OUTPUT_PREFIX,version)
delta_gctx = parse.parse(pert_fn)
pert_df = delta_gctx.row_metadata_df
delta_df = delta_gctx.data_df

graph_d_fn = '%s/project_graph_d%s.pkl' % (OUTPUT_PREFIX,version)
graph_d = pd.read_pickle(graph_d_fn)
nd_data_head = ['cell_type','cell_line','treatment','genotype']


## NEED TO WRITE A function TO PLOT THE R^2 results (for linear & nonadditive methods)
## May consider creating a table to average over datasets?
def r2_hexbin(stat_df,pred_df):
    di_tup_bpj_ser = pd.read_pickle('%s/dev2bioproject%s.pkl' % (OUTPUT_PREFIX,version))
    INDS = di_tup_bpj_ser[di_tup_bpj_ser=='Dox_Digit_Torin'].index
    exp_valid = stat_df.loc[INDS]
    NINDS = di_tup_bpj_ser[di_tup_bpj_ser!='Dox_Digit_Torin'].index
    stat_df = stat_df.loc[NINDS.intersection(stats_df.index)]
    
    plt.rcParams['svg.fonttype']='none'
    vcm = cm.get_cmap('plasma')
    diff = stat_df.r_est-stat_df.r_lin
    avg = (stat_df.r_est+stat_df.r_lin)/2.
    h_lbl_spc = 0.1
    h_sm_spc = 0.02
    width = 0.72
    fig_width = 3.75
    fig_height = 6.25
    win_size=100
    ar = fig_height/fig_width
    v_lbl_spc = h_lbl_spc/ar
    v_sm_spc = h_sm_spc/ar
    height = width/ar
    DeltaX = pred_df.apply(np.linalg.norm,axis=1)
    fig = plt.figure(figsize=(fig_width,fig_height),dpi=180)
    pdiff = pd.concat([(stat_df.r_est - stat_df.r_lin),stat_df.r_lin],axis=1).sort_values('r_lin')
    ixMU_l = []
    iyMU_l = []
    iSTD_l = []
    for nelt in range(1,win_size):
        iyMU = pdiff.iloc[:nelt,0].sum()/nelt
        iyMU_l.append(iyMU)
        ixMU = pdiff.iloc[:nelt,1].sum()/nelt
        ixMU_l.append(ixMU)
        iMUSQR = pdiff.iloc[:nelt,0].apply(np.square).sum()/nelt
        iSTD = np.sqrt((iMUSQR-iyMU*iyMU)/(nelt-1)) if nelt > 1 else 0
        iSTD_l.append(iSTD)
    roll_stats = pdiff.rolling(win_size)
    MUS = roll_stats.mean()
    SIGS = roll_stats.std()
    SIGS = SIGS[~MUS.r_lin.isna()]/np.sqrt(win_size)
    MUS = MUS[~MUS.r_lin.isna()]
    
    xMUS = np.r_[ixMU_l,MUS.iloc[:,1].values]
    yMUS = np.r_[iyMU_l,MUS.iloc[:,0].values]
    ySIGS = np.r_[iSTD_l,SIGS.iloc[:,0].values]
    ySIGS[0] = np.amax(ySIGS)
    
    rect_scatter = [h_lbl_spc, 2*v_lbl_spc+0.5*height, width, height]
    cbax = fig.add_axes([h_lbl_spc + width + h_sm_spc, 2*v_lbl_spc+0.5*height,0.04,height])
    #isect0 = diff.index.intersection(corrections.index)
    rect_histtop = [h_lbl_spc, 2*v_lbl_spc+1.5*height+v_sm_spc , width, 0.5*height]
    rect_histbot = [h_lbl_spc, v_lbl_spc, width, 0.5*height]
    
    ax = fig.add_axes(rect_scatter)
    hb = ax.hexbin(stat_df.r_lin,stat_df.r_est,norm=LogNorm(vmin=1),cmap=vcm,mincnt=1)
    ax.scatter(exp_valid.r_lin,exp_valid.r_est,color='C9',label='RNA-seq Validation',alpha=0.75)
    ax.fill_betweenx(np.linspace(-0.025,1.01),xMUS[np.where(yMUS<0)[0][0]],xMUS[np.where(yMUS>0)[0][-1]],color='#edb120',alpha=0.75)
    #ax.axvline(xMUS[np.where(yMUS<0)[0][0]],ls=':',color='C3')
    #ax.axvline(xMUS[np.where(yMUS>0)[0][-1]],ls=':',color='#edb120')
    leg = ax.legend(prop={'size':7},frameon=False)
    cb = fig.colorbar(hb,cax=cbax)
    cb.set_label('Count')
    ax.set_ylim(-0.025,1.01)
    #ax.set_xlim(-0.02,1.02)
    ax.set_xlabel(r'$R^2_{\mathrm{A}}$')
    ax.set_ylabel(r'$R^2_{\mathrm{NA}}$')
    
    bg_ln = ax.plot(np.linspace(0,1),np.linspace(0,1),ls='--',color='C7')
    ax_histtop = fig.add_axes(rect_histtop, sharex=ax)
    
    ax_histtop.fill_between(xMUS,(yMUS-2*ySIGS),(yMUS+2*ySIGS),alpha=0.7,color='C4')
    ax_histtop.plot(xMUS,yMUS,color='C4')
    ax_histtop.fill_betweenx(np.linspace(-0.025,0.11),xMUS[np.where(yMUS<0)[0][0]],xMUS[np.where(yMUS>0)[0][-1]],color='#edb120',alpha=0.75)
    #ax_histtop.axvline(xMUS[np.where(yMUS<0)[0][0]],ls=':',color='C3') #0.8619167894487159
    #ax_histtop.axvline(xMUS[np.where(yMUS>0)[0][-1]],ls=':',color='#edb120') #0.938224559985451
    ax_histtop.scatter(exp_valid.r_lin,exp_valid.r_est-exp_valid.r_lin,color='C9',alpha=0.75)
    ax_histtop.tick_params(axis="x", labelbottom=False)
    ax_histtop.set_ylabel(r'$R^2_{\mathrm{NA}}-R^2_{\mathrm{A}}$')
    ax_histtop.axhline(0,ls='--',color='C7')
    ax_histtop.set_ylim(-0.025,0.11)
    roll_delta_stats = pd.concat([(stat_df.r_est - stat_df.r_lin),DeltaX.loc[stat_df.index]],axis=1).sort_values(1).rolling(win_size)
    delta_MUS = roll_delta_stats.mean()
    delta_SIGS = roll_delta_stats.std()
    delta_SIGS = delta_SIGS[~delta_MUS.iloc[:,1].isna()]/np.sqrt(win_size)
    delta_MUS = delta_MUS[~delta_MUS.iloc[:,1].isna()]
    delta_xMUS = delta_MUS.iloc[:,1].values
    delta_yMUS = delta_MUS.iloc[:,0].values
    delta_ySIGS = delta_SIGS.iloc[:,0].values
    ax_histbot = fig.add_axes(rect_histbot)
    ax_histbot.axhline(0,ls='--',color='C7')
    ax_histbot.fill_between(delta_xMUS,(delta_yMUS-2*delta_ySIGS),(delta_yMUS+2*delta_ySIGS),alpha=0.7,color='C4')
    ax_histbot.plot(delta_xMUS,delta_yMUS,color='C4')
    ax_histbot.scatter(DeltaX.loc[exp_valid.index],exp_valid.r_est-exp_valid.r_lin,color='C9',alpha=0.75)
    #ax_histbot.tick_params(axis="x", labelbottom=False)
    ax_histbot.set_ylabel(r'$R^2_{\mathrm{NA}}-R^2_{\mathrm{A}}$')
    ax_histbot.set_xlabel(r'$||\mathbf{\Delta X}||$')
    ax_histbot.set_ylim(-0.025,0.11)
    #ax_histy.hist(stat_df.r_est, bins=binsy, orientation='horizontal',histtype='step',color='C0')
    #ax_histy.axhline(0,ls=':',color='C7')
    #ax_histy.set_xlabel('Count')
    #ax_histy.set_xscale('log')
    #fig.tight_layout()
    plt.setp(ax.get_yticklabels(),size=7)
    plt.setp(ax.get_xticklabels(),size=7)
    plt.setp(ax_histtop.get_yticklabels(),size=7)
    plt.setp(ax_histbot.get_yticklabels(),size=7)
    plt.setp(ax_histbot.get_xticklabels(),size=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(),size=7)
    fig.savefig('%s/diff_v_avg_updated%s.svg' % (OUTPUT_PREFIX,version))
    return

    
if __name__ == '__main__':
    FN = '%s/%s_stats%s.pkl' % (OUTPUT_PREFIX,KIND,version)
    FN2 = '%s/predictions_%s%s.pkl' % (OUTPUT_PREFIX,KIND,version)
    stats_df = pd.read_pickle(FN)
    pred_df = pd.read_pickle(FN2)
    r2_hexbin(stats_df,pred_df)
    
    