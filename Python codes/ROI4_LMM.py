# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:13:07 2023

@author: CENs-Admin
"""

'''
Exract ROI4 similarity values for LMM
'''

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt

lpz_path= r'E:\SocialDetection7T\Nahid_Project\IS_RSA\Lorazepam\all_sub_similarity_lpz.npy'
plc_path= r'E:\SocialDetection7T\Nahid_Project\IS_RSA\Placebo\all_sub_similarity_plc.npy'


output_dir= r'E:\SocialDetection7T\Nahid_Project\IS_RSA\ROI_LMM'


plc_dict= np.load(plc_path, allow_pickle=True).item()
lpz_dict= np.load(lpz_path, allow_pickle=True).item()

plc_keys= list(plc_dict.keys())

lpz_keys= list(lpz_dict.keys())


#%% Get all LPZ RSMs for ROI4
sub_id= []
treatment= []
group= []
similarity_score= []


for paths in lpz_keys:
    
    
    lpz_rsm_ls= lpz_dict[paths][3].to_numpy()
    




    #intra stim_type (intra_stim) (higher in plc)
    
    u1_u1= lpz_rsm_ls[0:31, 0:31][np.triu_indices(31, k=1)]
    u2_u2= lpz_rsm_ls[31:66, 31:66][np.triu_indices(35, k=1)]
    f1_f1= lpz_rsm_ls[66:98, 66:98][np.triu_indices(32, k=1)]
    f2_f2= lpz_rsm_ls[98:128, 98:128][np.triu_indices(30, k=1)]
    nt_nt= lpz_rsm_ls[128:162, 128:162][np.triu_indices(34, k=1)]
    
    
    #np archtanh for fisher's z transformation
    intra_stim= np.arctanh(np.concatenate((u1_u1, u2_u2, f1_f1, f2_f2, nt_nt), axis= 0)).tolist()
    similarity_score.extend(intra_stim)
    
    intra_stim_ls=['E_Intra_Stimuli']*len(intra_stim)
    group.extend(intra_stim_ls)
    
    #intra class (intra_cls)
    
    u1_u2= lpz_rsm_ls[31:66, 0:31].flatten()
    f1_f2= lpz_rsm_ls[98:128, 66:98].flatten()
    
    intra_class=np.arctanh(np.concatenate((u1_u2, f1_f2), axis= 0)).tolist()
    similarity_score.extend(intra_class)
    
    intra_class_ls= ['D_Intra_Class']* len(intra_class)
    group.extend(intra_class_ls)
    
    
    
    #between class(inter_cls)
    
    u1_f1= lpz_rsm_ls[66:98, 0:31].flatten()
    u1_f2= lpz_rsm_ls[98:128, 0:31].flatten()
    u2_f1= lpz_rsm_ls[66:98,31:66 ].flatten()
    u2_f2= lpz_rsm_ls[98:128, 31:66].flatten()
    
    bet_class= np.arctanh(np.concatenate((u1_f1, u1_f2, u2_f1, u2_f2), axis= 0)).tolist()
    similarity_score.extend(bet_class)
    
    bet_class_ls= ['C_Inter_Class']*len(bet_class)
    group.extend(bet_class_ls)
    
    #nt vs U&F (nt_othr)
    u1_nt= lpz_rsm_ls[128:162, 0:31].flatten()
    u2_nt= lpz_rsm_ls[128:162, 31:66].flatten()
    f1_nt= lpz_rsm_ls[128:162,66:98].flatten()
    f2_nt= lpz_rsm_ls[128:162,98:128].flatten()
    
    nt_uf= np.arctanh(np.concatenate((u1_nt, u2_nt, f1_nt, f2_nt), axis= 0)).tolist()
    similarity_score.extend(nt_uf)
    
    nt_uf_ls= ['B_Neutral_Other']*len(nt_uf)
    group.extend(nt_uf_ls)
    
    #sr vs all (sr_all)
    u1_sr= lpz_rsm_ls[162:194, 0:31].flatten()
    u2_sr= lpz_rsm_ls[162:194, 31:66].flatten()
    f1_sr= lpz_rsm_ls[162:194, 66:98].flatten()
    f2_sr= lpz_rsm_ls[162:194,98:128].flatten()
    nt_sr= lpz_rsm_ls[162:194, 128:162].flatten()
    
    sr_all= np.arctanh(np.concatenate((u1_sr, u2_sr, f1_sr, f2_sr, nt_sr))).tolist()
    similarity_score.extend(sr_all)
    
    sr_all_ls= ['A_Scramble_Other']*len(sr_all)
    group.extend(sr_all_ls)
    
    sub= paths
    sub_id.extend([sub]*len(intra_stim+ intra_class+ bet_class+ nt_uf+ sr_all))
    
    treatment.extend(['LPZ']* len(intra_stim+ intra_class+ bet_class+ nt_uf+ sr_all))

df_drug= pd.DataFrame({'Sub_ID': sub_id, 'Treatment': treatment, 'Group': group, 'Similarity':similarity_score })


#%% Get all the PLC RSMs
sub_id= []
treatment= []
group= []
similarity_score= []


for paths in plc_keys:
    
    
    plc_rsm_ls= plc_dict[paths][3].to_numpy()

    
    #intra stim_type
    u1_u1= plc_rsm_ls[0:31, 0:31][np.triu_indices(31, k=1)]
    u2_u2= plc_rsm_ls[31:66, 31:66][np.triu_indices(35, k=1)]
    f1_f1= plc_rsm_ls[66:98, 66:98][np.triu_indices(32, k=1)]
    f2_f2= plc_rsm_ls[98:128, 98:128][np.triu_indices(30, k=1)]
    nt_nt= plc_rsm_ls[128:162, 128:162][np.triu_indices(34, k=1)]
    
    
    #intra_stim
    
    intra_stim= np.arctanh(np.concatenate((u1_u1, u2_u2, f1_f1, f2_f2, nt_nt), axis= 0)).tolist()
    similarity_score.extend(intra_stim)
    
    intra_stim_ls=['E_Intra_Stimuli']*len(intra_stim)
    group.extend(intra_stim_ls)
    
    #intra class
    
    u1_u2= plc_rsm_ls[31:66, 0:31].flatten()
    f1_f2= plc_rsm_ls[98:128, 66:98].flatten()
    
    intra_class=np.arctanh(np.concatenate((u1_u2, f1_f2), axis= 0)).tolist()
    similarity_score.extend(intra_class)
    
    intra_class_ls= ['D_Intra_Class']* len(intra_class)
    group.extend(intra_class_ls)
    
    
    #between class
    
    u1_f1= plc_rsm_ls[66:98, 0:31].flatten()
    u1_f2= plc_rsm_ls[98:128, 0:31].flatten()
    u2_f1= plc_rsm_ls[66:98,31:66 ].flatten()
    u2_f2= plc_rsm_ls[98:128, 31:66].flatten()
    
    bet_class= np.arctanh(np.concatenate((u1_f1, u1_f2, u2_f1, u2_f2), axis= 0)).tolist()
    similarity_score.extend(bet_class)
    
    bet_class_ls= ['C_Inter_Class']*len(bet_class)
    group.extend(bet_class_ls)
    
    #nt vs U&F
    u1_nt= plc_rsm_ls[128:162, 0:31].flatten()
    u2_nt= plc_rsm_ls[128:162, 31:66].flatten()
    f1_nt= plc_rsm_ls[128:162,66:98].flatten()
    f2_nt= plc_rsm_ls[128:162,98:128].flatten()
    
    nt_uf= np.arctanh(np.concatenate((u1_nt, u2_nt, f1_nt, f2_nt), axis= 0)).tolist()
    similarity_score.extend(nt_uf)
    
    nt_uf_ls= ['B_Neutral_Other']*len(nt_uf)
    group.extend(nt_uf_ls)
    
    #sr vs all
    u1_sr= plc_rsm_ls[162:194, 0:31].flatten()
    u2_sr= plc_rsm_ls[162:194, 31:66].flatten()
    f1_sr= plc_rsm_ls[162:194, 66:98].flatten()
    f2_sr= plc_rsm_ls[162:194,98:128].flatten()
    nt_sr= plc_rsm_ls[162:194, 128:162].flatten()
    
    
    
    sr_all= np.arctanh(np.concatenate((u1_sr, u2_sr, f1_sr, f2_sr, nt_sr))).tolist()
    similarity_score.extend(sr_all)
    
    sr_all_ls= ['A_Scramble_Other']*len(sr_all)
    group.extend(sr_all_ls)
    
    sub= paths
    sub_id.extend([sub]*len(intra_stim+ intra_class+ bet_class+ nt_uf+ sr_all))
    
    treatment.extend(['PLC']* len(intra_stim+ intra_class+ bet_class+ nt_uf+ sr_all))

df_control= pd.DataFrame({'Sub_ID': sub_id, 'Treatment': treatment, 'Group': group, 'Similarity':similarity_score })

#%%

df_roi4_lmm= pd.concat([df_control, df_drug])

df_roi4_lmm.to_csv(os.path.join(output_dir, 'ROI_4_LMM.csv'))


#%%Make raincloud plot

df_roi4_lmm= pd.read_csv(os.path.join(output_dir, 'ROI_4_LMM.csv'), sep= ',', index_col=0)


df_roi4_lmm.replace(to_replace='LPZ', value='LZP', inplace=True)

params = {'axes.labelsize': 14,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          'font.size':14}

plt.rcParams.update(params)


kwbox= {'box_showfliers':False}
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
dx="Group"; dy="Similarity"; hue= "Treatment"; ort="h"; pal = "Set2"; sigma = .05
f, ax = plt.subplots(figsize=(7, 7))

ax=pt.RainCloud(x = dx, y = dy, hue= hue, data = df_roi4_lmm, palette = pal, bw = sigma,
                 width_viol = 1, ax = ax, orient = ort, alpha= .5, dodge=True, move= .2, point_size=2, **kwbox)

plt.text(1.2, 4, 'a', color= 'g')
plt.text(1.2, 4.25, 'a', color= 'tab:orange')

plt.text(1.2, 3, 'bd', color= 'g')
plt.text(1.2, 3.25, 'bc', color= 'tab:orange')

plt.text(1.2, 2, 'bd', color= 'g')
plt.text(1.2, 2.25, 'bc', color= 'tab:orange')

plt.text(1.2, 1, 'bd', color= 'g')
plt.text(1.2, 1.25, 'bc', color= 'tab:orange')

plt.text(1.2, 0,'ce', color= 'g')
plt.text(1.2, 0.25,'de', color= 'tab:orange')

plt.xlabel('Similarity (z-transformed)')

ax.set_yticklabels(['Intra_Stimuli','Intra_Class', 'Inter_Class', 'Neutral_Other','Scramble_Other'])

plt.savefig(os.path.join(output_dir,'Raincloud_ROI4.png'), dpi= 300, bbox_inches= 'tight')