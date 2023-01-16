# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:08:56 2022

@author: CENs-Admin
"""

'''
Backgroud of the study:

1. Biological motion task
2.59 participants [30 LPZ, 29 PLC]
3. 6 conditions in total [U1, U2, F1, F2, NT, SR]
4. 194 trials/run [31, 35, 32, 30, 34, 32]
5. RSM dimension (194x194)
'''
#%%
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from nilearn import image
import pingouin as pg

#%% Get the paths
mask_img_path_name = r'F:\SocialDetection7T\Nahid_Project\Masks\NIm2019_BM_BMscram_0p05FWEclus_0p001u_k40.nii'
rsm_glm_path= r'F:\SocialDetection7T\Nahid_Project\RSM_GLMSingle\BioM_mask_across_r_noavg'    
treatment_grp_path= 'F:\SocialDetection7T\BehavioralData\LOXY_LZPtrial_forJohannes_noMissing.csv'
treatment_grp= pd.read_csv(treatment_grp_path, sep= ';', on_bad_lines='skip')
min_tr= [31, 35, 32, 30, 34, 32] #minimum no. of trials per condition
labels= ['U1', 'U2', 'F1', 'F2', 'NT', 'SR']


#%% Mask Image
#let's see the mask 

mask_img= image.binarize_img(nib.load(mask_img_path_name))
plotting.plot_glass_brain(mask_img, colorbar=True)

#%% Create tick labels for plotting 
#x_y ticks required for plotting
x_y_ticks= [0]*len(min_tr)

for i in range(0, len(min_tr)):
   x_y_ticks[i]= min_tr[i]+ x_y_ticks[i-1]
   

#%% Grouping subjects based on treatement    
#create 2 columns with subject id and group assignment
lpz_plc= pd.concat([treatment_grp['Probandencode'], treatment_grp['treatment_code']], axis= 1, keys=['Sub','Group'])

#add the missing sub ids
lpz_plc.loc[len(lpz_plc)]=['LOXY_110', 1]
lpz_plc.loc[len(lpz_plc)]=['LOXY_168', 1]

#remove sub139
lpz_plc=lpz_plc.drop([13])

#divide between group lpz=1 and plc=0
lpz_grp= lpz_plc[lpz_plc['Group']==1]['Sub'].tolist()

plc_grp= lpz_plc[lpz_plc['Group']==0]['Sub'].tolist()

#make path names based on groups
lpz_grp_path_ls= [os.path.join(rsm_glm_path, sub) for sub in lpz_grp ]

plc_grp_path_ls= [os.path.join(rsm_glm_path, sub) for sub in plc_grp ]

#%% Get all LPZ RSMs
sub_id= []
treatment= []
group= []
similarity_score= []


for paths in range(0, len(lpz_grp_path_ls)):
    
    rsm_path_lpz= glob.glob(lpz_grp_path_ls[paths] + '\*RSM.csv')
    lpz_rsm_ls= pd.read_csv(rsm_path_lpz[0], sep=',').to_numpy()
    




    #intra stim_type (intra_stim) (higher in plc)
    
    u1_u1= lpz_rsm_ls[0:31, 0:31][np.triu_indices(31, k=1)]
    u2_u2= lpz_rsm_ls[31:66, 31:66][np.triu_indices(35, k=1)]
    f1_f1= lpz_rsm_ls[66:98, 66:98][np.triu_indices(32, k=1)]
    f2_f2= lpz_rsm_ls[98:128, 98:128][np.triu_indices(30, k=1)]
    nt_nt= lpz_rsm_ls[128:162, 128:162][np.triu_indices(34, k=1)]
    
    
    #np archtanh for fisher's z transformation
    intra_stim= np.arctanh(np.concatenate((u1_u1, u2_u2, f1_f1, f2_f2, nt_nt), axis= 0)).tolist()
    similarity_score.extend(intra_stim)
    
    intra_stim_ls=['Intra_Stimuli']*len(intra_stim)
    group.extend(intra_stim_ls)
    
    #intra class (intra_cls)
    
    u1_u2= lpz_rsm_ls[31:66, 0:31].flatten()
    f1_f2= lpz_rsm_ls[98:128, 66:98].flatten()
    
    intra_class=np.arctanh(np.concatenate((u1_u2, f1_f2), axis= 0)).tolist()
    similarity_score.extend(intra_class)
    
    intra_class_ls= ['Intra_Class']* len(intra_class)
    group.extend(intra_class_ls)
    
    
    
    #between class(inter_cls)
    
    u1_f1= lpz_rsm_ls[66:98, 0:31].flatten()
    u1_f2= lpz_rsm_ls[98:128, 0:31].flatten()
    u2_f1= lpz_rsm_ls[66:98,31:66 ].flatten()
    u2_f2= lpz_rsm_ls[98:128, 31:66].flatten()
    
    bet_class= np.arctanh(np.concatenate((u1_f1, u1_f2, u2_f1, u2_f2), axis= 0)).tolist()
    similarity_score.extend(bet_class)
    
    bet_class_ls= ['Inter_Class']*len(bet_class)
    group.extend(bet_class_ls)
    
    #nt vs U&F (nt_othr)
    u1_nt= lpz_rsm_ls[128:162, 0:31].flatten()
    u2_nt= lpz_rsm_ls[128:162, 31:66].flatten()
    f1_nt= lpz_rsm_ls[128:162,66:98].flatten()
    f2_nt= lpz_rsm_ls[128:162,98:128].flatten()
    
    nt_uf= np.arctanh(np.concatenate((u1_nt, u2_nt, f1_nt, f2_nt), axis= 0)).tolist()
    similarity_score.extend(nt_uf)
    
    nt_uf_ls= ['Neutral_Other']*len(nt_uf)
    group.extend(nt_uf_ls)
    
    #sr vs all (sr_all)
    u1_sr= lpz_rsm_ls[162:194, 0:31].flatten()
    u2_sr= lpz_rsm_ls[162:194, 31:66].flatten()
    f1_sr= lpz_rsm_ls[162:194, 66:98].flatten()
    f2_sr= lpz_rsm_ls[162:194,98:128].flatten()
    nt_sr= lpz_rsm_ls[162:194, 128:162].flatten()
    
    sr_all= np.arctanh(np.concatenate((u1_sr, u2_sr, f1_sr, f2_sr, nt_sr))).tolist()
    similarity_score.extend(sr_all)
    
    sr_all_ls= ['Scramble_Other']*len(sr_all)
    group.extend(sr_all_ls)
    
    sub= lpz_grp[paths]
    sub_id.extend([sub]*len(intra_stim+ intra_class+ bet_class+ nt_uf+ sr_all))
    
    treatment.extend(['Drug']* len(intra_stim+ intra_class+ bet_class+ nt_uf+ sr_all))

df_drug= pd.DataFrame({'Sub_ID': sub_id, 'Treatment': treatment, 'Group': group, 'Similarity_Score':similarity_score })

#%% Make a dataframe with col_names: Treatment, RSM_Group, Similarity

lpz_df_anova= pd.DataFrame({'RSM_Group': intra_stim_ls+ intra_class_ls+ bet_class_ls+nt_uf_ls+ sr_all_ls, 
                            'Similarity': intra_stim+ intra_class+ bet_class+ nt_uf+ sr_all })

lpz_df_anova.insert(0, 'Treatment', ['LPZ']*len(lpz_df_anova['RSM_Group']), True)

sn.violinplot(data= lpz_df_anova, x= 'RSM_Group', y= 'Similarity')
#%% Show the mean LPZ RSM
ax1= sn.heatmap(lpz_rsm_ls, cmap= 'viridis', vmin=0, vmax=1)
ax1.set_xticks(x_y_ticks)
ax1.set_yticks(x_y_ticks)
ax1.set_xticklabels(labels, minor= False, rotation=90)
ax1.set_yticklabels(labels, minor= False, rotation=90)
plt.show()

#%% Get all the PLC RSMs
sub_id= []
treatment= []
group= []
similarity_score= []


for paths in range(0, len(plc_grp_path_ls)):
    
    rsm_path_plc= glob.glob(plc_grp_path_ls[paths] + '\*RSM.csv')
    plc_rsm_ls= pd.read_csv(rsm_path_plc[0], sep=',').to_numpy()

    
    #intra stim_type
    u1_u1= plc_rsm_ls[0:31, 0:31][np.triu_indices(31, k=1)]
    u2_u2= plc_rsm_ls[31:66, 31:66][np.triu_indices(35, k=1)]
    f1_f1= plc_rsm_ls[66:98, 66:98][np.triu_indices(32, k=1)]
    f2_f2= plc_rsm_ls[98:128, 98:128][np.triu_indices(30, k=1)]
    nt_nt= plc_rsm_ls[128:162, 128:162][np.triu_indices(34, k=1)]
    sr_sr= plc_rsm_ls[162:194,162:194][np.triu_indices(32, k=1)]
    
    #intra_stim
    
    intra_stim= np.arctanh(np.concatenate((u1_u1, u2_u2, f1_f1, f2_f2, nt_nt), axis= 0)).tolist()
    similarity_score.extend(intra_stim)
    
    intra_stim_ls=['Intra_Stimuli']*len(intra_stim)
    group.extend(intra_stim_ls)
    
    #intra class
    
    u1_u2= plc_rsm_ls[31:66, 0:31].flatten()
    f1_f2= plc_rsm_ls[98:128, 66:98].flatten()
    
    intra_class=np.arctanh(np.concatenate((u1_u2, f1_f2), axis= 0)).tolist()
    similarity_score.extend(intra_class)
    
    intra_class_ls= ['Intra_Class']* len(intra_class)
    group.extend(intra_class_ls)
    
    
    #between class
    
    u1_f1= plc_rsm_ls[66:98, 0:31].flatten()
    u1_f2= plc_rsm_ls[98:128, 0:31].flatten()
    u2_f1= plc_rsm_ls[66:98,31:66 ].flatten()
    u2_f2= plc_rsm_ls[98:128, 31:66].flatten()
    
    bet_class= np.arctanh(np.concatenate((u1_f1, u1_f2, u2_f1, u2_f2), axis= 0)).tolist()
    similarity_score.extend(bet_class)
    
    bet_class_ls= ['Inter_Class']*len(bet_class)
    group.extend(bet_class_ls)
    
    #nt vs U&F
    u1_nt= plc_rsm_ls[128:162, 0:31].flatten()
    u2_nt= plc_rsm_ls[128:162, 31:66].flatten()
    f1_nt= plc_rsm_ls[128:162,66:98].flatten()
    f2_nt= plc_rsm_ls[128:162,98:128].flatten()
    
    nt_uf= np.arctanh(np.concatenate((u1_nt, u2_nt, f1_nt, f2_nt), axis= 0)).tolist()
    similarity_score.extend(nt_uf)
    
    nt_uf_ls= ['Neutral_Other']*len(nt_uf)
    group.extend(nt_uf_ls)
    
    #sr vs all
    u1_sr= plc_rsm_ls[162:194, 0:31].flatten()
    u2_sr= plc_rsm_ls[162:194, 31:66].flatten()
    f1_sr= plc_rsm_ls[162:194, 66:98].flatten()
    f2_sr= plc_rsm_ls[162:194,98:128].flatten()
    nt_sr= plc_rsm_ls[162:194, 128:162].flatten()
    
    
    
    sr_all= np.arctanh(np.concatenate((u1_sr, u2_sr, f1_sr, f2_sr, nt_sr))).tolist()
    similarity_score.extend(sr_all)
    
    sr_all_ls= ['Scramble_Other']*len(sr_all)
    group.extend(sr_all_ls)
    
    sub= plc_grp[paths]
    sub_id.extend([sub]*len(intra_stim+ intra_class+ bet_class+ nt_uf+ sr_all))
    
    treatment.extend(['Control']* len(intra_stim+ intra_class+ bet_class+ nt_uf+ sr_all))

df_control= pd.DataFrame({'Sub_ID': sub_id, 'Treatment': treatment, 'Group': group, 'Similarity_Score':similarity_score })

df_lmm= pd.concat([df_control, df_drug])

df_lmm.to_csv(r'E:\SocialDetection7T\Nahid_Project\RSM_GLMSingle\BioM_mask_lmm\BioM_mask_df_lmm.csv')





#%% Show the mean PLC RSM
ax1= sn.heatmap(plc_rsm_ls, cmap= 'viridis', vmin= 0, vmax=1)
ax1.set_xticks(x_y_ticks)
ax1.set_yticks(x_y_ticks)
ax1.set_xticklabels(labels, minor= False, rotation=90)
ax1.set_yticklabels(labels, minor= False, rotation=90)
plt.show()

'''
Now we divide the RSM (194*194) into different groups based stimulation type:

1. Intra_stimulation: U1_U1, U2_U2, F1_F1....
2. Intra_class: U1_U2, F1_F2
3. Inter_class: U1_F1, U1_F2, U2_F1, U2_F2
4. NT_ohter: NT_U1....
5. SR_all: SR_U1

Z transform the data for paramatric tests eg. ANOVA
'''







#%% Make a dataframe for PLC too
plc_df_anova= pd.DataFrame({'RSM_Group': intra_stim_ls+ intra_class_ls+ bet_class_ls+nt_uf_ls+ sr_all_ls, 
                            'Similarity': intra_stim+ intra_class+ bet_class+ nt_uf+ sr_all })

plc_df_anova.insert(0, 'Treatment', ['PLC']*len(plc_df_anova['RSM_Group']), True)

sn.violinplot(data= plc_df_anova, x= 'RSM_Group', y= 'Similarity')


#%% Combine two dataframes for running ANOVA
rsm_df_anova= pd.concat([lpz_df_anova, plc_df_anova])

sn.boxplot(data= rsm_df_anova, x= 'RSM_Group', y= 'Similarity', hue= 'Treatment')

#Run factorial anova

fact_anova= pg.anova(dv='Similarity', between=['Treatment', 'RSM_Group'], data= rsm_df_anova, detailed= True)
print(fact_anova)

194*194*2
59*36500


arrays1= np.array([[1,2,3], [4,5,6]])
arrays_2= np.array([[7,8,9], [10,11,12]])
axis_3= np.stack((arrays1, arrays_2), axis=2)
axis_3[0,0:2,0]

x=[]

x.extend(['test']*4)

