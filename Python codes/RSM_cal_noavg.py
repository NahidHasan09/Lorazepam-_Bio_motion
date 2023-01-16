# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:40:41 2022

@author: CENs-Admin
"""

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from  nilearn import image
from nilearn.maskers import NiftiMasker

root_path= r'E:\SocialDetection7T\Nahid_Project\RSM_GLMSingle\BioM_mask_across_r_noavg'
data_path= r'E:\SocialDetection7T\Nahid_Project\GLMSingle_Outputs'
eventfs_path= r'E:\SocialDetection7T\onsets_Logfiles'
mask_img_path= r'E:\SocialDetection7T\Nahid_Project\Masks\NIm2019_BM_BMscram_0p05FWEclus_0p001u_k40.nii'
mask_img= nib.load(mask_img_path)

list_of_folder_data_path = sorted(filter(lambda x: os.path.isdir(os.path.join(data_path, x)),
                                   os.listdir(data_path)))

temp_path_ls_glm = []
for folder_name in list_of_folder_data_path:
    temp_path_ls_glm.append(os.path.join(data_path, folder_name))


for imgs in range(0, len(temp_path_ls_glm)):
    sub_id= temp_path_ls_glm[imgs][53:61]
    print(f'Now working with subject# {sub_id} ')
    nifti_path= glob.glob(temp_path_ls_glm[imgs]+ '\L*.nii')
    
    eventfs= glob.glob(eventfs_path + '\socialDetection-fMRI-'+ sub_id + '*.txt')


    flt_bio_scrm= []
    flt_cons= []
    
    for  eventf in  eventfs:
    
    
        dm_read= pd.read_csv(eventf, sep=",", header=None, names= ["trial_no", "stimtype", "no_need", "biomotion", 
                                                               "accuracy", "onsets", "offsets", "correct", "RT"])
    
        bio_m_scrm= dm_read["biomotion"].values
        conditions= dm_read['stimtype'].values
    
        flt_bio_scrm.append(bio_m_scrm)
        flt_cons.append(conditions)

    flt_bio_scrm_all= flt_bio_scrm[0].tolist() + flt_bio_scrm[1].tolist()
    
    flt_cons_all= flt_cons[0].tolist() + flt_cons[1].tolist()

    betas_3d= nib.funcs.four_to_three(nib.load(nifti_path[0]))
    
    df_across_run= pd.DataFrame({'trial_type': flt_cons_all, 'bio_m': flt_bio_scrm_all, 'nifti_3d': betas_3d})
    

    
    # step 2
    #run1
    run1_con1= df_across_run[(df_across_run.bio_m != 0) & (df_across_run.trial_type==1)]
    
    run1_con2= df_across_run[(df_across_run.bio_m != 0) & (df_across_run.trial_type==2)]
    
    run1_con3= df_across_run[(df_across_run.bio_m != 0) & (df_across_run.trial_type==3)]
    
    run1_con4= df_across_run[(df_across_run.bio_m != 0) & (df_across_run.trial_type==4)]
    
    run1_con5= df_across_run[(df_across_run.bio_m != 0) & (df_across_run.trial_type==5)]
    
    run1_con6= df_across_run[(df_across_run.bio_m == 0)]
    
   
    #step 3
    
    #run1
    run1_con1_mimg= nib.concat_images(run1_con1['nifti_3d'].tolist())
    
    run1_con2_mimg= nib.concat_images(run1_con2['nifti_3d'].tolist())
    
    run1_con3_mimg= nib.concat_images(run1_con3['nifti_3d'].tolist())
    
    run1_con4_mimg= nib.concat_images(run1_con4['nifti_3d'].tolist())
    
    run1_con5_mimg= nib.concat_images(run1_con5['nifti_3d'].tolist())
    
    run1_con6_mimg= nib.concat_images(run1_con6['nifti_3d'].tolist())
    
    
    
    
    #step 4 masking
    affine= run1_con1_mimg.affine
    shape=run1_con1_mimg.get_fdata()[:,:,:,0].shape
    masker= NiftiMasker( target_affine=affine, target_shape=shape,
                             mask_strategy='background')
    masker.fit(mask_img)
    
    #run1
    run1_con1_voxles= masker.transform_single_imgs(run1_con1_mimg).T
    run1_con2_voxles= masker.transform_single_imgs(run1_con2_mimg).T
    run1_con3_voxles= masker.transform_single_imgs(run1_con3_mimg).T
    run1_con4_voxles= masker.transform_single_imgs(run1_con4_mimg).T
    run1_con5_voxles= masker.transform_single_imgs(run1_con5_mimg).T
    run1_con6_voxles= masker.transform_single_imgs(run1_con6_mimg).T
    
    
    com_run= np.concatenate((run1_con1_voxles[:, 0:31], run1_con2_voxles[:, 0:35], run1_con3_voxles[:, 0:32], run1_con4_voxles[:, 0:30], run1_con5_voxles[:, 0:34], run1_con6_voxles[:, 0:32]), axis=1)
 
   
    
    corr_df_across_run= pd.DataFrame(com_run)
    
    
    rsm= corr_df_across_run.corr('spearman')
    rdm= 1-rsm
    
    
    #Saving RSMs as csv
    print(f'Now saving {sub_id}')
    output_dir= os.path.join(root_path, sub_id)
    os.makedirs(output_dir)
    rsm.to_csv(os.path.join(output_dir, sub_id + 'RSM.csv'), header= True, index= True, index_label= False)
    rdm.to_csv(os.path.join(output_dir, sub_id + 'RDM.csv'), header= True, index= True, index_label= False)




'''
Group level rsm analysis for glmsingle
'''

import os
import glob
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

import pingouin as pg

rsm_glm_path= r'E:\SocialDetection7T\Nahid_Project\RSM_GLMSingle\BioM_mask_across_r_noavg'    
treatment_grp_path= 'E:\SocialDetection7T\BehavioralData\LOXY_LZPtrial_forJohannes_noMissing.csv'
treatment_grp= pd.read_csv(treatment_grp_path, sep= ';', on_bad_lines='skip')
min_tr= [31, 35, 32, 30, 34, 32] #minimum no. of trials per condition
labels= ['U1', 'U2', 'F1', 'F2', 'NT', 'SR']



#x_y ticks required for plotting
x_y_ticks= [0]*len(min_tr)

for i in range(0, len(min_tr)):
   x_y_ticks[i]= min_tr[i]+ x_y_ticks[i-1]

#create 2 columns with subject id and group assignment
lpz_plc= pd.concat([treatment_grp['Probandencode'], treatment_grp['treatment_code']], axis= 1, keys=['Sub','Group'])

#add the missing sub ids
lpz_plc.loc[len(lpz_plc)]=['LOXY_110', 1]
lpz_plc.loc[len(lpz_plc)]=['LOXY_168', 1]

#remove sub139
lpz_plc=lpz_plc.drop([13])

#group by group assignment
lpz_plc.sort_values(by=['Group', 'Sub'], ascending= True)

#divide between group lpz=1 and plc=0
lpz_grp= lpz_plc[lpz_plc['Group']==1]['Sub'].tolist()

plc_grp= lpz_plc[lpz_plc['Group']==0]['Sub'].tolist()

#make path names based on groups
lpz_grp_path_ls= [os.path.join(rsm_glm_path, sub) for sub in lpz_grp ]

plc_grp_path_ls= [os.path.join(rsm_glm_path, sub) for sub in plc_grp ]

#get all lpz rdms 
lpz_rdm= []

for paths in range(0, len(lpz_grp_path_ls)):
    
    rdm_path_lpz= glob.glob(lpz_grp_path_ls[paths] + '\*RSM.csv')
    lpz_rdm_ls= pd.read_csv(rdm_path_lpz[0], sep=',').to_numpy()
    lpz_rdm.append(lpz_rdm_ls)


#make average lpz rdm using numpy
mean_lpz_rdm= np.mean(lpz_rdm, axis=0)

mean_lpz_rdm.max()
#get upper triangle
up_tri_lpz= mean_plc_rdm[np.triu_indices(194, k=1)]

#get paired values

#intra stim_type (intra_stim)

u1_u1= mean_lpz_rdm[0:31, 0:31][np.triu_indices(31, k=1)]
u2_u2= mean_lpz_rdm[31:66, 31:66][np.triu_indices(35, k=1)]
f1_f1= mean_lpz_rdm[66:98, 66:98][np.triu_indices(32, k=1)]
f2_f2= mean_lpz_rdm[98:128, 98:128][np.triu_indices(30, k=1)]
nt_nt= mean_lpz_rdm[128:162, 128:162][np.triu_indices(34, k=1)]
sr_sr= mean_lpz_rdm[162:194,162:194][np.triu_indices(32, k=1)]

#np archtanh for fisher's z transformation
intra_stim= np.arctanh(np.concatenate((u1_u1, u2_u2, f1_f1, f2_f2, nt_nt), axis= 0)).tolist()
intra_stim_ls=['intra_stim']*len(intra_stim)


res_1= intra_stim.reshape(len(intra_stim), 1)
scp.normaltest(intra_stim)
sn.histplot(intra_stim)

#intra class (intra_cls)

u1_u2= mean_lpz_rdm[31:66, 0:31].flatten()
f1_f2= mean_lpz_rdm[98:128, 66:98].flatten()

intra_class=np.arctanh(np.concatenate((u1_u2, f1_f2), axis= 0)).tolist()
intra_class_ls= ['intra_cls']* len(intra_class)


res_2= intra_class.reshape(len(intra_class), 1)
scp.normaltest(intra_class)
sn.histplot(intra_class)


#between class(inter_cls)

u1_f1= mean_lpz_rdm[66:98, 0:31].flatten()
u1_f2= mean_lpz_rdm[98:128, 0:31].flatten()
u2_f1= mean_lpz_rdm[66:98,31:66 ].flatten()
u2_f2= mean_lpz_rdm[98:128, 31:66].flatten()

bet_class= np.arctanh(np.concatenate((u1_f1, u1_f2, u2_f1, u2_f2), axis= 0)).tolist()
bet_class_ls= ['inter_cls']*len(bet_class)

res_3= bet_class.reshape(len(bet_class), 1)
scp.normaltest(bet_class)
sn.histplot(bet_class)

#nt vs U&F (nt_othr)
u1_nt= mean_lpz_rdm[128:162, 0:31].flatten()
u2_nt= mean_lpz_rdm[128:162, 31:66].flatten()
f1_nt= mean_lpz_rdm[128:162,66:98].flatten()
f2_nt= mean_lpz_rdm[128:162,98:128].flatten()

nt_uf= np.arctanh(np.concatenate((u1_nt, u2_nt, f1_nt, f2_nt), axis= 0)).tolist()
nt_uf_ls= ['nt_othr']*len(nt_uf)

res_4= nt_uf.reshape(len(nt_uf), 1)
scp.normaltest(nt_uf)
sn.histplot(nt_uf)

#sr vs all (sr_all)
u1_sr= mean_lpz_rdm[162:194, 0:31].flatten()
u2_sr= mean_lpz_rdm[162:194, 31:66].flatten()
f1_sr= mean_lpz_rdm[162:194, 66:98].flatten()
f2_sr= mean_lpz_rdm[162:194,98:128].flatten()
nt_sr= mean_lpz_rdm[162:194, 128:162].flatten()

sr_all= np.arctanh(np.concatenate((u1_sr, u2_sr, f1_sr, f2_sr, nt_sr))).tolist()
sr_all_ls= ['sr_all']*len(sr_all)

res_5= sr_all.reshape(len(sr_all), 1)

scp.normaltest(sr_all)
sn.histplot(sr_all)

# make a dataframe for lpz labelling groups[col_names: Treatment, RSM_Group, Similarity]

lpz_df_anova= pd.DataFrame({'RSM_Group': intra_stim_ls+ intra_class_ls+ bet_class_ls+nt_uf_ls+ sr_all_ls, 
                            'Similarity': intra_stim+ intra_class+ bet_class+ nt_uf+ sr_all })

lpz_df_anova.insert(0, 'Treatment', ['LPZ']*len(lpz_df_anova['RSM_Group']), True)

sn.boxenplot(data= lpz_df_anova, x= 'RSM_Group', y= 'Similarity')

anova_con= np.concatenate((res_1, res_2, res_3, res_4, res_5), axis=1)

scp.f_oneway(intra_stim, intra_class, bet_class, nt_uf, sr_all, axis=0)

#plot lpz rdm 
ax1= sn.heatmap(mean_lpz_rdm, cmap= 'viridis')
ax1.set_xticks(x_y_ticks)
ax1.set_yticks(x_y_ticks)
ax1.set_xticklabels(labels, minor= False, rotation=90)
ax1.set_yticklabels(labels, minor= False, rotation=90)
plt.show()

#get all plc rdms 
plc_rdm= []

for paths in range(0, len(plc_grp_path_ls)):
    
    rdm_path_plc= glob.glob(plc_grp_path_ls[paths] + '\*RSM.csv')
    plc_rdm_ls= pd.read_csv(rdm_path_plc[0], sep=',').to_numpy()
    plc_rdm.append(plc_rdm_ls)

#make average lpz rdm using numpy
mean_plc_rdm= np.mean(plc_rdm, axis=0)

up_tri_plc= mean_plc_rdm[np.triu_indices(194, k=1)]


#get paired values

#intra stim_type
u1_u1= mean_plc_rdm[0:31, 0:31][np.triu_indices(31, k=1)]
u2_u2= mean_plc_rdm[31:66, 31:66][np.triu_indices(35, k=1)]
f1_f1= mean_plc_rdm[66:98, 66:98][np.triu_indices(32, k=1)]
f2_f2= mean_plc_rdm[98:128, 98:128][np.triu_indices(30, k=1)]
nt_nt= mean_plc_rdm[128:162, 128:162][np.triu_indices(34, k=1)]
sr_sr= mean_plc_rdm[162:194,162:194][np.triu_indices(32, k=1)]

#intra_stim

intra_stim= np.arctanh(np.concatenate((u1_u1, u2_u2, f1_f1, f2_f2, nt_nt), axis= 0)).tolist()
intra_stim_ls=['intra_stim']*len(intra_stim)

#intra class

u1_u2= mean_plc_rdm[31:66, 0:31].flatten()
f1_f2= mean_plc_rdm[98:128, 66:98].flatten()

intra_class=np.arctanh(np.concatenate((u1_u2, f1_f2), axis= 0)).tolist()
intra_class_ls= ['intra_cls']* len(intra_class)
#between class

u1_f1= mean_plc_rdm[66:98, 0:31].flatten()
u1_f2= mean_plc_rdm[98:128, 0:31].flatten()
u2_f1= mean_plc_rdm[66:98,31:66 ].flatten()
u2_f2= mean_plc_rdm[98:128, 31:66].flatten()

bet_class= np.arctanh(np.concatenate((u1_f1, u1_f2, u2_f1, u2_f2), axis= 0)).tolist()
bet_class_ls= ['inter_cls']*len(bet_class)

#nt vs U&F
u1_nt= mean_plc_rdm[128:162, 0:31].flatten()
u2_nt= mean_plc_rdm[128:162, 31:66].flatten()
f1_nt= mean_plc_rdm[128:162,66:98].flatten()
f2_nt= mean_plc_rdm[128:162,98:128].flatten()

nt_uf= np.arctanh(np.concatenate((u1_nt, u2_nt, f1_nt, f2_nt), axis= 0)).tolist()
nt_uf_ls= ['nt_othr']*len(nt_uf)

#sr vs all
u1_sr= mean_plc_rdm[162:194, 0:31].flatten()
u2_sr= mean_plc_rdm[162:194, 31:66].flatten()
f1_sr= mean_plc_rdm[162:194, 66:98].flatten()
f2_sr= mean_plc_rdm[162:194,98:128].flatten()
nt_sr= mean_plc_rdm[162:194, 128:162].flatten()



sr_all= np.arctanh(np.concatenate((u1_sr, u2_sr, f1_sr, f2_sr, nt_sr))).tolist()
sr_all_ls= ['sr_all']*len(sr_all)

plc_df_anova= pd.DataFrame({'RSM_Group': intra_stim_ls+ intra_class_ls+ bet_class_ls+nt_uf_ls+ sr_all_ls, 
                            'Similarity': intra_stim+ intra_class+ bet_class+ nt_uf+ sr_all })

plc_df_anova.insert(0, 'Treatment', ['PLC']*len(plc_df_anova['RSM_Group']), True)

sn.boxenplot(data= plc_df_anova, x= 'RSM_Group', y= 'Similarity')

rsm_df_anova= pd.concat([lpz_df_anova, plc_df_anova])

sn.boxenplot(data= rsm_df_anova, x= 'RSM_Group', y= 'Similarity', hue= 'Treatment')

sn.violinplot(data= rsm_df_anova, x= 'RSM_Group', y= 'Similarity', hue= 'Treatment')

#Run factorial anova

fact_anova= pg.anova(dv='Similarity', between=['Treatment', 'RSM_Group'], data= rsm_df_anova, detailed= True)

round(fact_anova, 4)

#plot plc rdm
ax2= sn.heatmap(mean_plc_rdm, vmin= 0, vmax=1, cmap= 'viridis')
ax2.set_xticks(x_y_ticks)
ax2.set_yticks(x_y_ticks)
ax2.set_xticklabels(labels, minor= False, rotation=90)
ax2.set_yticklabels(labels, minor= False, rotation=90)
plt.show()













sub_files= os.listdir(rsm_glm_path)

temp_path_ls_glm = []
for folder_name in sub_files:
    temp_path_ls_glm.append(os.path.join(rsm_glm_path, folder_name))

all_rdm= []

for paths in range(0, len(temp_path_ls_glm)):
    
    rdm_path_lpz= glob.glob(temp_path_ls_glm[paths] + '\*RDM.csv')
    lpz_rdm_ls= pd.read_csv(rdm_path_lpz[0], sep=',').to_numpy()
    all_rdm.append(lpz_rdm_ls)

mean_all_rdm= np.mean(all_rdm, axis=0)
mean_df_all= pd.DataFrame(mean_all_rdm).round(3)



ax1= sn.heatmap(mean_df_all, cmap= 'viridis')
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
ax1.set_xticklabels(labels, minor= False, rotation=90)
ax1.set_yticklabels(labels, minor= False, rotation=90)

plt.show()

ax1.vlines(x=xticks, ymin=0, ymax= 194, color='g', ls='dotted')    




ax1= sn.heatmap(rdm, cmap= 'viridis')
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.show()
len(run1_con1_voxles[0,:])



r1=31
r2= 35
r3= 32
r4= 30
r5= 34
r6= 32

min_tr[-1]

x_yticks=[r1, r1+r2, r1+r2+r3, r1+r2+r3+r4, r1+r2+r3+r4+r5, r1+r2+r3+r4+r5+r6]


import numpy as np 

a= np.array([[0,2,3], [4,0,5], [6,7,0]])   

a[0:2,0:2]
a[np.triu_indices(3, k=0)]



