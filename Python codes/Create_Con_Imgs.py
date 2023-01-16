# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:45:25 2022

@author: CENs-Admin
"""


''' script plan:
    This script will create contrast images of biological vs. scrambled 
    motion conditions.
    
    Inputs: 4d Nifti images with around 216 volumes 
            each volumes is a distinct condition ( bio or scrambled)
    Output: Single Nifti image of the contrast per subject (Con_1= bio motion vs. scramble)
    
    Structure:  
            1. Load the 4d Nifti image from placebo group only
            2. Separate to 3d Nifti images
            3. Load the onset logfiles specific to subject
            4. Make a pandas dataframe with 2 columns- col1: conditions
            col2: corresponding nifti images 
            5. Filter the images based on conditions (bio vs. scrambled)
            6. Make 4d of bio images and scrambled images
            7. Create mean bio and scrambled image
            8. contrast these two mean images'''
            

#%% Import required packages  

import os
import nibabel as nib
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nilearn.plotting import plot_design_matrix
from nilearn import image
from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table
from nltools.stats import fdr, threshold, fisher_r_to_z, one_sample_permutation
from nilearn.maskers import NiftiMasker
from scipy import stats
import seaborn as sns
from nilearn.reporting import get_clusters_table

#%% Set the paths
glm_all_path= r'E:\SocialDetection7T\Nahid_Project\GLMSingle_Outputs_all'
eventfs_path= r'E:\SocialDetection7T\onsets_Logfiles'
spm_path = r'E:\SocialDetection7T\FFX1_Norm_notSmoothed'
mask_img_path= r'E:\SocialDetection7T\Nahid_Project\Masks\com_mask_biom_act_men_emo.nii'
treatment_grp_path= 'E:\SocialDetection7T\BehavioralData\LOXY_LZPtrial_forJohannes_noMissing.csv'

output_dir= r'E:\SocialDetection7T\Nahid_Project\Figures\GLMSingle_vs_SPM'


#%% Separate placebo group
treatment_grp= pd.read_csv(treatment_grp_path, sep= ';', on_bad_lines='skip')

lpz_plc= pd.concat([treatment_grp['Probandencode'], treatment_grp['treatment_code']], axis= 1, keys=['Sub','Group'])

#add the missing sub ids
lpz_plc.loc[len(lpz_plc)]=['LOXY_110', 1]
lpz_plc.loc[len(lpz_plc)]=['LOXY_168', 1]

#remove sub139
lpz_plc=lpz_plc.drop([13])

#group by group assignment
lpz_plc.sort_values(by=['Group', 'Sub'], ascending= True)

#divide between group lpz=1 and plc=0
#lpz_grp= lpz_plc[lpz_plc['Group']==1]['Sub'].tolist()

plc_grp= lpz_plc[lpz_plc['Group']==0]['Sub'].tolist()


#list_of_folder_glm = sorted(filter(lambda x: os.path.isdir(os.path.join(glm_all_path, x)),
                                   #os.listdir(glm_all_path)))

#%% Create contrast images (bio_m - src) of GLMSingle
temp_path_ls_glm = []
for folder_name in plc_grp:
    temp_path_ls_glm.append(os.path.join(glm_all_path, folder_name))





con_img_glm= []

for imgs in range(0, len(temp_path_ls_glm)):
    sub_id= plc_grp[imgs]
    print(f'Now working with subject# {sub_id} ')
    nifti_path= glob.glob(temp_path_ls_glm[imgs]+ '\L*.nii')
    
    eventfs= glob.glob(eventfs_path + '\socialDetection-fMRI-'+ sub_id + '*.txt')
    
    stim_type= []
    
    for  eventf in  eventfs:
        
        
        dm_read= pd.read_csv(eventf, sep=",", header=None, names= ["trial_no", "stimtype", "no_need", "biomotion", "accuracy", "onsets", "offsets", "correct", "RT"])
        
        items= dm_read["biomotion"].values
        
        stim_type.append(items)
    
    stim_type_all= stim_type[0].tolist() + stim_type[1].tolist()
        
    print(f'Length of stimulation time is {len(stim_type_all)}')
    
    nifti_4d= nib.load(nifti_path[0])        
    nifti_3d= nib.funcs.four_to_three(nifti_4d)

    df= pd.DataFrame({'trials': stim_type_all, 'beta_maps': nifti_3d})

    scram= df.loc[df['trials']==0]    

    bio_m=  df.loc[df['trials']!= 0]    
        
    scram= nib.funcs.concat_images(scram['beta_maps'].values.tolist())
    bio_m= nib.funcs.concat_images(bio_m['beta_maps'].values.tolist())
        
    con_img= image.math_img('img1 - img2', img1= image.mean_img(bio_m), img2= image.mean_img(scram))
    
    print(f'Now saving subject # {sub_id}')    
    
    con_img_glm.append(con_img)
        
        

#list_of_folder_spm = sorted(filter(lambda x: os.path.isdir(os.path.join(spm_path, x)),
                                   #os.listdir(spm_path)))


#%% Import contrast images of SPM

temp_path_ls_spm = []
for folder_name in plc_grp:
    temp_path_ls_spm.append(os.path.join(spm_path, folder_name))

       



con_img_spm = []

for paths_spm in temp_path_ls_spm:

    nifti_path_spm = glob.glob(paths_spm + '\con_0001*.nii')
    nifti_spm = nib.load(nifti_path_spm[0])
    con_img_spm.append(nifti_spm)   



#%% Run 2nd level analysis on contrast images and save figure

mask_img = nib.load(mask_img_path)


n_subjects = len(con_img_glm)

design_matrix = pd.DataFrame([1] * n_subjects, columns=['intercept'])


second_level_model_glm = SecondLevelModel(smoothing_fwhm=3).fit(
    con_img_glm, design_matrix=design_matrix)


z_map_glm = second_level_model_glm.compute_contrast(output_type='z_score')





_, threshold = threshold_stats_img(
    z_map_glm, alpha=.05, height_control='fdr', cluster_threshold=10)
print('False Discovery rate= 0.05 threshold: %.3f' % threshold)
glm_contrast= plotting.plot_glass_brain(z_map_glm, threshold=threshold,
                          colorbar=True, plot_abs=False)

#save fig
glm_contrast.savefig(os.path.join(
    output_dir, 'Bio_Src_GLM_fdr0p05_fwhm3_cls10.png'), dpi= 300)

table_glm = get_clusters_table(z_map_glm, stat_threshold=threshold,
                           cluster_threshold=10)

#save table
table_glm.to_csv(os.path.join(output_dir, 'Bio_Src_GLM_fdr0p05_fwhm3_cls10.csv'))



second_level_model_spm = SecondLevelModel(mask_img, smoothing_fwhm=3).fit(
    con_img_spm, design_matrix=design_matrix)


z_map_spm = second_level_model_spm.compute_contrast(output_type='z_score')




_, threshold = threshold_stats_img(
    z_map_spm, alpha=.05, height_control='fdr', cluster_threshold=10)
print('False Discovery rate= 0.05 threshold: %.3f' % threshold)
spm_contrast= plotting.plot_glass_brain(z_map_spm, threshold=threshold,
                          colorbar=True, plot_abs=False)

spm_contrast.savefig(os.path.join(
    output_dir, 'Bio_Src_SPM_fdr0p05_fwhm3_cls10.png'), dpi= 300)

table_spm = get_clusters_table(z_map_spm, stat_threshold=threshold,
                           cluster_threshold=10)

table_spm.to_csv(os.path.join(output_dir, 'Bio_Src_SPM_fdr0p05_fwhm3_cls10.csv'))


#%% Correlation test of SPM and GLmSingle contrast images

mask_img = nib.load(mask_img_path)


n_subjects = len(con_img_glm)

glm_4d= nib.concat_images(con_img_glm)

spm_4d= nib.concat_images(con_img_spm)

affine= mask_img.affine
shape= mask_img.get_fdata()[:,:,:].shape

masker= NiftiMasker(target_affine=affine, target_shape=shape,
                         mask_strategy='background')

masker.fit(mask_img)

masked_glm= masker.transform_single_imgs(glm_4d)
masked_spm= masker.transform_single_imgs(spm_4d)


sub_r_scores= []

for rsms in range(0, len(masked_glm[:,1])):
    sim_score= stats.spearmanr(a= masked_glm[rsms,:], b= masked_spm[rsms,:])
    sub_r_scores.append(sim_score[0])

#%% Create regression plot for 3 randomly picked subjects amd save figures


df_sub_6= pd.DataFrame({'SPM': masked_spm[7,:], 'GLMSingle': masked_glm[7,:]})

df_sub_25= pd.DataFrame({'SPM': masked_spm[26,:], 'GLMSingle': masked_glm[26,:]})

df_sub_18= pd.DataFrame({'SPM': masked_spm[19,:], 'GLMSingle': masked_glm[19,:]})


params = {'axes.labelsize': 14,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14}

plt.rcParams.update(params)

sns.lmplot(x= 'SPM', y='GLMSingle', data= df_sub_18,line_kws={"color": "C1"})

plt.xlim(-200, 300)
plt.ylim(-10, 15)
plt.xlabel("Contrast value (SPM)")
plt.ylabel("Contrast value (GLMSingle)")
score_label = f"Spearman's rho: {sub_r_scores[19]:.2f}"
plt.text(50, -5, score_label, fontsize=14)

plt.savefig(os.path.join(
    output_dir, 'Sub_3_con_corr.png'), bbox_inches= 'tight', dpi= 600)

#%% Run 1 sample permutation test and save figures
r_stats= one_sample_permutation(sub_r_scores, return_perms=True )   

plt.figure(figsize=(5, 5), dpi= 600)
plt.hist(r_stats['perm_dist'], bins=50)
plt.xlabel(f"Speaman's rho")
plt.ylabel("Frequency")
plt.axvline(r_stats['mean'], color= 'b', linestyle= '--')
plt.axvline(0, color= 'k', alpha= 0.5, linestyle= '--')
score_label = f"Mean rho: {r_stats['mean']:.2f}\n(p-value: {r_stats['p']:.4f})"
plt.text(0.10, 300, score_label, fontsize=14)
plt.savefig(os.path.join(
    output_dir, 'permutation_5k_con_corr_spm_glms.png'), bbox_inches= 'tight')
plt.show()   





