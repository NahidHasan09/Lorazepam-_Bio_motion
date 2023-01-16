# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 18:07:23 2022

@author: CENs-Admin
"""
'''

 No needed
#For renaming folder names 
os.chdir(r'E:\SocialDetection7T\FFX1_Norm')

folders= os.listdir()[2:]

for fol_names in folders:
    old_name= fol_names
    new_name= fol_names[9:]+ '_'+ fol_names[:8]
    os.rename(old_name, new_name)

os.listdir()
#to move 103 at the end for synchronizing with glmsingle list
temp_path_ls_spm.insert(58, temp_path_ls_spm.pop(0))'''


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
os.chdir(glm_path)
spm_path = r'E:\SocialDetection7T\FFX1_Norm_notSmoothed'
glm_path = r'E:\SocialDetection7T\Nahid_Project\GLMSingle_Outputs'
mask_img_path_name = r'E:\SocialDetection7T\Nahid_Project\Masks\NIm2019_BM_BMscram_0p05FWEclus_0p001u_k40.nii'
root_dir = r'E:\SocialDetection7T\Nahid_Project'
# Listing directory paths of glmsingle outputs
# sorting directories by name
list_of_folder_glm = sorted(filter(lambda x: os.path.isdir(os.path.join(glm_path, x)),
                                   os.listdir(glm_path)))
temp_path_ls_glm = []
for folder_name in list_of_folder_glm:
    temp_path_ls_glm.append(os.path.join(glm_path, folder_name))


# Listing directory paths of spm not smoothed  outputs
list_of_folder_spm = sorted(filter(lambda x: os.path.isdir(os.path.join(spm_path, x)),
                                   os.listdir(spm_path)))

temp_path_ls_spm = []
for folder_name in list_of_folder_spm:
    temp_path_ls_spm.append(os.path.join(spm_path, folder_name))


# to remonve sub 139 because couldn't fit glmsingle
temp_path_ls_spm.remove(temp_path_ls_spm[25])


con_img_glm = []

for paths_glm in temp_path_ls_glm:

    nifti_path_glm = glob.glob(paths_glm + '\Con_1*.nii')
    nifti_glm = nib.load(nifti_path_glm[0])
    con_img_glm.append(nifti_glm)

con_img_spm = []

for paths_spm in temp_path_ls_spm:

    nifti_path_spm = glob.glob(paths_spm + '\con_0001*.nii')
    nifti_spm = nib.load(nifti_path_spm[0])
    con_img_spm.append(nifti_spm)


# Second level
# mask image
mask_img = nib.load(mask_img_path_name)
resample_mask_img = image.resample_img(
    mask_img, target_affine=nifti_glm.affine, target_shape=nifti_glm.dataobj.shape)
binary_mask_img = image.binarize_img(resample_mask_img)

n_subjects = 59

second_level_input = con_img_glm + con_img_spm


condition_effect = np.hstack(([1] * n_subjects, [- 1] * n_subjects))

subject_effect = np.vstack((np.eye(n_subjects), np.eye(n_subjects)))
subjects = os.listdir(glm_path)

paired_design_matrix = pd.DataFrame(
    np.hstack((condition_effect[:, np.newaxis], subject_effect)),
    columns=['Bio Motion vs Scrambled'] + subjects)

plot_design_matrix(paired_design_matrix, rescale=False)

second_level_model_paired = SecondLevelModel(binary_mask_img).fit(
    second_level_input, design_matrix=paired_design_matrix)

stat_maps_paired = second_level_model_paired.compute_contrast(
    'Bio Motion vs Scrambled',
    output_type='all')


# saving results
output_dir = os.path.join( root_dir,'Contrast_GLMSingle_SPM')
# os.makedirs(output_dir)
np.save(os.path.join(
    output_dir, 'con_stat_glmsingle_spm_unsmoothed.npy'), stat_maps_paired)

#smoothing z score image
smoothed_z_img = image.smooth_img(stat_maps_paired['z_score'], fwhm=3)


# FWER corrected glass brain
_, threshold = threshold_stats_img(
    stat_maps_paired['z_score'], alpha=.05, height_control='bonferroni')
print('Bonferroni-corrected, p<0.05 threshold: %.3f' % threshold)
fwer_corr_glsb= plotting.plot_glass_brain(stat_maps_paired['z_score'], threshold=threshold, 
                          colorbar=True, plot_abs=False)
fwer_corr_glsb.savefig(os.path.join(
    output_dir, 'con_glmsingle_vs_spm_uns_glsb_fwer0p05.png'), dpi= 300)




# FWER corrected stat map (Bonferroni-corrected, p<0.05 threshold: 4.563)
_, threshold = threshold_stats_img(
    stat_maps_paired['z_score'], alpha=.05, height_control='bonferroni')
print('Bonferroni-corrected, p<0.05 threshold: %.3f' % threshold)
fwer_corr_stat_map= plotting.plot_stat_map(stat_maps_paired['z_score'], threshold=threshold, 
                          colorbar=True, cut_coords=[56,-52,16], draw_cross=False)
fwer_corr_stat_map.savefig(os.path.join(
    output_dir, 'con_glmsingle_vs_spm_uns_statm_fwer0p05.png'), dpi= 300)






# FDR corrected
_, threshold = threshold_stats_img(
    stat_maps_paired['z_score'], alpha=.05, height_control='fdr')
print('False Discovery rate= 0.05 threshold: %.3f' % threshold)
plotting.plot_glass_brain(stat_maps_paired['z_score'], threshold=threshold,
                          colorbar=True, plot_abs=False,
                          title='Bio Motion vs Scrambled (FDR= 0.05)')

# FDR corrected with small clustered removed
clean_map, threshold = threshold_stats_img(
    stat_maps_paired['z_score'], alpha=.05, height_control='fdr', cluster_threshold=10)
print('False Discovery rate= 0.05 threshold: %.3f' % threshold)
plotting.plot_glass_brain(clean_map, threshold=threshold,
                          colorbar=True, plot_abs=False,
                          title='Bio Motion vs Scrambled (FDR= 0.05), clusters > 10 voxels')



table = get_clusters_table(
    stat_maps_paired['z_score'], stat_threshold=threshold, cluster_threshold=10)

table
