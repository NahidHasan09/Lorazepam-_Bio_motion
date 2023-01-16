#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:44:03 2022

@author: johannesschultz
"""

import os
import glob
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
import pandas as pd


os.chdir("/Volumes/WD_TRANSFER/SocialDetection7T/Nahid's_Project/Analysis_Scripts/GLMSingle")    


sub102_dict= np.load('Sub_102_results_masked.npy', allow_pickle= True).item()

f_data_path= '/Volumes/WD_TRANSFER/SocialDetection7T/freshData/16_12_19_LOXY_102/19_fMRI_wholehead_MOVE_1_E00_M/DICOM'
ls_f_data= glob.glob(f_data_path+ '/wr*.nii') #get 3d niftis

sub_102= nib.load(ls_f_data[0])

mask_img_path_name= "/Volumes/WD_TRANSFER/SocialDetection7T/Nahid's_Project/Masks/NIm2019_BM_BMscram_0p05FWEclus_0p001u_k40.nii"

#mask image
mask_img= nib.load(mask_img_path_name)

masker= NiftiMasker( target_affine=sub_102.affine, target_shape=sub_102.shape,
                        mask_strategy='background')
masker.fit(mask_img)


tr_betas= sub102_dict['betasmd'].T
betas_102_nifti= masker.inverse_transform(tr_betas)

betas_102_3d= nib.funcs.four_to_three(betas_102_nifti)


eventfs_path= '/Volumes/WD_TRANSFER/SocialDetection7T/onsets_Logfiles'

eventfs= glob.glob(eventfs_path + '/socialDetection-fMRI-LOXY_102*.txt')


stim_type= []

for i in eventfs:
    dm_read= pd.read_csv(i, sep=",", header=None, names= ["trial_no", "stimtype", "no_need", "biomotion", "accuracy", "onsets", "offsets", "correct", "RT"])
    
    items= dm_read["stimtype"].values
    
    stim_type.append(items)



joined= stim_type[0].tolist() + stim_type[1].tolist()

df= pd.DataFrame({'trials': joined, 'beta_maps': betas_102_3d})

non_bio= df.loc[df['trials']==5]    

non_bio= non_bio['beta_maps'].values.tolist()      

bio_m=  df.loc[df['trials']!= 5]    

bio_m= bio_m['beta_maps'].values.tolist()  


len(bio_m)


#second level analysis

second_level_input= bio_m + non_bio    


condition_effect= np.hstack(([1]* len(bio_m), [-1]* len(non_bio)))    

unpaired_design_matrix= pd.DataFrame(
    condition_effect[:, np.newaxis],
    columns=['Bio Motion vs. Scramble'])

from nilearn.plotting import plot_design_matrix

plot_design_matrix(unpaired_design_matrix, rescale=False)



from nilearn.glm.second_level import SecondLevelModel


second_level_model_unpaired= SecondLevelModel().fit(second_level_input, design_matrix= unpaired_design_matrix)

stat_maps_unpaired= second_level_model_unpaired.compute_contrast('Bio Motion vs. Scramble', output_type= 'all')


from nilearn import plotting


plotting.plot_glass_brain(stat_maps_unpaired['z_score'], threshold= 3.1, colorbar=True, title= 'Bio vs. Nonbio')


  
np.save('Sub_102_Contrasts.npy', stat_maps_unpaired)
