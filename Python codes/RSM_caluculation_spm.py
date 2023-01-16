# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:05:50 2022

@author: CENs-Admin
"""

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from  nilearn import image
from nilearn.maskers import NiftiMasker


root_path= r'E:\SocialDetection7T\Nahid_Project\RSM_SPM'
data_path= r'E:\SocialDetection7T\FFX1_Norm_notSmoothed'
mask_img_path= r'E:\SocialDetection7T\Nahid_Project\Masks\NIm2019_BM_BMscram_0p05FWEclus_0p001u_k40.nii'
mask_img= nib.load(mask_img_path)

list_of_folder_data_path = sorted(filter(lambda x: os.path.isdir(os.path.join(data_path, x)),
                                   os.listdir(data_path)))

temp_path_ls_spm = []
for folder_name in list_of_folder_data_path:
    temp_path_ls_spm.append(os.path.join(data_path, folder_name))
 
temp_path_ls_spm.remove(temp_path_ls_spm[25])    

for imgs in range(2, 3):
    sub_id= temp_path_ls_spm[0][43:51]
    

    #run1
    run1_con1_mimg= nib.load(glob.glob(temp_path_ls_spm[imgs]+ '\*beta_0001.nii')[0])
    
    run1_con2_mimg= nib.load(glob.glob(temp_path_ls_spm[imgs]+ '\*beta_0002.nii')[0])
    
    run1_con3_mimg= nib.load(glob.glob(temp_path_ls_spm[imgs]+ '\*beta_0003.nii')[0])
    
    run1_con4_mimg= nib.load(glob.glob(temp_path_ls_spm[imgs]+ '\*beta_0004.nii')[0])
    
    run1_con5_mimg= nib.load(glob.glob(temp_path_ls_spm[imgs]+ '\*beta_0005.nii')[0])
    
    run1_con6_mimg= nib.load(glob.glob(temp_path_ls_spm[imgs]+ '\*beta_0006.nii')[0])
    
    #run2
    run2_con1_mimg= nib.load(glob.glob(temp_path_ls_spm[imgs]+ '\*beta_0013.nii')[0])
    
    run2_con2_mimg= nib.load(glob.glob(temp_path_ls_spm[imgs]+ '\*beta_0014.nii')[0])
    
    run2_con3_mimg= nib.load(glob.glob(temp_path_ls_spm[imgs]+ '\*beta_0015.nii')[0])
    
    run2_con4_mimg= nib.load(glob.glob(temp_path_ls_spm[imgs]+ '\*beta_0016.nii')[0])
    
    run2_con5_mimg= nib.load(glob.glob(temp_path_ls_spm[imgs]+ '\*beta_0017.nii')[0])
    
    run2_con6_mimg= nib.load(glob.glob(temp_path_ls_spm[imgs]+ '\*beta_0018.nii')[0])
    


    #step 4 masking
    affine= run1_con1_mimg.affine
    shape=run1_con1_mimg.get_fdata().shape
    masker= NiftiMasker( target_affine=affine, target_shape=shape,
                            mask_strategy='background')
    masker.fit(mask_img)
    
     
     #run1
    run1_con1_voxles= masker.transform_single_imgs(run1_con1_mimg).tolist()
    run1_con2_voxles= masker.transform_single_imgs(run1_con2_mimg).tolist()
    run1_con3_voxles= masker.transform_single_imgs(run1_con3_mimg).tolist()
    run1_con4_voxles= masker.transform_single_imgs(run1_con4_mimg).tolist()
    run1_con5_voxles= masker.transform_single_imgs(run1_con5_mimg).tolist()
    run1_con6_voxles= masker.transform_single_imgs(run1_con6_mimg).tolist()
    
    #run2
    run2_con1_voxles= masker.transform_single_imgs(run2_con1_mimg).tolist()
    run2_con2_voxles= masker.transform_single_imgs(run2_con2_mimg).tolist()
    run2_con3_voxles= masker.transform_single_imgs(run2_con3_mimg).tolist()
    run2_con4_voxles= masker.transform_single_imgs(run2_con4_mimg).tolist()
    run2_con5_voxles= masker.transform_single_imgs(run2_con5_mimg).tolist()
    run2_con6_voxles= masker.transform_single_imgs(run2_con6_mimg).tolist()
    
   
    tuples_run1= list(zip(run1_con1_voxles[0], run1_con2_voxles[0], run1_con3_voxles[0], run1_con4_voxles[0], run1_con5_voxles[0], run1_con6_voxles[0] ))
    tuples_run2= list(zip(run2_con1_voxles[0], run2_con2_voxles[0], run2_con3_voxles[0], run2_con4_voxles[0], run2_con5_voxles[0], run2_con6_voxles[0] ))
    
    corr_df_run1= pd.DataFrame(tuples_run1, columns= ['U1_R1', 'U2_R1', 'F1_R1', 'F2_R1', 'NT_R1','SC_R1'])
    
    corr_df_run2= pd.DataFrame(tuples_run2, columns= ['U1_R2', 'U2_R2', 'F1_R2', 'F2_R2', 'NT_R2','SC_R2'])
    
    concate_corr_df= pd.concat([corr_df_run1, corr_df_run2], axis=1)
    rsm= concate_corr_df.corr('spearman').filter(corr_df_run1.columns).filter(corr_df_run2.columns, axis=0)
    rdm= 1-rsm

rsm_run1= corr_df_run1.corr('spearman')
rsm_run2= corr_df_run2.corr('spearman')
    #Saving RSMs as csv
    print(f'Now saving {sub_id}')
    output_dir= os.path.join(root_path, sub_id)
    os.makedirs(output_dir)
    rsm.to_csv(os.path.join(output_dir, sub_id + 'RSM.csv'), header= True, index= True, index_label= False)
    rdm.to_csv(os.path.join(output_dir, sub_id + 'RDM.csv'), header= True, index= True, index_label= False)
       
    

import seaborn as sn
import matplotlib.pyplot as plt

sn.heatmap(rdm, vmin= 0, vmax=1, cmap= 'viridis',annot= True)    

glm_105_rsm= pd.read_csv(r'E:\SocialDetection7T\Nahid_Project\RSM_GLMSingle\LOXY_105\LOXY_105RSM.csv')

sn.scatterplot(data= run2_con1_voxles)   
sn.histplot(run1_con1_voxles, )    

con1_r1= np.nan_to_num(run1_con1_voxles).tolist()
con1_r2= np.nan_to_num(run2_con1_voxles).tolist()
from scipy.stats.stats import pearsonr, spearmanr
print(spearmanr(con1_r1[0], con1_r1[0]))


run1_con1_brain= masker.inverse_transform(run1_con1_voxles)
run2_con1_brain= masker.inverse_transform(run2_con1_voxles)


from nilearn import plotting
plotting.plot_glass_brain(run1_con1_brain, colorbar=True,threshold=7)
