# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 14:28:44 2022

@author: CENs-Admin
"""

'''
Objectives:

1. Combine all glmsingle estimates from 3 different masks (bio motion,
   action observation, mentalizing, amydala/emotion)
2. Filter with combined mask and extract data 
3. Save as NIFTI
'''

'''
Structure of the algorithm

1. Get the path of all three regions glmsingle outputs.
2. Load 3 images per subject
3. Combine images
4. Filter with the combined mask
5. Transform and save final image in each subjects directory


'''


import os
import glob
import nibabel as nib
from nilearn.maskers import NiftiMasker
from nilearn import image, plotting



com_mask_path= r'E:\SocialDetection7T\Nahid_Project\Masks\com_mask_biom_act_men_emo.nii'
biom_glm_path= r'E:\SocialDetection7T\Nahid_Project\GLMSingle_Outputs_biom'
ac_mn_glm_path= r'E:\SocialDetection7T\Nahid_Project\GLMSingle_Outputs_Action_Mentalizing'
amg_glm_path= r'E:\SocialDetection7T\Nahid_Project\GLMSingle_Outputs_amygdala'
output_path= r'E:\SocialDetection7T\Nahid_Project\GLMSingle_Outputs_all'

list_of_folder_glm = sorted(filter(lambda x: os.path.isdir(os.path.join(biom_glm_path, x)),
                                   os.listdir(biom_glm_path)))

temp_path_ls_biom = []
for folder_name in list_of_folder_glm:
    temp_path_ls_biom.append(os.path.join(biom_glm_path, folder_name))

temp_path_ls_ac_mn = []
for folder_name in list_of_folder_glm:
    temp_path_ls_ac_mn.append(os.path.join(ac_mn_glm_path, folder_name))


temp_path_ls_amg = []
for folder_name in list_of_folder_glm:
    temp_path_ls_amg.append(os.path.join(amg_glm_path, folder_name))





com_mask= nib.load(com_mask_path)
affine= com_mask.affine
shape= com_mask.get_fdata()[:,:,:].shape


masker= NiftiMasker( target_affine=affine, target_shape=shape,
                         mask_strategy='background')

masker.fit(com_mask)



for paths in range(0, len(temp_path_ls_biom)):
    
    sub_id= list_of_folder_glm[paths]
    print(f'Now working on Sub# {sub_id}')
    
    glm1= nib.load(glob.glob(temp_path_ls_biom[paths] + '\L*.nii')[0])
    glm2= nib.load(glob.glob(temp_path_ls_ac_mn[paths] + '\L*.nii')[0])
    glm3= nib.load(glob.glob(temp_path_ls_amg[paths] + '\L*.nii')[0])
    
    full_glm= image.math_img('a+b+c', a= glm1, b=glm2, c=glm3)
    
    masked_full_glm= masker.transform_single_imgs(full_glm)

    masked_full_glm= masker.inverse_transform(masked_full_glm)
    
    print(f'Saving files of {sub_id}')
    output_dir= os.path.join(output_path,sub_id)
    os.makedirs(output_dir)
    nib.save(masked_full_glm, os.path.join(output_dir, sub_id +'.nii'))
    
    


masked_data= masker.transform_single_imgs(mask1_2)

masked_data= masker.inverse_transform(masked_data)


plotting.plot_glass_brain(masked_full_glm.slicer[:,:,:,1])
plotting.plot_glass_brain(mask2.slicer[:,:,:,1])

mask1_2= image.math_img("a+b", a= mask1, b= mask2)
plotting.plot_glass_brain(mask)
 