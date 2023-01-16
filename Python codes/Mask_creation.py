# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 16:55:58 2022

@author: CENs-Admin
"""

import os
import glob 
import nibabel as nib
from nilearn import image
from nilearn import plotting
from nilearn import glm


mask_path= r'C:\Users\CENs-Admin\Downloads\Action_observatio_mask'

mask_img_path= r'E:\SocialDetection7T\Nahid_Project\Masks\NIm2019_BM_BMscram_0p05FWEclus_0p001u_k40.nii'
mask_img= nib.load(mask_img_path)

resmp_mask= image.resample_img(mask_img, affine, shape)
bi_mask= image.binarize_img(resmp_mask)

plotting.plot_glass_brain(bi_mask, colorbar=True)                           

affine= fig_b.affine
shape= fig_b.get_fdata().shape

fig_b_path=glob.glob(mask_path + '\Fig1B*.nii') 

fig_c_path= glob.glob(mask_path + '\Fig1C*.nii')

both_mask_path= glob.glob(mask_path + '\b*.nii')

fig_b= nib.load(fig_b_path[0])
bi_fig_b= image.binarize_img(fig_b)
fig_c= nib.load(fig_c_path[0])
bi_fig_c= image.binarize_img(fig_c)

thres_fig_b= glm.threshold_stats_img(fig_b, alpha= 0.0001)
thres_fig_c= glm.threshold_stats_img(fig_c, alpha= 0.0001)
plotting.plot_glass_brain(thres_fig_b[0], colorbar=True, threshold= 6)

plotting.plot_glass_brain(thres_fig_b[0], colorbar=True)
plotting.plot_glass_brain(thres_fig_c[0], colorbar=True)

s_fig_b= image.smooth_img(fig_b, fwhm= 2)
s_fig_c= image.smooth_img(fig_c, fwhm= 2)

plotting.plot_glass_brain(s_thres_fig_b, colorbar=True)
plotting.plot_glass_brain(s_thres_fig_c, colorbar=True)

thres_s_fig_b= glm.threshold_stats_img(s_fig_b, alpha= 0.0001)
thres_s_fig_c= glm.threshold_stats_img(s_fig_c, alpha= 0.0001)


plotting.plot_glass_brain(thres_s_fig_b[0], colorbar=True)
plotting.plot_glass_brain(thres_s_fig_c[0], colorbar=True)

bi_s_fig_b= image.binarize_img(thres_s_fig_b[0])
bi_s_fig_c= image.binarize_img(thres_s_fig_c[0])

plotting.plot_glass_brain(bi_s_fig_b, colorbar=True)
plotting.plot_glass_brain(bi_s_fig_c, colorbar=True)

plotting.plot_glass_brain(fig_c_s_3, threshold=3, colorbar= True)
plotting.plot_glass_brain(mask_img, colorbar= True)

fig_c_s_3= image.smooth_img(thres_fig_c[0], fwhm= 4)
bi_fig_c= image.binarize_img(thres_fig_c[0])


com_fig_b_c= image.math_img("(a + b)>0", a= bi_s_fig_b, b= bi_s_fig_c)
plotting.plot_glass_brain(com_fig_b_c, colorbar=True)

sub_mask_fig_b_c= image.math_img("(a-b) >0 ", a= com_fig_b_c, b= bi_mask)

com= image.math_img('(a+b)', a=sub_mask_fig_b_c , b= bi_mask)

plotting.plot_glass_brain(com, colorbar=True, threshold=0)

output_dir= r'E:\SocialDetection7T\Nahid_Project\Masks'
nib.save(sub_mask_fig_b_c,  os.path.join( output_dir, 'action_mentalizing_mask.nii'))

plotting.plot_glass_brain(sub_mask_fig_b_c, colorbar= True)
plotting.plot_stat_map(sub_mask_fig_b_c)
plotting.plot_glass_brain(fig_c_sub_mask, colorbar=True)


cereb_mask_path= r'C:\Users\CENs-Admin\Downloads\spm12\spm12\toolbox\AAL3\cerebellumMask.nii'
cereb_mask= nib.load(cereb_mask_path)

resam_cereb_mask= image.resample_img(cereb_mask, affine, shape)
plotting.plot_glass_brain(resam_cereb_mask, colorbar= True)

sub_cereb= image.math_img("(a-b)> 0", a= sub_mask_fig_b_c, b= resam_cereb_mask)

plotting.plot_glass_brain(sub_cereb, colorbar= True)

plotting.plot_glass_brain(bi_fig_c, colorbar=True)
plotting.plot_glass_brain(bi_fig_b, colorbar=True)
plotting.plot_glass_brain(bi_mask)




'''
Creation of amygdala mask for emotion association task


'''
import nibabel as nib
from nilearn import image
from nilearn import plotting
from nilearn import glm
import os

emo_mask_path= r'E:\SocialDetection7T\Nahid_Project\Masks'
    
emo_path= os.path.join(emo_mask_path, 'emotional_association.nii')
emo_mask= nib.load(emo_path)
data_path= r'E:\SocialDetection7T\freshData\LOXY_102_16_12_19\19_fMRI_wholehead_MOVE_1_E00_M\DICOM\wr019-fMRI_wholehead_MOVE_1_00006.nii'

data= nib.load(data_path)

affine= data.affine
shape= data.get_fdata().shape

resmp_emo_mask= image.resample_img(emo_mask, affine, shape)

ths_emo_mask= glm.threshold_stats_img(resmp_emo_mask, alpha= 0.000000001)
plotting.plot_glass_brain(ths_emo_mask[0], colorbar=True)

bi_emo_mask= image.binarize_img(resmp_emo_mask, threshold= 6)
plotting.plot_glass_brain(bi_emo_mask, colorbar=True)

masker= NiftiMasker(mask_strategy='background')
masker.fit(bi_emo_mask)

masker.n_elements_

masked_data= masker.transform_single_imgs(data)
masked_data_t= masker.inverse_transform(masked_data)
plotting.plot_glass_brain(masked_data_t, colorbar=True)
#save amygdala mask
nib.save(bi_emo_mask,  os.path.join( emo_mask_path, 'amygdala_mask'))


import nibabel as nib
from nilearn.maskers import NiftiMasker

mask_path= r'E:\SocialDetection7T\Nahid_Project\Masks\action_mentalizing_mask.nii'
data_path= r'E:\SocialDetection7T\freshData\LOXY_102_16_12_19\19_fMRI_wholehead_MOVE_1_E00_M\DICOM\wr019-fMRI_wholehead_MOVE_1_00006.nii'

mask_ac= nib.load(mask_path)
data= nib.load(data_path)

affine= data.affine
shape= data.get_fdata().shape

masker_b= NiftiMasker(mask_strategy='background')

masker_c= NiftiMasker(mask_strategy='background')
masker_com= NiftiMasker(mask_strategy='background')

masker_sub= NiftiMasker(mask_strategy='background')

masker_b.fit(bi_s_fig_b)
masker_c.fit(bi_s_fig_c)
masker_com.fit(com_fig_b_c)
masker_sub.fit(sub_mask_fig_b_c)


masker_b.n_elements_ + masker_c.n_elements_
masker_c.n_elements_
masker_com.n_elements_
masker_sub.n_elements_
masker_ac.n_elements_
masker_ac= NiftiMasker(mask_strategy='background')

masker_ac.fit(mask_ac)




masked_data= masker_ac.transform_single_imgs(data)

masked_data_t= masker_ac.inverse_transform(masked_data)

from nilearn import plotting
plotting.plot_glass_brain(masked_data_t, colorbar=True)



import nibabel as nib
from nilearn.maskers import NiftiMasker

mask_img_path_name= r'E:\SocialDetection7T\Nahid_Project\Masks\NIm2019_BM_BMscram_0p05FWEclus_0p001u_k40.nii'

data_path= r'E:\SocialDetection7T\freshData\LOXY_102_16_12_19\19_fMRI_wholehead_MOVE_1_E00_M\DICOM\wr019-fMRI_wholehead_MOVE_1_00006.nii'

mask_img= nib.load(mask_img_path_name)

data= nib.load(data_path)

affine= data.affine
shape= data.get_fdata().shape

masker= NiftiMasker( target_affine=affine, target_shape=shape,
                         mask_strategy='background')
masker.fit(mask_img)

masker.n_elements_

masker_test= NiftiMasker(mask_strategy='background')
masker_test.fit(mask_img)

masker_test.n_elements_


#############################

'''
Combine all masked glmsingle outputs excluding overlaps



'''
import os
import nibabel as nib
from nilearn import image, plotting
from nilearn.maskers import NiftiMasker
import glob


#1st add the 3 masks

mask_root_path= r'E:\SocialDetection7T\Nahid_Project\Masks'

mask_bio_path= r'E:\SocialDetection7T\Nahid_Project\Masks\NIm2019_BM_BMscram_0p05FWEclus_0p001u_k40.nii'

mask_action_path= r'E:\SocialDetection7T\Nahid_Project\Masks\action_mentalizing_mask.nii'

mask_amygdala_path= r'E:\SocialDetection7T\Nahid_Project\Masks\amygdala_mask.nii'

data_path= r'E:\SocialDetection7T\freshData\LOXY_102_16_12_19\19_fMRI_wholehead_MOVE_1_E00_M\DICOM\wr019-fMRI_wholehead_MOVE_1_00006.nii'







data= nib.load(data_path)
affine= data.affine
shape= data.get_fdata().shape


mask_bio= image.binarize_img(image.resample_img(nib.load(mask_bio_path), affine, shape))

plotting.plot_glass_brain(mask_bio, colorbar=True)

mask_action= nib.load(mask_action_path)

mask_action= image.resample_img(mask_action, affine, shape)

plotting.plot_glass_brain(mask_action, colorbar=True)

mask_amygdala= nib.load(mask_amygdala_path)

plotting.plot_glass_brain(mask_amygdala, colorbar=True)





mask_all= image.math_img('(a+b+c)==1', a= mask_action, b= mask_bio, c= mask_amygdala )



view= plotting.view_img(mask_all)
view.open_in_browser()


nib.save(mask_all, os.path.join(mask_root_path, 'com_mask_biom_act_men_emo') )


'''Get ROIs from combining AAL and mask image'''


import os
import nibabel as nib
from nilearn import image, plotting

com_mask_path= r'E:\SocialDetection7T\Nahid_Project\Masks\com_mask_biom_act_men_emo.nii'
aal_path= r'C:\Users\CENs-Admin\Downloads\spm12\spm12\toolbox\AAL3\AAL3v1_1mm.nii'
output_dir= r'E:\SocialDetection7T\Nahid_Project\ROIs'

com_mask= nib.load(com_mask_path)
affine= com_mask.affine
shape= com_mask.get_fdata().shape


aal_img= nib.load(aal_path)


aal_19= image.math_img('(a>18) & (a<23) ', a= aal_img) #for mpfc
plotting.plot_glass_brain(aal_19)
aal_19_resamp= image.resample_img(aal_19, affine, shape)

plotting.plot_glass_brain(aal_19_resamp)

mask_mpfc= image.math_img('(a+b)==2', a= com_mask, b= aal_19_resamp)
plotting.plot_glass_brain(mask_mpfc)

nib.save(image.binarize_img(mask_mpfc), os.path.join(output_dir, 'ROI_1_MPFC.nii') )

plotting.plot_stat_map(mask_mpfc, display_mode= "z", cut_coords=5 )

aal_3= image.math_img('(a>0) & (a<10)', a= aal_img) #for premotor
plotting.plot_glass_brain(aal_3)
aal_3_resamp= image.resample_img(aal_3, affine, shape)

mask_premotor= image.math_img('(a+b)==2', a= com_mask, b= aal_3_resamp)
plotting.plot_glass_brain(mask_premotor)
view= plotting.view_img(mask_premotor)
view.open_in_browser()

nib.save(image.binarize_img(mask_premotor), os.path.join(output_dir, 'ROI_2_Premotor.nii') )

aal_61= image.math_img('(a>60) & (a<69)', a = aal_img) #for parietal lobe
plotting.plot_glass_brain(aal_61)
aal_61_resamp= image.resample_img(aal_61, affine, shape)

mask_parietal= image.math_img('(a+b)==2', a= com_mask, b= aal_61_resamp)
plotting.plot_glass_brain(mask_parietal)
plotting.plot_glass_brain(com_mask)

nib.save(image.binarize_img(mask_parietal), os.path.join(output_dir, 'ROI_3_Parietal.nii') )

aal_85= image.math_img('(a>84) & (a<95)', a = aal_img) #for STS
plotting.plot_glass_brain(aal_85)
aal_85_resamp= image.resample_img(aal_85, affine, shape)

mask_sts= image.math_img('(a+b)==2', a= com_mask, b= aal_85_resamp)
plotting.plot_glass_brain(mask_sts)

nib.save(image.binarize_img(mask_sts), os.path.join(output_dir, 'ROI_4_STS.nii') )
aal_71= image.math_img('(a>70) & (a<73)', a= aal_img) #for precuneus
plotting.plot_glass_brain(aal_71)
aal_71_resamp= image.resample_img(aal_71, affine, shape)

mask_precuneus= image.math_img('(a+b)==2', a= com_mask, b= aal_71_resamp)
plotting.plot_glass_brain(mask_precuneus)

nib.save(image.binarize_img(mask_precuneus), os.path.join(output_dir, 'ROI_5_Precuneus.nii') )

aal_49= image.math_img('(a>48) & (a<51)', a= aal_img) #for cuneus
plotting.plot_glass_brain(aal_49)
aal_49_resamp= image.resample_img(aal_49, affine, shape)

mask_cuneus= image.math_img('(a+b)==2', a= com_mask, b= aal_49_resamp)
plotting.plot_glass_brain(mask_cuneus)


aal_45= image.math_img('(a>44) & (a<47)', a = aal_img) #for amygdala
plotting.plot_glass_brain(aal_45)
aal_45_resamp= image.resample_img(aal_45, affine, shape)

mask_amygdala= image.math_img('(a+b)==2', a= com_mask, b= aal_45_resamp)
plotting.plot_glass_brain(mask_amygdala)

nib.save(image.binarize_img(mask_amygdala), os.path.join(output_dir, 'ROI_6_Amygdala.nii') )





############## Make Figures #############


import os 
from nilearn import plotting, image
import nibabel as nib


whole_mask_path= r'E:\SocialDetection7T\Nahid_Project\Masks\com_mask_biom_act_men_emo.nii'
biom_mask_path= r'E:\SocialDetection7T\Nahid_Project\Masks\Bio_m_mask.nii'
act_men_mask_path= r'E:\SocialDetection7T\Nahid_Project\Masks\action_mentalizing_mask.nii'
amg_mask_path= r'E:\SocialDetection7T\Nahid_Project\Masks\amygdala_mask.nii'

output_dir= r'E:\SocialDetection7T\Nahid_Project\Figures\Mask_Fig'



biom_mask= nib.load(biom_mask_path)
act_men_mask= nib.load(act_men_mask_path)
amg_mask= nib.load(amg_mask_path)
whole_mask= nib.load(whole_mask_path)






biom_mask_fig= plotting.plot_glass_brain(biom_mask, cmap= 'Blues', display_mode= 'xz', threshold='auto', alpha= 0.7)

act_men_mask_fig= plotting.plot_glass_brain(act_men_mask, cmap= 'Blues', display_mode= 'xz', threshold='auto', alpha= 0.7)
amg_mask_fig= plotting.plot_glass_brain(amg_mask, cmap= 'Blues', display_mode= 'xz', threshold='auto', alpha= 0.7)
whole_mask_fig= plotting.plot_glass_brain(whole_mask, cmap= 'Blues', display_mode= 'xz', threshold='auto', alpha= 0.7)





biom_mask_fig.savefig(os.path.join(
    output_dir, 'bio_m_mask.png'), dpi= 300)

act_men_mask_fig.savefig(os.path.join(
    output_dir, 'act_men_mask.png'), dpi= 300)

amg_mask_fig.savefig(os.path.join(
    output_dir, 'amg_mask.png'), dpi= 300)

whole_mask_fig.savefig(os.path.join(
    output_dir, 'whole_mask_mask.png'), dpi= 300)




'Saving ROIs figures'

import os
import nibabel as nib
from nilearn import plotting

roi_path= r'E:\SocialDetection7T\Nahid_Project\ROIs\ROI_all_4d.nii'
output_dir= r'E:\SocialDetection7T\Nahid_Project\Figures\ROIs'


roi= nib.load(roi_path)

roi= nib.four_to_three(roi)

roi_1= plotting.plot_glass_brain(roi[0], cmap= 'Blues', display_mode= 'xz', threshold='auto', alpha= 0.7)

roi_2= plotting.plot_glass_brain(roi[1], cmap= 'Blues', display_mode= 'xz', threshold='auto', alpha= 0.7)

roi_3= plotting.plot_glass_brain(roi[2], cmap= 'Blues', display_mode= 'xz', threshold='auto', alpha= 0.7)

roi_4= plotting.plot_glass_brain(roi[3], cmap= 'Blues', display_mode= 'xz', threshold='auto', alpha= 0.7)

roi_5= plotting.plot_glass_brain(roi[4], cmap= 'Blues', display_mode= 'xz', threshold='auto', alpha= 0.7)

roi_6= plotting.plot_glass_brain(roi[5], cmap= 'Blues', display_mode= 'xz', threshold='auto', alpha= 0.7) 


roi_1.savefig(os.path.join(
    output_dir, 'ROI_1.png'), dpi= 300)

roi_2.savefig(os.path.join(
    output_dir, 'ROI_2.png'), dpi= 300)

roi_3.savefig(os.path.join(
    output_dir, 'ROI_3.png'), dpi= 300)

roi_4.savefig(os.path.join(
    output_dir, 'ROI_4.png'), dpi= 300)

roi_5.savefig(os.path.join(
    output_dir, 'ROI_5.png'), dpi= 300)

roi_6.savefig(os.path.join(
    output_dir, 'ROI_6.png'), dpi= 300)


















