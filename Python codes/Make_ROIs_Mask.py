# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:03:47 2022

@author: CENs-Admin
"""

'''
This script is for extracting ROIs from our combined mask
(Bio_motion+mentalizing+action observation+ amygdala).

To extract a particular brain region:
1. Get the voxels of a particular region from AAL atlas
2. Overlay those voxels on our combined mask
3. Finalize the mask by getting the  common voxels


'''

#%% Import packages
import os
import nibabel as nib
from nilearn import image, plotting


#%% get paths 
com_mask_path= r'E:\SocialDetection7T\Nahid_Project\Masks\com_mask_biom_act_men_emo.nii'
aal_path= r'C:\Users\CENs-Admin\Downloads\spm12\spm12\toolbox\AAL3\AAL3v1_1mm.nii'


#%% Load combined mask and AAL atlas
com_mask= nib.load(com_mask_path)
affine= com_mask.affine
shape= com_mask.get_fdata().shape

plotting.plot_glass_brain(com_mask)

aal_img= nib.load(aal_path)

cuneus_path= r'E:\SocialDetection7T\Nahid_Project\Masks\cuneus.nii'

#%% Mask MPFC
'''
composed of:
19 Frontal_Sup_Medial_L 19
20 Frontal_Sup_Medial_R 20
21 Frontal_Med_Orb_L 21
22 Frontal_Med_Orb_R 22

'''
aal_19= image.math_img('(a>18) & (a<23) ', a= aal_img) #for mpfc
plotting.plot_glass_brain(aal_19)
aal_19_resamp= image.resample_img(aal_19, affine, shape)

plotting.plot_glass_brain(aal_19_resamp)

mask_mpfc= image.math_img('(a+b)==2', a= com_mask, b= aal_19_resamp)
plotting.plot_glass_brain(mask_mpfc)


#%% Mask premotor

'''
1 Precentral_L 1
2 Precentral_R 2
3 Frontal_Sup_2_L 3
4 Frontal_Sup_2_R 4
5 Frontal_Mid_2_L 5
6 Frontal_Mid_2_R 6
7 Frontal_Inf_Oper_L 7
8 Frontal_Inf_Oper_R 8
9 Frontal_Inf_Tri_L 9
10 Frontal_Inf_Tri_R 10

'''

aal_3= image.math_img('(a>0) & (a<11) ', a= aal_img) #for premotor
plotting.plot_glass_brain(aal_3)
aal_3_resamp= image.resample_img(aal_3, affine, shape)

mask_premotor= image.math_img('(a+b)==2', a= com_mask, b= aal_3_resamp)
plotting.plot_glass_brain(mask_premotor)
view= plotting.view_img(mask_premotor)
view.open_in_browser()

#%% Mask parietal_Sup

'''
61 Postcentral_L 61
62 Postcentral_R 62
63 Parietal_Sup_L 63
64 Parietal_Sup_R 64
65 Parietal_Inf_L 65
66 Parietal_Inf_R 66
67 SupraMarginal_L 67
68 SupraMarginal_R 68

'''

aal_61= image.math_img('(a>60) & (a<69)', a = aal_img) #for parietal lobe
plotting.plot_glass_brain(aal_61)
aal_61_resamp= image.resample_img(aal_61, affine, shape)

mask_parietal= image.math_img('(a+b)==2', a= com_mask, b= aal_61_resamp)
plotting.plot_glass_brain(mask_parietal)
plotting.plot_glass_brain(com_mask)


#%% Mask STS

'''
69 Angular_L 69
70 Angular_R 70

85 Temporal_Sup_L 85
86 Temporal_Sup_R 86
87 Temporal_Pole_Sup_L 87
88 Temporal_Pole_Sup_R 88
89 Temporal_Mid_L 89
90 Temporal_Mid_R 90
91 Temporal_Pole_Mid_L 91
92 Temporal_Pole_Mid_R 92
93 Temporal_Inf_L 93
94 Temporal_Inf_R 94

'''
aal_85= image.math_img('(a>84) & (a<95)', a = aal_img) #for STS +angular 
plotting.plot_glass_brain(aal_85)
aal_85_resamp= image.resample_img(aal_85, affine, shape)

mask_sts= image.math_img('(a+b)==2', a= com_mask, b= aal_85_resamp)
plotting.plot_glass_brain(mask_sts)


aal_69= image.math_img('(a>68) & (a<71)', a = aal_img)
plotting.plot_glass_brain(aal_69)
aal_85_69= image.math_img('(a+b)', a= aal_85, b= aal_69)
plotting.plot_glass_brain(aal_85_69)

aal_85_69_resamp= image.resample_img(aal_85_69, affine, shape)
mask_sts= image.math_img('(a+b)==2', a= com_mask, b= aal_85_69_resamp)
plotting.plot_glass_brain(mask_sts)


#%% Mask Precuneus
'''
71 Precuneus_L 71
72 Precuneus_R 72


'''

aal_71= image.math_img('(a>70) & (a<73)', a= aal_img) #for precuneus
plotting.plot_glass_brain(aal_71)
aal_71_resamp= image.resample_img(aal_71, affine, shape)

mask_precuneus= image.math_img('(a+b)==2', a= com_mask, b= aal_71_resamp)
plotting.plot_glass_brain(mask_precuneus)

#%% Mask Amygdala
'''
45 Amygdala_L 45
46 Amygdala_R 46

'''

aal_45= image.math_img('(a>44) & (a<47)', a = aal_img) #for amygdala
plotting.plot_glass_brain(aal_45)
aal_45_resamp= image.resample_img(aal_45, affine, shape)

mask_amygdala= image.math_img('(a+b)==2', a= com_mask, b= aal_45_resamp)
plotting.plot_glass_brain(mask_amygdala)

