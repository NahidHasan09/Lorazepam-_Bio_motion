# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:54:37 2022

@author: Nahid Hasan
"""

#test for rsm

import numpy as np
import pandas as pd


mat1= pd.DataFrame(np.random.rand(10,2), columns= list('ab'))

                 
mat1.reshape(4,2)

mat2= pd.DataFrame(np.random.rand(10,2), columns= list('AB'))
mat2
mat2.reshape(4,2)

pearson= np.corrcoef(mat1, mat2)

print(pearson)

concate_mat= pd.concat([mat1, mat2], axis=1)

concate_mat.corr().filter(mat1.columns).filter(mat2.columns, axis=0)

corr_coef= concate_mat.corr()

rdm= 1- corr_coef

'''
Script plan:
    This script will calculate representational similarity matrices between 
    runs. Average across subjects and compare between glmsingle and spm outputs.
    
    Structure:
        1. load 4d nifti beta images for each subject
        2. extract betas per condition using onset logfiles
        3. make mean images per condition
        4. extract desired voxels beta values in 2D form using nifti masker object
        5. make a dataframe from 2d data
        6. calculate RSM using pd.corr() function and filter for run1 and run2
        7. average RSMs across subjects
        '''


import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from  nilearn import image
from nilearn.maskers import NiftiMasker

root_path= r'E:\SocialDetection7T\Nahid_Project\RSM_GLMSingle'
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
    
    df_run1= pd.DataFrame({'trial_type': flt_cons[0].tolist(), 'bio_m': flt_bio_scrm[0].tolist(), 'nifti_3d': betas_3d[0:len(flt_cons[0].tolist())]})
    
    df_run2= pd.DataFrame({'trial_type': flt_cons[1].tolist(), 'bio_m': flt_bio_scrm[1].tolist(), 'nifti_3d': betas_3d[len(flt_cons[0].tolist()):len(flt_cons_all)]})
    
    # step 2
    #run1
    run1_con1= df_run1[(df_run1.bio_m != 0) & (df_run1.trial_type==1)]
    
    run1_con2= df_run1[(df_run1.bio_m != 0) & (df_run1.trial_type==2)]
    
    run1_con3= df_run1[(df_run1.bio_m != 0) & (df_run1.trial_type==3)]
    
    run1_con4= df_run1[(df_run1.bio_m != 0) & (df_run1.trial_type==4)]
    
    run1_con5= df_run1[(df_run1.bio_m != 0) & (df_run1.trial_type==5)]
    
    run1_con6= df_run1[(df_run1.bio_m == 0)]
    
    #run2
    run2_con1= df_run2[(df_run2.bio_m != 0) & (df_run2.trial_type==1)]
    
    run2_con2= df_run2[(df_run2.bio_m != 0) & (df_run2.trial_type==2)]
    
    run2_con3= df_run2[(df_run2.bio_m != 0) & (df_run2.trial_type==3)]
    
    run2_con4= df_run2[(df_run2.bio_m != 0) & (df_run2.trial_type==4)]
    
    run2_con5= df_run2[(df_run2.bio_m != 0) & (df_run2.trial_type==5)]
    
    run2_con6= df_run2[(df_run2.bio_m == 0)]
    
    #step 3
    
    #run1
    run1_con1_mimg= image.mean_img(nib.concat_images(run1_con1['nifti_3d'].tolist()))
    
    run1_con2_mimg= image.mean_img(nib.concat_images(run1_con2['nifti_3d'].tolist()))
    
    run1_con3_mimg= image.mean_img(nib.concat_images(run1_con3['nifti_3d'].tolist()))
    
    run1_con4_mimg= image.mean_img(nib.concat_images(run1_con4['nifti_3d'].tolist()))
    
    run1_con5_mimg= image.mean_img(nib.concat_images(run1_con5['nifti_3d'].tolist()))
    
    run1_con6_mimg= image.mean_img(nib.concat_images(run1_con6['nifti_3d'].tolist()))
    
    #run2
    run2_con1_mimg= image.mean_img(nib.concat_images(run2_con1['nifti_3d'].tolist()))
    
    run2_con2_mimg= image.mean_img(nib.concat_images(run2_con2['nifti_3d'].tolist()))
    
    run2_con3_mimg= image.mean_img(nib.concat_images(run2_con3['nifti_3d'].tolist()))
    
    run2_con4_mimg= image.mean_img(nib.concat_images(run2_con4['nifti_3d'].tolist()))
    
    run2_con5_mimg= image.mean_img(nib.concat_images(run2_con5['nifti_3d'].tolist()))
    
    run2_con6_mimg= image.mean_img(nib.concat_images(run2_con6['nifti_3d'].tolist()))
    
    
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
    
    #Saving RSMs as csv
    print(f'Now saving {sub_id}')
    output_dir= os.path.join(root_path, sub_id)
    os.makedirs(output_dir)
    rsm.to_csv(os.path.join(output_dir, sub_id + 'RSM.csv'), header= True, index= True, index_label= False)
    rdm.to_csv(os.path.join(output_dir, sub_id + 'RDM.csv'), header= True, index= True, index_label= False)





import seaborn as sn
import matplotlib.pyplot as plt

sn.heatmap(rdm, vmin= 0, vmax=1, cmap= 'viridis', annot=True)

rsm.to_csv(os.path.join(root_path, 'rsm.csv'), header= True, index= True, index_label= False)
rsm_path= os.path.join(root_path, 'rsm.csv')
rdm_102= pd.read_csv(os.path.join(output_dir, sub_id + 'RDM.csv'), sep=',' )


test_app= []

test_app.append(rdm)


'''
Group level rsm analysis for glmsingle
'''

import os
import glob
import pandas as pd
import numpy as np
rsm_glm_path= r'E:\SocialDetection7T\Nahid_Project\RSM_GLMSingle'
treatment_grp_path= 'E:\SocialDetection7T\BehavioralData\LOXY_LZPtrial_forJohannes_noMissing.csv'
treatment_grp= pd.read_csv(treatment_grp_path, sep= ';', on_bad_lines='skip')

print(treatment_grp)

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
    
    rdm_path_lpz= glob.glob(lpz_grp_path_ls[paths] + '\*RDM.csv')
    lpz_rdm_ls= pd.read_csv(rdm_path_lpz[0], sep=',').to_numpy()
    lpz_rdm.append(lpz_rdm_ls)

#make average lpz rdm using numpy
mean_lpz_rdm= np.mean(lpz_rdm, axis=0)

#covert it to dataframe for visualization
mean_df_lpz= pd.DataFrame(mean_lpz_rdm, columns= ['U1_R1', 'U2_R1', 'F1_R1', 'F2_R1', 'NT_R1','SC_R1'], index= ['U1_R2', 'U2_R2', 'F1_R2', 'F2_R2', 'NT_R2','SC_R2'])

import seaborn as sn

sn.heatmap(mean_df_lpz, vmin=0, vmax= 1, cmap= 'viridis')

#get all plc rdms 
plc_rdm= []

for paths in range(0, len(plc_grp_path_ls)):
    
    rdm_path_plc= glob.glob(plc_grp_path_ls[paths] + '\*RDM.csv')
    plc_rdm_ls= pd.read_csv(rdm_path_plc[0], sep=',').to_numpy()
    plc_rdm.append(plc_rdm_ls)

#make average lpz rdm using numpy
mean_plc_rdm= np.mean(plc_rdm, axis=0)

#covert it to dataframe for visualization
mean_df_plc= pd.DataFrame(mean_plc_rdm, columns= ['U1_R1', 'U2_R1', 'F1_R1', 'F2_R1', 'NT_R1','SC_R1'], index= ['U1_R2', 'U2_R2', 'F1_R2', 'F2_R2', 'NT_R2','SC_R2'])

sn.heatmap(mean_df_plc, vmin=0, vmax= 1, cmap= 'viridis', annot)

flat_plc= mean_plc_rdm.flatten()
flat_lpz= mean_lpz_rdm.flatten()

import scipy.stats as stats

print(stats.spearmanr(flat_lpz, flat_plc, ))

df_scatter= pd.DataFrame({'LPZ_RDM': flat_lpz, 'PLC_RDM': flat_plc})
sn.scatterplot(x= flat_lpz, y= flat_plc)
1*len(flat_lpz)
np.ones((1,36), dtype= int).tolist()

print(np.array([1,2,3,4]))
hue1= np.ones([1,36], dtype= int).flatten()
hue2= np.zeros([1,36], dtype= int).flatten()
hue_all= np.concatenate((hue1, hue2), axis= 0).tolist()

flat_all= np.concatenate((flat_lpz, flat_plc), axis= 0).tolist()
len(flat_lpz)

sn.regplot(x= 'LPZ_RDM', y= 'PLC_RDM',data=df_scatter)
sn.jointplot(x= 'LPZ_RDM', y= 'PLC_RDM',data=df_scatter, kind='kde')

sn.scatterplot(data= df_scatter)

