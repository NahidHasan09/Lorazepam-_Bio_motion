# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:28:42 2022

@author: CENs-Admin
"""

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


list_of_folder_data_path = sorted(filter(lambda x: os.path.isdir(os.path.join(data_path, x)),
                                   os.listdir(data_path)))

temp_path_ls_glm = []
for folder_name in list_of_folder_data_path:
    temp_path_ls_glm.append(os.path.join(data_path, folder_name))



con1= []
con2= []
con3= []
con4= []
con5= []
con6= []

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

    
    
    df_across_run= pd.DataFrame({'trial_type': flt_cons_all, 'bio_m': flt_bio_scrm_all})
    

    
    # step 2
    #run1
    run1_con1= len(df_across_run[(df_across_run.bio_m != 0) & (df_across_run.trial_type==1)])
    con1.append(run1_con1)
    
    run1_con2= len(df_across_run[(df_across_run.bio_m != 0) & (df_across_run.trial_type==2)])
    con2.append(run1_con2)
    
    run1_con3= len(df_across_run[(df_across_run.bio_m != 0) & (df_across_run.trial_type==3)])
    con3.append(run1_con3)
    
    run1_con4= len(df_across_run[(df_across_run.bio_m != 0) & (df_across_run.trial_type==4)])
    con4.append(run1_con4)
    
    run1_con5= len(df_across_run[(df_across_run.bio_m != 0) & (df_across_run.trial_type==5)])
    con5.append(run1_con5)
    
    run1_con6= len(df_across_run[(df_across_run.bio_m == 0)])
    con6.append(run1_con6)
    
min(con2) 
min(con3) 
min(con4)
min(con5)
min(con6)

max(con2) 
max(con3) 
max(con4)
max(con5)
max(con6)
