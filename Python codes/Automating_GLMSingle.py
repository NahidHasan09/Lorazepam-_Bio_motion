#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:06:45 2022

@author: johannesschultz
"""

'''
For renaming folder names

import os

os.getcwd()

os.chdir('/Volumes/WD_TRANSFER/SocialDetection7T/freshData')

len(os.listdir())

folders= os.listdir()[2:]



for fol_names in folders:
    old_name= fol_names
    new_name= fol_names[9:]+ '_'+ fol_names[:8]
    os.rename(old_name, new_name)
    

'''



import os
import glob
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
from slicetime.main import run_slicetime
from slicetime.make_image_stack import make_image_stack
from slicetime.tseriesinterp import tseriesinterp
import pandas as pd
from glmsingle.design.make_design_matrix import make_design
from nilearn.image import resample_img
from nilearn.image import binarize_img
import sys
import urllib.request
import warnings
from tqdm import tqdm
import glmsingle
from glmsingle.glmsingle import GLM_single







stimdur= 3.35
tr= 0.63

#Get the data paths


#change working directory to the data directory


root_path= r'E:\SocialDetection7T\freshData' 

eventfs_path= r'E:\SocialDetection7T\onsets_Logfiles'

output_path= r'E:\SocialDetection7T\Nahid_Project\GLMSingle_Outputs_amygdala'

mask_img_path_name= r'E:\SocialDetection7T\Nahid_Project\Masks\amygdala_mask.nii'

#mask image
mask_img= nib.load(mask_img_path_name)


# get temp data path list by joing directories and subdirectories

temp_path_list= []

for dirs, subdir, file in os.walk(root_path):
    for name in subdir:
        temp_path_list.append(os.path.join(dirs,name))

#filter the subdirectories where data is

runs_path=[]

for i in range(len(temp_path_list)):
    if temp_path_list[i][-2:]=='_M':
        add_dicom= temp_path_list[i] + '\DICOM'
        runs_path.append(add_dicom)


# join per subject runs in this case it's two

runs= []

for run_1, run_2 in zip(runs_path[::2],runs_path[1::2]):
    runs.append([run_1, run_2])

runs.remove(runs[30])  

# get 3d_niftis per subject in list [[niftis...],[niftis....]]
# for loop will stop at index 30 which is LOXY 139
for imgs in range(1, len(runs)):
    sub_id= runs[imgs][0][31:39]
    print(f'Now working with subject# {sub_id} ')
    niftis_3d_run_1= glob.glob(runs[imgs][0]+ '\wr*.nii')
    niftis_3d_run_2= glob.glob(runs[imgs][1]+ '\wr*.nii')
    ls_nifti=[]
    ls_nifti.append(niftis_3d_run_1)
    ls_nifti.append(niftis_3d_run_2)
    eventfs= glob.glob(eventfs_path + '\socialDetection-fMRI-'+ sub_id + '*.txt')
    
    data=[]
    design= []
    
    for item, (run, eventf) in enumerate(zip(ls_nifti, eventfs)):
        
        
        
        # concatenate to one 4D nifti
        
        brain_vol= nib.funcs.concat_images(run) 
        
        #get affine and shape for resampling the mask image
        affine= brain_vol.affine
        shape= brain_vol.dataobj[:,:,:,0].shape
        
        #mask the data to make model A faster
        masker= NiftiMasker( target_affine=affine, target_shape=shape,
                                 mask_strategy='background')
        masker.fit(mask_img)

        
        
        #interpolate the masked data (4d nii 3 times)
        
        brain_vol= run_slicetime(inpath=brain_vol,slicetimes=None, tr_old=1.9, tr_new= tr, time_dim=2, offset=0)
        
        
        
        #for selecting specific voxels 
        #outputs num of scans, num of voxels
        brain_vol= masker.transform_single_imgs(brain_vol)
        
        #transpose the output to (num of voxels, num of scans)
        brain_vol= brain_vol.T.astype(np.float32)
        # get number of volumes for making design matrix
        n_volumes= brain_vol.shape[-1]

        
        #making design matrix
        
        # read eventfiles 
        dm_read= pd.read_csv(eventf, sep=",", header=None, names= ["trial_no", "stimtype", "no_need", "biomotion", "accuracy", "onsets", "offsets", "correct", "RT"])
        
        onsets= dm_read["onsets"].values
        items= dm_read["stimtype"].values
        n_events= len(onsets)
        
        events= pd.DataFrame()
        events["duration"]= [stimdur]* n_events
        events["onset"]= np.round(onsets)
        events["trial_type"]= items
        
        design.append(make_design(events, tr, n_volumes))
        data.append(brain_vol)

    #set glmsingle parameters

    opt= dict()

    opt['wantlibrary']= 1
    opt['wantglmdenoise']=1
    opt['wantfracridge']=1
    #opt['hrffitmask']= mask

    opt['wantfileoutputs']= [0,0,0,0]
    opt['wantmemoryoutputs']= [0,0,0,1]

    glmsingle_obj= GLM_single(opt)

    glmsingle_obj.params


    #run glmsingle parameters
    results_glmsingle= glmsingle_obj.fit(design, data, stimdur, tr)
    
    
    type_d= results_glmsingle['typed']
    
    #again transpose back to (num of scans, num of voxels) 
    #inverse to original brian space
    
    tr_betas= type_d['betasmd'].T
    betas= masker.inverse_transform(tr_betas)
    
    #saving files
    print(f'Saving files of {sub_id}')
    output_dir= os.path.join(output_path,sub_id)
    os.makedirs(output_dir)
    np.save(os.path.join( output_dir, sub_id +'.npy'), type_d)
    nib.save(betas, os.path.join(output_dir, sub_id +'.nii'))


# LOXY 139 gave error in making design matrix
#We will exclude this for now, will see later

len(os.listdir(output_path))












