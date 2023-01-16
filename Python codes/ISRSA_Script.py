# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 14:22:12 2022

@author: CENs-Admin
"""

"ISRSA group level"

import os
import glob
import numpy as np
import pandas as pd
from nltools.data import Brain_Data, Adjacency
from nltools.mask import expand_mask, roi_to_brain, collapse_mask
import nibabel as nib
from nilearn import image
from nilearn import plotting
from nltools.stats import fdr, threshold, fisher_r_to_z, one_sample_permutation, holm_bonf
import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns
import fitter
from scipy.stats import shapiro 
import ptitprince as pt

#%% Set paths

glm_all_path= r'E:\SocialDetection7T\Nahid_Project\GLMSingle_Outputs_all'
eventfs_path= r'E:\SocialDetection7T\onsets_Logfiles'
treatment_grp_path= 'E:\SocialDetection7T\BehavioralData\LOXY_LZPtrial_forJohannes_noMissing.csv'

nl_mask_path= r'E:\SocialDetection7T\Nahid_Project\Masks\Whole_brain_mask_nltools.nii'
mni_template1mm_path= r'E:\SocialDetection7T\Nahid_Project\Templates\MNI152_T1_1mm_brain\MNI152_T1_1mm_brain.nii'
roi_all_path= r'E:\SocialDetection7T\Nahid_Project\ROIs\ROI_all_4d.nii'
output_dir= r'E:\SocialDetection7T\Nahid_Project\IS_RSA\Lorazepam'
output_dir_plc= r'E:\SocialDetection7T\Nahid_Project\IS_RSA\Placebo'
bio_motion_rsm_path= r'E:\SocialDetection7T\Nahid_Project\Model_RSMs\Model_RSM_Biological_Motion.npy'
action_cat_rsm_path= r'E:\SocialDetection7T\Nahid_Project\Model_RSMs\Model_RSM_Action_Category.npy'
emotion_rsm_path= r'E:\SocialDetection7T\Nahid_Project\Model_RSMs\Model_RSM_Emotion.npy'
bio_motion_fig_path= r'E:\SocialDetection7T\Nahid_Project\Figures\Representation_Hypo\Lorazepam\Bio_motion'
action_cat_fig_path= r'E:\SocialDetection7T\Nahid_Project\Figures\Representation_Hypo\Lorazepam\Action_cat'
emotion_fig_path= r'E:\SocialDetection7T\Nahid_Project\Figures\Representation_Hypo\Lorazepam\Emotion'
rep_hyp_path= r'E:\SocialDetection7T\Nahid_Project\IS_RSA\Rep_hypothesis'
#%% Load model rsms

bio_motion= Adjacency(np.load(bio_motion_rsm_path), matrix_type= 'similarity')
action_cat= Adjacency(np.load(action_cat_rsm_path), matrix_type= 'similarity')
emotion= Adjacency(np.load(emotion_rsm_path), matrix_type= 'similarity')


#%% Create 6 ROI percellation mask
nl_mask= nib.load(nl_mask_path)

mni_template1mm= nib.load(mni_template1mm_path)

roi_all= nib.load(roi_all_path)
bi_roi_all= image.binarize_img(roi_all)



roi_mask= Brain_Data(bi_roi_all, mask= nl_mask)

roi_mask= roi_mask.astype('int32')

roi_mask= collapse_mask(roi_mask, custom_mask=nl_mask)

roi_mask.plot()

plotting.plot_roi(roi_mask.to_nifti(), bg_img= mni_template1mm)

# expand mask
roi_mask_x= expand_mask(roi_mask)






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
lpz_grp= lpz_plc[lpz_plc['Group']==1]['Sub'].tolist()

plc_grp= lpz_plc[lpz_plc['Group']==0]['Sub'].tolist()



#%%

plc_sub_list = []
for folder_name in plc_grp:
    plc_sub_list.append(os.path.join(glm_all_path, folder_name))



lpz_sub_list = []
for folder_name in lpz_grp:
    lpz_sub_list.append(os.path.join(glm_all_path, folder_name))
    
 
    
all_sub_similarity = {}; 
all_sub_bio_motion_rsa = {};
all_sub_action_cat_rsa= {}; all_sub_emotion_rsa = {};

for imgs in range(0, len(lpz_sub_list)):
    sub_id= lpz_grp[imgs]
    print(f'Now working with subject# {sub_id} ')
    nifti_path= glob.glob(lpz_sub_list[imgs]+ '\L*.nii')
    
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
    
    df_across_run= pd.DataFrame({'trial_type': flt_cons_all, 'bio_m': flt_bio_scrm_all, 'nifti_3d': betas_3d})
    

    
    # step 2
    #run1
    run1_con1= df_across_run[(df_across_run.bio_m != 0) & (df_across_run.trial_type==1)][0:31]
    
    run1_con2= df_across_run[(df_across_run.bio_m != 0) & (df_across_run.trial_type==2)][0:35]
    
    run1_con3= df_across_run[(df_across_run.bio_m != 0) & (df_across_run.trial_type==3)][0:32]
    
    run1_con4= df_across_run[(df_across_run.bio_m != 0) & (df_across_run.trial_type==4)][0:30]
    
    run1_con5= df_across_run[(df_across_run.bio_m != 0) & (df_across_run.trial_type==5)][0:34]
    
    run1_con6= df_across_run[(df_across_run.bio_m == 0)][0:32]
    
    data_list= run1_con1['nifti_3d'].tolist() + run1_con2['nifti_3d'].tolist() + run1_con3['nifti_3d'].tolist() + run1_con4['nifti_3d'].tolist() +run1_con5['nifti_3d'].tolist() + run1_con6['nifti_3d'].tolist()
    
    data_4d= nib.concat_images(data_list)
    
    beta_data= Brain_Data(data_4d, mask= nl_mask)
    
    
    sub_pattern = []; 
    bio_motion_sim_r = []; 
    
    action_cat_sim_r = []; emotion_sim_r = [];
    
    for m in roi_mask_x:
        masked_data = beta_data.apply_mask(m)
        df_sub_pattern= pd.DataFrame(np.transpose(masked_data.data))
        sub_pattern_similarity= df_sub_pattern.corr('spearman')
        
        sub_pattern_similarity_ad= Adjacency(sub_pattern_similarity, matrix_type='similarity')
        s_bio_motion = sub_pattern_similarity_ad.similarity(bio_motion, metric='spearman', n_permute=0)
        s_action_cat = sub_pattern_similarity_ad.similarity(action_cat, metric='spearman', n_permute=0)
        s_emotion = sub_pattern_similarity_ad.similarity(emotion, metric='spearman', n_permute=0)
        sub_pattern.append(sub_pattern_similarity)
        bio_motion_sim_r.append(s_bio_motion['correlation'])
        action_cat_sim_r.append(s_action_cat['correlation'])
        emotion_sim_r.append(s_emotion['correlation'])
    
    all_sub_similarity[sub_id] = sub_pattern
    all_sub_bio_motion_rsa[sub_id] = bio_motion_sim_r
    all_sub_action_cat_rsa[sub_id] = action_cat_sim_r  
    all_sub_emotion_rsa[sub_id] = emotion_sim_r
    
np.save(os.path.join(output_dir, 'all_sub_similarity_lpz.npy'), all_sub_similarity)    

#%% Save bio_motion figures 


df_all_sub_bio_motion_rsa_lpz = pd.DataFrame(all_sub_bio_motion_rsa).T

rsa_stats_bio_motion = []

for i in df_all_sub_bio_motion_rsa_lpz:
    rsa_stats_bio_motion.append(one_sample_permutation(df_all_sub_bio_motion_rsa_lpz[i], return_perms=True))  



bonf = 0.05/len(roi_mask_x.data[:,0])
print(bonf)

rsa_bio_motion_r = Brain_Data([x*y['mean'] for x,y in zip(roi_mask_x, rsa_stats_bio_motion)]).sum()
rsa_bio_motion_p = Brain_Data([x*y['p'] for x,y in zip(roi_mask_x, rsa_stats_bio_motion)]).sum()

thresholded = threshold(rsa_bio_motion_r, rsa_bio_motion_p, thr=bonf)

plotting.plot_glass_brain(thresholded.to_nifti(), cmap='coolwarm')
    
bio_motion_fig_1= plotting.plot_stat_map(thresholded.to_nifti(), bg_img= mni_template1mm,display_mode='z', 
                       cut_coords=[-20, 0, 10], black_bg=False, colorbar=False) 
    
bio_motion_fig_2= plotting.plot_stat_map(thresholded.to_nifti(), bg_img= mni_template1mm,display_mode='z', 
                         cut_coords=[20, 30, 40], black_bg=False) 
bio_motion_fig_1.savefig(os.path.join(bio_motion_fig_path, 'Representation_bio_motion_brain_percel_1.png'), dpi=300)

bio_motion_fig_2.savefig(os.path.join(bio_motion_fig_path, 'Representation_bio_motion_brain_percel_2.png'), dpi=300)
   
params = {'axes.labelsize': 14,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14}

plt.rcParams.update(params)


for i in range(0, len(rsa_stats_bio_motion)):
    plt.figure(figsize=(5, 5), dpi= 600)
    plt.hist(rsa_stats_bio_motion[i]['perm_dist'], bins=50)
    plt.xlabel(f"Speaman's rho")
    plt.ylabel("Frequency")
    plt.xlim(-0.3, 0.4)
    plt.ylim(0, 300)
    plt.axvline(rsa_stats_bio_motion[i]['mean'], color= 'b', linestyle= '--')
    plt.axvline(0, color= 'k', alpha= 0.5, linestyle= '--')
    score_label = f"rho: {rsa_stats_bio_motion[i]['mean']:.2f}\n p < 0.05 \n(Bonf.)"
    plt.text(0.13, 200, score_label, fontsize=14)
    plt.savefig(os.path.join(
        bio_motion_fig_path, 'ROI_'+ str(i) +'Bio_motion_permutation_dist.png'), bbox_inches= 'tight')
    

plt.show() 



#%% save action_cat figures

df_all_sub_action_cat_rsa_lpz = pd.DataFrame(all_sub_action_cat_rsa).T
rsa_stats_action_cat = []

for i in df_all_sub_action_cat_rsa_lpz:
    rsa_stats_action_cat.append(one_sample_permutation(df_all_sub_action_cat_rsa_lpz[i], return_perms=True))  

bonf = 0.05/len(roi_mask_x.data[:,0])
print(bonf)




rsa_action_cat_r = Brain_Data([x*y['mean'] for x,y in zip(roi_mask_x, rsa_stats_action_cat)]).sum()
rsa_action_cat_p = Brain_Data([x*y['p'] for x,y in zip(roi_mask_x, rsa_stats_action_cat)]).sum()

thresholded_action_cat = threshold(rsa_action_cat_r, rsa_action_cat_p, thr=bonf)

plotting.plot_glass_brain(thresholded_action_cat.to_nifti(), cmap='coolwarm')



action_cat_fig_1= plotting.plot_stat_map(thresholded_action_cat.to_nifti(), bg_img= mni_template1mm,display_mode='z', 
                       cut_coords=[-20, 0, 10], black_bg=False, colorbar=False) 
    
action_cat_fig_2= plotting.plot_stat_map(thresholded_action_cat.to_nifti(), bg_img= mni_template1mm,display_mode='z', 
                         cut_coords=[20, 30, 40], black_bg=False) 
action_cat_fig_1.savefig(os.path.join(action_cat_fig_path, 'Representation_action_cat_brain_percel_1.png'), dpi=300)

action_cat_fig_2.savefig(os.path.join(action_cat_fig_path, 'Representation_action_cat_brain_percel_2.png'), dpi=300)


params = {'axes.labelsize': 14,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14}

plt.rcParams.update(params)


for i in range(0, len(rsa_stats_action_cat)):
    plt.figure(figsize=(5, 5), dpi= 600)
    plt.hist(rsa_stats_action_cat[i]['perm_dist'], bins=50)
    plt.xlabel(f"Speaman's rho")
    plt.ylabel("Frequency")
    plt.xlim(-0.3, 0.30)
    plt.ylim(0, 300)
    plt.axvline(rsa_stats_action_cat[i]['mean'], color= 'b', linestyle= '--')
    plt.axvline(0, color= 'k', alpha= 0.5, linestyle= '--')
    score_label = f"rho: {rsa_stats_action_cat[i]['mean']:.2f}\n p < 0.05 \n(Bonf.)"
    plt.text(0.09, 200, score_label, fontsize=14)
    plt.savefig(os.path.join(
        action_cat_fig_path, 'ROI_'+ str(i) +'Action_cat_permutation_dist.png'), bbox_inches= 'tight')
    

plt.show() 


#%% Save emotion figures 

df_all_sub_emotion_rsa_lpz = pd.DataFrame(all_sub_emotion_rsa).T
rsa_stats_emotion = []

for i in df_all_sub_emotion_rsa_lpz:
    rsa_stats_emotion.append(one_sample_permutation(df_all_sub_emotion_rsa_lpz[i], return_perms=True))  

bonf = 0.05/len(roi_mask_x.data[:,0])
print(bonf)




rsa_emotion_r = Brain_Data([x*y['mean'] for x,y in zip(roi_mask_x, rsa_stats_emotion)]).sum()
rsa_emotion_p = Brain_Data([x*y['p'] for x,y in zip(roi_mask_x, rsa_stats_emotion)]).sum()

thresholded_emotion = threshold(rsa_emotion_r, rsa_emotion_p, thr=bonf)

plotting.plot_glass_brain(thresholded_emotion.to_nifti(), cmap='coolwarm')



emotion_fig_1= plotting.plot_stat_map(thresholded_emotion.to_nifti(), bg_img= mni_template1mm,display_mode='z', 
                       cut_coords=[-20, 0, 10], black_bg=False, colorbar=False) 
    
emotion_fig_2= plotting.plot_stat_map(thresholded_emotion.to_nifti(), bg_img= mni_template1mm,display_mode='z', 
                         cut_coords=[20, 30, 40], black_bg=False) 
emotion_fig_1.savefig(os.path.join(emotion_fig_path, 'Representation_emotion_brain_percel_1.png'), dpi=300)

emotion_fig_2.savefig(os.path.join(emotion_fig_path, 'Representation_emotion_brain_percel_2.png'), dpi=300)


params = {'axes.labelsize': 14,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14}

plt.rcParams.update(params)


for i in range(0, len(rsa_stats_emotion)):
    plt.figure(figsize=(5, 5), dpi= 600)
    plt.hist(rsa_stats_emotion[i]['perm_dist'], bins=50)
    plt.xlabel(f"Speaman's rho")
    plt.ylabel("Frequency")
    plt.xlim(-0.3, 0.30)
    plt.ylim(0, 300)
    plt.axvline(rsa_stats_emotion[i]['mean'], color= 'b', linestyle= '--')
    plt.axvline(0, color= 'k', alpha= 0.5, linestyle= '--')
    score_label = f"rho: {rsa_stats_emotion[i]['mean']:.2f}\n p < 0.05 \n(Bonf.)"
    plt.text(0.061, 200, score_label, fontsize=14)
    plt.savefig(os.path.join(
        emotion_fig_path, 'ROI_'+ str(i) +'emotion_permutation_dist.png'), bbox_inches= 'tight')
    

plt.show() 


##save for anova analysis

df_all_sub_bio_motion_rsa.to_csv(os.path.join(output_dir, 'All_sub_bio_motion.csv'))
df_all_sub_action_cat_rsa.to_csv(os.path.join(output_dir, 'All_sub_action_cat.csv'))
df_all_sub_emotion_rsa.to_csv(os.path.join(output_dir, 'All_sub_emotion.csv'))



df_all_sub_bio_motion_rsa_lpz.to_csv(os.path.join(output_dir, 'All_sub_bio_motion_lpz.csv'))
df_all_sub_action_cat_rsa_lpz.to_csv(os.path.join(output_dir, 'All_sub_action_cat_lpz.csv'))
df_all_sub_emotion_rsa_lpz.to_csv(os.path.join(output_dir, 'All_sub_emotion_lpz.csv'))



#%%Make CSV files of bio_motion for LMM

#Read Bio_motion LPZ file
df_all_sub_bio_motion_rsa_lpz= pd.read_csv(os.path.join(output_dir, 'All_sub_bio_motion_lpz.csv'), sep=',', index_col=0)

#Insert Treatment column
df_all_sub_bio_motion_rsa_lpz.insert(6, 'Treatment', ['LPZ']*30, True)

#Add subject col as index
df_all_sub_bio_motion_rsa_lpz['Sub']= df_all_sub_bio_motion_rsa_lpz.index 


#Rename cols
df_all_sub_bio_motion_rsa_lpz.columns=['Similarity_1','Similarity_2','Similarity_3','Similarity_4','Similarity_5','Similarity_6', 
                                       'Treatment', 'Sub']

#Convert dataframe to long format
l_df_lpz= pd.wide_to_long(df_all_sub_bio_motion_rsa_lpz, stubnames= 'Similarity', i= 'Sub', j= 'ROI', sep='_')




#For plc group

df_all_sub_bio_motion_rsa_plc= pd.read_csv(os.path.join(output_dir_plc, 'All_sub_bio_motion.csv'), sep=',', index_col=0)

df_all_sub_bio_motion_rsa_plc.insert(6, 'Treatment', ['PLC']*29, True)

df_all_sub_bio_motion_rsa_plc['Sub']= df_all_sub_bio_motion_rsa_plc.index 

df_all_sub_bio_motion_rsa_plc.columns=['Similarity_1','Similarity_2','Similarity_3','Similarity_4','Similarity_5','Similarity_6', 
                                       'Treatment', 'Sub']


l_df_plc= pd.wide_to_long(df_all_sub_bio_motion_rsa_plc, stubnames= ['Similarity'], i='Sub', j= 'ROI', sep='_')

#concatenate both groups 
l_df_bio_motion_lpz_plc= pd.concat([l_df_plc, l_df_lpz])

#reindex 
l_df_bio_motion_lpz_plc=l_df_bio_motion_lpz_plc.reset_index().reindex(["Sub", "Treatment", "ROI", "Similarity"], axis=1)

l_df_bio_motion_lpz_plc.to_csv(os.path.join(rep_hyp_path,'Bio_motion','bio_motion_long_lpz_plc.csv'))



#%% Make CSV files of action_cat for LMM


#Read action_cat LPZ file
df_all_sub_action_cat_rsa_lpz= pd.read_csv(os.path.join(output_dir, 'All_sub_action_cat_lpz.csv'), sep=',', index_col=0)

#Insert Treatment column
df_all_sub_action_cat_rsa_lpz.insert(6, 'Treatment', ['LPZ']*30, True)

#Add subject col as index
df_all_sub_action_cat_rsa_lpz['Sub']= df_all_sub_action_cat_rsa_lpz.index 


#Rename cols
df_all_sub_action_cat_rsa_lpz.columns=['Similarity_1','Similarity_2','Similarity_3','Similarity_4','Similarity_5','Similarity_6', 
                                       'Treatment', 'Sub']

#Convert dataframe to long format
l_df_lpz= pd.wide_to_long(df_all_sub_action_cat_rsa_lpz, stubnames= 'Similarity', i= 'Sub', j= 'ROI', sep='_')




#For plc group

df_all_sub_action_cat_rsa_plc= pd.read_csv(os.path.join(output_dir_plc, 'All_sub_action_cat.csv'), sep=',', index_col=0)

df_all_sub_action_cat_rsa_plc.insert(6, 'Treatment', ['PLC']*29, True)

df_all_sub_action_cat_rsa_plc['Sub']= df_all_sub_action_cat_rsa_plc.index 

df_all_sub_action_cat_rsa_plc.columns=['Similarity_1','Similarity_2','Similarity_3','Similarity_4','Similarity_5','Similarity_6', 
                                       'Treatment', 'Sub']


l_df_plc= pd.wide_to_long(df_all_sub_action_cat_rsa_plc, stubnames= ['Similarity'], i='Sub', j= 'ROI', sep='_')

#concatenate both groups 
l_df_action_cat_lpz_plc= pd.concat([l_df_plc, l_df_lpz])

#reindex 
l_df_action_cat_lpz_plc=l_df_action_cat_lpz_plc.reset_index().reindex(["Sub", "Treatment", "ROI", "Similarity"], axis=1)

l_df_action_cat_lpz_plc.to_csv(os.path.join(rep_hyp_path,'Action_cat','action_cat_long_lpz_plc.csv'))


#%% Make CSV files of emotion for LMM


#Read emotion LPZ file
df_all_sub_emotion_rsa_lpz= pd.read_csv(os.path.join(output_dir, 'All_sub_emotion_lpz.csv'), sep=',', index_col=0)

#Insert Treatment column
df_all_sub_emotion_rsa_lpz.insert(6, 'Treatment', ['LPZ']*30, True)

#Add subject col as index
df_all_sub_emotion_rsa_lpz['Sub']= df_all_sub_emotion_rsa_lpz.index 


#Rename cols
df_all_sub_emotion_rsa_lpz.columns=['Similarity_1','Similarity_2','Similarity_3','Similarity_4','Similarity_5','Similarity_6', 
                                       'Treatment', 'Sub']

#Convert dataframe to long format
l_df_lpz= pd.wide_to_long(df_all_sub_emotion_rsa_lpz, stubnames= 'Similarity', i= 'Sub', j= 'ROI', sep='_')




#For plc group

df_all_sub_emotion_rsa_plc= pd.read_csv(os.path.join(output_dir_plc, 'All_sub_emotion.csv'), sep=',', index_col=0)

df_all_sub_emotion_rsa_plc.insert(6, 'Treatment', ['PLC']*29, True)

df_all_sub_emotion_rsa_plc['Sub']= df_all_sub_emotion_rsa_plc.index 

df_all_sub_emotion_rsa_plc.columns=['Similarity_1','Similarity_2','Similarity_3','Similarity_4','Similarity_5','Similarity_6', 
                                       'Treatment', 'Sub']


l_df_plc= pd.wide_to_long(df_all_sub_emotion_rsa_plc, stubnames= ['Similarity'], i='Sub', j= 'ROI', sep='_')

#concatenate both groups 
l_df_emotion_lpz_plc= pd.concat([l_df_plc, l_df_lpz])

#reindex 
l_df_emotion_lpz_plc=l_df_emotion_lpz_plc.reset_index().reindex(["Sub", "Treatment", "ROI", "Similarity"], axis=1)

l_df_emotion_lpz_plc.to_csv(os.path.join(rep_hyp_path,'Emotion','emotion_long_lpz_plc.csv'))

#%% Test for Normality and identifying distributions (do not need this)

params = {'axes.labelsize': 14,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          'font.size':14}

plt.rcParams.update(params)

emotion_qq= pg.qqplot(l_df_emotion_lpz_plc['Similarity'])
plt.savefig(os.path.join(rep_hyp_path,'emotion','Emotion_qqplot.png'), dpi= 300, bbox_inches= 'tight')


emotion_fit= fitter.Fitter(l_df_emotion_lpz_plc['Similarity'])
emotion_fit.fit()

sum_fit= emotion_fit.summary()
plt.savefig(os.path.join(rep_hyp_path,'Emotion','Emotion_data_dist.png'), dpi= 300, bbox_inches= 'tight')
sum_fit.to_excel(os.path.join(rep_hyp_path,'Emotion','emotion_data_dist.xlsx'))


#%%Save raincloud plots
l_df_bio_motion_lpz_plc= pd.read_csv(os.path.join(rep_hyp_path,'Bio_motion','bio_motion_long_lpz_plc.csv'), sep=',', index_col=0)
l_df_action_cat_lpz_plc= pd.read_csv(os.path.join(rep_hyp_path,'Action_cat','action_cat_long_lpz_plc.csv'), sep=',', index_col=0)
l_df_emotion_lpz_plc= pd.read_csv(os.path.join(rep_hyp_path,'Emotion','emotion_long_lpz_plc.csv'), sep=',', index_col=0)


l_df_bio_motion_lpz_plc.replace(to_replace='LPZ', value='LZP', inplace=True)
l_df_action_cat_lpz_plc.replace(to_replace='LPZ', value='LZP', inplace=True)
l_df_emotion_lpz_plc.replace(to_replace='LPZ', value='LZP', inplace=True)

l_df_bio_motion_lpz_plc['Similarity']= fisher_r_to_z(l_df_bio_motion_lpz_plc['Similarity'])

l_df_action_cat_lpz_plc['Similarity']= fisher_r_to_z(l_df_action_cat_lpz_plc['Similarity'])

l_df_emotion_lpz_plc['Similarity']= fisher_r_to_z(l_df_emotion_lpz_plc['Similarity'])



params = {'axes.labelsize': 14,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          'font.size':14}

plt.rcParams.update(params)



custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
dx="ROI"; dy="Similarity"; hue= "Treatment"; ort="h"; pal = "Set2"; sigma = .05
f, ax = plt.subplots(figsize=(7, 7))

ax=pt.RainCloud(x = dx, y = dy, hue= hue, data = l_df_bio_motion_lpz_plc, palette = pal, bw = sigma,
                 width_viol = 1, ax = ax, orient = ort, alpha= .5, dodge=True, move= .2)

plt.text(0.1, 5, 'ab', color= 'g')
plt.text(0.1, 5.25, 'a', color= 'tab:orange')

plt.text(0.35, 4, 'abc', color= 'g')
plt.text(0.35, 4.25, 'abc', color= 'tab:orange')

plt.text(0.65, 3, 'd', color= 'g')
plt.text(0.65, 3.25, 'd', color= 'tab:orange')

plt.text(0.53, 2, 'c', color= 'g')
plt.text(0.53, 2.25, 'bc', color= 'tab:orange')

plt.text(0.35, 1, 'abc', color= 'g')
plt.text(0.35, 1.25, 'bc', color= 'tab:orange')

plt.text(0.2, 0, 'ab', color= 'g')
plt.text(0.2, 0.25, 'abc', color= 'tab:orange')

plt.xlabel('Similarity (z-transformed)')
plt.savefig(os.path.join(rep_hyp_path,'Bio_motion','Raincloud_Bio_motion.png'), dpi= 300, bbox_inches= 'tight')


plt. hist(l_df_all['Similarity'])
shapiro(l_df_all['Similarity'])


