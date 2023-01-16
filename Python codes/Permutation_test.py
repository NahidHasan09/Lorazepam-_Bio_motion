# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:16:29 2022

@author: CENs-Admin
"""

''' Let's run a permutation test'''

#%%  Define a test-statistics
def test_stat(x,y, axis):
    return np.mean(x, axis= axis)- np.mean(y, axis= axis)


#%%% Collect our data ( replicate jwilber blog)
import numpy as np
x= np.array([4.1, 7.2, 6.2, 7.1, 4.8, 7.7, 8.3, 4.3, 8.3, 5.8, 4.4,7.4])
y= np.array([3.8, 3.9, 4.6, 4.2, 4.6, 5.1, 5.4, 4.1, 5.6, 5.8, 4.4, 2.8])

obs_dif=test_stat(x, y, axis= 0)

#%% Run the permutation test
from scipy.stats import permutation_test


res= permutation_test((x,y), statistic= test_stat, vectorized=True, n_resamples= 10000, alternative= 'two-sided')

print(res.statistic)
print(res.pvalue)

import matplotlib.pyplot as plt
plt.hist(res.null_distribution, bins=50)
plt.title("Permutation distribution of test statistic")
plt.xlabel("Value of Statistic")
plt.ylabel("Frequency")
plt.axvline(obs_dif, color= 'b', linestyle= '--')
plt.show()


#%%RSA group inference algorithm 

'''
########### This algorithm will be applicable to all ROIs (n= 6) ###########
1. Create neural RSM for each subject (n= 59)
2. Create model RSM for testing representational hypotheses (n= 3)
3. Get similarity scores (e.g. spearman) by comparing neural and model RSMs (n_sim_s= 59/model RSM)
4. Transform the 'r' values to fisher's z
5. Run an one sample permutation (n=10000) test to check 'r' values are 
    consistently different from zero
6. With test statistics and p value, also get permutation distribution
7. Perform multiple comparison correction (e.g. fdr)
8. Plot the 'r' and 'p' values on the brain space


##### Finally, to check the LPZ effect, run a two sample permutation test #####
1. Divide PLC and LPZ 'r' values
2. Calculate a test statistic( meand dif., median dif., t-statistics, etc.)'''





#%% Let's implement this algorithm

''' import the data'''

import os
import glob
import pandas as pd
import numpy as np
from nltools.stats import fdr, threshold, fisher_r_to_z, one_sample_permutation
import seaborn as sns
from scipy import stats


rsm_glm_path= r'E:\SocialDetection7T\Nahid_Project\RSM_GLMSingle\BioM_mask_r1_r2_avg'
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

lpz_rsm= []

for paths in range(0, len(lpz_grp_path_ls)):
    
    rsm_path_lpz= glob.glob(lpz_grp_path_ls[paths] + '\*RSM.csv')
    lpz_rsm_ls= pd.read_csv(rsm_path_lpz[0], sep=',').to_numpy()
    lpz_rsm.append(lpz_rsm_ls)
    

''' model RSM'''

bio_sc= np.zeros((6,6))

bio_sc[np.diag_indices(6)]= 1


bio_sc[0:5, 0:5]= 1

xtick= ['U1', 'U2', 'F1', 'F2', 'NT', 'SC']
ytick= ['U1', 'U2', 'F1', 'F2', 'NT', 'SC']
sns.heatmap( bio_sc, cmap= 'viridis', xticklabels=xtick, yticklabels=ytick)

model_rsm= bio_sc[np.triu_indices(6, k=1)]

z= np.array([1,2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

v=stats.spearmanr(model_rsm, z)
v[0]
''' get similarity scores'''

sub_r_scores= []

for rsms in range(0, len(lpz_rsm)):
    sim_score= stats.spearmanr(a= fisher_r_to_z (lpz_rsm[rsms][np.triu_indices(6, k=1)]), b= model_rsm)
    sub_r_scores.append(sim_score[0])


r_stats= one_sample_permutation(fisher_r_to_z(sub_r_scores), return_perms=True )   

plt.hist(r_stats['perm_dist'], bins=50)
plt.title("Permutation distribution of test statistic")
plt.xlabel("Value of Statistic")
plt.ylabel("Frequency")
plt.axvline(r_stats['mean'], color= 'b', linestyle= '--')
plt.show()   

print(r_stats['p'])







