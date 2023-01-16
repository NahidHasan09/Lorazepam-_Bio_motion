# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:34:58 2023

@author: CENs-Admin
"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plc_path= r'E:\SocialDetection7T\Nahid_Project\IS_RSA\Placebo\all_sub_similarity_plc.npy'

output_dir= r'E:\SocialDetection7T\Nahid_Project\Figures\ROI_RSMs\Placebo'
os.listdir()

plc_dict= np.load(plc_path, allow_pickle=True).item()





plc_keys= list(plc_dict.keys())

roi1=[]; roi2=[]; roi3= []; roi4=[]; 
roi5=[]; roi6=[]



for i in plc_keys:
    roi1.append(plc_dict[i][0].to_numpy())
    roi2.append(plc_dict[i][1].to_numpy())
    roi3.append(plc_dict[i][2].to_numpy())
    roi4.append(plc_dict[i][3].to_numpy())
    roi5.append(plc_dict[i][4].to_numpy())
    roi6.append(plc_dict[i][5].to_numpy())

mean_roi1= np.mean(roi1, axis=0)    
mean_roi2= np.mean(roi2, axis=0)
mean_roi3= np.mean(roi3, axis=0)
mean_roi4= np.mean(roi4, axis=0)
mean_roi5= np.mean(roi5, axis=0)
mean_roi6= np.mean(roi6, axis=0)

mask= np.triu(np.ones_like(mean_roi1, dtype= bool))

#%% Make figures
min_tr= [31, 35, 32, 30, 34, 32] #minimum no. of trials per condition
labels= ['U1', 'U2', 'F1', 'F2', 'NT', 'SR']

x_y_ticks= [0]*len(min_tr)

for i in range(0, len(min_tr)):
    x_y_ticks[i]= min_tr[i]+ x_y_ticks[i-1]    


sns.set(style='ticks', font_scale=1.4)

ax= sns.heatmap(mean_roi4, mask= mask, cmap= 'viridis', vmin=0, vmax=1, cbar_kws= {'label': 'Similarity'})
ax.set_xticks(x_y_ticks)
ax.set_yticks(x_y_ticks)
ax.set_xticklabels(labels, minor= False, rotation='horizontal')
ax.set_yticklabels(labels, minor= False)

plt.savefig(os.path.join(
    output_dir, 'ROI4half_RSM_PLC.png'), dpi= 600, bbox_inches= 'tight')
        

plt.show()


((194*194)/2)*59



