# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:28:43 2022

@author: CENs-Admin
"""

import numpy as np
import seaborn as sns


bio_sc= np.zeros((6,6))

bio_sc[np.diag_indices(6)]= 1


bio_sc[0:5, 0:5]= 1

xtick= ['U1', 'U2', 'F1', 'F2', 'NT', 'SC']
ytick= ['U1', 'U2', 'F1', 'F2', 'NT', 'SC']
sns.heatmap( bio_sc, cmap= 'viridis', xticklabels=xtick, yticklabels=ytick)


act_cat= np.zeros((6,6))

act_cat[0:2, 0:2]=1
act_cat[2:4, 0:2]=0.8
act_cat[2:4, 2:4]=1
act_cat[0:2,2:4]=0.8
act_cat[4:5,4:5]=1
act_cat[4:5,0:4]=0.6
act_cat[0:4,4:5]=0.6
sns.heatmap( act_cat, cmap= 'viridis', xticklabels=xtick, yticklabels=ytick)
