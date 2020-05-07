#!/usr/bin/env python 
from __future__ import print_function

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np  
from matplotlib import colors
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed #, dump, load
import pandas as pd 
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
parser.add_argument("--top_n",action="store_true")
parser.add_argument("--random_n",action="store_true")
args  = parser.parse_args() 

print(args.file1)
data=pd.read_csv(args.file1)

if args.top_n is True: 
    data=data.head(n=int(1e4))#.sample(n=int(2e4),random_state=1)
if args.random_n is True: 
    data=data.sample(n=int(1e4),random_state=1)


features=['op_voronoi_vol','op_eng','op_cnp','nn_si_bonds_average','nn_vv_average','nn_dist_average','nn_eng_average','nn_cnp_average',
          'nn_vv_std','nn_dist_std','nn_eng_std','nn_cnp_std','nn_si_bonds_std',
          'nn_vv_min','nn_dist_min','nn_eng_min','nn_cnp_min','nn_si_bonds_min',
          'nn_vv_max','nn_dist_max','nn_eng_max','nn_cnp_max','nn_si_bonds_max']

data=data.loc[data['deltaE']<0.4]
data=data.loc[data['deltaE']>-0.2]


#drop_list=[]
#print(np.var(data))
#for i in trange(len(features)): 
#	var_init = np.var(data[features[i]]) 
#	for j in trange(len(data)):
#		test_var=data[features[i]].drop(index=j) 
#		var_new=np.var(test_var)
#		if var_new < var_init:
#                        #print("dropped %i" %j)
#                        drop_list.append(j) 
var_init=Parallel(n_jobs=-1)(delayed(np.var)(data[features[i]]) for i in range(len(features))) 
var_new=Parallel(n_jobs=-1)(delayed(np.var)(data[features[i]].drop(index=j)) for j in range(len(data)) for i in range(len(features)))

np.save('var_init.npy',var_init)
np.save('var_new.npy',var_new)

print(len(drop_list)) 


