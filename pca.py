#!/usr/bin/env python 
from __future__ import print_function

#import pickle
import matplotlib 
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np  
#from joblib import dump, load
#from sklearn import datasets
from sklearn.decomposition import PCA 
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import RandomizedSearchCV, GridSearchCV 
#from sklearn.metrics import mean_squared_error, r2_score 
#from sklearn.kernel_ridge import KernelRidge
import pandas as pd 
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
args  = parser.parse_args() 

data=pd.read_csv(args.file1)

#features=['cnp','centro','vor_vol','vor_neigh']
#features=['Misorientation', 'Sigma', 'Stoichiometry']
features=['op_vornoi_vol','op_si-si_bonds','op_si-c_bonds','op_c-c_bonds','nn_1_distance','nn_1_eng','nn_1_cnp','nn_1_vornoi_vol','nn_1_si-si_bonds','nn_1_si-c_bonds','nn_2_distance','nn_2_eng','nn_2_cnp','nn_2_vornoi_vol','nn_2_si-si_bonds','nn_2_si-c_bonds','nn_2_c-c_bonds','nn_3_distance','nn_3_eng','nn_3_cnp','nn_3_vornoi_vol','nn_3_si-si_bonds','nn_3_si-c_bonds','nn_3_c-c_bonds','nn_4_distance','nn_4_eng','nn_4_cnp','nn_4_vornoi_vol','nn_4_si-si_bonds','nn_4_si-c_bonds']
#endpoint='total energy'
endpoint='deltaE'

X=data[features]

holder=[]
for feature in features: 
    for x in X[feature]:
        if np.isnan(x) == True: 
            x = 0 
        holder.append(x) 
    X[feature]=holder
    holder=[]

#principle component analysis 
pca = PCA(n_components=len(features))
pca.fit(X)
print('pca')
pca_ex_var=pca.explained_variance_ratio_

print(pca_ex_var)
for i in range(len(features)):
    print('%s:\t%.4f' %(features[i],pca_ex_var[i]))
