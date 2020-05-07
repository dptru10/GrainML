#!/usr/bin/env python 
from __future__ import print_function

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np  
from matplotlib import colors
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from joblib import dump, load
import pandas as pd 
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
#parser.add_argument("file2", help="file to be read in", type=str)
parser.add_argument("--pca",action="store_true")
parser.add_argument("--top_n",action="store_true")
parser.add_argument("--random_n",action="store_true")
args  = parser.parse_args() 

print(args.file1)
data=pd.read_csv(args.file1)

if args.top_n is True: 
    data=data.head(n=int(1e4))#.sample(n=int(2e4),random_state=1)
if args.random_n is True: 
    data=data.sample(n=int(1e4),random_state=1)


features=['op_voronoi_vol','nn_si_bonds_average','nn_vv_average','nn_dist_average','nn_eng_average','nn_cnp_average',
          'nn_vv_std','nn_dist_std','nn_eng_std','nn_cnp_std','nn_si_bonds_std',
          'nn_vv_min','nn_dist_min','nn_eng_min','nn_cnp_min','nn_si_bonds_min',
          'nn_vv_max','nn_dist_max','nn_eng_max','nn_cnp_max','nn_si_bonds_max']

data=data.loc[data['deltaE']<0.4]
data=data.loc[data['deltaE']>-0.2]

name=args.file1
name=name.split('/')
name=name[len(name)-1]
name=name.split('.')[0]

endpoint='deltaE'

X=data[features]
Y=data[endpoint]
print('X.columns') 
print(X.columns) 

combined=pd.DataFrame()
for item in features:
    print(item)
    combined[item]=X[item]
combined['deltaE']=Y

print('len(X)')
print(np.shape(X))
pca=PCA()#'mle',svd_solver='full')
X_pca = pca.fit_transform(X)

if args.pca is True: 
	pca = PCA(n_components=len(features))
	pca.fit(X)
	print('pca')
	pca_ex_var=pca.explained_variance_ratio_
	labels=features#pca.components_

	plt_data = pca_ex_var
	y_pos = np.arange(len(labels))

	plt.figure()
	plt.bar(y_pos, plt_data, align='center', alpha=0.5)
	plt.xticks(y_pos, labels)
	plt.ylabel('Explained Variance Ratio')
	plt.xticks(rotation=90)
	plt.tight_layout() 
	
	plt.savefig('exp_var.png')

