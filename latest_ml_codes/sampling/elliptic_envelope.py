#!/usr/bin/env python 
from __future__ import print_function

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np  
from matplotlib import colors
from sklearn.model_selection import GridSearchCV
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.model_selection import train_test_split
from collections import Counter
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


features=['op_voronoi_vol','nn_si_bonds_average','nn_vv_average','nn_dist_average','nn_eng_average','nn_cnp_average',
          'nn_vv_std','nn_dist_std','nn_eng_std','nn_cnp_std','nn_si_bonds_std',
          'nn_vv_min','nn_dist_min','nn_eng_min','nn_cnp_min','nn_si_bonds_min',
          'nn_vv_max','nn_dist_max','nn_eng_max','nn_cnp_max','nn_si_bonds_max']

data=data.loc[data['deltaE']<0.4]
data=data.loc[data['deltaE']>-0.2]

endpoint='deltaE'
X=data[features]
Y=data[endpoint]
print('X.columns') 
print(X.columns) 


EllipticEnvelope=EllipticEnvelope(contamination=0.15,random_state=0)
EllipticEnvelope.fit(data[features])

data['anomaly'] = EllipticEnvelope.predict(data[features])

print(data['anomaly'].value_counts())
data.to_csv('data_anomaly_ee.csv')

