#!/usr/bin/env python 
from __future__ import print_function

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np  
from matplotlib import colors
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd 
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
parser.add_argument("--negative",action="store_true")
parser.add_argument("--positive",action="store_true")
parser.add_argument("--insert",action="store_true")
parser.add_argument("--remove",action="store_true")
parser.add_argument("--top_n",action="store_true")
parser.add_argument("--random_n",action="store_true")
parser.add_argument("--split_zero",action="store_true")
parser.add_argument("--multi_split",action="store_true")
parser.add_argument("--stencil_split",action="store_true")
parser.add_argument("--cluster_centroids",action="store_true")
parser.add_argument("--random_undersample",action="store_true")
parser.add_argument("--cnn",action="store_true")
parser.add_argument("--near_miss",action="store_true")
parser.add_argument("--instance_hardness",action="store_true")
parser.add_argument("--one_sided",action="store_true")
parser.add_argument("--tomek_links",action="store_true")
args  = parser.parse_args() 

print(args.file1)
data=pd.read_csv(args.file1)

if args.top_n is True: 
    data=data.head(n=int(1e4))#.sample(n=int(2e4),random_state=1)
if args.random_n is True: 
    data=data.sample(n=int(1e4),random_state=1)

if args.insert is True: 
    features=['op_voronoi_vol','nn_si_bonds_average','nn_vv_average','nn_dist_average','nn_eng_average',
              'nn_cnp_average','nn_vv_std','nn_dist_std','nn_eng_std','nn_cnp_std','nn_si_bonds_std',
              'nn_vv_min','nn_dist_min','nn_eng_min','nn_cnp_min','nn_si_bonds_min','nn_vv_max','nn_dist_max',
              'nn_eng_max','nn_cnp_max','nn_si_bonds_max','deltaE']
    
if args.remove is True: 
    features=['op_voronoi_vol','op_eng','op_cnp','nn_si_bonds_average','nn_vv_average','nn_dist_average',
              'nn_eng_average','nn_cnp_average','nn_vv_std','nn_dist_std','nn_eng_std','nn_cnp_std',
              'nn_si_bonds_std','nn_vv_min','nn_dist_min','nn_eng_min','nn_cnp_min','nn_si_bonds_min',
              'nn_vv_max','nn_dist_max','nn_eng_max','nn_cnp_max','nn_si_bonds_max','deltaE']

#data=data.loc[data['deltaE']<0.4]

data['idx'] = range(len(data))

name=args.file1
name=name.split('/')
name=name[len(name)-1]
name=name.split('.')[0]

if args.negative is True: 
    data=data.loc[data['deltaE'] < 0.0] 

if args.positive is True: 
    data=data.loc[data['deltaE'] > 0.0] 

if args.split_zero is True: 
    range0=data.loc[data['deltaE'] < 0.0] 
    range1=data.loc[data['deltaE'] > 0.0] 
    
    print('range0')
    print("%.3f:%.3f" %(np.min(range0['deltaE']),np.max(range0['deltaE'])))
    range0["class"]=np.full(len(range0),0)#False,dtype=bool)
    
    print('range1')
    print("%.3f:%.3f" %(np.min(range1['deltaE']),np.max(range1['deltaE'])))
    range1["class"]=np.full(len(range1),1)#True,dtype=bool)
    
    total = len(range0) + len(range1)
    print('total') 
    print(total)
    
    data_new=pd.DataFrame(columns=data.columns) 
    data_new=data_new.append(range0)
    data_new=data_new.append(range1)

if args.stencil_split is True: 
    min_deltaE=np.min(data['deltaE']) 
    max_deltaE=np.max(data['deltaE'])
    mean_deltaE=np.average(data['deltaE'])
    
    stencil_point0=min_deltaE
    stencil_point1=np.average([min_deltaE,mean_deltaE])
    stencil_point2=mean_deltaE
    stencil_point3=np.average([mean_deltaE,max_deltaE])
    stencil_point4=max_deltaE
    
    centroid0=np.average([stencil_point0,stencil_point1])
    centroid1=np.average([stencil_point1,stencil_point2])
    centroid2=np.average([stencil_point2,stencil_point3])
    centroid3=np.average([stencil_point3,stencil_point4])
    
    range0=data.loc[data['deltaE'] < stencil_point1]     #between stencil points 0 and 1 
    range1=data.loc[data['deltaE'] < stencil_point2]
    range1=range1.loc[range1['deltaE'] > stencil_point1] #between stencil points 1 and 2 
    range2=data.loc[data['deltaE'] < stencil_point3]
    range2=range2.loc[data['deltaE'] > stencil_point2]   #between stencil points 2 and 3
    range3=data.loc[data['deltaE'] < stencil_point4]
    range3=range3.loc[data['deltaE'] > stencil_point3]   #between stencil points 3 and 4
    
    print('range0')
    print("%.3f:%.3f" %(np.min(range0['deltaE']),np.max(range0['deltaE'])))
    range0["class"]=np.full(len(range0),0)
    
    print('range1')
    print("%.3f:%.3f" %(np.min(range1['deltaE']),np.max(range1['deltaE'])))
    range1["class"]=np.full(len(range1),1)
    
    print('range2')
    print("%.3f:%.3f" %(np.min(range2['deltaE']),np.max(range2['deltaE'])))
    range2["class"]=np.full(len(range2),2)
    
    print('range3')
    print("%.3f:%.3f" %(np.min(range3['deltaE']),np.max(range3['deltaE'])))
    range3["class"]=np.full(len(range3),3)
    
    total = len(range0) + len(range1) + len(range2) + len(range3) 
    print('total') 
    print(total)

    data_new=pd.DataFrame(columns=data.columns) 
    data_new=data_new.append(range0)
    data_new=data_new.append(range1)
    data_new=data_new.append(range2)
    data_new=data_new.append(range3)

if args.multi_split is True: 
    #min_deltaE=np.min(data['deltaE']) 
    #max_deltaE=np.max(data['deltaE'])
    #mean_deltaE=np.average(data['deltaE'])
    #
    #stencil_point0=min_deltaE
    #stencil_point1=np.average([min_deltaE,mean_deltaE])
    #stencil_point2=mean_deltaE
    #stencil_point3=max_deltaE
    #
    #centroid0=np.average([stencil_point0,stencil_point1])
    #centroid1=np.average([stencil_point1,stencil_point2])
    #centroid2=np.average([stencil_point2,stencil_point3])
    #centroid3=np.average([stencil_point3,stencil_point4])
    
    range0=data.loc[data['deltaE'] < -0.15]#[data['deltaE'] < stencil_point1]     #between stencil points 0 and 1 
    range1=data.loc[data['deltaE'] < -0.02]#[data['deltaE'] < stencil_point2]
    range1=range1.loc[range1['deltaE'] > -0.15] #between stencil points 1 and 2 
    range2=data.loc[data['deltaE'] < 0.0]
    range2=range2.loc[data['deltaE'] > -0.02]   #between stencil points 2 and 3
    
    print('range0')
    print("%.3f:%.3f" %(np.min(range0['deltaE']),np.max(range0['deltaE'])))
    range0["class"]=np.full(len(range0),0)
    
    print('range1')
    print("%.3f:%.3f" %(np.min(range1['deltaE']),np.max(range1['deltaE'])))
    range1["class"]=np.full(len(range1),1)
    
    print('range2')
    print("%.3f:%.3f" %(np.min(range2['deltaE']),np.max(range2['deltaE'])))
    range2["class"]=np.full(len(range2),2)
    
    total = len(range0) + len(range1) + len(range2) #+ len(range3) 
    print('total') 
    print(total)
    
    data_new=pd.DataFrame(columns=data.columns) 
    data_new=data_new.append(range0)
    data_new=data_new.append(range1)
    data_new=data_new.append(range2)
    #data_new=data_new.append(range3)

endpoint='class'

X=data_new[features]
Y=data_new[endpoint]

print('Original dataset shape %s' % Counter(Y))

if args.cnn is True:
    from imblearn.under_sampling import CondensedNearestNeighbour
    cnn = CondensedNearestNeighbour(random_state=0,n_jobs=-1)
    print(cnn.get_params)
    X_new,Y_new=cnn.fit_resample(X,Y)
    print('Resampled dataset shape %s' % Counter(Y_new))
    X_new.to_csv('X_sample_cnn.csv')

if args.cluster_centroids is True: 
    from imblearn.under_sampling import ClusterCentroids 
    cluster_centroids = ClusterCentroids(random_state=0,n_jobs=-1)
    print(cluster_centroids.get_params)
    X_new,Y_new=cluster_centroids.fit_resample(X,Y)
    print('Resampled dataset shape %s' % Counter(Y_new))
    X_new.to_csv('X_sample_cluster_centroids.csv')

if args.random_undersample is True: 
    from imblearn.under_sampling import RandomUnderSampler 
    random = RandomUnderSampler(random_state=0)
    print(random.get_params)
    X_new,Y_new=random.fit_resample(X,Y)
    print('Resampled dataset shape %s' % Counter(Y_new))
    X_new.to_csv('X_sample_random_undersampler.csv')

if args.near_miss is True: 
    from imblearn.under_sampling import NearMiss 
    near_miss = NearMiss(n_jobs=-1)
    print(near_miss.get_params)
    X_new,Y_new=near_miss.fit_resample(X,Y)
    print('Resampled dataset shape %s' % Counter(Y_new))
    X_new.to_csv('X_sample_near_miss.csv')

if args.instance_hardness is True: 
    from imblearn.under_sampling import InstanceHardnessThreshold 
    IHT = InstanceHardnessThreshold(random_state=0,n_jobs=-1)
    print(IHT.get_params)
    X_new,Y_new=IHT.fit_resample(X,Y)
    print('Resampled dataset shape %s' % Counter(Y_new))
    X_new.to_csv('X_sample_instance_hardness.csv')

if args.one_sided is True: 
    from imblearn.under_sampling import OneSidedSelection 
    OSS = OneSidedSelection(random_state=0,n_jobs=-1)
    print(OSS.get_params)
    X_new,Y_new=OSS.fit_resample(X,Y)
    print('Resampled dataset shape %s' % Counter(Y_new))
    X_new.to_csv('X_sample_one_sided.csv')

if args.tomek_links is True: 
    from imblearn.under_sampling import TomekLinks 
    Tomek = TomekLinks(n_jobs=-1)
    print(Tomek.get_params)
    X_new,Y_new=Tomek.fit_resample(X,Y)
    print('Resampled dataset shape %s' % Counter(Y_new))
    X_new.to_csv('X_sample_tomek.csv')

