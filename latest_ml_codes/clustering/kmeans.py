#!/usr/bin/env python 
from __future__ import print_function

import argparse 
import matplotlib 
import numpy as np  
import pandas as pd 
import seaborn as sns 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.cluster import KMeans 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import calinski_harabasz_score 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
parser.add_argument("--insert",action="store_true")
parser.add_argument("--remove",action="store_true")
parser.add_argument("--top_n",action="store_true")
parser.add_argument("--random_n",action="store_true")
args  = parser.parse_args() 

name = args.file1 
name = name.split('.')[0]
name = name.split('/')
name = name[len(name)-1]
print(name) 

print(args.file1)
data=pd.read_csv(args.file1)

if args.top_n is True: 
    data=data.head(n=int(1e4))#.sample(n=int(2e4),random_state=1)
if args.random_n is True: 
    data=data.sample(n=int(1e4),random_state=1)


if args.insert is True: 
    features=['op_voronoi_vol','nn_si_bonds_average','nn_vv_average','nn_dist_average','nn_eng_average','nn_cnp_average',
              'nn_vv_std','nn_dist_std','nn_eng_std','nn_cnp_std','nn_si_bonds_std',
              'nn_vv_min','nn_dist_min','nn_eng_min','nn_cnp_min','nn_si_bonds_min',
              'nn_vv_max','nn_dist_max','nn_eng_max','nn_cnp_max','nn_si_bonds_max']

if args.remove is True: 
    features=['op_voronoi_vol','op_eng','op_cnp','nn_si_bonds_average','nn_vv_average','nn_dist_average','nn_eng_average','nn_cnp_average',
               'nn_vv_std','nn_dist_std','nn_eng_std','nn_cnp_std','nn_si_bonds_std',
               'nn_vv_min','nn_dist_min','nn_eng_min','nn_cnp_min','nn_si_bonds_min',
               'nn_vv_max','nn_dist_max','nn_eng_max','nn_cnp_max','nn_si_bonds_max']


endpoint='deltaE'
X = data[endpoint]
X = np.array(X) 
X = X.reshape(-1,1)
#mms = MinMaxScaler()
#mms.fit(X) 

#data_transform=mms.transform(X) 
summed_square_distance=[]
calinski_score=[]

clusters=range(2,15) 

for i in clusters: 
    kmeans=KMeans(n_clusters=i)
    kmeans=kmeans.fit(X)#data_transform)
    summed_square_distance.append(kmeans.inertia_)
    calinski_score.append(calinski_harabasz_score(X,kmeans.labels_))

plt.figure() 
plt.plot(clusters, summed_square_distance,'bx-',label='summed_square_distance')
#plt.plot(clusters, calinski_score,'rx-',label='calinski_harabasz')
plt.legend()
plt.xlabel('k')
#plt.yscale('log') 
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.tight_layout() 
plt.savefig('elbow_method_%s.png' %name)
