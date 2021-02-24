import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
sns.set(color_codes=True)
from scipy import stats as st
from numpy import genfromtxt
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
parser.add_argument("--negative",action="store_true")
parser.add_argument("--positive",action="store_true")
parser.add_argument("--split_zero",action="store_true")
parser.add_argument("--multi_split",action="store_true")
parser.add_argument("--insert",action="store_true")
parser.add_argument("--remove",action="store_true")
parser.add_argument("--top_n",action="store_true")
parser.add_argument("--random_n",action="store_true")
args  = parser.parse_args() 

print(args.file1)
data=pd.read_csv(args.file1)

if args.top_n is True: 
    data=data.head(n=int(5e4))#.sample(n=int(2e4),random_state=1)
if args.random_n is True: 
    data=data.sample(n=int(5e4),random_state=1)

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
    data = data_new 

if args.multi_split is True: 
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
    data = data_new

#distribution of data points
sns.distplot(data[features], kde=False)
#low_data = data.loc[data['deltaE'] < 0.0]

pca = PCA()
transformed_data = pca.fit_transform(data[features])
my_kde = st.gaussian_kde(data[features])
sample = my_kde.resample() #int(1e8)).reshape((int(1e8),))

num_sample = int(5e3)  
nrows1, npc = transformed_data.shape
sampled_data = np.zeros((num_sample, npc))

#set the gaussian width as desired, or leave it default 
## A very tight bandwidth is used here 
for pc in range(npc):
    kde_obj = st.gaussian_kde(transformed_data[:,pc], bw_method=0.02)
    #print(kde_obj.factor)
    sampled = kde_obj.resample(num_sample).reshape((num_sample,))
    for i in range(num_sample):
        sampled_data[i][pc] = sampled[i]

#sns.distplot(sampled_data[:,10])
#sns.distplot(transformed_data[:,10])

sampled_data_original_space = pca.inverse_transform(sampled_data)
#sampled_data_original_space.shape
#sns.distplot(data[:,ncols-1], kde=False)
#sns.distplot(sampled_data_original_space[:,ncols-1], kde=False)
#sampled_data_original_space
np.savetxt("generated_data.csv", sampled_data_original_space, delimiter=",", fmt='%.4f')
