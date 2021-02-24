from __future__ import print_function

import matplotlib 
import numpy as np  
import pandas as pd 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import datasets 
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
parser.add_argument("--insert",action="store_true")
parser.add_argument("--remove",action="store_true")
parser.add_argument("--negative",action="store_true")
parser.add_argument("--positive",action="store_true")
parser.add_argument("--outliers",action="store_true")
parser.add_argument("--featurized",action="store_true")
parser.add_argument("--top_n",action="store_true")
parser.add_argument("--random_n",action="store_true")
parser.add_argument("--step_weights",action="store_true")
parser.add_argument("--exp_weights",action="store_true")
args  = parser.parse_args() 

print(args.file1)
data=pd.read_csv(args.file1)

if args.outliers:
    data=data.loc[data['anomaly']==1]

if args.positive is True: 
    data=data.loc[data['deltaE']>0.0]

if args.negative is True: 
    data=data.loc[data['deltaE']<0.0]

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

if args.featurized is True: 
    features=list(data.columns)
    features.remove('deltaE')

if args.step_weights is True:
    weights=[]
    for i in range(len(data)): 
        if data['deltaE'].iloc[i] < 0: 
    	     weights.append(1) 
        if data['deltaE'].iloc[i] > 0: 
             weights.append(0) 
    data['weights']=pd.Series(weights)

if args.exp_weights is True:
    weights=[]
    for i in range(len(data)): 
        weights.append(np.exp(data['deltaE'].iloc[i])) 
    data['weights']=pd.Series(weights)


low_bias=data.loc[data['deltaE']<0.0]
data=data.append(low_bias)
print('final dataset size')
print(len(data))

name=args.file1
name=name.split('/')
name=name[len(name)-1]
name=name.split('.')[0]

endpoint='labels'

X=data[features]
Y=data[endpoint]
print('X.columns') 
print(X.columns) 

combined=pd.DataFrame()
for item in features:
    print(item)
    combined[item]=X[item]
combined['deltaE']=Y

combined=combined.replace([np.inf, -np.inf], np.nan)
combined=combined.replace(['inf', '-inf'], np.nan)
combined=combined.dropna()#fillna(0.0)
combined.to_csv('sampled_data.csv')
Y=combined['deltaE']
X=combined.drop(labels='deltaE',axis=1)

#X=StandardScaler().fit_transform(X)

# make training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=1)

clusters=range(3,4)
scores = ['accuracy']
tuned_parameters = [{'n_neighbors':clusters}]

cv_list = [5]
for cv in cv_list: 
	print('using %i fold validation:' %cv)
	for score in scores:
		neighbors   = GridSearchCV(KNeighborsClassifier(),tuned_parameters,verbose=0,cv=cv,n_jobs=-1,scoring='%s' %score)
		neighbors.fit(X_train,y_train)
		model_train = neighbors.predict(X_train)
		model_test  = neighbors.predict(X_test)
		print("Best parameters set found on development set:")
		print(str(neighbors.best_params_))
		print('Accuracy Score:')
		print(str(neighbors.best_score_))
		print() 

