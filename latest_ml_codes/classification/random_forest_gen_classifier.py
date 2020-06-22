#!/usr/bin/env python 
from __future__ import print_function

import argparse 
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd 
import seaborn as sns
from matplotlib import colors
from joblib import dump, load
from collections import Counter
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
parser.add_argument("--negative",action="store_true")
parser.add_argument("--positive",action="store_true")
parser.add_argument("--insert",action="store_true")
parser.add_argument("--remove",action="store_true")
parser.add_argument("--outliers",action="store_true")
parser.add_argument("--top_n",action="store_true")
parser.add_argument("--random_n",action="store_true")
args  = parser.parse_args() 

print(args.file1)
name=args.file1
name=name.split('/')
name=name[len(name)-1]
extension = name.split('.')[1]
name=name.split('.')[0]
print(name) 


if extension == 'pkl':
	data=pd.read_pickle(args.file1)

if extension == 'csv':
	data=pd.read_csv(args.file1)


#neg=data.loc[data['deltaE']<0.0] 
#pos=data.loc[data['deltaE']>0.0] 

data['labels'] = 0

data.loc[data['deltaE'] < 0.0, 'labels'] = -1
data.loc[data['deltaE'] > 0.0, 'labels'] = 1

print(Counter(data['labels']))

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


endpoint='labels'

X=data[features]
Y=data[endpoint]
print('X.columns') 
print(X.columns) 

combined=pd.DataFrame()
for item in features:
    combined[item]=X[item]
combined['deltaE']=Y

combined=combined.replace([np.inf, -np.inf], np.nan)
combined=combined.replace(['inf', '-inf'], np.nan)
combined=combined.dropna()
X.to_csv('sampled_data.csv')
Y=combined['deltaE']
X=combined.drop(labels='deltaE',axis=1)
#X=StandardScaler().fit_transform(X)

from imblearn.under_sampling import RandomUnderSampler 
random = RandomUnderSampler(random_state=0)
print(random.get_params)
X_new,Y_new=random.fit_resample(X,Y)

# make training and test set
X_train, X_test, y_train, y_test = train_test_split(X_new, Y_new, test_size=0.1,random_state=1)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2200, num = 11)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 8]
# Method of selecting samples for training each tree
bootstrap = [True,False]
# Create the random grid
tuned_parameters = {'n_estimators': n_estimators,'max_features': max_features,'max_depth': max_depth,
                'min_samples_split': min_samples_split,'min_samples_leaf': min_samples_leaf,'bootstrap': bootstrap}
scores = ['accuracy','precision','recall']

#forest = ExtraTreesClassifier(n_estimators=1000,random_state=1)
#forest.fit(X,Y)
#importances = forest.feature_importances_
#std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#             axis=0)
#indices = np.argsort(importances)[::-1]
#
#
#print("Feature ranking:")
#
#ranked_features=[]
#for f in range(X.shape[1]):
#    ranked_features.append(features[indices[f]])
#    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))
#
#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(X.shape[1]), importances[indices],
#       color="r", yerr=std[indices], align="center")
#plt.xticks(range(X.shape[1]),ranked_features,rotation=45,fontsize=10,fontweight='bold')
#plt.xlim([-1, X.shape[1]])
#plt.tight_layout()
#plt.savefig('random_forest_feature_importance.png')


for score in scores:
    print(score) 
    forest = GridSearchCV(ExtraTreesClassifier(random_state=1),tuned_parameters,verbose=0,cv=3,n_jobs=-1,scoring='%s' %score)
    
    forest.fit(X_train,y_train)

    print("Best parameters set found on development set:")    
    print()
    print(str(forest.best_params_))
    print()
    print('Score:')
    print(str(forest.best_score_))
    print()
    print('done!...')

    model_train=forest.predict(X_train)
    model_test=forest.predict(X_test)
    
    accuracy_train=accuracy_score(y_train,model_train)
    accuracy_test=accuracy_score(y_test,model_test)
   
    print('determining confusion matrix...')
   
    
    classes=['negative','positive']
    size=20
    confusion=confusion_matrix(y_train,model_train)
    print(confusion)
    plt.figure()
    confusion=plot_confusion_matrix(forest,X_test,y_test,display_labels=classes,cmap="coolwarm",normalize='all')
    #fig, ax = plt.subplots(figsize=(size, size))
    #sns.set(font_scale=1.5)
    #sns_plot=sns.heatmap(confusion,cmap="coolwarm",square=True,annot=True, fmt=".1f",annot_kws={"size": 20})
    #plt.yticks(rotation=0,fontsize=16,fontweight='bold')
    #plt.xticks(rotation=90,fontsize=16,fontweight='bold')
    #plt.tight_layout()
    plt.savefig('confusion_matrix_%s_%s.png' %(name,score))


    print('writing pickle file...')
    dump(forest,'%s_class_trees_class_%s.pkl' %(name,score))
    #importances = forest.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in forest.estimators_],
    #             axis=0)
    #indices = np.argsort(importances)[::-1]
    
    f=open('hyperpameters_'+name+'.txt',mode='w')
    f.write('Train: \naccuracy:%.3f \nTest:\naccuracy: %.3f' %(accuracy_train,accuracy_test))
    f.close() 
    
    print('score metrics...')
    print('Train: \naccuracy:%.3f \nTest:\naccuracy: %.3f' %(accuracy_train,accuracy_test))
    
    df1=pd.DataFrame()
    df1['train']=y_train
    df1['train_model']=model_train
    df1.to_csv('forest_model_vs_endpoint_train_'+name+'_'+score+'.csv')
    
    
    df2=pd.DataFrame()
    df2['test']=y_test
    df2['test_model']=model_test
    df2.to_csv('forest_model_vs_endpoint_test_'+name+'_'+score+'.csv')
