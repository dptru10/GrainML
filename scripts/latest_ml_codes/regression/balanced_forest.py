#!/usr/bin/env python 
from __future__ import print_function

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np  
from matplotlib import colors
from sklearn.model_selection import GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from joblib import dump 
import pandas as pd 
import argparse 

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

data_neg=data.loc[data['deltaE']< 0.0]
data_pos=data.loc[data['deltaE']> 0.0]

data_neg['deltaE'] = -1 
data_pos['deltaE'] =  1

X = data_neg[features]
X = X.append(data_pos[features])

endpoint='deltaE'
Y=data_neg[endpoint]
Y=Y.append(data_pos[endpoint])

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

# make training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=1,stratify=Y)

tuned_parameters=[{'n_estimators':[2000,10000]}]
scores=['accuracy']
print('training...')
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    forest = GridSearchCV(BalancedRandomForestClassifier(random_state=1),tuned_parameters,verbose=10,cv=5,n_jobs=-1,scoring='%s' %score)
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
    #confusion=confusion_matrix(y_train,model_train)
    plt.figure()
    confusion=plot_confusion_matrix(forest,X_test,y_test,display_labels=classes,cmap="coolwarm",normalize='true')
    #fig, ax = plt.subplots(figsize=(size, size))
    #sns.set(font_scale=1.5)
    #sns_plot=sns.heatmap(confusion,cmap="coolwarm",square=True,annot=True, fmt=".1f",annot_kws={"size": 20})
    #plt.yticks(rotation=0,fontsize=16,fontweight='bold')
    #plt.xticks(rotation=90,fontsize=16,fontweight='bold')
    #plt.tight_layout()
    plt.savefig('confusion_matrix_%s.png' %name)


    print('writing pickle file...')
    dump(forest,'%s_class_bf.pkl' %name)
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
    df1.to_csv('forest_model_vs_endpoint_train_'+name+'.csv')
    
    
    df2=pd.DataFrame()
    df2['test']=y_test
    df2['test_model']=model_test
    df2.to_csv('forest_model_vs_endpoint_test_'+name+'.csv')
