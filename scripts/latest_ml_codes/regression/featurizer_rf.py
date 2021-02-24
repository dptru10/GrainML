#!/usr/bin/env python 
from __future__ import print_function
import os 
import pickle
import matplotlib 
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  
from matplotlib import colors
from joblib import dump, load
from sklearn.preprocessing import normalize
from sklearn import datasets
from sklearn.decomposition import PCA 
from matminer.featurizers.function import FunctionFeaturizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor 
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
#parser.add_argument("file2", help="file to be read in", type=str)
parser.add_argument("--insert",action="store_true")
parser.add_argument("--remove",action="store_true")
parser.add_argument("--all",action="store_true")
parser.add_argument("--average",action="store_true")
parser.add_argument("--top_n",action="store_true")
parser.add_argument("--random_n",action="store_true")
args  = parser.parse_args() 

print(args.file1)
data=pd.read_csv(args.file1)

print(args.file1)
data=pd.read_csv(args.file1)

#if args.outliers:
#    data=data.loc[data['anomaly']==1]
#
#if args.positive is True: 
#    data=data.loc[data['deltaE']>0.0]
#
#if args.negative is True: 
#    data=data.loc[data['deltaE']<0.0]

if args.top_n is True: 
    data=data.head(n=int(1e3))#.sample(n=int(2e4),random_state=1)
if args.random_n is True: 
    data=data.sample(n=int(1e3),random_state=1)

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

combined=combined.replace([np.inf, -np.inf], np.nan)
combined=combined.dropna()#fillna(0.0)
combined.to_csv('sampled_data.csv')
Y=combined['deltaE']
X=combined.drop(labels='deltaE',axis=1)


X_init=data[features]
Y=data[endpoint]

Featurizer=FunctionFeaturizer(multi_feature_depth=2,combo_function=np.sum,expressions=["x","1/x","x**2","x**-2","x**3","x**-3","exp(x)","exp(-x)"],latexify_labels=False)
Featurizer.fit(X_init)
df=Featurizer.featurize_dataframe(X_init, features)
df.to_csv('frame.csv')
labels=Featurizer.feature_labels()
print(labels)

X=df
combined=pd.DataFrame()
for item in labels:
    combined[item]=df[item]
combined['deltaE']=Y

combined=combined.replace([np.inf, -np.inf], np.nan)
combined=combined.replace(['inf', '-inf'], np.nan)
combined=combined.dropna()
combined.to_csv('sampled_data_test.csv')
Y=combined['deltaE']
X=combined.drop(labels='deltaE',axis=1)
X.to_csv('sampled_data.csv')

#X=normalize(X)

correlated_data=pd.DataFrame()
f=open('corr_coeffs.csv', mode='w')
for item in labels:
        f.write("%s,%.3f\n" %(item,np.corrcoef(X[item],Y)[0][1]))
f.close()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=0)

scaler=StandardScaler()
scaler.fit(X_train.fillna(0))

selector=SelectFromModel(Lasso(alpha=0))

selector.fit(scaler.transform(X_train),y_train)

selected_features=X_train.columns[(selector.get_support())]

print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_features)))
print('features with coefficients shrank to zero: {}'.format(
          np.sum(selector.estimator_.coef_ == 0)))

important_features=pd.DataFrame() 
important_features['feature']=selected_features

important_features.to_csv('important_features.csv')

X_train = selector.transform(X_train.fillna(0))
X_test  = selector.transform(X_test.fillna(0))

print('training set size: %s' %len(X_train))

n_estimators=1000
print('training...')
forest = ExtraTreesRegressor(n_estimators=n_estimators,random_state=1,n_jobs=-1)
forest.fit(X_train,y_train)
print('done!...')
model_train=forest.predict(X_train)
model_test=forest.predict(X_test)
r2_score_train=r2_score(y_train,model_train)
mse_score_train=mean_squared_error(y_train,model_train)
mae_score_train=mean_absolute_error(y_train,model_train)
rmse_score_train=np.sqrt(mse_score_train)
r2_score_test=r2_score(y_test,model_test)
mse_score_test=mean_squared_error(y_test,model_test)
mae_score_test=mean_absolute_error(y_test,model_test)
rmse_score_test=np.sqrt(mse_score_test)

#print('writing pickle file...')
dump(forest,'%s.pkl' %name)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

f=open('hyperpameters_'+name+'.txt',mode='w')
f.write('Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,mae_score_train,r2_score_test,mse_score_test,rmse_score_test,mae_score_test)) 
f.close() 

print('score metrics...')
print('Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,mae_score_train,r2_score_test,mse_score_test,rmse_score_test,mae_score_test)) 
 
# Print the feature ranking
print("Feature ranking:")

#ranked_features=[]
#for f in range(X.shape[1]):
#    ranked_features.append(features[indices[f]])
#    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

print('making plots...')
# Plot the feature importances of the forest
#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(X.shape[1]), importances[indices],
#       color="r", yerr=std[indices], align="center")
#plt.xticks(range(X.shape[1]),ranked_features,fontsize=8,rotation=45)
#plt.xlim([-1, X.shape[1]])
#plt.tight_layout()
#plt.savefig('random_forest_feature_importance.png')

df1=pd.DataFrame()
df1['train']=y_train
df1['train_model']=model_train
#i=0
#for item in X_train: 
    #print(item)
    #df1[features[i]]=item
    #i+=1
df1.to_csv('forest_model_vs_endpoint_train_'+name+'.csv')


df2=pd.DataFrame()
df2['test']=y_test
df2['test_model']=model_test
#i=0
#for item in X_test: 
    #print(item)
    #df2[features[i]]=item
    #i+=1 
df2.to_csv('forest_model_vs_endpoint_test_'+name+'.csv')

#plot figures
plt.figure()
plt.title('Histogram forest Train')
plt.hist2d(x=y_train,y=model_train,bins=100,norm=colors.LogNorm())   
plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
plt.colorbar() 
plt.xlabel('Reported Energy (eV/$\AA^{2}$)')
plt.ylabel('Predicted Energy (eV/$\AA^{2}$)')
plt.tight_layout()
plt.savefig('forest_histogram_train_'+name+'.png')

plt.figure()
plt.title('Histogram forest Test')
plt.hist2d(x=y_test,y=model_test,bins=100,norm=colors.LogNorm())
plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
plt.colorbar() 
plt.xlabel('Reported Energy (eV/$\AA^{2}$)')
plt.ylabel('Predicted Energy (eV/$\AA^{2}$)')
plt.tight_layout()
plt.savefig('forest_histogram_test_'+name+'.png')

