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
from imblearn.under_sampling import CondensedNearestNeighbour
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
          'nn_vv_max','nn_dist_max','nn_eng_max','nn_cnp_max','nn_si_bonds_max','deltaE']

#data=data.loc[data['deltaE']<0.4]
#data=data.loc[data['deltaE']>-0.2]

data['idx'] = range(len(data))

name=args.file1
name=name.split('/')
name=name[len(name)-1]
name=name.split('.')[0]

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
#print(len(data_new))

endpoint='class'

X=data_new[features]
Y=data_new[endpoint]
print('Original dataset shape %s' % Counter(Y))

cnn = CondensedNearestNeighbour(n_jobs=-1,random_state=42)
X_new,Y_new=cnn.fit_resample(X,Y)


print('Resampled dataset shape %s' % Counter(Y_new))

#regression_endpoint='deltaE'
#X=X_new#[features]
#X=X.append(range2)
#X=X.append(range3)
X_new.to_csv('X_sample.csv')

#Y=data_new[regression_endpoint]
#
#print('X.columns') 
#print(X.columns) 
#
#combined=pd.DataFrame()
#for item in features:
#    print(item)
#    combined[item]=X[item]
#combined['deltaE']=Y
#
#combined=combined.replace([np.inf, -np.inf], np.nan)
#combined=combined.replace(['inf', '-inf'], np.nan)
#combined=combined.dropna()#fillna(0.0)
#combined.to_csv('sampled_data.csv')
#Y=combined['deltaE']
#X=combined.drop(labels='deltaE',axis=1)
#
## make training and test set
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=1)
#
#print('training set size: %s' %len(X_train))
#
#n_estimators=1000
#print('training...')
#forest = ExtraTreesRegressor(n_estimators=n_estimators,random_state=1,n_jobs=-1)
#forest.fit(X_train,y_train)
#print('done!...')
#model_train=forest.predict(X_train)
#model_test=forest.predict(X_test)
#r2_score_train=r2_score(y_train,model_train)
#mse_score_train=mean_squared_error(y_train,model_train)
#mae_score_train=mean_absolute_error(y_train,model_train)
#rmse_score_train=np.sqrt(mse_score_train)
#r2_score_test=r2_score(y_test,model_test)
#mse_score_test=mean_squared_error(y_test,model_test)
#mae_score_test=mean_absolute_error(y_test,model_test)
#rmse_score_test=np.sqrt(mse_score_test)
#
##print('writing pickle file...')
##dump(forest,'%s.pkl' %name)
#importances = forest.feature_importances_
#std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#             axis=0)
#indices = np.argsort(importances)[::-1]
#
#f=open('hyperpameters_'+name+'.txt',mode='w')
#f.write('Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,mae_score_train,r2_score_test,mse_score_test,rmse_score_test,mae_score_test)) 
#f.close() 
#
#print('score metrics...')
#print('Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,mae_score_train,r2_score_test,mse_score_test,rmse_score_test,mae_score_test)) 
# 
## Print the feature ranking
#print("Feature ranking:")
#
#ranked_features=[]
#for f in range(X.shape[1]):
#    ranked_features.append(features[indices[f]])
#    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))
#
#print('making plots...')
## Plot the feature importances of the forest
#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(X.shape[1]), importances[indices],
#       color="r", yerr=std[indices], align="center")
#plt.xticks(range(X.shape[1]),ranked_features,fontsize=8,rotation=80)
#plt.xlim([-1, X.shape[1]])
#plt.tight_layout()
#plt.savefig('random_forest_feature_importance.png')
#
#df1=pd.DataFrame()
#df1['train']=y_train
#df1['train_model']=model_train
##i=0
##for item in X_train: 
#    #print(item)
#    #df1[features[i]]=item
#    #i+=1
#df1.to_csv('forest_model_vs_endpoint_train_'+name+'.csv')
#
#
#df2=pd.DataFrame()
#df2['test']=y_test
#df2['test_model']=model_test
##i=0
##for item in X_test: 
#    #print(item)
#    #df2[features[i]]=item
#    #i+=1 
#df2.to_csv('forest_model_vs_endpoint_test_'+name+'.csv')
#
##plot figures
#plt.figure()
#plt.title('Histogram forest Train')
#plt.hist2d(x=y_train,y=model_train,bins=100,norm=colors.LogNorm())   
#plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
#plt.colorbar() 
#plt.xlabel('Reported Energy (eV/$\AA^{2}$)')
#plt.ylabel('Predicted Energy (eV/$\AA^{2}$)')
#plt.tight_layout()
#plt.savefig('forest_histogram_train_'+name+'.png')
#
#plt.figure()
#plt.title('Histogram forest Test')
#plt.hist2d(x=y_test,y=model_test,bins=100,norm=colors.LogNorm())
#plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
#plt.colorbar() 
#plt.xlabel('Reported Energy (eV/$\AA^{2}$)')
#plt.ylabel('Predicted Energy (eV/$\AA^{2}$)')
#plt.tight_layout()
#plt.savefig('forest_histogram_test_'+name+'.png')
#
##print('validate model with %s dataset' %args.file2)
##data=pd.read_csv(args.file2)
##
##actual=data['deltaE']
##predicted=forest.predict(data[features])
##
##df=pd.DataFrame()
##df['true']=actual
##df['predicted']=predicted
##df.to_csv('forest_model_vs_endpoint_validate_'+name+'.csv')
##
##name=args.data
##name=name.split('.')[0]
##name=name.split('/')
##name=name[len(name)-1]
##            
##r2_score_validate=r2_score(actual,predicted)
##mse_score_validate=mean_squared_error(actual,predicted)
##mae_score_validate=mean_absolute_error(actual,predicted)
##rmse_score_validate=np.sqrt(mse_score_validate)
##            
##f=open('validation_score_'+name+'.txt',mode='w')
##f.write('Validation:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f' %(r2_score_validate,mse_score_validate,rmse_score_validate,mae_score_validate)) 
##f.close() 
##
##plt.figure()
##plt.title('Histogram Forest Validation')
##plt.hist2d(x=actual,y=predicted,bins=100,norm=colors.LogNorm())
##plt.axis([-0.2,0.4,-0.2,0.4])#[np.min(actual),np.max(actual),np.min(actual),np.max(actual)])
##plt.colorbar() 
##plt.xlabel('Reported Energy (eV/$\AA^{2}$)')
##plt.ylabel('Predicted Energy (eV/$\AA^{2}$)')
##plt.tight_layout()
##plt.savefig('forest_histogram_validation_'+name+'.png')
##
##print('completely done!')
#
