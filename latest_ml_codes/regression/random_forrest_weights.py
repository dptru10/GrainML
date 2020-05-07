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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd 
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

#print('generating averaged descriptors...')
#data['nn_vv_average']=data['nn_1_voronoi_vol'] + data['nn_2_voronoi_vol'] + data['nn_3_voronoi_vol'] + data['nn_4_voronoi_vol'] 
#data['nn_vv_average']=data['nn_vv_average'].divide(4)#len(data['nn_vv_average']))
#data['nn_dist_average']=data['nn_1_distance'] + data['nn_2_distance'] + data['nn_3_distance'] + data['nn_4_distance']
#data['nn_dist_average']=data['nn_dist_average'].divide(4)#len(data['nn_dist_average']))
#data['nn_eng_average']=data['nn_1_eng'] + data['nn_2_eng'] + data['nn_3_eng'] + data['nn_4_eng']
#data['nn_eng_average']=data['nn_eng_average'].divide(4)#len(data['nn_eng_average']))
#data['nn_cnp_average']=data['nn_1_cnp'] + data['nn_2_cnp'] + data['nn_3_cnp'] + data['nn_4_cnp']
#data['nn_cnp_average']=data['nn_cnp_average'].divide(4)#len(data['nn_cnp_average']))
#data['nn_si_bonds_average']=data['nn_1_si-si_bonds'] + data['nn_2_si-si_bonds'] + data['nn_3_si-si_bonds'] + data['nn_4_si-si_bonds']
#data['nn_si_bonds_average']=data['nn_si_bonds_average'].divide(4)#len(data['nn_cnp_average']))
#
#nn_vv_std      =[]
#nn_dist_std    =[]
#nn_eng_std     =[]
#nn_cnp_std     =[]
#nn_si_bonds_std=[]
#
#nn_vv_min      =[]
#nn_dist_min    =[]
#nn_eng_min     =[]
#nn_cnp_min     =[]
#nn_si_bonds_min=[]
#
#nn_vv_max      =[]
#nn_dist_max    =[]
#nn_eng_max     =[]
#nn_cnp_max     =[]
#nn_si_bonds_max=[]
#
#print('calculating statistical descriptors...')
#print('determining std...')
## determine variance 
#for i in tqdm(range(len(data))):
#    nn_vv_std.append(np.std([data['nn_1_voronoi_vol'].iloc[i],data['nn_2_voronoi_vol'].iloc[i],data['nn_3_voronoi_vol'].iloc[i],data['nn_4_voronoi_vol'].iloc[i]]))
#    nn_dist_std.append(np.std([data['nn_1_distance'].iloc[i],data['nn_2_distance'].iloc[i],data['nn_3_distance'].iloc[i],data['nn_4_distance'].iloc[i]])) 
#    nn_eng_std.append(np.std([data['nn_1_eng'].iloc[i],data['nn_2_eng'].iloc[i],data['nn_3_eng'].iloc[i],data['nn_4_eng'].iloc[i]]))
#    nn_cnp_std.append(np.std([data['nn_1_cnp'].iloc[i],data['nn_2_cnp'].iloc[i],data['nn_3_cnp'].iloc[i],data['nn_4_cnp'].iloc[i]]))
#    nn_si_bonds_std.append(np.std([data['nn_1_si-si_bonds'].iloc[i],data['nn_2_si-si_bonds'].iloc[i],data['nn_3_si-si_bonds'].iloc[i],data['nn_4_si-si_bonds'].iloc[i]])) 
#
#print('determining min...')
## determine min 
#for i in tqdm(range(len(data))): 
#    nn_vv_min.append(np.min([data['nn_1_voronoi_vol'].iloc[i],data['nn_2_voronoi_vol'].iloc[i],data['nn_3_voronoi_vol'].iloc[i],data['nn_4_voronoi_vol'].iloc[i]])) 
#    nn_dist_min.append(np.min([data['nn_1_distance'].iloc[i],data['nn_2_distance'].iloc[i],data['nn_3_distance'].iloc[i],data['nn_4_distance'].iloc[i]])) 
#    nn_eng_min.append(np.min([data['nn_1_eng'].iloc[i],data['nn_2_eng'].iloc[i],data['nn_3_eng'].iloc[i],data['nn_4_eng'].iloc[i]]))
#    nn_cnp_min.append(np.min([data['nn_1_cnp'].iloc[i],data['nn_2_cnp'].iloc[i],data['nn_3_cnp'].iloc[i],data['nn_4_cnp'].iloc[i]]))
#    nn_si_bonds_min.append(np.min([data['nn_1_si-si_bonds'].iloc[i],data['nn_2_si-si_bonds'].iloc[i],data['nn_3_si-si_bonds'].iloc[i],data['nn_4_si-si_bonds'].iloc[i]])) 
#
#print('determining max...')
## determine max 
#for i in tqdm(range(len(data))): 
#    nn_vv_max.append(np.max([data['nn_1_voronoi_vol'].iloc[i],data['nn_2_voronoi_vol'].iloc[i],data['nn_3_voronoi_vol'].iloc[i],data['nn_4_voronoi_vol'].iloc[i]])) 
#    nn_dist_max.append(np.max([data['nn_1_distance'].iloc[i],data['nn_2_distance'].iloc[i],data['nn_3_distance'].iloc[i],data['nn_4_distance'].iloc[i]])) 
#    nn_eng_max.append(np.max([data['nn_1_eng'].iloc[i],data['nn_2_eng'].iloc[i],data['nn_3_eng'].iloc[i],data['nn_4_eng'].iloc[i]]))
#    nn_cnp_max.append(np.max([data['nn_1_cnp'].iloc[i],data['nn_2_cnp'].iloc[i],data['nn_3_cnp'].iloc[i],data['nn_4_cnp'].iloc[i]]))
#    nn_si_bonds_max.append(np.max([data['nn_1_si-si_bonds'].iloc[i],data['nn_2_si-si_bonds'].iloc[i],data['nn_3_si-si_bonds'].iloc[i],data['nn_4_si-si_bonds'].iloc[i]])) 
#
#data['nn_vv_std']      =nn_vv_std      
#data['nn_dist_std']    =nn_dist_std    
#data['nn_eng_std']     =nn_eng_std     
#data['nn_cnp_std']     =nn_cnp_std      
#data['nn_si_bonds_std']=nn_si_bonds_std
#                        
#data['nn_vv_min']      =nn_vv_min      
#data['nn_dist_min']    =nn_dist_min    
#data['nn_eng_min']     =nn_eng_min     
#data['nn_cnp_min']     =nn_cnp_min      
#data['nn_si_bonds_min']=nn_si_bonds_min
#                        
#data['nn_vv_max']      =nn_vv_max      
#data['nn_dist_max']    =nn_dist_max    
#data['nn_eng_max']     =nn_eng_max     
#data['nn_cnp_max']     =nn_cnp_max      
#data['nn_si_bonds_max']=nn_si_bonds_max

#data=data.loc[data['deltaE']<0.4]
#data=data.loc[data['deltaE']>-0.2]


#if args.insert is True: 
#    high=data.loc[data['deltaE']>0.1]
#    low=data.loc[data['deltaE']<-0.1]
#if args.remove is True: 
#    high=data.loc[data['deltaE']>0.1]
#    low=data.loc[data['deltaE']<-0.1]
##
##
#length=len(high)+len(low)
#print('total dataset size')
#print(length)

low_bias=data.loc[data['deltaE']<0.0]
#low_bias=low_bias.loc[low_bias['deltaE']>-0.1]

#low_bias=low_bias.sample(n=int(1e4))
#print('low bias, (0.1 < 0.0 < -0.1) size')
#print(len(low_bias))
#if len(low_bias) > int(1e5): 
#    low_bias=low_bias.sample(n=int(1e5),random_state=1)
#
#data=high
#data=data.append(low)
##data=data.sample(n=len(low_bias))
data=data.append(low_bias)
print('final dataset size')
print(len(data))

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

if args.step_weights is True: 
    combined['weights'] = data['weights']
if args.exp_weights is True: 
    combined['weights'] = data['weights']
combined=combined.replace([np.inf, -np.inf], np.nan)
combined=combined.replace(['inf', '-inf'], np.nan)
combined=combined.dropna()#fillna(0.0)
combined.to_csv('sampled_data.csv')
Y=combined['deltaE']
X=combined.drop(labels='deltaE',axis=1)

#X=StandardScaler().fit_transform(X)
# make training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=1)

print('training set size: %s' %len(X_train))

n_estimators=1000
print('training...')
forest = ExtraTreesRegressor(n_estimators=n_estimators,random_state=1,n_jobs=-1)
if args.step_weights is True: 
    forest.fit(X=X_train[features],y=y_train,sample_weight=X_train['weights'])
if args.exp_weights is True: 
    forest.fit(X=X_train[features],y=y_train,sample_weight=X_train['weights'])
forest.fit(X=X_train,y=y_train)#,sample_weight=data['weights'])
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

print('writing pickle file...')
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

ranked_features=[]
for f in range(X.shape[1]):
    ranked_features.append(features[indices[f]])
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

print('making plots...')
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]),ranked_features,fontsize=8,rotation=80)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig('random_forest_feature_importance.png')

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

#print('validate model with %s dataset' %args.file2)
#data=pd.read_csv(args.file2)
#
#actual=data['deltaE']
#predicted=forest.predict(data[features])
#
#df=pd.DataFrame()
#df['true']=actual
#df['predicted']=predicted
#df.to_csv('forest_model_vs_endpoint_validate_'+name+'.csv')
#
#name=args.data
#name=name.split('.')[0]
#name=name.split('/')
#name=name[len(name)-1]
#            
#r2_score_validate=r2_score(actual,predicted)
#mse_score_validate=mean_squared_error(actual,predicted)
#mae_score_validate=mean_absolute_error(actual,predicted)
#rmse_score_validate=np.sqrt(mse_score_validate)
#            
#f=open('validation_score_'+name+'.txt',mode='w')
#f.write('Validation:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f' %(r2_score_validate,mse_score_validate,rmse_score_validate,mae_score_validate)) 
#f.close() 
#
#plt.figure()
#plt.title('Histogram Forest Validation')
#plt.hist2d(x=actual,y=predicted,bins=100,norm=colors.LogNorm())
#plt.axis([-0.2,0.4,-0.2,0.4])#[np.min(actual),np.max(actual),np.min(actual),np.max(actual)])
#plt.colorbar() 
#plt.xlabel('Reported Energy (eV/$\AA^{2}$)')
#plt.ylabel('Predicted Energy (eV/$\AA^{2}$)')
#plt.tight_layout()
#plt.savefig('forest_histogram_validation_'+name+'.png')
#
#print('completely done!')
