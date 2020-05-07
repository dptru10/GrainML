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
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
args  = parser.parse_args() 

print(args.file1)
data=pd.read_csv(args.file1)
data=data.loc[data['deltaE']<0.1]
data=data.loc[data['deltaE']>-0.1]
data=data.sample(n=int(1e4),random_state=1)

data['nn_vv_average']=data['nn_1_voronoi_vol'] + data['nn_2_voronoi_vol'] + data['nn_3_voronoi_vol'] + data['nn_4_voronoi_vol'] 
data['nn_vv_average']=data['nn_vv_average'].divide(4)#len(data['nn_vv_average']))
data['nn_dist_average']=data['nn_1_distance'] + data['nn_2_distance'] + data['nn_3_distance'] + data['nn_4_distance'] 
data['nn_dist_average']=data['nn_dist_average'].divide(4)#len(data['nn_dist_average']))
data['nn_eng_average']=data['nn_1_eng'] + data['nn_2_eng'] + data['nn_3_eng'] + data['nn_4_eng'] 
data['nn_eng_average']=data['nn_eng_average'].divide(4)#len(data['nn_eng_average']))
data['nn_cnp_average']=data['nn_1_cnp'] + data['nn_2_cnp'] + data['nn_3_cnp'] + data['nn_4_cnp'] 
data['nn_cnp_average']=data['nn_cnp_average'].divide(4)#len(data['nn_cnp_average']))
data['nn_si_bonds_average']=data['nn_1_si-si_bonds'] + data['nn_2_si-si_bonds'] + data['nn_3_si-si_bonds'] + data['nn_4_si-si_bonds'] 
data['nn_si_bonds_average']=data['nn_cnp_average'].divide(4)#len(data['nn_cnp_average']))


#features=['num_neigh','op_voronoi_vol','nn_si_bonds_average','nn_vv_average','nn_dist_average','nn_eng_average','nn_cnp_average']

features=['num_neigh','op_voronoi_vol','nn_1_distance','nn_1_eng','nn_1_cnp',
        'nn_1_voronoi_vol','num_neigh','nn_1_si-si_bonds','nn_2_distance',
        'nn_2_eng','nn_2_cnp','nn_2_voronoi_vol','nn_2_si-si_bonds',
        'nn_3_distance','nn_3_eng','nn_3_cnp','nn_3_voronoi_vol',
        'nn_3_si-si_bonds','nn_4_distance','nn_4_eng','nn_4_cnp',
        'nn_4_voronoi_vol','nn_4_si-si_bonds']
endpoint='deltaE'



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
        #print(np.corrcoef(df[item],Y)[0][1])
        #if np.corrcoef(np.array(X[item]),np.array(Y))[0][1] <= -0.3 or np.corrcoef(np.array(X[item]),np.array(Y))[0][1] >= 0.2: 
        #   correlated_data[item]=X[item]
#correlated_data.to_csv('correlated_data.csv')
f.close()
#X=correlated_data

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
#y_train_selected = selector.transform(y_train.fillna(0))
X_test  = selector.transform(X_test.fillna(0))
#y_test_selected  = selector.transform(y_test.fillna(0))

## Set the parameters by cross-validation
#tuned_parameters = {'alpha': [0.0,0.1]}#,0.2,0.3,0.4,0.5]}
tuned_parameters = {'kernel':['rbf'],'gamma': [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,3e-1,5e-1,7e-1,9e-1],'alpha': [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,3e-1,5e-1,7e-1,9e-1]}#,1e2,1e3]}
scores = ['neg_mean_absolute_error']
#
for score in scores:
#    print(kernel[i])
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = RandomizedSearchCV(KernelRidge(), tuned_parameters, cv=5, verbose=10, n_jobs=-1, scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(str(clf.best_params_))
    print()
    print('Score:')
    print(str(-clf.best_score_))
    print()



    #build models     
    model_test=clf.predict(X_test)
    model_train=clf.predict(X_train)
    r2_score_train=r2_score(y_train,model_train)
    mse_score_train=mean_squared_error(y_train,model_train)
    mae_score_train=mean_absolute_error(y_train,model_train)
    rmse_score_train=np.sqrt(mse_score_train)
    r2_score_test=r2_score(y_test,model_test)
    mse_score_test=mean_squared_error(y_test,model_test)
    mae_score_test=mean_absolute_error(y_test,model_test)
    rmse_score_test=np.sqrt(mse_score_test)


    name=args.file1
    name=name.split('/')
    name=name[len(name)-1]
    name=name.split('.')[0]
    #print(name)
#    name=name[2]

    #pickle model
    dump(clf,'%s.pkl' %name)

    f=open('hyperpameters_'+name+'.txt',mode='w')
    f.write("Best parameters set found on development set:")
    f.write('\n\n')
    f.write(str(clf.best_params_))
    f.write('\n\n')
    f.write('Score:')
    f.write(str(-clf.best_score_))
    f.write('\n\n')
    f.write(args.file1)
    f.write('\n\n')

    f.write('Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,mae_score_train,r2_score_test,mse_score_test,rmse_score_test,mae_score_test)) 
    f.close() 

    df1=pd.DataFrame()
    df1['train']=y_train
    df1['train_model']=model_train
    df1.to_csv('krr_model_vs_endpoint_train_'+name+'.csv')
    df2=pd.DataFrame()
    df2['test']=y_test
    df2['test_model']=model_test
    df2.to_csv('krr_model_vs_endpoint_test_'+name+'.csv')

    #plot figures
    plt.figure()
    plt.title('Histogram KRR Train')
    plt.hist2d(x=y_train,y=model_train,bins=100,norm=colors.LogNorm())   
    plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
    plt.colorbar() 
    plt.xlabel('Reported Energy (eV/$\AA^{2}$)')
    plt.ylabel('Predicted Energy (eV/$\AA^{2}$)')
    plt.tight_layout()
    plt.savefig('KRR_histogram_train_'+name+'.png')
    
    plt.figure()
    plt.title('Histogram KRR Test')
    plt.hist2d(x=y_test,y=model_test,bins=100,norm=colors.LogNorm())
    plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
    plt.colorbar() 
    plt.xlabel('Reported Energy (eV/$\AA^{2}$)')
    plt.ylabel('Predicted Energy (eV/$\AA^{2}$)')
    plt.tight_layout()
    plt.savefig('KRR_histogram_test_'+name+'.png')


