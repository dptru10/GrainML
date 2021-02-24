#!/usr/bin/env python 
from __future__ import print_function

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np  
from sklearn import datasets
from sklearn.preprocessing import normalize 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from matplotlib import colors
from joblib import dump, load
from sklearn.svm import SVR
import pandas as pd 
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
parser.add_argument("--insert",action="store_true")
parser.add_argument("--remove",action="store_true")
parser.add_argument("--all",action="store_true")
parser.add_argument("--average",action="store_true")
parser.add_argument("--top_n",action="store_true")
parser.add_argument("--random_n",action="store_true")
args  = parser.parse_args() 

print(args.file1)
data=pd.read_csv(args.file1)

if args.average is True:
    features=['num_neigh','op_voronoi_vol','nn_si_bonds_average','nn_vv_average','nn_dist_average','nn_eng_average','nn_cnp_average']

if args.all is True:
    features=['num_neigh','op_voronoi_vol','nn_1_distance','nn_1_eng','nn_1_cnp',
            'nn_1_voronoi_vol','nn_1_si-si_bonds','nn_2_distance',
            'nn_2_eng','nn_2_cnp','nn_2_voronoi_vol','nn_2_si-si_bonds',
            'nn_3_distance','nn_3_eng','nn_3_cnp','nn_3_voronoi_vol',
            'nn_3_si-si_bonds','nn_4_distance','nn_4_eng','nn_4_cnp',
            'nn_4_voronoi_vol','nn_4_si-si_bonds']

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
endpoint='deltaE'

if args.insert is True: 
    data=data.loc[data['deltaE']<0.2]
    data=data.loc[data['deltaE']>-0.25]
if args.remove is True: 
    data=data.loc[data['deltaE']<0.45]
    data=data.loc[data['deltaE']>-0.15]

if args.top_n is True: 
    data=data.head(n=int(2e4))#.sample(n=int(2e4),random_state=1)
if args.random_n is True: 
    data=data.sample(n=int(2e4),random_state=1)

X=data[features]
Y=data[endpoint]

combined=pd.DataFrame()
for item in features:
    combined[item]=X[item]
combined['deltaE']=Y

combined=combined.replace([np.inf, -np.inf], np.nan)
combined=combined.replace(['inf', '-inf'], np.nan)
combined=combined.dropna()
combined.to_csv('sampled_data.csv')
Y=combined['deltaE']
X=combined.drop(labels='deltaE',axis=1)


print('X.columns') 
print(X.columns) 

# make training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=0)

kernel=['rbf','poly']

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['poly'], 'epsilon':[1e-2, 1e-1],'degree':[3],'C': [1, 5, 10],'gamma': [1e-4, 1e-3, 1e-2, 1e-1]}]#, 
                    #{'kernel': ['rbf'],'epsilon':[1e-4, 1e-3, 1e-2, 1e-1],'C': [1, 10, 100, 1000],'gamma': [1e-4, 1e-3, 1e-2, 1e2, 1e3, 1e4]}]
		    #{'kernel': ['linear'],'epsilon':[1e-4, 1e-3, 1e-2, 1e-1],'C': [1, 10, 100, 1000]},
		    #{'kernel': ['sigmoid'],'epsilon':[1e-4, 1e-3, 1e-2, 1e-1],'C': [1, 10, 100, 1000],'coef0':[1,10,100],'gamma': [1e-4, 1e-3, 1e-2, 1e2, 1e3, 1e4]},
		    #{'kernel': ['poly'], 'epsilon':[1e-4, 1e-3, 1e-2, 1e-1],'degree':[2,3,4,5],'C': [1, 10, 100, 1000],'gamma': [1e-4, 1e-3, 1e-2, 1e2, 1e3, 1e4]}]
scores = ['neg_mean_absolute_error']

for score in scores:
#    print(kernel[i])
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVR(), tuned_parameters, cv=5, n_jobs=-1, verbose=10,scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print('Score:')
    print(-clf.best_score_)
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
    #name=name[2]
    
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
    df1.to_csv('svm_model_vs_endpoint_train_'+name+'.csv')
    df2=pd.DataFrame()
    df2['test']=y_test
    df2['test_model']=model_test
    df2.to_csv('svm_model_vs_endpoint_test_'+name+'.csv')
    
    #plot figures
    plt.figure()
    plt.title('Histogram SVM Train')
    plt.hist2d(x=y_train,y=model_train,bins=100,norm=colors.LogNorm())   
    plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
    plt.colorbar() 
    plt.xlabel('Reported Energy (eV/$\AA^{2}$)')
    plt.ylabel('Predicted Energy (eV/$\AA^{2}$)')
    plt.tight_layout()
    plt.savefig('SVM_histogram_train_'+name+'.png')
    
    plt.figure()
    plt.title('Histogram SVM Test')
    plt.hist2d(x=y_test,y=model_test,bins=100,norm=colors.LogNorm())
    plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
    plt.colorbar() 
    plt.xlabel('Reported Energy (eV/$\AA^{2}$)')
    plt.ylabel('Predicted Energy (eV/$\AA^{2}$)')
    plt.tight_layout()
    plt.savefig('SVM_histogram_test_'+name+'.png')
    
    
