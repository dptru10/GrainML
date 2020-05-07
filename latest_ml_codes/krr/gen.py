#!/usr/bin/env python 
from __future__ import print_function
import os 
import sys
import pickle
import matplotlib 
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  
from matplotlib import colors
from joblib import dump, load
from sklearn import datasets
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.kernel_ridge import KernelRidge
import pandas as pd 
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
parser.add_argument("--outliers",action="store_true")
parser.add_argument("--insert",action="store_true")
parser.add_argument("--remove",action="store_true")
parser.add_argument("--corr",action="store_true")
parser.add_argument("--pca",action="store_true")
parser.add_argument("--distribution",action="store_true")
parser.add_argument("--xy",action="store_true")
parser.add_argument("--ml",action="store_true")
parser.add_argument("--top_n",action="store_true")
parser.add_argument("--random_n",action="store_true")
args  = parser.parse_args() 

print(args.file1)
data=pd.read_csv(args.file1)

if args.outliers:
	data=data.loc[data['anomaly']==1]

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

if args.distribution:
    features.append('deltaE')
    for feature in features:
        plt.figure()
        histogram=plt.hist(data[feature])
        plt.xlabel('%s' %feature)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('%s_orig_dist.png' %(feature))

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

name=args.file1
name=name.split('/')
name=name[len(name)-1]
name=name.split('.')[0]

if args.distribution is True: 
    for feature in features:
        plt.figure()
        histogram=plt.hist(data[feature])
        plt.xlabel('%s' %feature)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('%s_dist.png' %(feature))
#        sys.exit() 

endpoint='deltaE'

X=data[features]
Y=data[endpoint]
print('X.columns') 
print(X.columns) 

#new_data=pd.DataFrame()
#for item in features: 
#	new_data[item]=data[item]
#new_data[endpoint]=data[endpoint]
#new_data.to_csv('sampled_data.csv')

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

#X=normalize(X)


if args.xy is True: 
	for item in features: 
		plt.figure() 
		plt.scatter(data[item],data[endpoint])
		plt.xlabel('%s' %item)
		plt.ylabel('deltaE')
		plt.savefig('%s.png' %item) 

if args.corr is True:
	corr_features=features
	corr_features.append(endpoint)
	corr_dat = data[corr_features]
	
	size=20
	corr=corr_dat.corr()
	plt.figure()
	fig, ax = plt.subplots(figsize=(size, size))
	sns.set(font_scale=1.5)
	sns_plot=sns.heatmap(corr,cmap="coolwarm",square=True,annot=True, fmt=".1f",annot_kws={"size": 20})
	plt.yticks(rotation=0,fontsize=16,fontweight='bold')
	plt.xticks(rotation=90,fontsize=16,fontweight='bold')
	plt.tight_layout()
	plt.savefig('cross_corr_insert.png')# %name.split('.')[0])

if args.pca is True: 
	pca = PCA(n_components=len(features))
	pca.fit(X)
	print('pca')
	pca_ex_var=pca.explained_variance_ratio_
	

	plt_data = pca_ex_var
	labels=features
	y_pos = np.arange(len(labels))

	plt.figure()
	plt.bar(y_pos, plt_data, align='center', alpha=0.5)
	plt.xticks(y_pos, labels)
	plt.ylabel('Explained Variance Ratio')
	plt.xticks(rotation=90)
	plt.tight_layout() 
	
	plt.savefig('exp_var.png')
	
correlated_data=pd.DataFrame()
for item in features:
	print(item)
	print(np.corrcoef(data[item],data[endpoint])[0][1])
	if np.corrcoef(np.array(data[item]),np.array(data[endpoint]))[0][1] > 0 or np.corrcoef(np.array(data[item]),np.array(data[endpoint]))[0][1] != np.nan: 
		correlated_data[item]=data[item]
X=correlated_data 

print('X.columns')
print(X.columns)

holder=[]
for feature in features: 
    for x in X[feature]:
        if np.isnan(x) == True: 
            x = 0 
        holder.append(x) 
    X[feature]=holder
    holder=[]

for y in Y: 
    if np.isnan(y) == True:
        y = 0
    holder.append(y)
Y=holder 


if args.ml is True: 
	# make training and test set
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=0)
	#print(X_test) 
	#print(y_test)
        # Set the parameters by cross-validation
        tuned_parameters = {'kernel':['rbf','poly','laplacian'],'degree':[2,3,4],'gamma': [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,3e-1,5e-1,7e-1,9e-1,1],'alpha': [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,3e-1,5e-1,7e-1,9e-1]}#,1e2,1e3]}
        #tuned_parameters = {'kernel':['rbf'],'gamma': [1,3e-1],'alpha': [1e-1]}#,1e2,1e3]}
        #tuned_parameters = {'kernel':['rbf'],'gamma': [1e-6,1e-5,1e-4],'alpha': [1e-6,1e-5,1e-4]}#,1e2,1e3]}
        #tuned_parameters = {'kernel':['rbf'],'gamma': [0.01],'alpha': [0.01]}#,1e2,1e3]}
        scores = ['neg_mean_absolute_error']
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
            clf = GridSearchCV(KernelRidge(), tuned_parameters, cv=10, verbose=10, n_jobs=-1, scoring='%s' % score)
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
            df1.append(X_train)
            df1.to_csv('krr_model_vs_endpoint_train_'+name+'.csv')
            df2=pd.DataFrame()
            df2['test']=y_test
            df2['test_model']=model_test
            df2.append(X_test)
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
	
	
