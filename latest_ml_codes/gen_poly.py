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
parser.add_argument("--corr",action="store_true")
parser.add_argument("--pca",action="store_true")
parser.add_argument("--xy",action="store_true")
parser.add_argument("--ml",action="store_true")
args  = parser.parse_args() 

print(args.file1)
data=pd.read_csv(args.file1)
data=data.loc[data['deltaE']<0.2]
data=data.loc[data['deltaE']>-0.2]
data=data.sample(n=int(1e4),random_state=1)

name=args.file1

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


#features=['op_voronoi_vol','nn_si_bonds_average','nn_vv_average','nn_dist_average','nn_eng_average','nn_cnp_average']
features=['op_voronoi_vol','nn_1_distance','nn_1_eng','nn_1_cnp',
        'nn_1_voronoi_vol','num_neigh','nn_1_si-si_bonds','nn_2_distance',
        'nn_2_eng','nn_2_cnp','nn_2_voronoi_vol','nn_2_si-si_bonds',
        'nn_3_distance','nn_3_eng','nn_3_cnp','nn_3_voronoi_vol',
        'nn_3_si-si_bonds','nn_4_distance','nn_4_eng','nn_4_cnp',
        'nn_4_voronoi_vol','nn_4_si-si_bonds']

#operations=[np.exp,np.sin]

#label_list=[]
#for obj in operations: 
#	i=0
#	for item in features:
#		label='%s_%s' %(item,str(obj).split('\'')[1])
#		label_list.append(label)
#		print(label) 
#		data[label] = data[item].apply(obj)
#		i+=1
#
#for item in label_list: 
#	features.append(item)

endpoint='deltaE'

new_data=pd.DataFrame()
for item in features: 
	new_data[item]=data[item]
new_data[endpoint]=data[endpoint]
new_data.to_csv('sampled_data.csv')

X=data[features]
print('X.columns') 
print(X.columns) 
Y=data[endpoint]


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
	#tuned_parameters = {'kernel':['rbf'],'gamma': [1e-6,1e-5,1e-4,1e-3,1e-2,0.1,0.3,0.5,0.7,1],'alpha': [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,2e-1,3e-1,4e-1,5e-1,6e-1,7e-1,9e-1]}#,1e2,1e3]}
	#tuned_parameters = {'kernel':['rbf'],'gamma': [1e-6,1e-5,1e-4],'alpha': [1e-6,1e-5,1e-4]}#,1e2,1e3]}
	tuned_parameters = {'kernel':['poly'],'degree':[2,3,4], 'gamma': [1e-6,1e-5,1e-4,1e-3,1e-2,0.1,0.3,0.5,0.7,1],'alpha': [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,2e-1,3e-1,4e-1,5e-1,6e-1,7e-1,9e-1]}#,1e2,1e3]}
	scores = ['neg_mean_absolute_error']
	
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
	
	
