#!/usr/bin/env python 
from __future__ import print_function

import argparse 
import matplotlib 
import numpy as np  
import pandas as pd 
import seaborn as sns 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from joblib import dump, load
from sklearn.cluster import KMeans 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 


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


endpoint='deltaE'
X = data[features]
Y = data[endpoint]
Y = np.array(Y) 
Y = Y.reshape(-1,1) 

# apply kmeans-silhouette
summed_square_distance=[]
clusters=3
model = KMeans(n_clusters=clusters, random_state=0)
model.fit(Y)#data_transform)
X['labels'] = model.labels_ 
X=X.dropna() 
X.to_csv('classes_data.csv')

big_X = X #pd.DataFrame(columns=features)
big_X['deltaE'] = Y 
for label in set(X['labels']):
    X_new = X.loc[X['labels'] == label]
    if len(X_new) > 1: 
        isolated_forest=IsolationForest(n_estimators=100,max_samples='auto',behaviour='new',contamination='auto',max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
        
        isolated_forest.fit(X_new[features])
        predicted=isolated_forest.predict(X_new[features]) 
        
        X_new['anomaly']=predicted
        outliers=X_new.loc[X_new['anomaly']==-1]
        
        print('Number of anomalies in cluster %i' %label)
        print(X_new['anomaly'].value_counts())
    else: 
        print("cluster %i has only 1 element, this cluster is an anomaly..." %label)
        X_new['anomaly'] = -1 
    big_X=big_X.append(X_new) 
big_X.to_csv('anomaly.csv')
big_X = big_X.loc[big_X['anomaly']==1]

ml_switch = True 
if ml_switch is True:
	cv_val = 5 
	for label in set(X['labels']):
		print("training model for cluster %i" %label)
		data = big_X.loc[big_X['labels'] == label]

		if len(data) > cv_val:
			data=data.dropna()
			data.to_csv('data.csv')
			
			X = data[features]
			Y = data[endpoint]

			# make training and test set
			X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=1)
			
			tuned_parameters = [{'n_estimators':[100,200,500,1000]}]
			scores = ['neg_mean_absolute_error']
			for score in scores:
			   forest = GridSearchCV(ExtraTreesRegressor(),tuned_parameters,verbose=10,cv=cv_val,n_jobs=-1,scoring='%s' %score)
			   
			   forest.fit(X_train, y_train)
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
			
			   dump(forest,'%s_cluster_%i.pkl' %(endpoint,label))
			   
			   f=open('%s_hyperpameters_cluster_%i.txt' %(endpoint,label),mode='w')
			   f.write("Best parameters set found on development set:")
			   f.write('\n\n')
			   f.write(str(forest.best_params_))
			   f.write('\n\n')
			   f.write('Score:')
			   f.write(str(-forest.best_score_))
			   f.write('\n\n')
			   
			   f.write('Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,mae_score_train,r2_score_test,mse_score_test,rmse_score_test,mae_score_test)) 
			   f.close() 
			   
			   #df1=pd.DataFrame()
			   #df1['train']=y_train
			   #df1['train_model']=model_train
			   #df1.append(X_train)
			   #df1.to_csv('forest_model_vs_endpoint_train.csv')
			   #df2=pd.DataFrame()
			   #df2['test']=y_test
			   #df2['test_model']=model_test
			   #df2.append(X_test)
			   #df2.to_csv('forest_model_vs_endpoint_test.csv')
			   #plot figures
			   plt.figure()
			   plt.title('Histogram forest Train')
			   plt.hist2d(x=y_train,y=model_train,bins=100,norm=colors.LogNorm())   
			   plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
			   plt.colorbar() 
			   plt.xlabel('Reported DeltaE')
			   plt.ylabel('Predicted DeltaE')
			   plt.tight_layout()
			   plt.savefig('%s_forest_histogram_train_cluster_%i.png' %(endpoint,label))
			   
			   plt.figure()
			   plt.title('Histogram forest Test')
			   plt.hist2d(x=y_test,y=model_test,bins=100,norm=colors.LogNorm())
			   plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
			   plt.colorbar() 
			   plt.xlabel('Reported DeltaE')
			   plt.ylabel('Predicted DeltaE')
			   plt.tight_layout()
			   plt.savefig('%s_forest_histogram_test_cluster_%i.png' %(endpoint,label))
	   
	   

