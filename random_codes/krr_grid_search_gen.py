#!/usr/bin/env python 
from __future__ import print_function

import pickle
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np  
from joblib import dump, load
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV 
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.kernel_ridge import KernelRidge
import pandas as pd 
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
args  = parser.parse_args() 

data=pd.read_csv(args.file1)

#features=['cnp','centro','vor_vol','vor_neigh']
#features=['Misorientation', 'Sigma', 'Stoichiometry']
features=['deltaE','op_eng','op_cnp','op_vornoi_vol','op_si-si_bonds','op_si-c_bonds','op_c-c_bonds','nn_1_distance','nn_1_eng','nn_1_cnp','nn_1_vornoi_vol','nn_1_si-si_bonds','nn_1_si-c_bonds','nn_2_distance','nn_2_eng','nn_2_cnp','nn_2_vornoi_vol','nn_2_si-si_bonds','nn_2_si-c_bonds','nn_2_c-c_bonds','nn_3_distance','nn_3_eng','nn_3_cnp','nn_3_vornoi_vol','nn_3_si-si_bonds','nn_3_si-c_bonds','nn_3_c-c_bonds','nn_4_distance','nn_4_eng','nn_4_cnp','nn_4_vornoi_vol','nn_4_si-si_bonds','nn_4_si-c_bonds']
#endpoint='total energy'
endpoint='deltaE'

X=data[features]

#print(X)

holder=[]
for feature in features: 
    for x in X[feature]:
        if np.isnan(x) == True: 
            x = 0 
        holder.append(x) 
    X[feature]=holder
    holder=[]

Y=data[endpoint]
for y in Y: 
    if np.isnan(y) == True:
        y = 0
    holder.append(y)
Y=holder 

# make training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=0)

#print(X_test) 
#print(y_test)


# Set the parameters by cross-validation
tuned_parameters = {'kernel':['rbf'],'coef0':[1,10,100],'gamma': [1e-6,1e-5,1e-4, 1e-3, 1e-2, 1e2, 1e3, 1e4],'alpha': [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]}#,0,1,1e1,1e2,1e3]}
scores = ['neg_mean_absolute_error']

for score in scores:
#    print(kernel[i])
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
    rmse_score_train=np.sqrt(mse_score_train)
    r2_score_test=r2_score(y_test,model_test)
    mse_score_test=mean_squared_error(y_test,model_test)
    rmse_score_test=np.sqrt(mse_score_test)


    name=args.file1
    name=name.split('.')[0]

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

    f.write('model_train,actual_train\n')
    for i in range(len(model_train)):
        f.write('%3.2f,%3.2f\n' %(model_train[i],y_train[i]))
    f.write('\n\n')

    f.write('model_test,actual_test\n')
    for i in range(len(model_test)):
        f.write('%3.6f,%3.6f\n' %(model_test[i],y_test[i]))
    f.close() 

    print("Endpoint:")
    print(np.transpose(np.array(y_train)))
    
    print("Model:")
    print(model_train)
    df1=pd.DataFrame((np.array(y_train)))
    df2=pd.DataFrame(np.transpose(np.array(model_train)))
    
    df1.to_csv('krr_model_vs_endpoint_'+name+'.csv')
    df2.to_csv('krr_model_vs_endpoint_'+name+'.csv',mode='a')

    #plot figures
    plt.figure() 
    #plt.title('ML Performed via MLP on Training Set')
    plt.text(np.max(y_train)-1.3*np.std(y_train),np.min(y_train)+0.5*(np.std(y_train)),'Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,r2_score_test,mse_score_test,rmse_score_test), style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10},fontsize=10) 
    plt.plot(y_train,model_train,'gs')
    #plt.plot(y_train,y_train,'k-')
    plt.plot(y_test,model_test,'ro')
    #plt.plot(y_test,y_test,'k-')
    plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
    plt.xlabel('True')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig('krr_train_test_'+name+'.png')
