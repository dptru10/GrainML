#!/usr/bin/env python3 
import os
import glob 
import time
import pymatgen
import argparse
import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('Agg')
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from pymatgen.core.structure import Structure
from sklearn.model_selection import train_test_split
#from pymatgen.analysis.bond_valence import BVAnalyzer 
from sklearn.model_selection import KFold, GridSearchCV
from matminer.featurizers.structure import CoulombMatrix
from matminer.featurizers.structure import OrbitalFieldMatrix
from sklearn.metrics import mean_squared_error, r2_score 

parser= argparse.ArgumentParser()
parser.add_argument("file",  help="file to be read in", type=str)
parser.add_argument("--debug",  action='store_true', default=False)
parser.add_argument("--cm",  action='store_true', default=False)
parser.add_argument("--ofm", action='store_true', default=False)
args  = parser.parse_args() 

cwd = os.getcwd()

csv=pd.read_csv(args.file)
flist=csv['Path']


file_list=[]
i=0
for file in flist:
    #print(file)
    file_list.append("%s/POSCAR/%s.POSCAR" %(cwd,file))
    i+=1

if args.debug == True:
    small_list=[]
    debug_lim=50
    for i in range(debug_lim):
        print(file_list[i])
        small_list.append(file_list[i])
    file_list=small_list


def read_vasp_data(id):
        print("Now reading system with id:", id)
        structure = pymatgen.Structure.from_file(id) 
        print("structure")
        print(structure) 
        return structure

structures = [ read_vasp_data(id) for id in file_list ]
ids = [ id for id in file_list ]
data = {'structures': structures, 'ids' : ids }
df = pd.DataFrame.from_dict(data)

if args.cm==True: 
    start = time.monotonic()
    cm = CoulombMatrix()
    cm.set_n_jobs(28)
    df  = cm.featurize_dataframe(df, 'structures')
    df['coulomb matrix'] = pd.Series([s.flatten() for s in df['coulomb matrix']], df.index)
    finish = time.monotonic()
    
    print("TIME TO FEATURIZE CM %f SECONDS" % (finish-start))
    print()

    dfdumb=pd.DataFrame()
    i=0
    for item in df['coulomb matrix']: 
        name = file_list[i]#"item %i" %i
        print(name) 
        dfdumb[name]=pd.Series(item) 
        i+=1 
    dfdumb.to_csv("cm.csv",mode='w')

if args.ofm==True:
    start = time.monotonic()
    ofm = OrbitalFieldMatrix()
    ofm.set_n_jobs(28)
    df  = ofm.featurize_dataframe(df, 'structures',ignore_errors=True)
    #df['orbital field matrix'] = pd.Series([s.flatten() for s in df['orbital field matrix']], df.index)
    finish = time.monotonic()
    
    #dfdumb=pd.DataFrame()
    #i=0
    #for item in df['orbital field matrix']: 
    #    name = file_list[i]#"item %i" %i
    #    print(name) 
    #    dfdumb[name]=pd.Series(item) 
    #    i+=1 
    #dfdumb.to_csv("ofm.csv",mode='w')
    
    
    print("TIME TO FEATURIZE OFM %f SECONDS" % (finish-start))
    print()

tuned_parameters = [{'kernel':['rbf'],'coef0':[1,10,100],'gamma': [1e-6,1e-5,1e-4, 1e-3, 1e-2, 1e2, 1e3, 1e4],'alpha': [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]}]
scores = ['neg_mean_absolute_error']

data_y=pd.read_csv(args.file)
Y=data_y['total energy']#.as_matrix()


if args.debug == True: 
    cleaned=[]
    for name in file_list: 
        first=name.split(".POSCAR")[0]
        cleaned.append(first.split(str(os.getcwd())+'/POSCAR/')[1])

    data_y=pd.read_csv(args.file)
    data_y=data_y.set_index('Path')

    print('data_y')
    print(data_y)
    dfbug=[]
    dfnew=pd.DataFrame()
    for item in cleaned:
        print(item)
        dfnew=data_y.loc[item]
        dfbug.append(dfnew['total energy'])
    print('dfbug')
    print(dfbug)
    Y=dfbug

#for item in cleaned: 
#    Y_cleaned.append(Y.loc[cleaned[i]]) 
#Y = Y_cleaned

if args.ofm==True:
    X = df['orbital field matrix']#.as_matrix()
    if args.cm==True:
        X.append(df['coulomb matrix'])

if args.cm==True:
    X = df['coulomb matrix']#.as_matrix()
    if args.ofm==True:
        X.append(df['coulomb matrix'])

# make training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

X_train_list=[]
for element in X_train: 
    X_train_list.append(len(element))
max_len=np.max(X_train_list)
print('max_len')
print(max_len)

X_test_list=[]
for element in X_test: 
    X_test_list.append(element.shape)
X_test_max_len=np.max(X_test_list)

if X_test_max_len > max_len: 
    max_len = X_test_max_len 

i=0
X_train_pad=[]
for element in X_train: 
    X_train_pad_width=max_len-len(element)
    X_train_pad.append(np.pad(element,(0,X_train_pad_width), 'constant'))
    print(len(X_train_pad[i]))
    i+=1 
X_train = X_train_pad 


#print('max_len')
#print(max_len)

i=0
X_test_pad=[]
for element in X_test: 
    X_test_pad_width=max_len-len(element)
    X_test_pad.append(np.pad(element,(0,X_test_pad_width), 'constant'))
    print(len(X_test_pad[i]))
    i+=1 
X_test = X_test_pad 

for item in X_test: 
    print(item)

for score in scores:
#    print(kernel[i])
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(KernelRidge(), tuned_parameters, cv=10, verbose=10,n_jobs=-1, scoring='%s' % score)
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
    rmse_score_train=np.sqrt(mse_score_train)
    r2_score_test=r2_score(y_test,model_test)
    mse_score_test=mean_squared_error(y_test,model_test)
    rmse_score_test=np.sqrt(mse_score_test)

    print("Endpoint:")
    print(np.transpose(np.array(y_train)))
    
    print("Model Train:")
    print(model_train)

    print("Model Train:")
    print(model_test)


    df1=pd.DataFrame()
    df2=pd.DataFrame()
    df3=pd.DataFrame()
    df4=pd.DataFrame()
	
    df1['model_train']=pd.Series(model_train)
    df2['true_train'] =pd.Series(y_train)
    df3['model_test']=pd.Series(model_test)
    df4['true_test'] =pd.Series(y_test)
    df1.to_csv('krr_model_train.csv',mode='w')
    df2.to_csv('krr_train_set.csv',mode='w')
    df3.to_csv('krr_model_test.csv',mode='w')
    df4.to_csv('krr_test_set.csv',mode='w')
    print('Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,r2_score_test,mse_score_test,rmse_score_test))

    name=args.file
    #name=name.split('/')[1]
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
    f.write(args.file)
    f.write('\n\n')

    #f.write('model_train,actual_train\n')
    #for i in range(len(model_train)):
    #    f.write('%3.2f,%3.2f\n' %(model_train[i],y_train[i]))
    #f.write('\n\n')

    #f.write('model_test,actual_test\n')
    #for i in range(len(model_test)):
    #    f.write('%3.6f,%3.6f\n' %(model_test[i],y_test[i]))
    f.close() 

    df1=pd.DataFrame((np.array(y_train)))
    df2=pd.DataFrame(np.transpose(np.array(model_train)))
    
    df1.to_csv('krr_model_vs_endpoint_'+name+'.csv')
    df2.to_csv('krr_model_vs_endpoint_'+name+'.csv',mode='a')

    #plot figures
    model='KRR'
    plt.figure() 
    plt.title('ML Performed via %s on Training Set'%model)
    plt.text(np.max(y_train)-2.2*np.std(y_train),np.min(y_train)+1.2*(np.std(y_train)),'Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,r2_score_test,mse_score_test,rmse_score_test), style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10},fontsize=10) 
    plt.scatter(x=y_train,y=model_train,c='green',marker='o',label='train')
    plt.plot(x=y_train,y=y_train)
    plt.scatter(x=y_test,y=model_test,c='red',marker='o',label='test')
    plt.legend(loc='upper left')
    plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
    plt.xlabel('True')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig('krr_train_test_'+name+'.png')
