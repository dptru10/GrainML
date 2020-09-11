import argparse 
import matplotlib 
import numpy as np  
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import colors
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 

def mae_acc(y_true,y_pred): 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.abs((y_true - y_pred)).mean() + accuracy_loss(y_true,y_pred) 


parser= argparse.ArgumentParser()
parser.add_argument("data", help="file to be read in", type=str)
parser.add_argument("model", help="file to be read in", type=str)
parser.add_argument("--classification",action="store_true")
parser.add_argument("--regression",action="store_true")
parser.add_argument("--negative",action="store_true")
parser.add_argument("--positive",action="store_true")
parser.add_argument("--insert",action="store_true")
parser.add_argument("--remove",action="store_true")
args  = parser.parse_args() 

data=pd.read_csv(args.data)

name=args.data
name=name.split('/')
name=name[len(name)-1]
name=name.split('.')[0]


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

print('generating averaged descriptors...')
data['nn_vv_average']=data['nn_1_voronoi_vol'] + data['nn_2_voronoi_vol'] + data['nn_3_voronoi_vol'] + data['nn_4_voronoi_vol'] 
data['nn_vv_average']=data['nn_vv_average'].divide(4)#len(data['nn_vv_average']))
data['nn_dist_average']=data['nn_1_distance'] + data['nn_2_distance'] + data['nn_3_distance'] + data['nn_4_distance']
data['nn_dist_average']=data['nn_dist_average'].divide(4)#len(data['nn_dist_average']))
data['nn_eng_average']=data['nn_1_eng'] + data['nn_2_eng'] + data['nn_3_eng'] + data['nn_4_eng']
data['nn_eng_average']=data['nn_eng_average'].divide(4)#len(data['nn_eng_average']))
data['nn_cnp_average']=data['nn_1_cnp'] + data['nn_2_cnp'] + data['nn_3_cnp'] + data['nn_4_cnp']
data['nn_cnp_average']=data['nn_cnp_average'].divide(4)#len(data['nn_cnp_average']))
data['nn_si_bonds_average']=data['nn_1_si-si_bonds'] + data['nn_2_si-si_bonds'] + data['nn_3_si-si_bonds'] + data['nn_4_si-si_bonds']
data['nn_si_bonds_average']=data['nn_si_bonds_average'].divide(4)#len(data['nn_cnp_average']))

nn_vv_std      =[]
nn_dist_std    =[]
nn_eng_std     =[]
nn_cnp_std     =[]
nn_si_bonds_std=[]

nn_vv_min      =[]
nn_dist_min    =[]
nn_eng_min     =[]
nn_cnp_min     =[]
nn_si_bonds_min=[]

nn_vv_max      =[]
nn_dist_max    =[]
nn_eng_max     =[]
nn_cnp_max     =[]
nn_si_bonds_max=[]

print('calculating statistical descriptors...')
print('determining std...')
# determine variance 
for i in tqdm(range(len(data))):
    nn_vv_std.append(np.std([data['nn_1_voronoi_vol'].iloc[i],data['nn_2_voronoi_vol'].iloc[i],data['nn_3_voronoi_vol'].iloc[i],data['nn_4_voronoi_vol'].iloc[i]]))
    nn_dist_std.append(np.std([data['nn_1_distance'].iloc[i],data['nn_2_distance'].iloc[i],data['nn_3_distance'].iloc[i],data['nn_4_distance'].iloc[i]])) 
    nn_eng_std.append(np.std([data['nn_1_eng'].iloc[i],data['nn_2_eng'].iloc[i],data['nn_3_eng'].iloc[i],data['nn_4_eng'].iloc[i]]))
    nn_cnp_std.append(np.std([data['nn_1_cnp'].iloc[i],data['nn_2_cnp'].iloc[i],data['nn_3_cnp'].iloc[i],data['nn_4_cnp'].iloc[i]]))
    nn_si_bonds_std.append(np.std([data['nn_1_si-si_bonds'].iloc[i],data['nn_2_si-si_bonds'].iloc[i],data['nn_3_si-si_bonds'].iloc[i],data['nn_4_si-si_bonds'].iloc[i]])) 

print('determining min...')
# determine min 
for i in tqdm(range(len(data))): 
    nn_vv_min.append(np.min([data['nn_1_voronoi_vol'].iloc[i],data['nn_2_voronoi_vol'].iloc[i],data['nn_3_voronoi_vol'].iloc[i],data['nn_4_voronoi_vol'].iloc[i]])) 
    nn_dist_min.append(np.min([data['nn_1_distance'].iloc[i],data['nn_2_distance'].iloc[i],data['nn_3_distance'].iloc[i],data['nn_4_distance'].iloc[i]])) 
    nn_eng_min.append(np.min([data['nn_1_eng'].iloc[i],data['nn_2_eng'].iloc[i],data['nn_3_eng'].iloc[i],data['nn_4_eng'].iloc[i]]))
    nn_cnp_min.append(np.min([data['nn_1_cnp'].iloc[i],data['nn_2_cnp'].iloc[i],data['nn_3_cnp'].iloc[i],data['nn_4_cnp'].iloc[i]]))
    nn_si_bonds_min.append(np.min([data['nn_1_si-si_bonds'].iloc[i],data['nn_2_si-si_bonds'].iloc[i],data['nn_3_si-si_bonds'].iloc[i],data['nn_4_si-si_bonds'].iloc[i]])) 

print('determining max...')
# determine max 
for i in tqdm(range(len(data))): 
    nn_vv_max.append(np.max([data['nn_1_voronoi_vol'].iloc[i],data['nn_2_voronoi_vol'].iloc[i],data['nn_3_voronoi_vol'].iloc[i],data['nn_4_voronoi_vol'].iloc[i]])) 
    nn_dist_max.append(np.max([data['nn_1_distance'].iloc[i],data['nn_2_distance'].iloc[i],data['nn_3_distance'].iloc[i],data['nn_4_distance'].iloc[i]])) 
    nn_eng_max.append(np.max([data['nn_1_eng'].iloc[i],data['nn_2_eng'].iloc[i],data['nn_3_eng'].iloc[i],data['nn_4_eng'].iloc[i]]))
    nn_cnp_max.append(np.max([data['nn_1_cnp'].iloc[i],data['nn_2_cnp'].iloc[i],data['nn_3_cnp'].iloc[i],data['nn_4_cnp'].iloc[i]]))
    nn_si_bonds_max.append(np.max([data['nn_1_si-si_bonds'].iloc[i],data['nn_2_si-si_bonds'].iloc[i],data['nn_3_si-si_bonds'].iloc[i],data['nn_4_si-si_bonds'].iloc[i]])) 

data['nn_vv_std']      =nn_vv_std      
data['nn_dist_std']    =nn_dist_std    
data['nn_eng_std']     =nn_eng_std     
data['nn_cnp_std']     =nn_cnp_std      
data['nn_si_bonds_std']=nn_si_bonds_std
                        
data['nn_vv_min']      =nn_vv_min      
data['nn_dist_min']    =nn_dist_min    
data['nn_eng_min']     =nn_eng_min     
data['nn_cnp_min']     =nn_cnp_min      
data['nn_si_bonds_min']=nn_si_bonds_min
                        
data['nn_vv_max']      =nn_vv_max      
data['nn_dist_max']    =nn_dist_max    
data['nn_eng_max']     =nn_eng_max     
data['nn_cnp_max']     =nn_cnp_max      
data['nn_si_bonds_max']=nn_si_bonds_max


if args.positive is True: 
    data=data.loc[data['deltaE']>0.0]

if args.negative is True: 
    data=data.loc[data['deltaE']<0.0]

combined=pd.DataFrame()
for item in features:
    combined[item]=data[item]
combined['deltaE']=data['deltaE']

combined=combined.replace([np.inf, -np.inf], np.nan)
combined=combined.replace(['inf', '-inf'], np.nan)
combined=combined.dropna()
data=combined
#data=data.sample(n=int(2e4),random_state=1)

print('loading model...')
model=load(args.model)

print('predicting features....')
print('done!') 

actual=data['deltaE']
predicted=model.predict(data[features])

df=pd.DataFrame()
df['true']=actual
df['predicted']=pd.Series(predicted)

binary_true=[]
binary_predicted=[]
for i in range(len(df)):
    if df['true'].iloc[i] < 0: 
        binary_true.append(-1)
    if df['true'].iloc[i] > 0: 
        binary_true.append(1)
    if df['predicted'].iloc[i] > 0: 
        binary_predicted.append(1)
    if df['predicted'].iloc[i] < 0:
        binary_predicted.append(-1)
df['binary_true']=pd.Series(binary_true)
df['binary_predicted']=pd.Series(binary_predicted)


df.to_csv('forest_model_vs_endpoint_validate_'+name+'.csv')

if args.classification is True:
    actual=df['binary_true']
    predicted=df['binary_predicted']
    from sklearn.metrics import accuracy_score, plot_confusion_matrix
    accuracy=accuracy_score(actual,predicted)
    
    print('determining confusion matrix...')
    
    size=20
    classes=['negative','positive']
    #confusion=confusion_matrix(y_train,model_train)
    plt.figure()
    confusion=plot_confusion_matrix(model,data[features],predicted,display_labels=classes,cmap="coolwarm",normalize='true')
    plt.savefig('confusion_matrix_%s.png' %name)

if args.regression is True:             
    r2_score_validate=r2_score(actual,predicted)
    mse_score_validate=mean_squared_error(actual,predicted)
    mae_score_validate=mean_absolute_error(actual,predicted)
    rmse_score_validate=np.sqrt(mse_score_validate)
                
    f=open('validation_score_'+name+'.txt',mode='w')
    f.write('Validation:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f' %(r2_score_validate,mse_score_validate,rmse_score_validate,mae_score_validate)) 
    f.close() 
    
    plt.figure()
    plt.title('Histogram Forest Validation')
    plt.hist2d(x=actual,y=predicted,bins=100,norm=colors.LogNorm())
    plt.axis([np.min(actual),np.max(actual),np.min(actual),np.max(actual)])
    plt.colorbar() 
    plt.xlabel('Reported Energy (eV/$\AA^{2}$)')
    plt.ylabel('Predicted Energy (eV/$\AA^{2}$)')
    plt.tight_layout()
    plt.savefig('forest_histogram_validation_'+name+'.png')
