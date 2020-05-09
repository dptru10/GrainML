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

parser= argparse.ArgumentParser()
parser.add_argument("data", help="file to be read in", type=str)
parser.add_argument("model", help="file to be read in", type=str)
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

combined=pd.DataFrame()
for item in features:
    combined[item]=data[item]
combined['deltaE']=data['deltaE']

combined=combined.replace([np.inf, -np.inf], np.nan)
combined=combined.replace(['inf', '-inf'], np.nan)
combined=combined.dropna()
data=combined

print('loading model...')
model=load(args.model)

print('predicting features....')
print('done!') 

actual=data['deltaE']
predicted=model.predict(data[features])

df=pd.DataFrame()
df['true']=actual
df['predicted']=pd.Series(predicted)
df.to_csv('forest_model_vs_endpoint_validate_'+name+'.csv')


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
