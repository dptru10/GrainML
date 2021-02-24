from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
import pandas as pd
import numpy as np
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("test",  help="file to be read in", type=str)
parser.add_argument("train", help="file to be read in", type=str) 
parser.add_argument("validate", help="file to be read in", type=str) 
args  = parser.parse_args() 

Test=pd.read_csv(args.test)
Train=pd.read_csv(args.train)
Validation=pd.read_csv(args.validate)

#test set 
Test=Test.loc[Test['test'] < 0]
y_test=Test['test']
model_test=Test['test_model']
r2_score_test=r2_score(y_test,model_test)
mse_score_test=mean_squared_error(y_test,model_test)
mae_score_test=mean_absolute_error(y_test,model_test)
rmse_score_test=np.sqrt(mse_score_test)

#training set 
Train=Train.loc[Train['train'] < 0]
y_train=Train['train']
model_train=Train['train_model']
r2_score_train=r2_score(y_train,model_train)
mse_score_train=mean_squared_error(y_train,model_train)
mae_score_train=mean_absolute_error(y_train,model_train)
rmse_score_train=np.sqrt(mse_score_train)

#validation set 
Validation=Validation.loc[Validation['true'] < 0]
y_validate=Validation['true']
model_validate=Validation['predicted']
r2_score_validate=r2_score(y_validate,model_validate)
mse_score_validate=mean_squared_error(y_validate,model_validate)
mae_score_validate=mean_absolute_error(y_validate,model_validate)
rmse_score_validate=np.sqrt(mse_score_train)


f=open('neg_hyperpameters.txt',mode='w')
f.write('Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f\n\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f\n\nValidate:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f' \
        %(r2_score_train,mse_score_train,rmse_score_train,mae_score_train, 
        r2_score_test,mse_score_test,rmse_score_test,mae_score_test,
        r2_score_validate,mse_score_validate,rmse_score_validate,mae_score_validate)) 
f.close() 

