import numpy as np  

def accuracy_loss(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    sum_error=0
    for i in range(len(y_true)):
        if y_true[i] < 0.0:
            binary_true = 0
        else: 
            binary_true = 1

        if y_pred[i] < 0.0:
            binary_pred = 0
        else: 
            binary_pred = 1
        if binary_true != binary_pred: 
            temp =  np.exp(-y_true[i])*np.exp(np.abs(y_true[i] - y_pred[i]))  
        else: 
            temp =  np.exp(-y_true[i])#0
        sum_error+=temp 
    return sum_error  

def accuracy_binary_loss(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    sum_error=0
    for i in range(len(y_true)):
        if y_true[i] < 0.0:
            binary_true = 0
        else: 
            binary_true = 1

        if y_pred[i] < 0.0:
            binary_pred = 0
        else: 
            binary_pred = 1
        if binary_true != binary_pred: 
            temp =  np.abs(y_true[i] - y_pred[i])
        else: 
            temp =  -(y_true[i]/100)
        sum_error+=temp 
    return sum_error  


def accuracy_binary_loss_orig(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    binary_true=[]
    binary_pred=[]
    for i in range(len(y_true)):
        if y_true[i] < 0.0:
            binary_true.append(0)
        else: 
            binary_true.append(1)
        if y_pred[i] < 0.0:
            binary_pred.append(0)
        else: 
            binary_pred.append(1)
        return accuracy_score(binary_true,binary_pred)

def mape(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.abs((y_true - y_pred)/y_true).mean()

def mae_acc(y_true,y_pred): 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.abs((y_true - y_pred)).mean() + accuracy_loss(y_true,y_pred) 

