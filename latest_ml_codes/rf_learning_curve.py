#!/usr/bin/env python 
print(__doc__)

import matplotlib 
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import pandas as pd 
import argparse 


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("RMSE")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    #plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                 train_scores_mean + train_scores_std, alpha=0.1,
    #                 color="r")
    #plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

    print("Training Set Size:")
    print(np.transpose(np.array(train_sizes)))
    
    print("RMSE Training Error:")
    print(np.sqrt(-train_scores_mean))

    print("Validation Error:")
    print(np.sqrt(-test_scores_mean))

    df1=pd.DataFrame(np.array(train_sizes))
    df2=pd.DataFrame(np.sqrt(-train_scores_mean))
    df3=pd.DataFrame(np.sqrt(-test_scores_mean))  

    df1.to_csv('rf_learning_curve.csv')
    df2.to_csv('rf_learning_curve.csv',mode='a')
    df3.to_csv('rf_learning_curve.csv',mode='a')
    plt.plot(train_sizes, np.sqrt(-train_scores_mean), 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, np.sqrt(-test_scores_mean), 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('learning_curve_rf.png')

    print("average difference between curves %.3f") %np.mean(np.sqrt(np.abs((-train_scores_mean)-(-test_scores_mean)))) 
    return plt 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
args  = parser.parse_args() 

print(args.file1)
data=pd.read_csv(args.file1)
data=data.sample(n=int(5e5),random_state=1)
features=['op_voronoi_vol','nn_si_bonds_average','nn_vv_average','nn_dist_average','nn_eng_average','nn_cnp_average',
          'nn_vv_std','nn_dist_std','nn_eng_std','nn_cnp_std','nn_si_bonds_std',
          'nn_vv_min','nn_dist_min','nn_eng_min','nn_cnp_min','nn_si_bonds_min',
          'nn_vv_max','nn_dist_max','nn_eng_max','nn_cnp_max','nn_si_bonds_max']

endpoint='deltaE'

X=data[features]
Y=data[endpoint]

n_estimators=1000 #len(np.array(data_x))
title = 'Learning Curves (RF)'
title_difference = 'Difference Learning Curves (RF)'
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
estimator = ExtraTreesRegressor(n_estimators=n_estimators)#,gamma=0.01,degree=5)
plot_learning_curve(estimator, title, X, Y, cv=cv, n_jobs=-1)

plt.show()
