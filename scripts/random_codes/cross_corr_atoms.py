#!/usr/bin/env python 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd 
import argparse

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
args  = parser.parse_args() 

data=pd.read_csv(args.file1)
features=['energy','cnp','vor_vol','vor_neigh']
corr_dat = data[features]

# determine cross correlation table matrix
size=10
corr=corr_dat.corr()
fig, ax = plt.subplots(figsize=(size, size))
sns.set(font_scale=1.5)
sns_plot=sns.heatmap(corr,cmap="coolwarm",square=True,annot=True, fmt=".1f",annot_kws={"size": 20})
plt.yticks(rotation=45,fontsize=16,fontweight='bold')
plt.xticks(rotation=45,fontsize=16,fontweight='bold')
plt.savefig('cross_corr.png')

features_pp=['cnp','vor_vol','energy']
plt.figure()
pairplot=sns.pairplot(data, hue='vor_neigh',diag_kind='kde', vars=features_pp)
plt.savefig('features_dist.png')

features_pp=['cnp','vor_vol','vor_neigh']
plt.figure()
pairplot=sns.pairplot(data,x_vars=features_pp,y_vars='energy')
plt.savefig('features_reg.png')

for feature in features_pp: 
    plt.figure()
    histogram=plt.hist(data[feature])
    plt.xlabel('%s' %feature)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('%s_dist.png' %feature)
