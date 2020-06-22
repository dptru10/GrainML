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
data=data.set_index('op_type')

index=data.index 

insert_all  = data.loc['insert']
remove_all  = data.loc['remove']
replace_all = data.loc['replace']

insert_all  = insert_all.set_index('op_species')
remove_all  = remove_all.set_index('op_species')
replace_all = replace_all.set_index('op_species')

insert_si = insert_all.loc['Si']
insert_c  = insert_all.loc['C']

remove_si = remove_all.loc['Si']
remove_c  = remove_all.loc['C']

replace_si = replace_all.loc['Si']
replace_c  = replace_all.loc['C']


insert_si.to_csv('insert_si.csv',mode='w')
insert_c.to_csv('insert_c.csv',mode='w') 

remove_si.to_csv('remove_si.csv',mode='w') 
remove_c.to_csv('remove_c.csv',mode='w') 

replace_si.to_csv('replace_si.csv',mode='w') 
replace_c.to_csv('replace_c.csv',mode='w') 





features=['deltaE','op_eng','op_cnp','op_vornoi_vol','op_si-si_bonds','op_si-c_bonds','op_c-c_bonds','nn_1_distance','nn_1_eng','nn_1_cnp','nn_1_vornoi_vol','nn_1_si-si_bonds','nn_1_si-c_bonds','nn_2_distance','nn_2_eng','nn_2_cnp','nn_2_vornoi_vol','nn_2_si-si_bonds','nn_2_si-c_bonds','nn_2_c-c_bonds','nn_3_distance','nn_3_eng','nn_3_cnp','nn_3_vornoi_vol','nn_3_si-si_bonds','nn_3_si-c_bonds','nn_3_c-c_bonds','nn_4_distance','nn_4_eng','nn_4_cnp','nn_4_vornoi_vol','nn_4_si-si_bonds','nn_4_si-c_bonds']

list_of_ops=[insert_si,insert_c,remove_si,remove_c,replace_si,replace_c]
op_names=['insert_si','insert_c','remove_si','remove_c','replace_si','replace_c']
i=0
f=open('data_set_sizes.txt')
for item in list_of_ops:

    f.write(op_names[i])
    f.write('\n')
    f.write(str(item.shape[0]))
    f.write('\n')

    corr_dat = item[features]
    #corr_dar = (corr_dat-corr_dat.mean())/corr_dat.std()
    # determine cross correlation table matrix
    size=20
    corr=corr_dat.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    sns.set(font_scale=1.5)
    sns_plot=sns.heatmap(corr,cmap="coolwarm",square=True,annot=True, fmt=".1f",annot_kws={"size": 10})
    plt.yticks(rotation=0,fontsize=16,fontweight='bold')
    plt.xticks(rotation=90,fontsize=16,fontweight='bold')
    plt.tight_layout()
    plt.savefig('cross_corr_%s.png' %op_names[i])
    i+=1
f=close('data_set_sizes.txt')

