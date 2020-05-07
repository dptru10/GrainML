from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import matplotlib 
import argparse 
matplotlib.use('Agg')

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
parser.add_argument("--test",action="store_true")
parser.add_argument("--train",action="store_true")
parser.add_argument("--validate",action="store_true")
args  = parser.parse_args() 

data=pd.read_csv(args.file1)

name = args.file1 
name = name.split('.')[0]
name = name.split('/')
name = name[len(name)-1]
print(name) 

if args.train is True: 
    true  = data['train']
    model = data['train_model']

if args.test is True: 
    true  = data['test']
    model = data['test_model']

if args.validate is True: 
    true  = data['binary_predicted']
    model = data['binary_true']

accuracy=accuracy_score(true,model)

if args.test is True: 
    f=open('scores_test_'+name+'.txt',mode='w')
    f.write('Test:\naccuracy: %.3f'  %(accuracy))
if args.train is True: 
    f=open('scores_train_'+name+'.txt',mode='w')
    f.write('Train: \naccuracy:%.3f' %(accuracy))
if args.validate is True: 
    f=open('scores_validate_'+name+'.txt',mode='w')
    f.write('Train: \naccuracy:%.3f' %(accuracy))
f.close() 

size=20
confusion=confusion_matrix(true,model)
plt.figure()
fig, ax = plt.subplots(figsize=(size, size))
sns.set(font_scale=1.5)
sns_plot=sns.heatmap(confusion,cmap="coolwarm",square=True,annot=True, fmt=".1f",annot_kws={"size": 20})
plt.yticks(rotation=0,fontsize=28,fontweight='bold')
plt.xticks(fontsize=28,fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_%s.png' %name)

