# Importing all relevant packages
import json
from implement_linreg import gd_lin_reg, cf_lin_reg
from implement_linreg import compute_mse
from proj1_data_loading import transform
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas as pd
import math
import numpy as np
import seaborn as sns



# Load json data file into data numpy array
with open("proj1_data.json") as fp:
    data = json.load(fp)

# Splitting data into training, validation and testing datasets
training = data[0:10000] #the list contains elements from position 0 to position 9999, total 10000
validation = data[10000:11000] #the list contains elements from position 10000 to position 10999, total 1000
testing = data[11000:12000] # the list contains elements from position 11000 to position 11999, total 1000

# Examine the top 10 most popular comments
newlist = sorted(training, key=itemgetter('popularity_score'), reverse=True)
for i in range(10):
    print(newlist[i])

# Examine the top 10 least popular comments
newlist = sorted(training, key=itemgetter('popularity_score'), reverse=False)
for i in range(10):
    print(newlist[i])

# For more analysis, it is best to convert the list of dicts to a DataFrame (with pandas)
train_df = pd.DataFrame(training)

# Analyze training set by pivoting features against each other
train_df[['controversiality', 'popularity_score']].groupby(['controversiality'], as_index=False).agg(['min', 'max', 'mean'])

train_df[['is_root', 'popularity_score']].groupby(['is_root'], as_index=False).agg(['count', 'min', 'max', 'mean'])

train_df[['children', 'popularity_score']].groupby(['children'], as_index=False).agg(['count', 'min', 'max', 'mean'])

# Adding new features and visualizing correlation
train_df['children'] = pd.to_numeric(train_df['children']).astype(float)
train_df['children^2'] = train_df['children'] ** 2  #children squared
train_df['children^3'] = train_df['children'] ** 3  #children cubed
train_df['sqrt_children'] = train_df['children'] ** 0.5 #log children
train_df['log_children'] = np.log(train_df['children'])
train_df['log_children*(1-controversiality)'] = np.log(train_df['children']) * (1-train_df['controversiality'])
train_df['log_children*is_root'] = np.log(train_df['children']) * train_df['is_root']

for i in range(len(train_df)):
    train_df.loc[i, 'length_text'] = len(train_df.loc[i, 'text'])
    train_df.loc[i, 'exclamation_count'] = (train_df.loc[i, 'text']).count('!')

# Analyze by visualizing data
grid=sns.FacetGrid(train_df,row='is_root', col='controversiality', size=2.2, aspect=2)
grid.map(plt.scatter, 'log_children*(1-controversiality)', 'popularity_score', edgecolor='w')
grid.add_legend


# Analyzing features based on correlation
corr = train_df.corr()
ax = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.tight_layout()