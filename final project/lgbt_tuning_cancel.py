import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import *
import seaborn as sns
import itertools
import math
import datetime
from datetime import timedelta
import warnings
from scipy.stats import zscore
import csv

# load data
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df=df.loc[(df['adr']<5300)] # remove shit data
df=df.loc[(df['adults']<5)]
df=df.loc[(df['children']<10)]
df=df.loc[(df['babies']<9)]

def feature_eng(df):
    ## drop unnecessary columns and rows with missing children, country values 
    drop_cols = ['ID','reservation_status_date', 'reservation_status', 'agent', 'company','adr']
    df = df.drop(columns=drop_cols)
    df = df.dropna(0, subset=['children']).reset_index(drop=True)
    # feature engineering -- first do one hot encodings of the hotel, meal, country, market segment, distribution
    #                         channel, deposit type, customer type columns 
    ohe = OneHotEncoder()
    to_encode = pd.concat([df['hotel'], df['meal'], df['market_segment'],
                           df['distribution_channel'], df['deposit_type'], df['customer_type'], df['arrival_date_year'],
                           df['assigned_room_type'], df['reserved_room_type']], axis=1)
    ## encoding the months... 
    # we can start January at (1,0), increasing by 30degrees (pi/6)
    months = ['January', 'February', 'March', 
              'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_enc = {}
    degr = 0
    for month in months: 
        month_enc[month] = (math.cos(degr), math.sin(degr))
        degr+=(math.pi)/6
    # build new column names for the encoded data
    unique_vals = []
    for col in to_encode:
        uniquevals = ['_'.join([str(val), col]) for val in list(to_encode[col].unique())]
        unique_vals += uniquevals
    # create a df of the encoded data
    encoded_arr = ohe.fit_transform(to_encode).toarray()
    encoded_df = pd.DataFrame(data=encoded_arr, columns = unique_vals)
    # add the encoded month information to the encoded_df with the one hot vectors
    all_reserved_months = df['arrival_date_month'].tolist()
    all_month_xs = []
    all_month_ys = []
    for month in all_reserved_months: 
        x, y = month_enc[month]
        all_month_xs.append(x)
        all_month_ys.append(y)
    encoded_df['arrival_date_month_X'] = all_month_xs
    encoded_df['arrival_date_month_Y'] = all_month_ys
    # z-transform the numerical columns and put them into the encoded df
    ss= StandardScaler()
    # add the original numerical columns back into the encoded dataframe
    # ignore columns that we just encoded! 
    encoded_cols = ['hotel', 'meal', 'country', 'market_segment', 
                    'distribution_channel', 'deposit_type', 'customer_type',
                    'assigned_room_type', 'reserved_room_type', 'arrival_date_month', 'arrival_date_year']
    for col in df.columns: 
        if col not in encoded_cols:
            if col!= 'is_canceled':
                encoded_df[col] = ss.fit_transform(np.asarray(df[col]).reshape(-1,1))
            else: 
                encoded_df[col] = df[col]
                
    return encoded_df

def get_feature_importances(tree_model, x_columns):
    df = pd.DataFrame({'weight':tree_model.feature_importances_.tolist(), 'features':list(x_columns)})
    df = df.sort_values(by='weight', ascending=False)
    return df

def evaluate_test(test_x, test_y, model):
    preds = model.predict(test_x)
    print("F1 score: {}, Recall Score: {}, Precision Score:{}, Accuracy: {} ".format(f1_score(test_y, preds), recall_score(test_y, preds), precision_score(test_y, preds),
                                                                                     accuracy_score(test_y, preds)))

import lightgbm as lgbm
from lightgbm import LGBMClassifier


# combine train.csv and test.csv then do features eng + one hot encoding
df_tmp = pd.concat([df, df_test])
encoded_df = feature_eng(df_tmp)

# split train and test into X and X_final
X, X_final = train_test_split(encoded_df, test_size=len(df_test)/(len(df_test)+len(df)), shuffle=False)
X_final.pop('is_canceled')

# bootstrap the original data 
bootstrapped = X.sample(frac=1, replace=True, random_state=321)
# shuffle in place
bootstrapped = bootstrapped.sample(frac=1, random_state=321)

# split into top-level train/test , 75-25 split
train = bootstrapped[:int(len(bootstrapped)*0.75)]
test = bootstrapped[int(len(bootstrapped)*0.75):]
x_data_cols = [col for col in bootstrapped.columns if col!= 'is_canceled']

# split into train/test
X_train = train[x_data_cols] # train data 0.75
y_train = train['is_canceled'] # train label 0.75
y = bootstrapped['is_canceled'] # all train label
X = bootstrapped[x_data_cols] # all train data

# tuning hyperparam
param_test1 = {
#'num_leaves':[100,150,200,250,300,350,400,450,500,550,600,650,700,750,800],
#'n_estimators':[100,150,200],
#'num_iterations':[100,150,200],
#'max_depth': [40,43]

}

random_grid = {

'boosting': ['dart','boosting'],
'max_depth': [5,10,15,20,30,40,50,60,70,80,90,100],
'reg_alpha':np.linspace(0,10,101),
'reg_lambda':np.linspace(0,10,101),
'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
'num_iterations': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
'subsample':np.linspace(0.5,1,101),
'colsample_bytree':np.linspace(0.5,1,11),
'num_leaves':[50, 100, 200, 300, 400, 500, 600, 700, 800],
'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
'subsample_for_bin': [3000,4000,5000,6000,7000,8000,9000],
'min_child_samples': [1, 3, 5, 10, 15, 20, 25, 30]

}
grid_lgbm = GridSearchCV(estimator = LGBMClassifier(boosting='dart',
    
                       max_depth=40,#
                       min_child_weight=0.1,#
                       min_child_samples=3,#
                       subsample=0.5,#
                       colsample_bytree=0.75,#
                       subsample_for_bin=5500,#
                       num_leaves=800,#
                       reg_alpha=0.06,#
                       reg_lambda=0,#
                       min_split_gain=0,#
                       num_iterations=100,
                       n_estimators=500,#
                       learning_rate=0.5,), #

                                            param_grid = param_test1, n_jobs=-1, iid=False, scoring='roc_auc', verbose=2,

                                            cv = 2 )


random_lgbm = RandomizedSearchCV(estimator = LGBMClassifier(boosting='dart',
    
                       max_depth=40,#
                       min_child_weight=0.1,#
                       min_child_samples=1,#
                       subsample=0.75,#
                       colsample_bytree=0.75,#
                       subsample_for_bin=6000,#
                       num_leaves=800,#
                       reg_alpha=0.03,
                       reg_lambda=0.15,
                       min_split_gain=0,#
                       num_iterations=100,
                       n_estimators=500,#
                       learning_rate=0.5,), #

  param_distributions = random_grid, n_iter = 2000, cv=5, n_jobs = -1, iid=False, scoring='roc_auc', verbose=2)

random_lgbm.fit(X, y)
#grid_lgbm.fit(X, y)

#print(" Results from Grid Search " )
#print("\n The best estimator across ALL searched params:\n",grid_lgbm.best_estimator_)
#print("\n The best score across ALL searched params:\n",grid_lgbm.best_score_)
#print("\n The best parameters across ALL searched params:\n",grid_lgbm.best_params_)

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",random_lgbm.best_estimator_)
print("\n The best score across ALL searched params:\n",random_lgbm.best_score_)
print("\n The best parameters across ALL searched params:\n",random_lgbm.best_params_)
