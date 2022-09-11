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
    drop_cols = ['ID','reservation_status_date', 'reservation_status', 'agent', 'company','is_canceled']
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
            if col!= 'adr':
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

import xgboost as xgb
from xgboost import XGBRegressor

# combine train.csv and test.csv then do features eng + one hot encoding
df_tmp = pd.concat([df, df_test])
encoded_df = feature_eng(df_tmp)

# split train and test into X and X_final
X, X_final = train_test_split(encoded_df, test_size=len(df_test)/(len(df_test)+len(df)), shuffle=False)
X_final.pop('adr')

# bootstrap the original data 
bootstrapped = X.sample(frac=1, replace=True, random_state=321)
# shuffle in place
bootstrapped = bootstrapped.sample(frac=1, random_state=321)

# split into top-level train/test , 75-25 split
train = bootstrapped[:int(len(bootstrapped)*0.75)]
test = bootstrapped[int(len(bootstrapped)*0.75):]
x_data_cols = [col for col in bootstrapped.columns if col!= 'adr']

# split into train/test
X_train = train[x_data_cols] # train data 0.75
y_train = train['adr'] # train label 0.75
y = bootstrapped['adr'] # all train label
X = bootstrapped[x_data_cols] # all train data

# tuning hyperparam
param_test1 = {
 'gamma':[0,0.01],
    
}

grid_xgb = GridSearchCV(estimator = XGBRegressor( 
                                                 learning_rate = 0.065, n_estimators=800, max_depth=57,
                                                 min_child_weight=1, gamma=0, subsample=0.95, colsample_bytree=0.75,
                                                 objective= 'reg:linear', reg_lambda = 1, reg_alpha = 0,
                                                 nthread=4, scale_pos_weight=1, seed=27), 
                                                 param_grid = param_test1, n_jobs=72, scoring='neg_mean_absolute_error', verbose=2,

                                                 cv = 5

                                                )

grid_xgb.fit(X, y)

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid_xgb.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_xgb.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_xgb.best_params_)
