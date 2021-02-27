#encoding=utf-8
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
#import tensorflow as tf
import numpy as np
#from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
#from deepctr.models import DeepFM

import datetime
from collections import defaultdict

test=pd.read_csv('../test')
print(test.head(5))

train=pd.read_csv('../train_sample')

print(train.head(5))

fea_names=[]
for i, fea in enumerate(train):
    print(i,fea)
    if fea != 'click':
        fea_names.append(fea)
print(fea_names)

train[fea_names]=train[fea_names].fillna('-1', )
train[fea_names]=train[fea_names].fillna(0, )

test[fea_names]=test[fea_names].fillna('-1', )
test[fea_names]=test[fea_names].fillna(0, )

train_sample_num=len(train.index)

#将训练集和测试集concat起来，然后对离散特征进行编码
merged_df = pd.concat([train.drop(columns=['click'],axis=1),test])

labelEncoder_list=defaultdict(list)
for col in merged_df:
    if col == 'id':
        continue
    lbe = LabelEncoder()
    if col == 'hour':
        lbe.fit(merged_df[col].astype(str).str[-2:])
        print(len(merged_df[col].astype(str).str[-2:]))
        merged_df[col] = lbe.transform(merged_df[col].astype(str).str[-2:])
    else:
        lbe.fit(merged_df[col])
        merged_df[col] = lbe.transform(merged_df[col])
    labelEncoder_list[col]=lbe

    
train_df= merged_df.iloc[:train_sample_num,:]
train_df['click']=train['click']

test_df=merged_df.iloc[train_sample_num:,:]

train_x= train_df.drop(columns=['click','id'],axis=1).as_matrix()
train_y= train_df['click']
test_x= test_df.drop(columns=['id'],axis=1).as_matrix()

print(type(train_x))
print(type(train_y))
print(type(test_x))

import xgboost as xgb

starttime = datetime.datetime.now()
model=xgb.XGBClassifier(max_depth=5,learning_rate= 0.5, verbosity=1, objective='binary:logistic',random_state=1)
model.fit(train_x, train_y)
endtime = datetime.datetime.now()
print( "train cost " + str((endtime - starttime).seconds))

starttime2 = datetime.datetime.now()
predict_test = model.predict(test_x)
endtime2 = datetime.datetime.now()
print( "predict cost " + str((endtime2 - starttime2).seconds))

print(type(predict_test))
print(predict_test[0:100])
