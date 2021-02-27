#encoding=utf-8
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
import numpy as np
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM

import datetime
from collections import defaultdict


import sys

if_debug=sys.argv[1]

if if_debug:
    train_file='../train_sample_2'
    test_file='../test_sample'
else:
    train_file='../train'
    test_file='../test'

#分别读取训练集和测试集
train=pd.read_csv(train)
print("训练集示例")

print(train.head(5))
print(train['id'].head())

test=pd.read_csv(test)
print("测试集示例")
print(test.head(5))



#搜集特征名
fea_names=[]
for i, fea in enumerate(train):
    print(i,fea)
    if fea != 'click' and fea != 'id':
        fea_names.append(fea)
print("特征列表:")
print(fea_names)

#填充默认值
train[fea_names]=train[fea_names].fillna('-1', )
train[fea_names]=train[fea_names].fillna(0, )

test[fea_names]=test[fea_names].fillna('-1', )
test[fea_names]=test[fea_names].fillna(0, )
train_sample_num=len(train.index)

#将训练集和测试集concat起来，然后对离散特征进行编码
merged_df = pd.concat([train.drop(columns=['click'],axis=1),test])

print(merged_df.dtypes)

print("合并数据集")
print(merged_df.head(5))


#对离散特征进行编码
#
labelEncoder_list=defaultdict(list)
for col in merged_df:
    if col == 'id':
        continue
    lbe = LabelEncoder()
    if col == 'hour':
        lbe.fit(merged_df[col].astype(str).str[-2:]) #提取出hour字段
#         print(len(merged_df[col].astype(str).str[-2:]))
        merged_df[col] = lbe.transform(merged_df[col].astype(str).str[-2:])
    else:
        lbe.fit(merged_df[col])
        merged_df[col] = lbe.transform(merged_df[col])
    labelEncoder_list[col]=lbe

#从合并后的dataframe中再抽取出训练集和测试集    
train_df= merged_df.iloc[:train_sample_num,:]
train_df.loc[:,'click']=train['click']

test_df=merged_df.iloc[train_sample_num:,:]

#生成模型的训练集输入x和y，及测试集输入
train_x= train_df.drop(columns=['click','id'],axis=1) # .as_matrix()
train_y= train_df['click']
test_x= test_df.drop(columns=['id'],axis=1) #.as_matrix()

# test_ids= pd.DataFrame(test_df['id'].values.reshape([-1,1]))

test_ids=test['id'] #.astype('int64')

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=train[feat].nunique(),
                                     embedding_dim=4,
                                     use_hash=True,
                                         embeddings_initializer=tf.initializers.identity(np.zeros([train[feat].nunique(),4])))
                           for i,feat in enumerate(fea_names)] 


model = DeepFM(fixlen_feature_columns, fixlen_feature_columns, task='binary')

model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )


train_input = {name:train_x[name].values for name in fea_names}

for i,v in train_input.items():
    print (i,type(v))
#print(type(train['click'].values))

test_input = { name:test_x[name].values for name in fea_names}



target='click'
starttime = datetime.datetime.now()
history = model.fit(train_input, train_y,batch_size=256, epochs=10, verbose=2, validation_split=0.1)
endtime = datetime.datetime.now()
print( "train cost " + str((endtime - starttime).seconds))

starttime2 = datetime.datetime.now()
pred_ans = model.predict(test_input, batch_size=256)
endtime2 = datetime.datetime.now()
print( "predict cost " + str((endtime2 - starttime2).seconds))

pred_ans_df = pd.DataFrame(data=pred_ans)
print(pred_ans_df.shape)
print(test_ids.shape)

submission=pd.concat([test_ids,pred_ans_df],axis=1)
print(submission.shape)
print(submission.dtypes)


submission.to_csv('deepfm_submission',header=['id','click'],index=False)
# print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
# print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

