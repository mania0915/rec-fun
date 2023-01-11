#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file        :GBDT_LR.py
@desc        :
@date        :2023/01/04 16:50:18
@author      :eason
@version     :1.0
'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
import tensorflow as tf
import gc

def lr_model(data, continuous_fea, category_fea):
    scaler = MinMaxScaler()
    for col in continuous_fea:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    for col in category_fea:
        onehot_fea = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis=1, inplace = True)
        data = pd.concat([data,onehot_fea], axis = 1)

    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis=1, inplace=True)

    x_train, x_val ,y_train, y_val = train_test_split(train, target, test_size=0.02, random_state = 2020)
    lr = LogisticRegression()
    lr.fit(x_train, y_train)

    train_loss = log_loss(y_train, lr.predict_proba(x_train)[:,1 ])
    test_loss = log_loss(y_val, lr.predict_proba(x_val)[:,1 ])

    print('tr_logloss: ', train_loss)
    print('val_logloss: ', test_loss)

    y_pred = lr.predict_proba(test)[:, 1]  # predict_proba 返回n行k列的矩阵，第i行第j列上的数值是模型预测第i个预测样本为某个标签的概率, 这里的1表示点击的概率
    print('predict: ', y_pred[:10]) 


def gbdt_model(data, continuous_fea, category_fea):

    for fea_name in category_fea:
        onehot_fea = pd.get_dummies(data[fea_name],prefix=fea_name)
        data.drop([fea_name], axis=1, inplace = True)
        data = pd.concat([data,onehot_fea], axis=1)

    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis=1, inplace=True)

    X_train ,X_val, y_train, y_val = train_test_split(train, target, test_size=0.02, random_state=2020)

    gbm = LGBMClassifier(boosting_type='gbdt',  # 这里用gbdt
                             objective='binary', 
                             subsample=0.8,
                             min_child_weight=0.5, 
                             colsample_bytree=0.7,
                             num_leaves=100,
                             max_depth=12,
                             learning_rate=0.01,
                             n_estimators=10000)
    auc_metric = tf.keras.metrics.AUC(name='auc', num_thresholds=20000, curve='ROC')
    gbm.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names = ['train', 'val'],
            eval_metric='binary_logloss',
            early_stopping_rounds=100,
            )
    tr_logloss = log_loss(y_train, gbm.predict_proba(X_train)[:, 1])   # −(ylog(p)+(1−y)log(1−p)) log_loss
    val_logloss = log_loss(y_val, gbm.predict_proba(X_val)[:, 1])
    print('tr_logloss: ', tr_logloss)
    print('val_logloss: ', val_logloss)
    
    # 模型预测
    y_pred = gbm.predict_proba(test)[:, 1]  # predict_proba 返回n行k列的矩阵，第i行第j列上的数值是模型预测第i个预测样本为某个标签的概率, 这里的1表示点击的概率
    print('predict: ', y_pred[:10]) #
    
#下面就是把上面两个模型进行组合， GBDT负责对各个特征进行交叉和组合， 把原始特征向量转换为新的离散型特征向量， 然后在使用逻辑回归模型
def gbdt_lr_model(data, continuouse_fea, category_fea):
    print('gbdt_lr')
    for fea_name in category_fea:
        onehot_fea = pd.get_dummies(data[fea_name],prefix=fea_name)
        data.drop([fea_name], axis=1, inplace = True)
        data = pd.concat([data,onehot_fea], axis=1)

    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis=1, inplace=True)

    X_train ,X_val, y_train, y_val = train_test_split(train, target, test_size=0.02, random_state=2020)

    gbm = LGBMClassifier(boosting_type='gbdt',  # 这里用gbdt
                             objective='binary', 
                             subsample=0.8,
                             min_child_weight=0.5, 
                             colsample_bytree=0.7,
                             num_leaves=100,
                             max_depth=12,
                             learning_rate=0.01,
                             n_estimators=10000)
    auc_metric = tf.keras.metrics.AUC(name='auc', num_thresholds=20000, curve='ROC')
    gbm.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names = ['train', 'val'],
            eval_metric='binary_logloss',
            early_stopping_rounds=100,
            )
    model = gbm.booster_
    gbdt_feat_train = model.predict(X_train, pred_leaf=True)
    gbdt_feat_val = model.predict(X_val, pred_leaf=True)
    gbdt_feats_name = ["gbdt_leaf"+str(i) for i in range(gbdt_feat_train.shape[1])]
    print(f'gbdt_feats_name: {gbdt_feats_name}')
    df_train_gbdt_feats = pd.DataFrame(gbdt_feat_train, columns = gbdt_feats_name) 
    df_test_gbdt_feats = pd.DataFrame(gbdt_feat_val, columns = gbdt_feats_name)

    train = pd.concat([train, df_train_gbdt_feats], axis = 1)
    test = pd.concat([test, df_test_gbdt_feats], axis = 1)
    train_len = train.shape[0]
    data = pd.concat([train, test])
    del train
    del test
    gc.collect()

    scaler = MinMaxScaler()
    for col in continuouse_fea:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    for col in gbdt_feats_name:
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)

    train = data[: train_len]
    test = data[train_len:]
    del data
    gc.collect()

    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.3, random_state = 2018)

    
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    print('tr-logloss: ', tr_logloss)
    val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print('val-logloss: ', val_logloss)
    y_pred = lr.predict_proba(test)[:, 1]
    print(y_pred[:10])








if __name__ == '__main__':
    print(f"gbdt_lr")
    path = 'data/'
    df_train = pd.read_csv(path + "kaggle_train.csv")
    df_test = pd.read_csv(path + "kaggle_test.csv")

    df_train.drop(['Id'], axis=1, inplace=True)
    df_test.drop(['Id'], axis=1, inplace=True)
    df_test['Label'] = -1
    data = pd.concat([df_train, df_test])
    data.fillna(-1, inplace=True)
    print(data[:10])

    # continuous_fea = ['I'+str(i+1) for i in range(13)]
    # category_fea = ['C'+str(i+1) for i in range(26)]
    cols = data.columns.values
    continuous_fea = [i for i in cols if i[0] == 'I']
    category_fea = [i for i in cols if i[0] == 'C']
    print(f'continuous_fea:{continuous_fea}\n category_fea:{category_fea}')

    ## 建模
    # 下面训练三个模型对数据进行预测， 分别是LR模型， GBDT模型和两者的组合模型， 然后分别观察它们的预测效果， 对于不同的模型， 特征会有不同的处理方式如下：
    # 1. 逻辑回归模型： 连续特征要归一化处理， 离散特征需要one-hot处理
    # 2. GBDT模型： 树模型连续特征不需要归一化处理， 但是离散特征需要one-hot处理
    # 3. LR+GBDT模型： 由于LR使用的特征是GBDT的输出， 原数据依然是GBDT进行处理交叉， 所以只需要离散特征one-hot处理

    lr_model(data, continuous_fea, category_fea)
    gbdt_model(data, continuous_fea, category_fea)
    gbdt_lr_model(data, continuous_fea, category_fea)
