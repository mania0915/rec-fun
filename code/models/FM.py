#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file        :FM.py
@desc        :
@date        :2022/12/30 16:32:13
@author      :eason
@version     :1.0
'''
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# dense特征取对数　　sparse特征进行类别编码
def process_feat(data, dense_feats, parse_feats):
    df = data.copy()
    df_dense = df[dense_feats].fillna(0.0)
    for f in tqdm(dense_feats):
        df_dense[f] = df_dense[f].apply(lambda x: np.log(1 + x)
                                        if x > -1 else -1)

    df_sparse = df[parse_feats].fillna('-1')
    for f in tqdm(parse_feats):
        lbe = LabelEncoder()
        df_sparse[f] = lbe.fit_transform(df_sparse[f])

    df_sparse_arr = []
    for f in tqdm(parse_feats):
        data_new = pd.get_dummies(df_sparse.loc[:, f].values)
        data_new.columns = [
            f + '_{}'.format(i) for i in range(data_new.shape[1])
        ]
        df_sparse_arr.append(data_new)
    df_new = pd.concat([df_dense] + df_sparse_arr, axis=1)
    return df_new

class crossLayer(layers.Layer):
    def __init__(self, input_dim, output_dim=10, **kwargs):
        super(crossLayer, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
       
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, x):  # 
        a = K.pow(K.dot(x, self.kernel), 2)
        b = K.dot(K.pow(x, 2), K.pow(self.kernel, 2))
        return 0.5 * K.mean(a - b, 1, keepdims=True)





def FM(feature_dim):
    inputs = Input(shape=(feature_dim,))
    #一阶特征
    linear = Dense(units=1, kernel_regularizer=regularizers.l2(0.01)   
                            ,bias_regularizer=regularizers.l2(0.01))(inputs)
    #二阶特征
    cross = crossLayer(feature_dim)(inputs)
    
    add = Add()([linear, cross])
    pred = Dense(units=1, activation='sigmoid')(add)
    model = Model(inputs=inputs, outputs=pred)
    model.summary()
    auc_metric = tf.keras.metrics.AUC(name='auc', num_thresholds=20000, curve='ROC')
    model.compile(loss='binary_crossentropy',
                optimizer = optimizers.Adam(),
                metrics=['binary_accuracy',auc_metric]
                )   
    return model



if __name__ == '__main__':
    print('load data')
    data = pd.read_csv('./data/kaggle_train.csv')
    print(data.shape)
    print(type(data.columns))  # Index
    print(data[:10])

    cols = data.columns.values
    print(cols)
    dense_feats = [f for f in cols if f[0] == 'I']
    parse_feats = [f for f in cols if f[0] == 'C']

    # data process
    print('processing features')
    feats = process_feat(data, dense_feats, parse_feats)
    print(feats)
    X_train, X_test, y_train, y_test = train_test_split(feats, data['Label'], test_size=0.2, random_state=2020)
    print(feats.shape[1])
    model = FM(feats.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))
