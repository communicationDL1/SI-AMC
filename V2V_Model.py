# -*- coding: utf-8 -*-
"""
Created on Mon Oct 9 14:50:49 2023

@author: Gray
"""

import scipy.io as scio
import numpy as np
import pandas as pd
from keras.models import Model
from sklearn.metrics import confusion_matrix
from numpy import array
from keras.utils import to_categorical,plot_model
from sklearn.model_selection import train_test_split
import keras.models as models
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import optimizers
import tensorflow as  tf
tf.compat.v1.disable_eager_execution()
def get_flops_params():
    sess = tf.compat.v1.Session()
    graph = sess.graph
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
KS=9
def CNN_paper_bndropout():
    cnn_input = Input(shape=[52, 2])
    x = Conv1D(32, KS, padding='same')(cnn_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(32, KS, padding='same')(x)
    # x = SeparableConv1D(32, KS, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(4)(x)
    # x = SeparableConv1D(32, KS, padding='same')(x
    x = Conv1D(32, KS, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # x = SeparableConv1D(32, KS, padding='same')(x)
    x = Conv1D(32, KS, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(32,activation="gelu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = GaussianDropout(0.5)(x)
    x = Dense(5, activation="softmax", kernel_regularizer=regularizers.l2(0.01))(x)
    adam=optimizers.Adam(lr=0.001)
    model = Model(inputs=cnn_input, outputs=x)
    model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['accuracy'])
    return model
def CNN_paper():
        cnn_input = Input(shape=[52, 2])
        x = Conv1D(32, KS, padding='same')(cnn_input)
        x = Activation('relu')(x)
        x = SeparableConv1D(32, KS, padding='same')(x)
        x = Activation('relu')(x)
        x = AveragePooling1D(4)(x)
        x = SeparableConv1D(32, KS, padding='same')(x)
        x = Activation('relu')(x)
        x = SeparableConv1D(32, KS, padding='same')(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(5, activation="softmax", kernel_regularizer=regularizers.l2(0.01))(x)
        # SGD = optimizers.SGD()
        model = Model(inputs=cnn_input, outputs=x)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
def DCNN():
    cnn_input = Input(shape=[52, 2])
    x = Conv1D(30, 15, padding='same')(cnn_input)
    x = Activation('relu')(x)
    x = Conv1D(20, 15, padding='same')(x)
    x = Activation('relu')(x)
    x = AveragePooling1D(4)(x)
    x = Conv1D(30, 15, padding='same')(x)
    x = Activation('relu')(x)
    x = AveragePooling1D(4)(x)
    x = Conv1D(20, 15, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(30, 15, padding='same')(x)
    x = Activation('relu')(x)
    x = AveragePooling1D(4)(x)
    x = Conv1D(30, 15, padding='same')(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(25,activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dense(5, activation="softmax", kernel_regularizer=regularizers.l2(0.01))(x)
    model = Model(inputs=cnn_input, outputs=x)
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model
data_path="V2V_H.mat"
data = scio.loadmat(data_path)
x=data.get('H')
Sample_num = 10000
y1=np.zeros([Sample_num,1])
y2=np.ones([Sample_num,1])
y3=np.ones([Sample_num,1])*2
y4=np.ones([Sample_num,1])*3
y5=np.ones([Sample_num,1])*4
y=np.vstack((y1,y2,y3,y4,y5))
y = array(y)
y = to_categorical(y)
X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size = 0.3, random_state= 1)
model = CNN_paper_bndropout()
model.summary()
checkpoint = ModelCheckpoint("CNN_GAP_gelu.hdf5", verbose=1, save_best_only=False)
hist=model.fit(
    X_train,
    Y_train,
    batch_size=100,
    epochs=100,
    verbose=1,
    validation_data=(X_val, Y_val),
    callbacks=[checkpoint]
    )
train_test_list = [hist.history['accuracy'],hist.history['val_accuracy'],hist.history['loss'],hist.history['val_loss']]
train_test_array=np.array(train_test_list).T
df = pd.DataFrame(train_test_array, columns=['Training Acc', 'Test Acc','Training Loss','Test Loss'])
df.to_excel("CNN_GAP_gelu.xlsx", index=False)

