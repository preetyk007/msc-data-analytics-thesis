# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 03:27:53 2019

@author: Preety
"""

import os

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

base_path = os.path.join(os.getcwd(), "./dataset/")

train_dir = base_path + "train/"

class_names = []
# LULC class names
for c in os.listdir(train_dir):
    class_names.append(c)

lr=[0.01, 0.001]
decay=[1e-4, 1e-5]
dropout = [0.3, 0.5]
epochs = [10, 30]
batch_size = [28, 32]

def create_base_model(lr=0.01, decay=1e-5, dropout=0.5):
    opt = optimizers.SGD(lr=lr, decay=decay, momentum=0.9)
                                                     
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(2048,)))
    model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu'))        
    model.add(Dropout(dropout))
    model.add(Dense(len(class_names), activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

def create_model():
    param_grid = dict(epochs=epochs, batch_size=batch_size, lr=lr, decay=decay, dropout=dropout)
    model = KerasClassifier(build_fn=create_base_model, verbose=1)
    
    return model, param_grid


def main():
    model, param_grid = create_model()

if __name__ == '__main__':
    main()