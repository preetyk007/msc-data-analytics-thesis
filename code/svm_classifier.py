# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 02:50:27 2019

@author: Preety
"""

from sklearn.svm import SVC

def create_best_model(kernel='rbf', gamma=1e-5, C=10):
    clf = SVC(decision_function_shape='ovo', random_state=9, probability=True, degree=3, kernel=kernel, gamma=gamma, C=C)
    return clf

def create_model():
    param_grid = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
               'gamma': [1e-4, 1e-5],
               'C': [1, 10, 100, 1000]
               }
    clf = SVC(decision_function_shape='ovo', random_state=9, probability=True, degree=3)
    
    return clf, param_grid


def main():
    model, param_grid = create_model()

if __name__ == '__main__':
    main()