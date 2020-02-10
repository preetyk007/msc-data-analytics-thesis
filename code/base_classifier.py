# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 20:31:52 2019

@author: Preety
"""

import argument_parser
import svm_classifier
import fc_classifier

import os
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

import warnings
warnings.filterwarnings('always')

args = argument_parser.classifier_args()
classifier = args.type

base_path = os.path.join(os.getcwd(), "./dataset/")
output_path = os.path.join(os.getcwd(), "./output/")

train_dir = base_path + "train/"

image_size = (224, 224)

class_names = []
# LULC class names
for c in os.listdir(train_dir):
    class_names.append(c)

def load_data(mode='train'):
    h5f_data  = h5py.File(output_path + mode + "_features_resnet152" + ".h5", 'r')
    train = h5f_data['dataset_1']
    features_train = np.array(train)
    h5f_data.close()
    
    h5f_data  = h5py.File(output_path + mode + "_labels_resnet152" + ".h5", 'r')
    train = h5f_data['dataset_1']
    train_labels = np.array(train)
    h5f_data.close()
    
    if(classifier == 'svm'):
        train_labels = [np.where(r==1)[0][0] for r in train_labels]
        train_labels = np.array(train_labels)
    
    return features_train, train_labels


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def calc_top_error(best_model, X, Y):
    rank_1 = 0
    rank_5 = 0
    
    for (features, label) in zip(X, Y):
        predictions = best_model.predict_proba(np.atleast_2d(features))[0]
        predictions = np.argsort(predictions)[::-1][:5]
    
        if label == predictions[0]:
            rank_1 += 1
    
        if label in predictions:
            rank_5 += 1
    
    rank_1 = (rank_1 / float(len(Y))) * 100
    rank_5 = (rank_5 / float(len(Y))) * 100
    
    return rank_1, rank_5


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


def store_results(model, X_test, actual, preds):
    rank_1, rank_5 = calc_top_error(model, X_test, actual)
    
    filename = output_path + "results_base_" + classifier + ".txt"
    
    AUC = multiclass_roc_auc_score(actual, preds)
    
    f = open(filename, "w")
    
    f.write("Rank-1: {:.2f}%\n".format(rank_1))
    f.write("Rank-5: {:.2f}%\n\n".format(rank_5))
    
    f.write("Classification report:\n \t{}\n".format(classification_report(actual, preds, target_names=class_names)))
    f.write("Confusion matrix:\n {}\n".format(confusion_matrix(actual, preds)))
    
    f.write("Accuracy score: {:.2f}%\n".format(accuracy_score(actual, preds) * 100))
    f.write("Precision score (macro): {:.2f}%\n".format(precision_score(actual, preds, average='macro') * 100))
    f.write("Recall score (macro): {:.2f}%\n".format(recall_score(actual, preds, average='macro') * 100))
    f.write("F1 score (macro): {:.2f}%\n".format(f1_score(actual, preds, average='macro') * 100))
    f.write("AUC ROC score: {:.2f}%\n".format(AUC * 100))
    f.write("Matthews Correlation/phi Coefficient is: {:.2f}\n".format(matthews_corrcoef(actual, preds)))
    
    f.close()


def build_model():
    X_train, Y_train = load_data()
    X_test, Y_test = load_data('test')
    
    print("[INFO] training model...")
    
    if(classifier == 'svm'):
        model = svm_classifier.create_best_model()
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)

    else:
        model = fc_classifier.create_base_model()
        output = model.fit(X_train, Y_train, batch_size=28, epochs=3)
        preds = model.predict(X_test)
        Y_test = Y_test.argmax(axis=1)
        preds = preds.argmax(axis=1)
        print("Training loss: {:.2f}%\n".format(output.history['loss'][-1] * 100))
        print("Training accuracy: {:.2f}%\n".format(output.history['accuracy'][-1] * 100))
        
    store_results(model, X_test, Y_test, preds)
    
    print("Classification report:\n")
    print(classification_report(Y_test, preds, target_names=class_names))
    
    conf_mat = confusion_matrix(Y_test, preds)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    print("MCC: {:.2f}".format(matthews_corrcoef(Y_test, preds)))

    
def main():
    build_model()


if __name__ == '__main__':
    main()