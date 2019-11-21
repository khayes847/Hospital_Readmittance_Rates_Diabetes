"""
This file contains assorted functions.
"""

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.dummy import DummyClassifier




def standardize_train_test_split(x_val, y_val, ts=.25, rs=42):
    """Standardizes X data and creates stratified train-test split"""
    scaler = MinMaxScaler()
    x_val = scaler.fit_transform(x_val)
    x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=ts,
                                                        random_state=rs, stratify=y_val)
    return x_train, x_test, y_train, y_test


def plot_confusion_matrix(cm_val, classes, normalize=False,
                          title="Confusion Matrix"):
    """Creates graph of confusion matrix"""
    plt.grid(None)
    plt.imshow(cm_val, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation='45')
    plt.yticks(tick_marks, classes)
    if normalize:
        cm_val = cm_val.astype('float') / cm_val.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm_val.max()/2
    for i, j in itertools.product(range(cm_val.shape[0]),
                                  range(cm_val.shape[1])):
        plt.text(j, i, cm_val[i, j], horizontalalignment="center",
                 color="white" if cm_val[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True \nlabel', rotation=0)
    plt.xlabel('Predicted label')


def dummy_regression(x_train, x_test, y_train, y_test):
    """Creates dummy logistic regression"""
    dummy = DummyClassifier(strategy='most_frequent', random_state=42).fit(x_train, y_train)
    dummy_pred = dummy.predict(x_test)
    print('Test Accuracy score: ', accuracy_score(y_test, dummy_pred))
    print('Test F1 score: ', f1_score(y_test, dummy_pred, average='micro'))
    cm_val = confusion_matrix(y_test, dummy_pred)
    cm_labels = ['No', '>6 months', '<=6 months']
    plot_confusion_matrix(cm_val, cm_labels, title="Dummy Regression")


