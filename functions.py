"""
This file contains assorted functions.
"""

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression




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
    print('Test F1 score: ', f1_score(y_test, dummy_pred, average='weighted'))
    print('Recall score: ', recall_score(y_test, dummy_pred, average='weighted'))
    cm_val = confusion_matrix(y_test, dummy_pred)
    cm_labels = ['No', 'Yes']
    plot_confusion_matrix(cm_val, cm_labels, title="Dummy Regression")


def log_gridsearch(x_train, y_train, x_test, y_test):
    """Conducts gridsearch for logistic regression. Compares estimated pre- and post-
    gridsearch f1 scores."""
    lr_clf = LogisticRegression(random_state=42)
    lr_cv_score = cross_val_score(lr_clf, x_train, y_train, cv=3, scoring='recall')
    mean_lr_cv_score = np.mean(lr_cv_score)
    print(f"Mean Pre-Gridsearch Recall: {mean_lr_cv_score :.2%}")
    lr_param_grid = {'penalty': ['l1', 'l2'],
                     'C': [.5, 1, 3, 5],
                     'tol': [0.00005, 0.0001, 0.00015, 0.0002],
                     'fit_intercept': [True, False],
                     'warm_start': [True, False]
                    }
    lr_grid_search = GridSearchCV(lr_clf, lr_param_grid, cv=3, n_jobs=-3, verbose=10,
                                  scoring='recall', return_train_score=True)
    lr_grid_search.fit(x_train, y_train)
    lr_gs_training_score = np.mean(lr_grid_search.cv_results_['mean_train_score'])
    print(f"Mean Training Score: {lr_gs_training_score :.2%}")
    print("Best Parameter Combination Found During Grid Search:")
    print(lr_grid_search.best_params_)
    lr_clf = LogisticRegression(random_state=42,
                                penalty = lr_grid_search.best_params_['penalty'],
                                C=lr_grid_search.best_params_['C'],
                                tol=lr_grid_search.best_params_['tol'],
                                fit_intercept=
                                lr_grid_search.best_params_['fit_intercept'],
                                warm_start=lr_grid_search.best_params_['warm_start'])
    lr_clf.fit(x_train, y_train)
    y_pred_test = lr_clf.predict(x_test)
    print('Test Accuracy score: ', accuracy_score(y_test, y_pred_test))
    print('Test F1 score: ', f1_score(y_test, y_pred_test, average='weighted'))
    print('Recall score: ', recall_score(y_test, y_pred_test, average='weighted'))
    cm_val = confusion_matrix(y_test, y_pred_test)
    cm_labels = ['No', 'Yes']
    plot_confusion_matrix(cm_val, cm_labels, title="Logistic Regression")
