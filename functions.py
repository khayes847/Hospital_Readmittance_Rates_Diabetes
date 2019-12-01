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
from sklearn.metrics import auc, precision_recall_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


def standardize_train_test_split(x_val, y_val, ts=.25, rs=42):
    """Creates stratified train-test split, and standardizes X data."""
    x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=ts,
                                                        random_state=rs, stratify=y_val)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
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
    thresh = cm_val.max()/2
    for i, j in itertools.product(range(cm_val.shape[0]),
                                  range(cm_val.shape[1])):
        plt.text(j, i, cm_val[i, j], horizontalalignment="center",
                 color="white" if cm_val[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True \nlabel', rotation=0)
    plt.xlabel('Predicted label')
    plt.show()


def scores(y_test, y_pred_test):
    """Prints Recall, Accuracy, and F1 Score"""
    print('Recall score: ', recall_score(y_test, y_pred_test))
    print('Test Accuracy score: ', accuracy_score(y_test, y_pred_test))
    print('Test F1 score: ', f1_score(y_test, y_pred_test))


def plot_pr_curve(model, x_test, y_test, title='Precision-Recall Curve'):
    """Plots Precision-Recall Curve"""
    model_probs = (model.predict_proba(x_test))[:,1]
    model_precision, model_recall, _ = precision_recall_curve(y_test, model_probs)
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(model_recall, model_precision, marker='.', label='Model')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.show()
    model_auc = auc(model_recall, model_precision)
    print(f'AUC: {model_auc}')


def scores_cm_pr(model, x_test, y_test, y_pred_test, cm_title):
    """Returns model scores, confusion matrix, precision-recall curve"""
    scores(y_test, y_pred_test)
    cm_val = confusion_matrix(y_test, y_pred_test)
    cm_labels = ['Not', 'Return']
    plot_confusion_matrix(cm_val, cm_labels, title=(f'{cm_title} Confusion Matrix'))
    plot_pr_curve(model, x_test, y_test, title=(f'{cm_title} Precision-Recall Curve'))


def dummy_regression(x_train, x_test, y_train, y_test):
    """Creates dummy logistic regression"""
    dummy = DummyClassifier(strategy='most_frequent', random_state=42).fit(x_train, y_train)
    dummy_pred = dummy.predict(x_test)
    scores(y_test, dummy_pred)
    cm_val = confusion_matrix(y_test, dummy_pred)
    cm_labels = ['Not', 'Return']
    plot_confusion_matrix(cm_val, cm_labels, title="Dummy Regression")


def log_gridsearch(x_train, y_train, x_test, y_test):
    """Conducts gridsearch for logistic regression, performs regression, and 
    creates confusion matrix and precision-recall curve."""
    lr_clf = LogisticRegression(random_state=42)
    lr_param_grid = {'penalty': ['l1', 'l2'],
                     'C': [.5, 1, 3, 5],
                     'tol': [0.00005, 0.0001, 0.00015, 0.0002],
                     'fit_intercept': [True, False],
                     'warm_start': [True, False]
                    }
    lr_grid_search = GridSearchCV(lr_clf, lr_param_grid, cv=3, n_jobs=-2,
                                  scoring='recall', return_train_score=True)
    lr_grid_search.fit(x_train, y_train)
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
    y_probs = (lr_clf.predict_proba(x_test))[:, 1]
    y_pred_test = lr_clf.predict(x_test)
    scores_cm_pr(lr_clf, x_test, y_test, y_pred_test, cm_title='Logistic Regression')


def forests_gridsearch(x_train, y_train, x_test, y_test):
    """Conducts gridsearch for random forests with bootstrapping, performs regression,
    and creates confusion matrix and precision-recall curve."""
    rf_clf = RandomForestClassifier(random_state=42, bootstrap=True)
    rf_param_grid = {'criterion': ['gini', 'entropy'],
                     'max_depth': [None, 2, 4, 6],
                     'min_samples_split': [2, 3, 4, 5],
                     'min_samples_leaf': [1, 2, 3, 4],
                     'warm_start': [True, False]}
    rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=3, n_jobs=-2,
                                  scoring='recall', return_train_score=True)
    rf_grid_search.fit(x_train, y_train)
    print("Best Parameter Combination Found During Grid Search:")
    print(rf_grid_search.best_params_)
    rf_clf = RandomForestClassifier(random_state = 42,
                                    bootstrap =
                                    True,
                                    criterion =
                                    rf_grid_search.best_params_['criterion'],
                                    max_depth =
                                    rf_grid_search.best_params_['max_depth'],
                                    min_samples_split =
                                    rf_grid_search.best_params_['min_samples_split'],
                                    min_samples_leaf =
                                    rf_grid_search.best_params_['min_samples_leaf'],
                                    warm_start =
                                    rf_grid_search.best_params_['warm_start'])
    rf_clf.fit(x_train, y_train)
    y_pred_test = rf_clf.predict(x_test)
    scores_cm_pr(rf_clf, x_test, y_test, y_pred_test, cm_title='Random Forests')


def adaboost_gridsearch(x_train, y_train, x_test, y_test):
    """Conducts gridsearch for adaboost, performs regression,
    and creates confusion matrix and precision-recall curve."""
    adaboost_clf = AdaBoostClassifier(random_state=42)
    adaboost_param_grid = {'n_estimators': [50, 60, 80, 100],
                           'learning_rate': [1.6, 1.4, 1.2, 1, 0.8]
                          }
    adaboost_grid_search = GridSearchCV(adaboost_clf, adaboost_param_grid, cv=3,
                                        n_jobs=-2, scoring='recall',
                                        return_train_score=True)
    adaboost_grid_search.fit(x_train, y_train)
    print("Best Parameter Combination Found During Grid Search:")
    print(adaboost_grid_search.best_params_)
    adaboost_clf = AdaBoostClassifier(random_state=42,
                                      n_estimators =
                                      adaboost_grid_search.best_params_['n_estimators'],
                                      learning_rate =
                                      adaboost_grid_search.best_params_['learning_rate'],
                                      )
    adaboost_clf.fit(x_train, y_train)
    y_pred_test = adaboost_clf.predict(x_test)
    scores_cm_pr(adaboost_clf, x_test, y_test, y_pred_test, cm_title='Adaboost')


def gboost_gridsearch(x_train, y_train, x_test, y_test):
    """Conducts gridsearch for adaboost, performs regression,
    and creates confusion matrix and precision-recall curve."""
    gboost_clf = GradientBoostingClassifier(random_state=42, validation_fraction=0.2,
                                            n_iter_no_change=5, tol=0.01)
    gboost_param_grid = {'n_estimators': [100, 300, 500],
                         'min_samples_split': [2, 3, 4],
                         'min_samples_leaf': [1, 2, 3],
                         'max_depth': [3, 5, 7],
                         'warm_start': [True, False]
                        }
    gboost_grid_search = GridSearchCV(gboost_clf, gboost_param_grid, cv=3,
                                      n_jobs=-2, scoring='recall',
                                      return_train_score=True)
    gboost_grid_search.fit(X_train, y_train)
    print("Best Parameter Combination Found During Grid Search:")
    print(gboost_grid_search.best_params_)
    gboost_clf = GradientBoostingClassifier(random_state=42,
                                            n_estimators =
                                            gboost_grid_search.best_params_['n_estimators'],
                                            min_samples_split =
                                            gboost_grid_search.best_params_['min_samples_split'],
                                            min_samples_leaf =
                                            gboost_grid_search.best_params_['min_samples_leaf'],
                                            max_depth =
                                            gboost_grid_search.best_params_['max_depth'],
                                            warm_start =
                                            gboost_grid_search.best_params_['warm_start'],
                                            validation_fraction=0.2,
                                            n_iter_no_change=5,
                                            tol=0.01
                                            )
    gboost_clf.fit(x_train, y_train)
    y_pred_test = gboost_clf.predict(x_test)
    scores_cm_pr(gboost_clf, x_test, y_test, y_pred_test, cm_title='Gradient Boosting')
