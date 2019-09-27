"""
Collects functions for use in creating models.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import resample
from yellowbrick.classifier import ConfusionMatrix
from imblearn.over_sampling import SMOTE
import functions as f
pd.set_option('display.max_columns', None)


def train_test_split_vals(x_val, y_val, test_size=0.25):
    """Splits X and y datasets into training and testing sets"""
    x_train, x_test, y_train, y_test = train_test_split(x_val, y_val,
                                                        test_size=test_size,
                                                        random_state=23)
    return x_train, x_test, y_train, y_test


def dummy_binary(x_train, y_train, x_test, y_test):
    """Creates dummy regression and confusion matrix for binary data"""
    print("DUMMY REGRESSION")
    dummy = DummyClassifier(strategy='most_frequent',
                            random_state=23).fit(x_train, y_train)
    dummy_pred = dummy.predict(x_test)
    print('Test Accuracy score: ', accuracy_score(y_test, dummy_pred))
    print('Test F1 score: ', f1_score(y_test, dummy_pred))
    cm_val = ConfusionMatrix(dummy, classes=[0, 1])
    cm_val.fit(x_train, y_train)
    cm_val.score(x_test, y_test)
    cm_val.poof()


def reg_unbalanced(x_train, y_train, x_test, y_test):
    """Creates unbalanced regression and confusion matrix for binary data"""
    print("UNBALANCED REGRESSION")
    lr_clf = LogisticRegression(solver='liblinear', random_state=23)
    lr_clf.fit(x_train, y_train)
    y_pred_test = lr_clf.predict(x_test)
    print('Test Accuracy score: ', accuracy_score(y_test, y_pred_test))
    print('Test F1 score: ', f1_score(y_test, y_pred_test))
    cm_val = ConfusionMatrix(lr_clf, classes=[0, 1])
    cm_val.fit(x_train, y_train)
    cm_val.score(x_test, y_test)
    cm_val.poof()


def y_split_fitting(x_train, y_train):
    """Splits training data into majority and minority target results
    for over and undersampling"""
    training = f.concat(x_train, y_train)
    no_val = training[training['readmitted'] == 0]
    yes = training[training['readmitted'] == 1]
    return no_val, yes


def upsample(x_train, y_train, x_test, y_test):
    """Upsamples minority target variable, creates regression and
    confusion matrix"""
    print("UPSAMPLED REGRESSION")
    no_val, yes = y_split_fitting(x_train, y_train)
    yes_upsampled = resample(yes, replace=True, n_samples=len(no_val),
                             random_state=23)
    upsampled = pd.concat([no_val, yes_upsampled])
    y_train_new = upsampled.readmitted
    x_train_new = upsampled.drop('readmitted', axis=1)
    upsampled_lr = LogisticRegression(solver='liblinear')
    upsampled_lr.fit(x_train_new, y_train_new)
    upsampled_pred = upsampled_lr.predict(x_test)
    print('Test Accuracy score: ', accuracy_score(y_test, upsampled_pred))
    print('Test F1 score: ', f1_score(y_test, upsampled_pred))
    cm_val = ConfusionMatrix(upsampled_lr, classes=[0, 1])
    cm_val.fit(x_train_new, y_train_new)
    cm_val.score(x_test, y_test)
    cm_val.poof()


def downsample(x_train, y_train, x_test, y_test):
    """Downsamples majority target variable, creates regression and
    confusion matrix"""
    print("DOWNSAMPLED REGRESSION")
    no_val, yes = y_split_fitting(x_train, y_train)
    no_downsampled = resample(no_val, replace=True, n_samples=len(yes),
                              random_state=23)
    downsampled = pd.concat([no_downsampled, yes])
    y_train_new = downsampled.readmitted
    x_train_new = downsampled.drop('readmitted', axis=1)
    downsampled_lr = LogisticRegression(solver='liblinear')
    downsampled_lr.fit(x_train_new, y_train_new)
    downsampled_pred = downsampled_lr.predict(x_test)
    print('Test Accuracy score: ', accuracy_score(y_test, downsampled_pred))
    print('Test F1 score: ', f1_score(y_test, downsampled_pred))
    cm_val = ConfusionMatrix(downsampled_lr, classes=[0, 1])
    cm_val.fit(x_train_new, y_train_new)
    cm_val.score(x_test, y_test)
    cm_val.poof()


def smote(x_val, y_val, test_size=0.25):
    """Balances target variable with synthetic samples, creates
    regression and confusion matrix"""
    print("SMOTE REGRESSION")
    x_train, x_test, y_train, y_test = train_test_split_vals(
        x_val, y_val, test_size=test_size)
    sm_val = SMOTE(random_state=23, ratio=1.0)
    x_train, y_train = sm_val.fit_sample(x_train, y_train)
    smote_lr = LogisticRegression(solver='liblinear')
    smote_lr.fit(x_train, y_train)
    smote_pred = smote_lr.predict(x_test)
    print('Test Accuracy score: ', accuracy_score(y_test, smote_pred))
    print('Test F1 score: ', f1_score(y_test, smote_pred))
    cm_val = ConfusionMatrix(smote_lr, classes=[0, 1])
    cm_val.fit(x_train, y_train)
    cm_val.score(x_test, y_test)
    cm_val.poof()


def run_all_regressions(x_val, y_val, unbalanced=True,
                        sampling=True, smote_test=True):
    """Runs all regressions"""
    x_train, x_test, y_train, y_test = train_test_split_vals(x_val,
                                                             y_val,
                                                             test_size=0.25)
    dummy_binary(x_train, y_train, x_test, y_test)
    if unbalanced:
        reg_unbalanced(x_train, y_train, x_test, y_test)
    if sampling:
        upsample(x_train, y_train, x_test, y_test)
        downsample(x_train, y_train, x_test, y_test)
    if smote_test:
        smote(x_val, y_val, test_size=0.25)
