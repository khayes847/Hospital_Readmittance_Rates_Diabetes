"""
Collects functions for use in creating models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.utils import resample
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.classifier import ConfusionMatrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
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


def smote(x_train, y_train, x_test, y_test):
    """Balances target variable with synthetic samples, creates
    regression and confusion matrix"""
    print("SMOTE REGRESSION")
    sm_val = SMOTE(random_state=42, ratio=1.0)
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
    return x_train, y_train


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


def dummy_classifier(x_train, y_train, x_test, y_test, metric_dicts):
    """Runs dummy classifier as baseline model"""
    dummy = DummyClassifier(strategy='most_frequent',
                            random_state=22424).fit(x_train, y_train)
    y_pred = dummy.predict(x_test)
    display_confusion_matrix(dummy, x_train, y_train, x_test, y_test, y_pred)
    display_roc_curve(dummy, x_train, y_train, x_test, y_test)
    performance_metrics = metrics_to_metric_dictionary('Dummy', dummy, x_train,
                                                       y_train, x_test, y_test, y_pred)
    metric_dicts.update(performance_metrics)
    return metric_dicts


def display_confusion_matrix(model, x_train, y_train,
                             x_test, y_test, y_pred):
    """Displays confusion matrix"""
    cm_val = ConfusionMatrix(model, classes=[0,1], fontsize=18, )
    cm_val.fit(x_train, y_train)
    cm_val.score(x_test, y_test)
    cm_val.poof()
    print('Accuracy: {:.4}'.format(accuracy_score(y_test, y_pred)))
    print('F1 Score: {:.4}'.format(f1_score(y_test, y_pred)))


def display_roc_curve(model, X_train, y_train, X_test, y_test):
    """Displays roc curve"""
    predictions = model.predict_proba(X_test)

    #print(roc_auc_score(y_test, predictions[:,1]))
    fpr, tpr, _ = roc_curve(y_test, predictions[:,1])

    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    plt.figure(figsize=(10,8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    print('AUC (Area Under Curve): {:.4}'.format(auc(fpr, tpr)))


def metrics_to_metric_dictionary(model_str, model, x_train, y_train,
                                 x_test, y_test, y_pred):
    """Produces metrics for accuracy and f1"""
    calc_accuracy = accuracy_score(y_test, y_pred)
    calc_f1_score = f1_score(y_test, y_pred)
    predictions = model.predict_proba(x_test)
    fpr, tpr, _ = roc_curve(y_test, predictions[:,1])
    calc_auc = auc(fpr, tpr)
    metrics = {model_str : {'Accuracy': calc_accuracy,
                            'F1 Score': calc_f1_score,
                             'AUC (Area Under Curve)': calc_auc}}
    return metrics


def decision_tree(x_train, y_train, x_test, y_test, metric_dicts):
    """Creates decision trees"""
    decisiontree = DecisionTreeClassifier(criterion='entropy', max_depth=10,
                                          max_features=25, random_state=22424)
    decisiontree.fit(x_train,y_train)
    y_pred = decisiontree.predict(x_test)
    display_confusion_matrix(decisiontree, x_train, y_train,
                             x_test, y_test, y_pred)
    display_roc_curve(decisiontree, x_train, y_train,
                      x_test, y_test)
    performance_metrics = metrics_to_metric_dictionary('Decision Tree',
                                                       decisiontree, x_train,
                                                       y_train, x_test,
                                                       y_test, y_pred)
    metric_dicts.update(performance_metrics)
    
    return metric_dicts


def decision_tree_first_three(x_train, y_train):
    """Shows first three branches of the decision tree"""
    decisiontree = DecisionTreeClassifier(criterion='entropy',
                                          random_state=22424, max_depth=3)
    decisiontree.fit(x_train,y_train)
    plt.figure(figsize=(30,30))
    tree.plot_tree(decisiontree, feature_names=x_train.columns,
                   filled=True, rounded=True)


def random_forest(x_train, y_train, x_test, y_test, metric_dicts):
    """Creates random forests"""
    forest = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    max_features=25, random_state=22424)
    forest.fit(x_train, y_train)
    y_pred = forest.predict(x_test)
    display_confusion_matrix(forest, x_train, y_train, x_test, y_test, y_pred)
    display_roc_curve(forest, x_train, y_train, x_test, y_test)
    performance_metrics = metrics_to_metric_dictionary('Random Forest',
                                                       forest, x_train, y_train,
                                                       x_test, y_test, y_pred)
    metric_dicts.update(performance_metrics)
    return metric_dicts


def logistic_regression(x_train, y_train, x_test, y_test, metric_dicts):
    """Creates logistic regression"""
    logreg = LogisticRegression(solver='lbfgs')
    model = logreg.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    display_confusion_matrix(logreg, x_train, y_train, x_test, y_test, y_pred)
    display_roc_curve(logreg, x_train, y_train, x_test, y_test)
    coefficients = list(zip(x_train.columns, list(logreg.coef_[0])))
    performance_metrics = metrics_to_metric_dictionary('Logistic Regression',
                                                       logreg, x_train, y_train,
                                                       x_test, y_test, y_pred)
    metric_dicts.update(performance_metrics)
    return metric_dicts


def k_nearest_neighbors(x_train, y_train, x_test, y_test, metric_dicts):
    """Runs K-nearest_neighbors"""
    knn=KNeighborsClassifier(n_neighbors = 100)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    display_confusion_matrix(knn, x_train, y_train, x_test, y_test, y_pred)
    display_roc_curve(knn, x_train, y_train, x_test, y_test)
    performance_metrics = metrics_to_metric_dictionary('K-Nearest Neighbors',
                                                       knn, x_train, y_train,
                                                       x_test, y_test, y_pred)
    metric_dicts.update(performance_metrics)
    return metric_dicts


def xgboost(x_train, y_train, x_test, y_test, metric_dicts):
    """Runs XGBoost on data"""
    xgboost = xgb.XGBClassifier()
    xgboost.fit(x_train, y_train)
    y_pred = xgboost.predict(x_test)
    display_confusion_matrix(xgboost, x_train, y_train, x_test, y_test, y_pred)
    display_roc_curve(xgboost, x_train, y_train, x_test, y_test)
    performance_metrics = metrics_to_metric_dictionary('XGBoost', xgboost,
                                                       x_train, y_train,
                                                       x_test, y_test, y_pred)
    metric_dicts.update(performance_metrics)
    return metric_dicts


def performance_metric_chart(metric_dicts):
    accuracy = []
    f1scores = []
    aucs = []
    algos = []
    for metric_dict in metric_dicts:
        accuracy.append(metric_dicts[metric_dict]['Accuracy'])
        f1scores.append(metric_dicts[metric_dict]['F1 Score'])
        aucs.append(metric_dicts[metric_dict]['AUC (Area Under Curve)'])
        algos.append(metric_dict)

    plt.figure(figsize=(15, 15))
    barWidth = 0.25
    accuracy = accuracy[0:]
    f1scores = f1scores[0:]
    aucs = aucs[0:]
    r1 = np.arange(len(accuracy))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    plt.bar(r1, accuracy, color='#7f6d5f', width=barWidth, edgecolor='white', label='Accuracy')
    plt.bar(r2, f1scores, color='#557f2d', width=barWidth, edgecolor='white', label='F1 Scores')
    plt.bar(r3, aucs, color='#2d7f5e', width=barWidth, edgecolor='white', label='Area Under Curve (AUC)')
    plt.xlabel('Supervised Learning Models', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(accuracy))], algos)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
