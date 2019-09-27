"""
This file contains assorted functions.
"""

import pandas as pd
import matplotlib as plt


def concat(x_val, y_val):
    """Concats given X and y datasets"""
    data = pd.concat([x_val, y_val], axis=1)
    return data


def split_xy(data):
    """Splits data along X and target"""
    x_val = data.drop(columns=['readmitted'])
    y_val = data.readmitted
    return x_val, y_val


def remove_repeats_deaths(x_val, y_val):
    """Removes repeated visits based on order. Removes
    patients discharged to hospice care or expired"""
    data = concat(x_val, y_val)
    data2 = data.sort_index()
    data3 = data2.loc[~(data2.duplicated(subset=['patient_nbr']))]
    discharge_list = [11, 13, 14, 19, 20, 21]
    data4 = data3.loc[~(data3.discharge_disposition_id.isin(discharge_list))]
    data5 = data4.drop(columns=['patient_nbr'])
    x_val_new, y_val_new = split_xy(data5)
    return x_val_new, y_val_new


def visualize_y(y_val, y_train, y_test):
    """Visualizes split data"""
    fig, ax_val = plt.subplots(1, 3, sharey=True, figsize=(12, 3))
    fig.set_facecolor('lightgrey')
    ax_val[0].bar(y_val.unique(), height=list(y_val.value_counts()))
    ax_val[0].set_title("y")
    ax_val[1].bar(y_val.unique(), height=list(y_train.value_counts()))
    ax_val[1].set_title("y_train")
    ax_val[2].bar(y_val.unique(), height=list(y_test.value_counts()))
    ax_val[2].set_title("y_test")
