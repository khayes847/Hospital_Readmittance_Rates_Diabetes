import matplotlib.pyplot as plt
import pandas as pd

def visualize_y(y, y_train, y_test):
    fig, ax = plt.subplots(1,3, sharey=True, figsize=(12,3))
    fig.set_facecolor('lightgrey')
    ax[0].bar(y.unique(), height=list(y.value_counts()))
    ax[0].set_title("y")
    ax[1].bar(y.unique(), height=list(y_train.value_counts()))
    ax[1].set_title("y_train")
    ax[2].bar(y.unique(), height=list(y_test.value_counts()))
    ax[2].set_title("y_test")
    return

def explore_x(X, X_train, X_test):
    for key in X.describe().keys():
        a = X.describe()[key]
        b = X_train.describe()[key]
        c = X_test.describe()[key]

        a_title = key
        b_title = key + "_train"
        c_title = key + "_test"
        comparison = pd.DataFrame([a,b,c]).T
        comparison.columns=[a_title, b_title, c_title]
        print(comparison)
