from yellowbrick.classifier import ConfusionMatrix
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, auc, roc_curve, roc_auc_score
import seaborn as sns


def display_confusion_matrix(model, X_train, y_train, X_test, y_test, y_pred):
    cm = ConfusionMatrix(model, classes=[0,1], fontsize=18, )
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    cm.poof()
    print('Accuracy: {:.4}'.format(accuracy_score(y_test, y_pred)))
    print('F1 Score: {:.4}'.format(f1_score(y_test, y_pred)))
    return

def display_roc_curve(model, X_train, y_train, X_test, y_test):
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
    return

def metrics_to_metric_dictionary(model_str, model, X_train, y_train, X_test, y_test, y_pred):
        calc_accuracy = accuracy_score(y_test, y_pred)
        calc_f1_score = f1_score(y_test, y_pred)
        predictions = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, predictions[:,1])
        calc_auc = auc(fpr, tpr)
        metrics = {model_str : {'Accuracy': calc_accuracy, 'F1 Score': calc_f1_score, 'AUC (Area Under Curve)': calc_auc}}
        return metrics
