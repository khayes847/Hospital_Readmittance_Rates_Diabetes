import numpy as np
import matplotlib.pyplot as plt

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

    plt.figure(figsize=(15,15))

    # source: https://python-graph-gallery.com/11-grouped-barplot/
    # set width of bar
    barWidth = 0.25

    # set height of bar
    accuracy = accuracy[0:]
    f1scores = f1scores[0:]
    aucs = aucs[0:]

    # Set position of bar on X axis
    r1 = np.arange(len(accuracy))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, accuracy, color='#7f6d5f', width=barWidth, edgecolor='white', label='Accuracy')
    plt.bar(r2, f1scores, color='#557f2d', width=barWidth, edgecolor='white', label='F1 Scores')
    plt.bar(r3, aucs, color='#2d7f5e', width=barWidth, edgecolor='white', label='Area Under Curve (AUC)')

    # Add xticks on the middle of the group bars
    plt.xlabel('Supervised Learning Models', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(accuracy))], algos)

    # Set upper limit at 1 (all metrics are on a 0-1 scale)
    plt.ylim(0,1)

    # Create legend & Show graphic
    plt.legend()
    plt.show()
