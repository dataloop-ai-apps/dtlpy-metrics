import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(item_title, filename, matrix_to_plot, axis_labels):
    """
    Plot confusion matrix between annotator pairs
    :param item_title: title of the item
    :param filename: path to save plot
    :param matrix_to_plot: confusion matrix
    :param axis_labels: list of labels for the axis
    :return filename: path to saved plot
    """
    # annotators matrix plot, per item
    mask = np.zeros_like(matrix_to_plot, dtype=bool)
    # mask[np.triu_indices_from(mask)] = True

    sns.set(rc={'figure.figsize': (20, 10), 'axes.facecolor': 'white'}, font_scale=2)
    sns_plot = sns.heatmap(
        matrix_to_plot,
        annot=True,
        mask=mask,
        cmap='Blues',
        xticklabels=axis_labels,
        yticklabels=axis_labels,
        vmin=0,
        vmax=1,
    )
    sns_plot.set(title=item_title)
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation=270)

    fig = sns_plot.get_figure()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename)
    plt.close()

    return filename
