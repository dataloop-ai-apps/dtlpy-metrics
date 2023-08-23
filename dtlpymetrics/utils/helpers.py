import os
import dtlpy as dl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dtlpymetrics.dtlpy_scores import Score
from typing import List


def check_if_video(item: dl.Item):
    if item.metadata.get('system', dict()):
        item_mimetype = item.metadata['system'].get('mimetype', None)
        is_video = 'video' in item_mimetype
    else:
        is_video = False
    return is_video


def add_score_context(score: Score,
                      annotation_id=None,
                      user_id=None,
                      assignment_id=None,
                      task_id=None,
                      item_id=None,
                      dataset_id=None):
    # update scores with context
    if annotation_id is not None:
        score.entity_id = annotation_id
    if user_id is not None:
        score.user_id = user_id
    if assignment_id is not None:
        score.assignment_id = assignment_id
    if task_id is not None:
        score.task_id = task_id
    if item_id is not None:
        score.item_id = item_id
    if dataset_id is not None:
        score.dataset_id = dataset_id
    return score


# @scorer.add_function(display_name='Plot annotators confusion matrix')
def plot_matrix(item_title, filename, matrix_to_plot, axis_labels):
    # annotators matrix plot, per item
    mask = np.zeros_like(matrix_to_plot, dtype=bool)
    # mask[np.triu_indices_from(mask)] = True

    sns.set(rc={'figure.figsize': (20, 10),
                'axes.facecolor': 'white'},
            font_scale=2)
    sns_plot = sns.heatmap(matrix_to_plot,
                           annot=True,
                           mask=mask,
                           cmap='Blues',
                           xticklabels=axis_labels,
                           yticklabels=axis_labels,
                           vmin=0,
                           vmax=1)
    sns_plot.set(title=item_title)
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation=270)

    fig = sns_plot.get_figure()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename)
    plt.close()

    return filename
