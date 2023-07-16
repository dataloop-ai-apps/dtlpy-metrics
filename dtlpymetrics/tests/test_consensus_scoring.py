# working script for checking the consensus functions

import dtlpy as dl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dtlpymetrics.scoring import calculate_task_score, create_task_item_score
from dtlpymetrics.dtlpy_scores import Score, Scores, ScoreType
import logging


def plot_matrix(item_title, filename, matrix_to_plot, axis_labels, item=None, local_path=None):
    import os
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # annotators matrix plot, per item
    mask = np.zeros_like(matrix_to_plot, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

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

    if item is not None:
        newax = fig.add_axes((0.8, 0.8, 0.2, 0.2), anchor='NE', zorder=-1)

        im = plt.imread(item.download())
        newax.imshow(im)
        newax.axis('off')

    if local_path is None:
        save_path = os.path.join(root_dir, '.output', filename)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
    else:
        plt.savefig(filename)
    plt.close()

    return True


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    dl.setenv('rc')
    # nir's tasks
    # item = dl.items.get(item_id='64a9b634a1198616d9dfc1bd')
    # task = dl.tasks.get(task_id='64a9b62e465fd3a73ef9f3fd')

    # consensus
    # item = dl.items.get(item_id='64aa74e3d99c9b255e3ced3b')
    # task = dl.tasks.get(task_id='64af242642cb1c6671e74b52')  # guy's task

    # qualification
    item = dl.items.get(item_id='64af4e6ab0095cf6b2e144f3')
    task = dl.tasks.get(task_id='64af4e686ddcb36a188c6fd2')  # guy's task

    # DEBUG
    task = dl.tasks.get(task_id='64abe4ff24d52d7748b8f0dd')  # qualification
    # task = dl.tasks.get(task_id='64abe6af24d52d73d3b8f0fb')  # honeypot
    # task = dl.tasks.get(task_id='64aa776da54a2acafd368370')  # consensus

    project = task.project
    # consensus_item = calculate_task_score(task=task)
    # create_task_item_score(task=task, item=item)
    calculate_task_score(task=task)

    print()
