# working script for checking the consensus functions

import dtlpy as dl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dtlpymetrics.scoring import calculate_consensus_task_score
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
    # project = dl.projects.get('feature vectors')
    project = dl.projects.get('new scoring 6')
    # dataset = project.datasets.get('waterfowl')
    # create new scores for consensus task
    # consensus_task = dataset.tasks.get('pipeline consensus test (test tasks)')  # 644a307ae052f434dab98ff3

    consensus_task = project.tasks.get(task_id='64a523cad0c94374c31c332f')
    consensus_task = calculate_consensus_task_score(consensus_task)

    # print(scores.to_json())
    # dl_scores = scores.create([score])
    # print([d.id for d in dl_scores])
    #
