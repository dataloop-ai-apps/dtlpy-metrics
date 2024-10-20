import os
import logging
import dtlpy as dl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dtlpymetrics.dtlpy_scores import Score, ScoreType
from typing import List


def check_if_video(item: dl.Item):
    """
    Check if item is a video
    :param item: dl.Item
    :return: True if item is video
    """
    if item.metadata.get('system', dict()):
        item_mimetype = item.metadata['system'].get('mimetype', None)
        is_video = 'video' in item_mimetype
    else:
        is_video = False
    return is_video


def add_score_context(score: Score,
                      relative=None,
                      user_id=None,
                      entity_id=None,
                      assignment_id=None,
                      task_id=None,
                      item_id=None,
                      dataset_id=None):
    """
    Add context to a score
    :param score: dl.Score
    :param relative: entity the score is compared to
    :param user_id: user or annotator who is being scored
    :param entity_id: dl entity being scored
    :param assignment_id: assignment id for the annotator's work to be scored
    :param task_id: task id for the annotator's work to be scored
    :param item_id: item id for the annotator's work to be scored
    :param dataset_id: dataset id for the annotator's work to be scored
    :return: dl.Score
    """
    if entity_id is not None:
        score.entity_id = entity_id
    if user_id is not None:
        score.user_id = user_id
    if relative is not None:
        score.relative = relative
    if assignment_id is not None:
        score.assignment_id = assignment_id
    if task_id is not None:
        score.task_id = task_id
    if item_id is not None:
        score.item_id = item_id
    if dataset_id is not None:
        score.dataset_id = dataset_id
    return score


def calculate_confusion_matrix_item(item: dl.Item,
                                    scores: List[Score],
                                    save_plot=True) -> pd.DataFrame:
    """
    Calculate confusion matrix from a set of label confusion scores
    :param item: dl.Item
    :param scores: list of scores
    :param save_plot: bool
    :return: confusion matrix as pd.DataFrame
    """
    scores_dl = []
    for score in scores:
        scores_dl.append(score)

    # ###############################
    # # create table of comparisons #
    # ###############################
    label_names = []
    for score in scores_dl:
        if score.type == ScoreType.LABEL_CONFUSION:
            if score.entity_id not in label_names:
                label_names.append(score.entity_id)
            if score.relative not in label_names:
                label_names.append(score.relative)

    conf_matrix = pd.DataFrame(index=label_names, columns=label_names)

    for score in scores_dl:
        if score.type == ScoreType.LABEL_CONFUSION:
            conf_matrix.loc[score.entity_id] = score.value
            conf_matrix.loc[score.entity_id, score.relative] = score.value

    conf_matrix.fillna(0, inplace=True)
    conf_matrix.rename(columns={None: 'unmatched'}, inplace=True)
    conf_matrix.rename(index={None: 'unmatched'}, inplace=True)
    label_names = ['unmatched' if label is None else label for label in label_names]

    if save_plot is True:
        if os.environ.get('SCORES_DEBUG_PATH', None) is not None:
            debug_path = os.environ.get('SCORES_DEBUG_PATH', None)

            plot_matrix(item_title=f'label confusion matrix {item.id}',
                        filename=os.path.join(debug_path, 'label_confusion',
                                              f'label_confusion_matrix_{item.id}.png'),
                        matrix_to_plot=conf_matrix,
                        axis_labels=label_names)

        else:
            plot_matrix(item_title=f'label confusion matrix {item.id}',
                        filename=os.path.join('.dataloop', 'label_confusion',
                                              f'label_confusion_matrix_{item.id}.png'),
                        matrix_to_plot=conf_matrix,
                        axis_labels=label_names)

    return conf_matrix


def plot_matrix(item_title, filename, matrix_to_plot, axis_labels):
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


def cleanup_annots_by_score(scores, annots_to_keep=None, logger: logging.Logger = None):
    """
    Clean up annotations based on a list of scores to keep.
    :param scores: list of scores
    :param annots_to_keep: list of annotation ids to keep
    :param logger: logging.Logger
    :return: None
    """

    annotations_to_delete = []
    for score in scores:
        if score.type == ScoreType.ANNOTATION_OVERALL:
            if score.entity_id in annots_to_keep:
                pass
            else:
                if score.entity_id not in annotations_to_delete:
                    annotations_to_delete.append(score.entity_id)

    if logger is not None:
        logger.info(f'Deleting annotations: {annotations_to_delete}')

    filters = dl.Filters(field='id', values=annotations_to_delete, operator=dl.FILTERS_OPERATIONS_IN)
    dl.annotations.delete(filters=filters)

    return


def get_scores_by_annotator(scores):
    """
    Function to return a dic with annotator name as key and assignment entity as value
    :param scores: list of scores
    :return scores_by_annotator: dict of scores organized by annotator
    """
    scores_by_annotator = dict()

    for score in scores:
        if score.type == ScoreType.ANNOTATION_OVERALL:
            if scores_by_annotator.get(score.context.get('assignmentId')) is None:
                scores_by_annotator[score.context.get('assignmentId')] = [score.value]
            else:
                scores_by_annotator[score.context.get('assignmentId')].append(score.value)

    return scores_by_annotator
