import logging

import dtlpy as dl
import pandas as pd

from .scoring import calc_task_item_score, create_model_score, calc_precision_recall
from .evaluating import get_consensus_agreement

dl.use_attributes_2()

scorer = dl.AppModule(name='Scoring and metrics app',
                      description='Functions for calculating scores and metrics and tools for evaluating with them.')

logger = logging.getLogger('scoring-and-metrics')

scores_debug = True


@scorer.add_function(display_name='Calculate task scores for quality tasks')
def create_task_item_score(item: dl.Item,
                           task: dl.Task = None,
                           context: dl.Context = None,
                           score_types=None,
                           upload=True):
    """
    Calculate scores for a quality task item. This is a wrapper function for _create_task_item_score.
    :param item: dl.Item
    :param task: dl.Task (optional) Task entity. If none provided, task will be retrieved from context.
    :param context: dl.Context (optional)
    :param score_types: list of ScoreType
    :param upload: bool

    """
    if item is None:
        raise ValueError('No item provided, please provide an item.')
    if task is None:
        if context is None:
            raise ValueError('Must provide either task or context.')
        else:
            task = context.task

    item = calc_task_item_score(item=item,
                                task=task,
                                score_types=score_types,
                                upload=upload)
    return item


@scorer.add_function(display_name='')
def consensus_agreement(item: dl.Item,
                        context: dl.Context,
                        task: dl.Task = None,
                        progress: dl.Progress = None,
                        **kwargs) -> dl.Item:
    """
    Calculate consensus agreement for a quality task item. This is a wrapper function for get_consensus_agreement.
    :param item:
    :param context:
    :param task:
    :param progress:
    :param kwargs:
    :return:
    """
    item = get_consensus_agreement(item=item,
                                   context=context,
                                   task=task,
                                   progress=progress,
                                   **kwargs)
    return item


@scorer.add_function(display_name='Create scores for model predictions on a dataset per annotation')
def model_score(dataset: dl.Dataset = None,
                model: dl.Model = None,
                filters: dl.Filters = None,
                ignore_labels=False,
                match_threshold=0.01,
                compare_types=None) -> dl.Model:
    """
    Creates scores for a set of model predictions compared against ground truth annotations. This is a wrapper function
    for _create_model_score.

    :param dataset:
    :param model:
    :param filters:
    :param ignore_labels:
    :param match_threshold:
    :param compare_types:
    :return:
    """
    model = create_model_score(dataset=dataset,
                               model=model,
                               filters=filters,
                               ignore_labels=ignore_labels,
                               match_threshold=match_threshold,
                               compare_types=compare_types)

    return model


@scorer.add_function(display_name='Calculate precision recall values for model predictions')
def precision_recall(dataset_id: str,
                     model_id: str,
                     iou_threshold=0.01,
                     method_type=None,
                     each_label=True,
                     n_points=None) -> pd.DataFrame:
    """
    Calculate precision recall values for model predictions, for a given metric threshold.
    :param dataset_id: str dataset ID
    :param model_id: str model ID
    :param iou_threshold: float Threshold for accepting matched annotations as a true positive
    :param method_type: str method for calculating precision and recall (i.e. every_point or n_point_interpolated)
    :param each_label: bool calculate precision recall for each one of the labels
    :param n_points: int number of points to interpolate in case of n point interpolation
    :return: dataframe with all the points to plot for the dataset and individual labels
    """
    df = calc_precision_recall(dataset_id=dataset_id,
                               model_id=model_id,
                               iou_threshold=iou_threshold,
                               method_type=method_type,
                               each_label=each_label,
                               n_points=n_points)
    return df
