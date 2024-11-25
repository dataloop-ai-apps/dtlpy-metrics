import logging

import dtlpy as dl
import pandas as pd

from .scoring import calc_task_item_score, calc_precision_recall
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
    :param score_types: list of ScoreType (optional)
    :param upload: bool flag to upload the scores to the platform (optional)
    :return: dl.Item
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
    :param item: dl.Item to calculate consensus agreement for
    :param context: dl.Context for the item
    :param task: dl.Task for the item (optional)
    :param progress: dl.Progress for the item
    :return: dl.Item
    """
    item = get_consensus_agreement(item=item,
                                   context=context,
                                   task=task,
                                   progress=progress,
                                   **kwargs)
    return item


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
    precision_recall_df = calc_precision_recall(dataset_id=dataset_id,
                                                model_id=model_id,
                                                iou_threshold=iou_threshold,
                                                method_type=method_type,
                                                each_label=each_label,
                                                n_points=n_points)
    return precision_recall_df
