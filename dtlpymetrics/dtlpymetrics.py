## this is where all the wrapper functions that are exposed to the platform will go
# we want only one module
import logging

import dtlpy as dl
import pandas as pd

from .scoring import calc_precision_recall

dl.use_attributes_2()

scorer = dl.AppModule(name='Scoring and metrics app',
                      description='Functions for calculating scores and metrics and tools for evaluating with them.'
                      )
logger = logging.getLogger('scoring-and-metrics')

scores_debug = True


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
    :param method_type: str method for calculating precision and recall. Options are: every_point and n_point_interpolated
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
