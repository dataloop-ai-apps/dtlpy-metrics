import logging

import dtlpy as dl
import pandas as pd

from dtlpymetrics.scoring import calc_task_item_score, calc_precision_recall
from dtlpymetrics.evaluating import get_consensus_agreement

logger = logging.getLogger('scoring-and-metrics')

class Scorer(dl.BaseServiceRunner):
    """
    Scorer class for scoring and metrics.
    Functions for calculating scores and metrics and tools for evaluating with them.
    """
    @staticmethod
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

    @staticmethod
    def consensus_agreement(item: dl.Item,
                            context: dl.Context,
                            progress: dl.Progress,
                            task: dl.Task = None) -> dl.Item:
        """
        Calculate consensus agreement for a quality task item.
        This is a wrapper function for get_consensus_agreement for use in pipelines.
        :param item: dl.Item for which to calculate consensus agreement
        :param context: dl.Context for the item
        :param task: dl.Task for the item (optional)
        :param progress: dl.Progress for the item
        :return: dl.Item
        """
        if item is None:
            raise ValueError('No item provided, please provide an item.')
        if context is None:
            raise ValueError('Must provide pipeline context.')
        if task is None and context.task is not None:
            task = context.task
        if task is None:
            # context task may still be none
            pipeline_id = context.pipeline_id
            task_node_id = None
            for node in reversed(context.pipeline_execution.nodes):
                if node.node_type == "task":
                    task_node_id = node.node_id
                    break
            if task_node_id is None:
                raise ValueError(f"Could not find task from pipeline, and task not provided.")
            filters = dl.Filters(resource=dl.FiltersResource.TASK)
            filters.add(field='metadata.system.nodeId', values=task_node_id)
            filters.add(field='metadata.system.pipelineId', values=pipeline_id)

            tasks = item.project.tasks.list(filters=filters)
            if tasks.items_count != 1:
                raise ValueError(f"Failed getting consensus task, found: {tasks.items_count} matches")
            task = tasks.items[0]

        agreement_config = dict()
        node = context.node
        agreement_config['agree_threshold'] = node.metadata.get('customNodeConfig', dict()).get('threshold', 0.5)
        agreement_config['keep_only_best'] = node.metadata.get('customNodeConfig', dict()).get('consensus_pass_keep_best', False)
        agreement_config['fail_keep_all'] = node.metadata.get('customNodeConfig', dict()).get('consensus_fail_keep_all', True)

        item = get_consensus_agreement(item=item,
                                       task=task,
                                       agreement_config=agreement_config,
                                       progress=progress)
        return item

    @staticmethod
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
