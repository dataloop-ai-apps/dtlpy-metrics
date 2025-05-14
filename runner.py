import logging
import dtlpy as dl
import pandas as pd

from dtlpymetrics.scoring import (
    calc_task_item_score,
    calc_precision_recall,
    calc_item_model_score,
)
from dtlpymetrics.evaluating import (
    get_consensus_agreement,
    get_model_agreement,
)

logger = logging.getLogger("scoring-and-metrics")


class Scorer(dl.BaseServiceRunner):
    """
    Scorer class for scoring and metrics.
    Functions for calculating scores and metrics and tools for evaluating with them.
    """

    def __init__(self):
        import dtlpymetrics
        import sys

        logger.info(f"This dtlpymetrics version is: {dtlpymetrics.__version__}")
        logger.info(f"This is the python executable: {sys.executable}")

    @staticmethod
    def create_task_item_score(
        item: dl.Item,
        task: dl.Task = None,
        context: dl.Context = None,
        score_types=None,
        upload=True,
    ) -> dl.Item:
        """
        Calculate scores for a quality task item. This is a wrapper function for calc_task_item_score.
        :param item: dl.Item
        :param task: dl.Task (optional) Task entity. If none provided, task will be retrieved from context.
        :param context: dl.Context (optional)
        :param score_types: list of ScoreType (optional)
        :param upload: bool flag to upload the scores to the platform (optional)
        :return: dl.Item
        """
        if item is None:
            raise ValueError("No item provided, please provide an item.")
        if task is None:
            if context is None:
                raise ValueError("Must provide either task or context.")
            else:
                task = context.task

        scores = calc_task_item_score(
            item=item, task=task, score_types=score_types, upload=upload
        )
        return item

    @staticmethod
    def get_previous_task_nodes(pipeline, start_node_id, previous_nodes):
        """
        Recursively collects previous nodes in the pipeline and stores them in previous_nodes.
        """
        for connection in pipeline.connections:
            connection: dl.PipelineConnection
            if connection.target.node_id in start_node_id:
                if connection.source.node_id not in previous_nodes:
                    node = pipeline.nodes.get(node_id=connection.source.node_id)
                    previous_nodes[connection.source.node_id] = node
                    if node.node_type == "task":
                        return connection.source.node_id
                    else:
                        return Scorer.get_previous_task_nodes(
                            pipeline, connection.source.node_id, previous_nodes
                        )

    @staticmethod
    def consensus_agreement(
        item: dl.Item, context: dl.Context, progress: dl.Progress, task: dl.Task = None
    ) -> dl.Item:
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
            raise ValueError("No item provided, please provide an item.")
        if context is None:
            raise ValueError("Must provide pipeline context.")
        if task is None and context.task is not None:
            task = context.task
        if task is None:
            # context task may still be none
            pipeline_id = context.pipeline_id
            pipeline = context.pipeline
            current_node_id = context.node.node_id

            previous_nodes = dict()
            logger.info(f"Finding task node recursively for pipeline: {pipeline_id}")
            task_node_id = Scorer.get_previous_task_nodes(
                pipeline=pipeline,
                start_node_id=current_node_id,
                previous_nodes=previous_nodes,
            )
            if task_node_id is None:
                raise ValueError(
                    f"Could not find task from pipeline, and task not provided."
                )
            filters = dl.Filters(resource=dl.FiltersResource.TASK)
            filters.add(field="metadata.system.nodeId", values=task_node_id)
            filters.add(field="metadata.system.pipelineId", values=pipeline_id)

            tasks = pipeline.project.tasks.list(filters=filters)
            if tasks.items_count != 1:
                raise ValueError(
                    f"Failed getting consensus task, found: {tasks.items_count} matches"
                )
            task = tasks.items[0]
        logger.info(f"Found task id: {task.id}")
        
        agreement_config = dict()
        node = context.node
        agreement_config["agree_threshold"] = node.metadata.get(
            "customNodeConfig", dict()
        ).get("threshold", 0.5)
        agreement_config["keep_only_best"] = node.metadata.get(
            "customNodeConfig", dict()
        ).get("consensus_pass_keep_best", False)
        agreement_config["fail_keep_all"] = node.metadata.get(
            "customNodeConfig", dict()
        ).get("consensus_fail_keep_all", True)

        agreement = get_consensus_agreement(
            item=item, task=task, agreement_config=agreement_config
        )

        if agreement is True:
            progress.update(action="consensus passed")
            logger.info(f"Consensus passed for item {item.id}")
        else:
            progress.update(action="consensus failed")
            logger.info(f"Consensus failed for item {item.id}")

        return item

    @staticmethod
    def precision_recall(
        dataset_id: str,
        model_id: str,
        iou_threshold=0.01,
        method_type=None,
        each_label=True,
        n_points=None,
    ) -> pd.DataFrame:
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
        precision_recall_df = calc_precision_recall(
            dataset_id=dataset_id,
            model_id=model_id,
            iou_threshold=iou_threshold,
            method_type=method_type,
            each_label=each_label,
            n_points=n_points,
        )
        return precision_recall_df

    @staticmethod
    def create_model_item_score(
        item: dl.Item, model: dl.Model, context: dl.Context = None, score_types=None
    ) -> dl.Item:
        """
        Calculate scores for a model's predictions on an item compared to ground truth annotations.
        This is a wrapper function for calc_item_model_score.
        :param item: dl.Item to score
        :param model: dl.Model whose predictions to evaluate
        :param context: dl.Context (optional)
        :param score_types: list of ScoreType (optional)
        :param upload: bool flag to upload the scores to the platform (optional)
        :return: dl.Item
        """
        if item is None:
            raise ValueError("No item provided, please provide an item.")
        if model is None:
            raise ValueError("No model provided, please provide a model.")

        scores = calc_item_model_score(
            item=item, model=model, score_types=score_types, upload=upload
        )
        return item

    @staticmethod
    def model_agreement(
        item: dl.Item,
        context: dl.Context,
        progress: dl.Progress,
        model: dl.Model = None,
    ) -> dl.Item:
        """
        Calculate agreement between model predictions and ground truth annotations.
        :param item: dl.Item to evaluate
        :param context: dl.Context for the item
        :param progress: dl.Progress for the item
        :param model: dl.Model to evaluate (optional)
        :return: dl.Item
        """
        if item is None:
            raise ValueError("No item provided, please provide an item.")
        if context is None:
            raise ValueError("Must provide pipeline context.")
        if model is None:
            if context.model is not None:
                model = context.model
            else:
                raise ValueError("Must provide either model or context with model.")

        agreement_config = dict()
        node = context.node
        agreement_config["agree_threshold"] = node.metadata.get(
            "customNodeConfig", dict()
        ).get("threshold", 0.5)
        agreement_config["keep_annots"] = node.metadata.get(
            "customNodeConfig", dict()
        ).get("model_keep_annots", False)

        agreement = get_model_agreement(
            item=item, model=model, agreement_config=agreement_config
        )

        # determine node output action
        if agreement is True:
            progress.update(action="model passed")
            logger.info(f"Model agreement passed for item {item.id}")
        else:
            progress.update(action="model failed")
            logger.info(f"Model agreement failed for item {item.id}")

        return item
