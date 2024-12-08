import logging
import os
import json
import dtlpy as dl

from ..dtlpy_scores import ScoreType, Score
from ..scoring import calc_task_item_score
from ..utils.dl_helpers import get_scores_by_annotator, cleanup_annots_by_score

logger = logging.getLogger('scoring-and-metrics')


def get_consensus_agreement(item: dl.Item,
                            context: dl.Context,
                            task: dl.Task = None,
                            progress: dl.Progress = None,
                            **kwargs) -> dl.Item:
    """
    Determine whether annotators agree on annotations for a given item. Only available in pipelines.
    :param item: dl.Item
    :param context: dl.Context
    :param task: dl.Task (optional)
    :param progress: dl.Progress (optional)
    :return: dl.Item
    """
    logger.info("Running consensus agreement")
    if item is None:
        raise ValueError('No item provided, please provide an item.')
    if task is None:
        if context is None:
            raise ValueError('Must provide either task or context.')
        elif context.task is not None:
            task = context.task
        else:
            pipeline_id = context.pipeline_id
            task_node_id = None
            for node in reversed(context.pipeline_execution.nodes):
                if node.node_type == "task":
                    task_node_id = node.node_id
                    break
            if task_node_id is None:
                raise ValueError(f"Could not find task from pipeline, and task not provided.")
            # project = context.project
            filters = dl.Filters(resource=dl.FiltersResource.TASK)
            filters.add(field='metadata.system.nodeId', values=task_node_id)
            filters.add(field='metadata.system.pipelineId', values=pipeline_id)

            tasks = item.project.tasks.list(filters=filters)
            if tasks.items_count != 1:
                raise ValueError(f"Failed getting consensus task, found: {tasks.items_count} matches")
            task = tasks.items[0]
            logger.info(f"Found consensus task: {task.id}")

    if context is not None:
        node = context.node
        agree_threshold = node.metadata.get('customNodeConfig', dict()).get('threshold', 0.5)
        keep_only_best = node.metadata.get('customNodeConfig', dict()).get('consensus_pass_keep_best', False)
        fail_keep_all = node.metadata.get('customNodeConfig', dict()).get('consensus_fail_keep_all', True)
    else:
        raise ValueError('Context cannot be none.')
    logger.info(f"task: {task}")

    # get scores and convert to dl.Score
    calc_task_item_score(task=task, item=item, upload=False)
    saved_filepath = os.path.join(os.getcwd(), '../.dataloop', task.id, f'{item.id}.json')
    with open(saved_filepath, 'r') as f:
        scores_json = json.load(f)
    all_scores = [Score.from_json(_json=s) for s in scores_json]

    agreement = check_annotator_agreement(scores=all_scores, threshold=agree_threshold)

    # determine node output action
    if progress is not None:
        if agreement is True:
            progress.update(action='consensus passed')
            logger.info(f'Consensus passed for item {item.id}')
            if keep_only_best is True:
                scores_by_annotator = get_scores_by_annotator(scores=all_scores)
                annot_scores = {key: sum(val) / len(val) for key, val, in scores_by_annotator.items()}
                best_annotator = annot_scores[max(annot_scores, key=annot_scores.get)]
                annots_to_keep = [score.entity_id for score in all_scores if
                                  (score.context.get('assignmentId') == best_annotator) and (
                                          score.type == ScoreType.ANNOTATION_OVERALL)]

                cleanup_annots_by_score(scores=all_scores,
                                        annots_to_keep=annots_to_keep,
                                        logger=logger)
        else:
            progress.update(action='consensus failed')
            logger.info(f'Consensus failed for item {item.id}')
            if fail_keep_all is False:
                cleanup_annots_by_score(scores=all_scores,
                                        annots_to_keep=None,
                                        logger=logger)

    return item


def check_annotator_agreement(scores, threshold=1):
    """
    Check agreement between all annotators

    Scores are averaged across users and compared to the threshold. If the average score is above the threshold,
    the function returns True.
    :param scores: list of Scores
    :param threshold: float, 0-1 (optional)
    :return: True if agreement is above threshold
    """
    if threshold < 0 or threshold > 1:
        raise ValueError('Threshold must be between 0 and 1. Please set a valid threshold.')
    # calculate agreement based on the average agreement across all annotators
    user_scores = [score.value for score in scores if score.type == ScoreType.USER_CONFUSION]
    if sum(user_scores) / len(user_scores) >= threshold:
        return True
    else:
        return False


def check_unanimous_agreement(scores, threshold=1):
    """
    Check unanimous agreement between all annotators above a certain threshold

    Scores are averaged across users and compared to the threshold. If the average score is above the threshold,
    the function returns True.
    :param scores: list of Scores
    :param threshold: float, 0-1 (optional)
    :return: True if all annotator pairs agree above threshold
    """
    if threshold < 0 or threshold > 1:
        raise ValueError('Threshold must be between 0 and 1. Please set a valid threshold.')
    # calculate unanimity based on whether each pair agrees
    for score in scores:
        if score.type == ScoreType.USER_CONFUSION:
            if score.value >= threshold:
                continue
            else:
                return False
    return True
