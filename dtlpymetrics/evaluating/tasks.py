import os
import json
import logging
import dtlpy as dl

from ..dtlpy_scores import ScoreType, Score
from ..scoring import calc_task_item_score
from ..utils.dl_helpers import get_scores_by_annotator, cleanup_annots_by_score

logger = logging.getLogger('scoring-and-metrics')


def get_consensus_agreement(item: dl.Item,
                            task: dl.Task,
                            agreement_config: dict,
                            progress: dl.Progress = None) -> dl.Item:
    """
    Determine whether annotators agree on annotations for a given item.
    :param item: dl.Item
    :param task: dl.Task
    :param agreement_config: dict that needs 3 keys: "agreement_threshold", "keep_only_best", and "fail_keep_all"
    :param progress: dl.Progress (optional)
    :return: dl.Item
    """
    agree_threshold = agreement_config.get("agree_threshold", 0.5)
    keep_only_best = agreement_config.get("keep_only_best", False)
    fail_keep_all = agreement_config.get("fail_keep_all", True)

    logger.info(f"Running consensus agreement using task {task.name} with ID {task.id}")
    logger.info(f"Configurations: agreement threshold = {agree_threshold}, "
                f"upon agreement pass, keep only best annotations: {keep_only_best}, "
                f"upon agreement fail keep all annotations: {fail_keep_all}")

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
