import logging
import dtlpy as dl

from ..dtlpy_scores import ScoreType
from ..scoring import calc_task_item_score

logger = logging.getLogger("scoring-and-metrics")


def get_consensus_agreement(
    item: dl.Item, task: dl.Task, agreement_config: dict
) -> bool:
    """
    Determine whether annotators agree on annotations for a given item.
    :param item: dl.Item
    :param task: dl.Task
    :param agreement_config: dict that needs 3 keys: "agreement_threshold", "keep_only_best", and "fail_keep_all"
    :return: bool
    """
    agree_threshold = agreement_config.get("agree_threshold", 0.5)
    keep_only_best = agreement_config.get("keep_only_best", False)
    fail_keep_all = agreement_config.get("fail_keep_all", True)

    logger.info(f"Running consensus agreement using task {task.name} with ID {task.id}")
    logger.info(
        f"Configurations: agreement threshold = {agree_threshold}, "
        f"upon agreement pass, keep only best annotations: {keep_only_best}, "
        f"upon agreement fail keep all annotations: {fail_keep_all}"
    )

    # get scores and convert to dl.Score
    all_scores = calc_task_item_score(task=task, item=item, upload=False)
    agreement = check_annotator_agreement(scores=all_scores, threshold=agree_threshold)

    return agreement


def check_annotator_agreement(scores, threshold: float = 1.0):
    """
    Check agreement between all annotators

    Scores are averaged across users and compared to the threshold. If the average score is above the threshold,
    the function returns True.
    :param scores: list of Scores
    :param threshold: float, 0-1 (optional)
    :return: True if agreement is above threshold
    """
    if threshold < 0 or threshold > 1:
        raise ValueError(
            "Threshold must be between 0 and 1. Please set a valid threshold."
        )
    # calculate agreement based on the average agreement across all annotators
    user_scores = [
        score.value for score in scores if score.type == ScoreType.USER_CONFUSION
    ]
    agreement = True if sum(user_scores) / len(user_scores) >= threshold else False
    return agreement


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
        raise ValueError(
            "Threshold must be between 0 and 1. Please set a valid threshold."
        )
    # calculate unanimity based on whether each pair agrees
    agreement = True
    for score in scores:
        if score.type == ScoreType.USER_CONFUSION:
            if score.value >= threshold:
                continue
            else:
                agreement = False
    return agreement
