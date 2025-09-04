import logging
import dtlpy as dl
import numpy as np
from collections import defaultdict
from typing import Dict, List, Union

from ..dtlpy_scores import ScoreType
from ..scoring import calc_task_item_score
from ..utils import get_scores_by_annotator, cleanup_annots_by_score

logger = logging.getLogger("scoring-and-metrics")


def get_consensus_agreement(item: dl.Item, task: dl.Task, agreement_config: dict) -> bool:
    """
    Determine whether annotators agree on annotations for a given item.
    :param item: dl.Item
    :param task: dl.Task
    :param agreement_config: dict that needs 3 keys: "agreement_threshold", "keep_only_best", and "fail_keep_all"
    :return: bool
    """
    agreement_threshold = agreement_config.get("agreement_threshold", 0.5)
    keep_only_best = agreement_config.get("keep_only_best", False)
    fail_keep_all = agreement_config.get("fail_keep_all", True)

    logger.info(f"Running consensus agreement using task {task.name} with ID {task.id}")
    logger.info(
        f"Configurations: agreement threshold = {agreement_threshold}, "
        f"upon agreement pass, keep only best annotations: {keep_only_best}, "
        f"upon agreement fail keep all annotations: {fail_keep_all}"
    )

    # get scores and convert to dl.Score
    all_scores = calc_task_item_score(task=task, item=item, upload=False)
    agreement = check_annotator_agreement(scores=all_scores, threshold=agreement_threshold)

    # determine node output action
    if agreement is True:
        logger.info(f'Consensus passed for item {item.id}')
        if keep_only_best is True:
            logger.info("Keeping the annotation with the highest score.")
            scores_by_annotator = get_scores_by_annotator(scores=all_scores)
            annot_scores = {key: sum(val) / len(val) for key, val, in scores_by_annotator.items()}
            # Get the annotator with the highest score
            max_score = max(annot_scores.values())
            best_annotator = None
            # Find the first key with the maximum value
            for key, value in annot_scores.items():
                if value == max_score:
                    best_annotator = key
                    break
            logger.info(f"Best annotator assignment ID: {best_annotator}")

            annots_to_keep = [
                score.entity_id
                for score in all_scores
                if (score.context.get('assignmentId') == best_annotator)
                and (score.type == ScoreType.ANNOTATION_OVERALL)
            ]
            logger.info(f"Annotations to keep: {annots_to_keep}")
            cleanup_annots_by_score(item=item, scores=all_scores, annots_to_keep=annots_to_keep, logger=logger)
    else:
        logger.info(f'Consensus failed for item {item.id}')
        if fail_keep_all is False:
            logger.info("Deleting all annotations.")
            cleanup_annots_by_score(item=item, scores=all_scores, annots_to_keep=None, logger=logger)

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
        raise ValueError("Threshold must be between 0 and 1. Please set a valid threshold.")
    # calculate agreement based on the average agreement across all annotators
    user_scores = [score.value for score in scores if score.type == ScoreType.USER_CONFUSION]
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
        raise ValueError("Threshold must be between 0 and 1. Please set a valid threshold.")
    # calculate unanimity based on whether each pair agrees
    agreement = True
    for score in scores:
        if score.type == ScoreType.USER_CONFUSION:
            if score.value >= threshold:
                continue
            else:
                agreement = False
    return agreement


def dynamic_consensus_agreement(
    item: dl.Item, task: dl.Task, min_agreers: int, iou_threshold: float = 0.5, agreement_threshold: float = 0.8
) -> bool:
    """
    Dynamic consensus agreement function for backend use to create dynamic consensus assignments.

    This function calculates item overall score agreement at runtime without retrieving scores from platform.
    Currently supports Item Overall Score agreement type only.

    :param item: dl.Item entity
    :param task: dl.Task entity
    :param min_agreers: minimum number of annotators required from the total annotator pool to calculate the item score
    :param iou_threshold: IoU threshold for annotations match scores (box, polygon, etc)
    :param agreement_threshold: item score threshold to pass agreement (0-1)
    :return: bool - True if consensus agreement passes, False otherwise
    """
    # Validate input parameters
    if min_agreers < 2:
        raise ValueError("Minimum number of annotators available to agree must be at least 2")
    if not (0 < iou_threshold <= 1):
        raise ValueError("IoU threshold must be between 0 and 1")
    if not (0 < agreement_threshold <= 1):
        raise ValueError("Agreement threshold must be between 0 and 1")

    # Calculate scores at runtime (no platform retrieval)
    all_scores = calc_task_item_score(task=task, item=item, upload=False)

    # Get unique annotators from the scores
    unique_annotators = set()
    for score in all_scores:
        if score.type == ScoreType.USER_CONFUSION and hasattr(score, 'user_id') and score.user_id:
            unique_annotators.add(score.user_id)

    agreement_passed = False
    if len(unique_annotators) >= min_agreers:
        # Get item overall score - this is already calculated in calc_task_item_score
        item_scores = [score.value for score in all_scores if score.type == ScoreType.ITEM_OVERALL]

        if item_scores:
            # Use the calculated item overall score (average of annotation overall scores)
            item_overall_score = item_scores[0]  # There should only be one ITEM_OVERALL score

            # Check if item score meets agreement threshold
            agreement_passed = item_overall_score >= agreement_threshold

            logger.info(
                f"Item overall score: {item_overall_score}, threshold: {agreement_threshold}, passed: {agreement_passed}"
            )
        else:
            logger.warning("No item overall scores found")
    else:
        logger.info(f"Insufficient annotators: found {len(unique_annotators)}, required {min_agreers}")

    return agreement_passed


def cohens_kappa(scores, annotator1_id: str, annotator2_id: str) -> float:
    """
    Calculate Cohen's kappa for agreement between two specific annotators.

    Cohen's kappa measures the agreement between two raters, accounting for the possibility
    of agreement occurring by chance. Values range from -1 to 1, where:
    - 1 indicates perfect agreement
    - 0 indicates no agreement beyond chance
    - Negative values indicate systematic disagreement

    :param scores: list of Scores
    :param annotator1_id: ID of first annotator (assignment ID or user ID)
    :param annotator2_id: ID of second annotator (assignment ID or user ID)
    :return: Cohen's kappa coefficient (-1 to 1)
    """
    # Extract annotation scores for each annotator
    annotator1_scores = {}
    annotator2_scores = {}

    # TODO: have an inbetween lookup table for matches and the corresponding consensus task annotation ID for each annotator
    for score in scores:
        if score.type == ScoreType.ANNOTATION_OVERALL:
            # Use assignment ID if available, otherwise use user ID
            annotator_id = score.context.get('assignmentId') or score.user_id
            annotation_id = score.entity_id

            if annotator_id == annotator1_id:
                annotator1_scores[annotation_id] = score.value
            elif annotator_id == annotator2_id:
                annotator2_scores[annotation_id] = score.value

    # Find common annotations rated by both annotators
    common_annotations = set(annotator1_scores.keys()) & set(annotator2_scores.keys())

    if len(common_annotations) == 0:
        raise ValueError("No common annotations found between the two annotators")

    # Convert scores to binary categories (agree/disagree) based on threshold
    # Use 0.5 as default threshold for agreement
    threshold = 0.5

    # Create agreement matrix
    agreements = []
    disagreements = []

    for annotation_id in common_annotations:
        score1 = annotator1_scores[annotation_id] >= threshold
        score2 = annotator2_scores[annotation_id] >= threshold

        if score1 == score2:
            agreements.append(1)
            disagreements.append(0)
        else:
            agreements.append(0)
            disagreements.append(1)

    # Calculate observed agreement
    po = sum(agreements) / len(agreements)

    # Calculate expected agreement by chance
    # Proportion of times each annotator says "pass"
    annotator1_passes = sum(1 for aid in common_annotations if annotator1_scores[aid] >= threshold)
    annotator2_passes = sum(1 for aid in common_annotations if annotator2_scores[aid] >= threshold)

    p1_pass = annotator1_passes / len(common_annotations)
    p2_pass = annotator2_passes / len(common_annotations)

    p1_fail = 1 - p1_pass
    p2_fail = 1 - p2_pass

    # Expected agreement by chance
    pe = (p1_pass * p2_pass) + (p1_fail * p2_fail)

    # Cohen's kappa
    if pe == 1.0:
        return 1.0  # Perfect agreement case

    kappa = (po - pe) / (1 - pe)
    return kappa


def fleiss_kappa(scores) -> float:
    """
    Calculate Fleiss' kappa for agreement among multiple annotators.

    Fleiss' kappa is a generalization of Scott's pi statistic for multiple raters.
    It measures the reliability of agreement between a fixed number of raters when
    assigning categorical ratings to a number of items or classifying items.

    Values range from 0 to 1, where:
    - 1 indicates perfect agreement
    - 0 indicates no agreement beyond chance
    - Values below 0 indicate systematic disagreement

    :param scores: list of Scores from multiple annotators
    :return: Fleiss' kappa coefficient (0 to 1)
    """
    # Group scores by annotation and annotator
    annotation_ratings = defaultdict(list)
    annotators = set()

    # Use 0.5 as threshold for binary classification (agree/disagree)
    threshold = 0.5

    for score in scores:
        if score.type == ScoreType.ANNOTATION_OVERALL:
            # Use assignment ID if available, otherwise use user ID
            annotator_id = score.context.get('assignmentId') or score.user_id
            annotation_id = score.entity_id

            # Convert score to binary rating
            rating = 1 if score.value >= threshold else 0

            annotation_ratings[annotation_id].append((annotator_id, rating))
            annotators.add(annotator_id)

    if len(annotators) < 2:
        raise ValueError("Fleiss' kappa requires at least 2 annotators")

    # Filter annotations that have ratings from multiple annotators
    valid_annotations = {
        ann_id: ratings
        for ann_id, ratings in annotation_ratings.items()
        if len(set(r[0] for r in ratings)) >= 2  # At least 2 different annotators
    }

    if len(valid_annotations) == 0:
        raise ValueError("No annotations found with ratings from multiple annotators")

    # Create rating matrix: annotations x categories (agree/disagree)
    n_annotations = len(valid_annotations)
    n_categories = 2  # Binary: agree (1) or disagree (0)
    n_raters = len(annotators)

    # Count matrix: each cell [i,j] contains count of raters who assigned category j to annotation i
    rating_matrix = np.zeros((n_annotations, n_categories))

    for i, (annotation_id, ratings) in enumerate(valid_annotations.items()):
        # Count ratings for each category
        category_counts = [0, 0]  # [disagree_count, agree_count]

        for annotator_id, rating in ratings:
            category_counts[rating] += 1

        rating_matrix[i] = category_counts

    # Calculate proportion of all assignments to each category
    p_j = np.sum(rating_matrix, axis=0) / (n_annotations * n_raters)

    # Calculate P_i (extent of agreement for annotation i)
    P_i = np.zeros(n_annotations)
    for i in range(n_annotations):
        r_ij = rating_matrix[i]
        # Number of raters who actually rated this annotation
        n_i = np.sum(r_ij)
        if n_i > 1:
            P_i[i] = (np.sum(r_ij * (r_ij - 1))) / (n_i * (n_i - 1))
        else:
            P_i[i] = 0  # Cannot calculate agreement with only one rater

    # Mean of P_i values
    P_bar = np.mean(P_i)

    # Expected agreement by chance
    P_e = np.sum(p_j**2)

    # Fleiss' kappa
    if P_e == 1.0:
        return 1.0  # Perfect agreement case

    kappa = (P_bar - P_e) / (1 - P_e)
    return max(0.0, kappa)  # Ensure non-negative result
