import dtlpy as dl
from dtlpymetrics.dtlpy_scores import Score, ScoreType


def check_annotator_agreement(scores, threshold):
    """
    Check agreement between all annotators

    Scores are averaged across users and compared to the threshold. If the average score is above the threshold,
    the function returns True.
    :param scores: list of Scores
    :param threshold: float, 0-1
    :return: True if agreement is above threshold
    """
    # calculate agreement based on the average agreement across all annotators
    user_scores = [score.value for score in scores if score.type == ScoreType.USER_CONFUSION]
    if sum(user_scores) / len(user_scores) >= threshold:
        return True
    else:
        return False


def check_unanimous_agreement(scores, threshold=1):
    """
    Check unanimous agreement between all annotators above a certain threshold
    :param scores: list of Scores
    :param threshold: float, 0-1 threshold for agreement
    :return: True if all annotator pairs agree above threshold
    """
    # calculate unanimity based on whether each pair agrees
    for score in scores:
        if score.type == ScoreType.USER_CONFUSION:
            if score.value >= threshold:
                continue
            else:
                return False
    return True


def get_best_annotator_by_score(scores):
    """
    Get the best annotator scores for a given item
    :param scores: list of scores
    :return: assignmentId of the best annotator
    """
    scores_by_annotator = dict()

    for score in scores:
        if score.type == ScoreType.ANNOTATION_OVERALL:
            if scores_by_annotator.get(score.context.get('assignmentId')) is None:
                scores_by_annotator[score.context.get('assignmentId')] = [score.value]
            else:
                scores_by_annotator[score.context.get('assignmentId')].append(score.value)

    annot_scores = {key: sum(val) / len(val) for key, val, in scores_by_annotator.items()}
    best_annotator = annot_scores[max(annot_scores, key=annot_scores.get)]

    return best_annotator
