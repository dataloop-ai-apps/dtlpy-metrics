import dtlpy as dl
from dtlpymetrics.dtlpy_scores import Score, ScoreType


def get_annotator_agreement(scores, threshold):
    # check all user confusion scores aka agreement are above threshold
    for score in scores:
        if score.type == ScoreType.USER_CONFUSION:
            if score.value >= threshold:
                continue
            else:
                return False
    return True


def get_best_annotator_scores(assignments_by_annotator, scores):
    """
    Get the best annotator scores for a given item

    @return:
    """
    # TODO get best annotator scores
    a_score = dl.Score()
    best_scores = []

    # iterate through scores
    # group scores by annotator
    # find annotator score?

    return best_scores
