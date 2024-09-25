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


def get_best_annotator_by_score(scores):
    """
    Get the best annotator scores for a given item

    @return:
    """
    # get all annotation scores by annotator
    # figure out who the best annotator is
    # return only their scores
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
