import dtlpy as dl
import logging
from dtlpymetrics.dtlpy_scores import Score, ScoreType

def cleanup_annots_by_score(scores, annots_to_keep=None, logger: logging.Logger = None):
    """
    Clean up annotations based on a list of scores to keep.

    @return:
    """

    annotations_to_delete = []
    for score in scores:
        if score.type == ScoreType.ANNOTATION_OVERALL:
            if score.entity_id in annots_to_keep:
                pass
            else:
                if score.entity_id not in annotations_to_delete:
                    annotations_to_delete.append(score.entity_id)

    if logger is not None:
        logger.info(f'Deleting annotations: {annotations_to_delete}')

    filters = dl.Filters(field='id', values=annotations_to_delete, operator=dl.FILTERS_OPERATIONS_IN)
    dl.annotations.delete(filters=filters)

    return


def get_scores_by_annotator(scores):
    """
    Function to return a dic with annotator name as key and assignment entity as value
    @param scores:
    @return:
    """
    scores_by_annotator = dict()

    for score in scores:
        if score.type == ScoreType.ANNOTATION_OVERALL:
            if scores_by_annotator.get(score.context.get('assignmentId')) is None:
                scores_by_annotator[score.context.get('assignmentId')] = [score.value]
            else:
                scores_by_annotator[score.context.get('assignmentId')].append(score.value)

    return scores_by_annotator



def check_annotator_agreement(scores, threshold):
    # calculate agreement based on the average agreement across all annotators
    user_scores = [score.value for score in scores if score.type == ScoreType.USER_CONFUSION]
    if sum(user_scores) / len(user_scores) >= threshold:
        return True
    else:
        return False


def check_unanimous_agreement(scores, threshold=1):
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

    @return: assignmentId of the best annotator
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

