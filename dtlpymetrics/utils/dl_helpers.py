import logging
import dtlpy as dl
from dtlpymetrics.dtlpy_scores import Score, ScoreType


def check_if_video(item: dl.Item):
    """
    Check if item is a video
    :param item: dl.Item
    :return: True if item is video
    """
    if item.metadata.get('system', dict()):
        item_mimetype = item.metadata['system'].get('mimetype', None)
        is_video = 'video' in item_mimetype
    else:
        is_video = False
    return is_video


def add_score_context(score: Score,
                      relative=None,
                      user_id=None,
                      entity_id=None,
                      assignment_id=None,
                      task_id=None,
                      item_id=None,
                      dataset_id=None):
    """
    Add context to a score
    :param score: dl.Score
    :param relative: entity the score is compared to (optional)
    :param user_id: user or annotator who is being scored (optional)
    :param entity_id: dl entity being scored (optional)
    :param assignment_id: assignment id for the annotator's work to be scored (optional)
    :param task_id: task id for the annotator's work to be scored (optional)
    :param item_id: item id for the annotator's work to be scored (optional)
    :param dataset_id: dataset id for the annotator's work to be scored (optional)
    :return: dl.Score
    """
    if entity_id is not None:
        score.entity_id = entity_id
    if user_id is not None:
        score.user_id = user_id
    if relative is not None:
        score.relative = relative
    if assignment_id is not None:
        score.assignment_id = assignment_id
    if task_id is not None:
        score.task_id = task_id
    if item_id is not None:
        score.item_id = item_id
    if dataset_id is not None:
        score.dataset_id = dataset_id
    return score


def cleanup_annots_by_score(scores, annots_to_keep=None, logger: logging.Logger = None):
    """
    Clean up annotations based on a list of scores to keep.
    :param scores: list of scores
    :param annots_to_keep: list of annotation ids to keep (optional)
    :param logger: logging.Logger (optional)
    :return: True
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

    filters = dl.Filters(resource=dl.FiltersResource.ANNOTATION)
    filters.add(field='id', values=annotations_to_delete, operator=dl.FILTERS_OPERATIONS_IN)
    dl.annotations.delete(filters=filters)

    return True


def get_scores_by_annotator(scores) -> dict:
    """
    Function to return a dic with annotator name as key and assignment entity as value
    :param scores: list of scores
    :return scores_by_annotator: dict of scores organized by annotator
    """
    scores_by_annotator = dict()

    for score in scores:
        if score.type == ScoreType.ANNOTATION_OVERALL:
            if scores_by_annotator.get(score.context.get('assignmentId')) is None:
                scores_by_annotator[score.context.get('assignmentId')] = [score.value]
            else:
                scores_by_annotator[score.context.get('assignmentId')].append(score.value)

    return scores_by_annotator


def get_best_annotator_by_score(scores) -> int:
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
