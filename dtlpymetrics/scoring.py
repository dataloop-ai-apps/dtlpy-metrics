import logging
import os
import json
import dtlpy as dl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dtlpy import Item
from tqdm import tqdm
from typing import List, Union
from dtlpymetrics.metrics_utils import measure_annotations, all_compare_types, mean_or_default
from dtlpymetrics.dtlpy_scores import Score, Scores, ScoreType

dl.use_attributes_2()
results_columns = {'annotation_iou': 'geometry_score',
                   'annotation_label': 'label_score',
                   'annotation_attribute': 'attribute_score',
                   'annotation_overall': 'annotation_score'
                   }

scorer = dl.AppModule(name='Scoring and metrics function',
                      description='Functions for calculating scores between annotations.'
                      )
logger = logging.getLogger('scoring-and-metrics')

scores_debug = True


def check_if_video(item: dl.Item):
    if item.metadata.get('system', dict()):
        item_mimetype = item.metadata['system'].get('mimetype', None)
        is_video = 'video' in item_mimetype
    else:
        is_video = False
    return is_video


@scorer.add_function(display_name='Calculate task scores for quality tasks')
def calculate_task_score(task: dl.Task, score_types=None) -> dl.Task:
    """
    Calculate scores for all items in a quality task, based on the item scores from each assignment.

    :param task: dl.Task entity
    :param score_types: optional list of ScoreTypes to calculate (e.g. [ScoreType.ANNOTATION_IOU, ScoreType.ANNOTATION_LABEL])
    :return: dl.Task entity
    """
    # determine task type
    if task.metadata['system'].get('consensusTaskType') not in ['qualification', 'honeypot', 'consensus']:
        raise ValueError(f'Task type is not suitable for scoring')

    if task.metadata['system'].get('consensusTaskType') in ['honeypot', 'qualification']:
        # qualification and honeypot scoring is completed on cloned items for each assignee
        filters = dl.Filters()
        filters.add(field='hidden', values=True)  # return only the clones
        pages = task.get_items(filters=filters)

    else:  # for consensus
        # default filter for quality tasks is hidden = True, so this hides the clones and returns only original items
        filters = dl.Filters()
        filters.add(field='hidden', values=False)
        pages = task.get_items(filters=filters, get_consensus_items=True)

    for item in pages.all():
        all_item_tasks = item.metadata['system']['refs']
        for item_task_dict in all_item_tasks:
            if item_task_dict['id'] != task.id:
                continue
            # for testing tasks, check if the item is complete via metadata
            elif item_task_dict.get('metadata', None) is None:
                continue
            elif item_task_dict.get('metadata').get('status', None) in ['completed', 'consensus_done']:
                if check_if_video(item) is False:
                    create_task_item_score(item=item, task=task, score_types=score_types)
                else:
                    create_task_video_score(item=item, task=task, score_types=score_types)
            else:
                logger.info(f'Item {item.id} is not complete, skipping scoring')
                continue

    return task


# def create_comparison_scores(annots_by_assignment: dict,
#                              task_type,
#                              score_types,
#                              task,
#                              assignments_by_annotator,
#                              item,
#                              confusion_by_label):
#     """
#     Compares sets of annotations from assignments in pairs.
#
#     @param annots_by_assignment: dict with assignment id as key and list of annotations as value
#     @param task_type: the task type (testings vs consensus)
#     @param score_types: the score types to include when calculating scores
#     @param task: dl.Task entity
#     @param assignments_by_annotator: dict lookup of assignments by annotator name
#     @param item: dl.Item to be scored
#     @param confusion_by_label: empty dict with all possible label confusion combinations
#     @return:
#     """
#     compared_scores = list()
#
#     # do pairwise comparisons of each assignment for all annotations on the item
#     for i_assignment, assignment_annotator_i in enumerate(annots_by_assignment):
#         if task_type == "testing" and assignment_annotator_i != 'ref':
#             # if "testing", compare only to ref
#             continue
#         for j_assignment, assignment_annotator_j in enumerate(annots_by_assignment):
#             # dont compare a set to itself
#             if i_assignment == j_assignment:
#                 continue
#             # skip ref in inner loop
#             if assignment_annotator_j == 'ref':
#                 continue
#             logger.info(
#                 f'Comparing assignee: {assignment_annotator_i!r} with assignee: {assignment_annotator_j!r}')
#             annot_collection_1 = annots_by_assignment[assignment_annotator_i]
#             annot_collection_2 = annots_by_assignment[assignment_annotator_j]
#             # score types that can be returned: ANNOTATION_IOU, ANNOTATION_LABEL, ANNOTATION_ATTRIBUTE
#             pairwise_scores = calculate_annotation_score(annot_collection_1=annot_collection_1,
#                                                          annot_collection_2=annot_collection_2,
#                                                          ignore_labels=True,
#                                                          match_threshold=0.01,
#                                                          score_types=score_types)
#
#             # update scores with context
#             for score in pairwise_scores:
#                 score.user_id = assignment_annotator_j
#                 score.task_id = task.id
#                 score.assignment_id = assignments_by_annotator[assignment_annotator_j].id
#                 score.item_id = item.id
#
#             raw_annotation_scores = [score for score in pairwise_scores if score.type != ScoreType.LABEL_CONFUSION]
#             confusion_scores = [score for score in pairwise_scores if score.type == ScoreType.LABEL_CONFUSION]
#
#             # calc general label confusion for the entire annotation collection/item
#             for score in confusion_scores:
#                 if score.entity_id not in confusion_by_label:
#                     confusion_by_label[score.entity_id] = dict()
#                 if score.relative not in confusion_by_label[score.entity_id]:
#                     confusion_by_label[score.entity_id][score.relative] = 0
#                 confusion_by_label[score.entity_id][score.relative] += score.value
#
#             # calc overall annotation
#             user_annotation_overalls = list()
#             for annotation in annot_collection_2:  # go over all annotations from the "test" set
#                 single_annotation_scores = mean_or_default(arr=[score.value
#                                                                 for score in raw_annotation_scores
#                                                                 if score.entity_id == annotation.id],
#                                                            default=1)
#                 # ANNOTATION_OVERALL
#                 user_annotation_overalls.append(single_annotation_scores)
#                 annotation_overall = Score(type=ScoreType.ANNOTATION_OVERALL,
#                                            value=single_annotation_scores,
#                                            entity_id=annotation.id,
#                                            task_id=task.id,
#                                            item_id=item.id,
#                                            user_id=assignment_annotator_j,
#                                            dataset_id=item.dataset.id)
#                 compared_scores.append(annotation_overall)
#
#             # calc user confusion
#             user_confusion_score = Score(type=ScoreType.USER_CONFUSION,
#                                          value=mean_or_default(arr=user_annotation_overalls,
#                                                                default=1),
#                                          entity_id=assignment_annotator_j,
#                                          user_id=assignment_annotator_j,
#                                          relative=assignment_annotator_i,  # this can be "ref"
#                                          task_id=task.id,
#                                          item_id=item.id,
#                                          dataset_id=item.dataset.id)
#             compared_scores.append(user_confusion_score)
#             compared_scores.extend(raw_annotation_scores)
#
#     # label confusion at the item level
#     for label_a, rest in confusion_by_label.items():
#         for label_b, value in rest.items():
#             item_confusion_score = Score(type=ScoreType.LABEL_CONFUSION,
#                                          value=value,
#                                          entity_id=label_a,
#                                          relative=label_b,
#                                          task_id=task.id,
#                                          item_id=item.id,
#                                          dataset_id=item.dataset.id)
#             compared_scores.append(item_confusion_score)
#     return compared_scores


@scorer.add_function(display_name='Create scores for image items in a quality task')
def create_task_item_score(item: dl.Item = None,
                           task: dl.Task = None,
                           context: dl.Context = None,
                           score_types=None) -> dl.Item:
    """
    Create scores for items in a task.

    In the case of qualification and honeypot, the first set of annotations is considered the reference set.
    In the case of consensus, annotations are compared twice-- once as a reference set, and once as a test set.
    :param item: dl.Item entity (optional)
    :param task: dl.Task entity (optional)
    :param context: dl.Context entity that includes references to associated entities (optional)
    :param score_types: list of ScoreTypes to calculate (e.g. [ScoreType.ANNOTATION_IOU, ScoreType.ANNOTATION_LABEL]) (optional)
    :return: item
    """
    ####################################
    # collect assignments for grouping #
    ####################################
    if item is None:
        raise KeyError('No dataset provided, please provide a dataset.')
    if task is None:
        if context is None:
            raise ValueError('Must provide either task or context.')
        else:
            task = context.task
    assignments = task.assignments.list()

    # create lookup dictionaries getting assignments by id or annotator
    assignments_by_id = {assignment.id: assignment for assignment in assignments}
    assignments_by_annotator = {assignment.annotator: assignment for assignment in assignments}

    if task.metadata['system'].get('consensusTaskType') == 'consensus':
        task_type = 'consensus'
    elif task.metadata['system'].get('consensusTaskType') in ['qualification', 'honeypot']:
        task_type = 'testing'
    else:
        raise ValueError(f'Task type is not suitable for scoring.')

    ################################
    # sort annotations by assignee #
    ################################
    annotations = item.annotations.list()
    annots_by_assignment = {assignment.annotator: [] for assignment in assignments}
    logger.info(f'Starting scoring for assignments: {list(annots_by_assignment.keys())}')

    # group by some field (e.g. 'creator' or 'assignment id'), here we use assignment id
    for annotation in annotations:
        # default is "ref"
        # TODO handle models
        assignment_id = annotation.metadata['system'].get('assignmentId', 'ref')
        task_id = annotation.metadata['system'].get('taskId', None)
        if task_id == task.id:
            assignment_annotator = assignments_by_id[assignment_id].annotator
            annots_by_assignment[assignment_annotator].append(annotation)
        else:
            # TODO comparing annotations from another task
            continue

    if task_type == 'testing':
        # get all ref from the src item
        src_item = dl.items.get(item_id=item._src_item)
        annots_by_assignment['ref'] = src_item.annotations.list()

    labels = dl.recipes.get(task.recipe_id).ontologies.list()[0].labels_flat_dict.keys()
    confusion_by_label = {l: {m: 0 for m in labels} for l in labels}
    # compare between each assignment and create Score entities
    # all_scores = create_comparison_scores(annots_by_assignment, task_type, score_types, task, assignments_by_annotator,
    #                                       item, confusion_by_label)
    all_scores = list()

    # do pairwise comparisons of each assignment for all annotations on the item
    for i_assignment, assignment_annotator_i in enumerate(annots_by_assignment):
        if task_type == "testing" and assignment_annotator_i != 'ref':
            # if "testing", compare only to ref
            continue
        for j_assignment, assignment_annotator_j in enumerate(annots_by_assignment):
            # dont compare a set to itself
            if i_assignment == j_assignment:
                continue
            # skip ref in inner loop
            if assignment_annotator_j == 'ref':
                continue
            logger.info(
                f'Comparing assignee: {assignment_annotator_i!r} with assignee: {assignment_annotator_j!r}')
            annot_collection_1 = annots_by_assignment[assignment_annotator_i]
            annot_collection_2 = annots_by_assignment[assignment_annotator_j]
            # score types that can be returned: ANNOTATION_IOU, ANNOTATION_LABEL, ANNOTATION_ATTRIBUTE
            pairwise_scores = calculate_annotation_score(annot_collection_1=annot_collection_1,
                                                         annot_collection_2=annot_collection_2,
                                                         ignore_labels=True,
                                                         match_threshold=0.01,
                                                         score_types=score_types)

            # update scores with context
            for score in pairwise_scores:
                score.user_id = assignment_annotator_j
                score.task_id = task.id
                score.assignment_id = assignments_by_annotator[assignment_annotator_j].id
                score.item_id = item.id

            raw_annotation_scores = [score for score in pairwise_scores if score.type != ScoreType.LABEL_CONFUSION]
            confusion_scores = [score for score in pairwise_scores if score.type == ScoreType.LABEL_CONFUSION]

            # calc general label confusion for the entire annotation collection/item
            for score in confusion_scores:
                if score.entity_id not in confusion_by_label:
                    confusion_by_label[score.entity_id] = dict()
                if score.relative not in confusion_by_label[score.entity_id]:
                    confusion_by_label[score.entity_id][score.relative] = 0
                confusion_by_label[score.entity_id][score.relative] += score.value

            # calc overall annotation
            user_annotation_overalls = list()
            for annotation in annot_collection_2:  # go over all annotations from the "test" set
                single_annotation_scores = mean_or_default(arr=[score.value
                                                                for score in raw_annotation_scores
                                                                if score.entity_id == annotation.id],
                                                           default=1)
                # ANNOTATION_OVERALL
                user_annotation_overalls.append(single_annotation_scores)
                annotation_overall = Score(type=ScoreType.ANNOTATION_OVERALL,
                                           value=single_annotation_scores,
                                           entity_id=annotation.id,
                                           task_id=task.id,
                                           item_id=item.id,
                                           user_id=assignment_annotator_j,
                                           dataset_id=item.dataset.id)
                all_scores.append(annotation_overall)

            # calc user confusion
            user_confusion_score = Score(type=ScoreType.USER_CONFUSION,
                                         value=mean_or_default(arr=user_annotation_overalls,
                                                               default=1),
                                         entity_id=assignment_annotator_j,
                                         user_id=assignment_annotator_j,
                                         relative=assignment_annotator_i,  # this can be "ref"
                                         task_id=task.id,
                                         item_id=item.id,
                                         dataset_id=item.dataset.id)
            all_scores.append(user_confusion_score)
            all_scores.extend(raw_annotation_scores)

    # label confusion at the item level
    for label_a, rest in confusion_by_label.items():
        for label_b, value in rest.items():
            item_confusion_score = Score(type=ScoreType.LABEL_CONFUSION,
                                         value=value,
                                         entity_id=label_a,
                                         relative=label_b,
                                         task_id=task.id,
                                         item_id=item.id,
                                         dataset_id=item.dataset.id)
            all_scores.append(item_confusion_score)

    # calc overall item score as an average of all overall annotation scores
    item_overall = [score.value for score in all_scores if score.type == ScoreType.ANNOTATION_OVERALL.value]
    item_score = Score(type=ScoreType.ITEM_OVERALL,
                       value=mean_or_default(arr=item_overall, default=1),
                       entity_id=item.id,
                       task_id=task.id,
                       item_id=item.id,
                       dataset_id=item.dataset.id)
    all_scores.append(item_score)

    #############################
    # upload scores to platform #
    #############################
    # clean previous scores before creating new ones
    logger.info(f'About to delete all scores with context item ID: {item.id} and task ID: {task.id}')
    dl_scores = Scores(client_api=dl.client_api)
    dl_scores.delete(context={'itemId': item.id,
                              'taskId': task.id})
    dl_scores = dl_scores.create(all_scores)
    logger.info(f'Uploaded {len(dl_scores)} scores to platform.')

    if os.environ.get('SCORES_DEBUG_PATH', None) is not None:
        debug_path = os.environ.get('SCORES_DEBUG_PATH', None)
        logger.debug('Saving scores locally')

        save_filepath = os.path.join(debug_path, task.id, f'{item.id}.json')
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
        scores_json = list()
        for score in all_scores:
            scores_json.append(score.to_json())
        with open(save_filepath, 'w', encoding='utf-8') as f:
            json.dump(scores_json, f, ensure_ascii=False, indent=4)

        logger.debug(f'SAVED score to: {save_filepath}')
    return item


def get_annotations_from_frames(annotations: List[dl.Annotation]):
    """
    Split video annotations by frame
    @param annotations: a list of annotations from a video item
    @return: annotations_by_frame, dict with frame number as key and list of annotations as value
    """
    annotations_by_frame = dict()
    for annotation in annotations:
        # then, in each frame, split by assignment (dict in dict)
        for frame, frame_annotation in annotation.frames.items():
            frame_annotation.annotation.set_frame(frame=frame)
            if frame not in annotations_by_frame:
                annotations_by_frame[frame] = [frame_annotation.annotation]
            else:
                annotations_by_frame[frame].append(frame_annotation.annotation)
    return annotations_by_frame


@scorer.add_function(display_name='Create scores for video items in a quality task')
def create_task_video_score(item: dl.Item = None,
                            task: dl.Task = None,
                            context: dl.Context = None,
                            score_types=None) -> dl.Item:
    """
    Create scores for a video item in a task.

    @param item:
    @param task:
    @param context:
    @param score_types:
    @return:
    """
    # check if the item is a video
    # if it's a video, then go through all the annotations and split them frame by frame
    # then, in each frame, split by assignment (dict in dict)
    # within each frame, do the pairwise comparison
    # once each frame's score is calculated, take the average score of all frames (between the two assignees)
    # return this score as the score for the item
    # the common function for this could be called, "compare annotation collections"

    if item is None:
        raise KeyError('No dataset provided, please provide a dataset.')
    if task is None:
        if context is None:
            raise ValueError('Must provide either task or context.')
        else:
            task = context.task
    assignments = task.assignments.list()

    # create lookup dictionaries getting assignments by id or annotator
    assignments_by_id = {assignment.id: assignment for assignment in assignments}
    assignments_by_annotator = {assignment.annotator: assignment for assignment in assignments}

    if task.metadata['system'].get('consensusTaskType') == 'consensus':
        task_type = 'consensus'
    elif task.metadata['system'].get('consensusTaskType') in ['qualification', 'honeypot']:
        task_type = 'testing'
    else:
        raise ValueError(f'Task type is not suitable for scoring.')

    ################################
    # sort annotations by assignee #
    ################################
    annotations = item.annotations.list()
    logger.info(f'Starting scoring for assignments')

    # sort all annotations by frame
    all_annotation_slices = get_annotations_from_frames(annotations)
    annotations_by_frame = {}

    # within each frame, sort all annotation slices to their corresponding assignment/annotator
    for frame, annotation_slices in all_annotation_slices.items():
        frame_annots_by_assignment = {assignment.annotator: [] for assignment in assignments}
        for annotation_slice in annotation_slices:
            # TODO compare annotations between models
            # default is "ref", if no assignment ID is found
            assignment_id = annotation_slice.metadata['system'].get('assignmentId', 'ref')
            task_id = annotation_slice.metadata['system'].get('taskId', None)
            if task_id == task.id:
                assignment_annotator = assignments_by_id[assignment_id].annotator
                frame_annots_by_assignment[assignment_annotator].append(annotation_slice)
            else:
                # TODO comparing annotations from another task
                continue
        annotations_by_frame[frame] = frame_annots_by_assignment

    # add in reference annotations if testing task
    if task_type == 'testing':
        # get all ref from the src item
        src_item = dl.items.get(item_id=item._src_item)
        ref_annots_by_frame = get_annotations_from_frames(src_item.annotations.list())
        for frame, annotation_slices in ref_annots_by_frame.items():
            # print(len(annotation_slices))
            annotations_by_frame[frame]['ref'] = annotation_slices

    ####################
    # calculate scores #
    ####################
    all_scores_by_annotation = dict()

    for frame, annots_by_assignment in annotations_by_frame.items():
        # within each frame, do the pairwise comparison
        frame_scores = list()

        # do pairwise comparisons of each assignment for all annotations on the item
        for i_assignment, assignment_annotator_i in enumerate(annots_by_assignment):
            if task_type == "testing" and assignment_annotator_i != 'ref':
                # if "testing", compare only to ref
                continue
            for j_assignment, assignment_annotator_j in enumerate(annots_by_assignment):
                # dont compare a set to itself
                if i_assignment == j_assignment:
                    continue
                # skip ref in inner loop
                if assignment_annotator_j == 'ref':
                    continue
                logger.info(
                    f'Comparing assignee: {assignment_annotator_i!r} with assignee: {assignment_annotator_j!r}')
                annot_collection_1 = annots_by_assignment[assignment_annotator_i]
                annot_collection_2 = annots_by_assignment[assignment_annotator_j]

                # score types that can be returned: ANNOTATION_IOU, ANNOTATION_LABEL, ANNOTATION_ATTRIBUTE
                pairwise_scores = calculate_annotation_score(annot_collection_1=annot_collection_1,
                                                             annot_collection_2=annot_collection_2,
                                                             ignore_labels=True,
                                                             include_confusion=False,
                                                             match_threshold=0.01,
                                                             score_types=score_types)

                # update scores with context
                for score in pairwise_scores:
                    score.user_id = assignment_annotator_j
                    score.task_id = task.id
                    score.assignment_id = assignments_by_annotator[assignment_annotator_j].id
                    score.item_id = item.id

                raw_annotation_scores = [score for score in pairwise_scores if score.type != ScoreType.LABEL_CONFUSION]
                frame_scores.extend(raw_annotation_scores)

                # calc overall annotation
                user_annotation_overalls = list()
                for annotation in annot_collection_2:  # go over all annotations from the "test" set
                    single_annotation_slice_score = mean_or_default(arr=[score.value
                                                                         for score in raw_annotation_scores
                                                                         if score.entity_id == annotation.id],
                                                                    default=1)
                    # overall slice score
                    user_annotation_overalls.append(single_annotation_slice_score)
                    annotation_overall = Score(type=ScoreType.ANNOTATION_OVERALL,
                                               value=single_annotation_slice_score,
                                               entity_id=annotation.id,
                                               task_id=task.id,
                                               item_id=item.id,
                                               user_id=assignment_annotator_j,
                                               dataset_id=item.dataset.id)
                    frame_scores.append(annotation_overall)

        for score in frame_scores:
            if score.entity_id not in all_scores_by_annotation:
                all_scores_by_annotation[score.entity_id] = list()
            all_scores_by_annotation[score.entity_id].append(score)

    # once each frame's score is calculated, take the average score of all frames
    all_scores = list()
    for annotation_id, annotation_frame_scores in all_scores_by_annotation.items():
        all_scores.append(Score(type=ScoreType.ANNOTATION_OVERALL,
                                value=mean_or_default(arr=[score.value for score in annotation_frame_scores if
                                                           score.type == ScoreType.ANNOTATION_OVERALL.value],
                                                      default=1)))
        all_scores.append(Score(type=ScoreType.ANNOTATION_LABEL,
                                value=mean_or_default(arr=[score.value for score in annotation_frame_scores if
                                                           score.type == ScoreType.ANNOTATION_LABEL.value],
                                                      default=1)))
        all_scores.append(Score(type=ScoreType.ANNOTATION_IOU,
                                value=mean_or_default(arr=[score.value for score in annotation_frame_scores if
                                                           score.type == ScoreType.ANNOTATION_IOU.value],
                                                      default=1)))
        all_scores.append(Score(type=ScoreType.ANNOTATION_ATTRIBUTE,
                                value=mean_or_default(arr=[score.value for score in annotation_frame_scores if
                                                           score.type == ScoreType.ANNOTATION_ATTRIBUTE.value],
                                                      default=1)))

    # calc overall video item score as an average of all frame scores
    item_overall = [score.value for score in all_scores if score.type == ScoreType.ANNOTATION_OVERALL.value]

    item_score = Score(type=ScoreType.ITEM_OVERALL,
                       value=mean_or_default(arr=item_overall, default=1),
                       entity_id=item.id,
                       task_id=task.id,
                       item_id=item.id,
                       dataset_id=item.dataset.id)
    all_scores.append(item_score)

    #############################
    # upload scores to platform #
    #############################
    # clean previous scores before creating new ones
    logger.info(f'About to delete all scores with context item ID: {item.id} and task ID: {task.id}')
    dl_scores = Scores(client_api=dl.client_api)
    dl_scores.delete(context={'itemId': item.id,
                              'taskId': task.id})
    dl_scores = dl_scores.create(all_scores)  # internal 500 error
    logger.info(f'Uploaded {len(dl_scores)} scores to platform.')

    if os.environ.get('SCORES_DEBUG_PATH', None) is not None:
        debug_path = os.environ.get('SCORES_DEBUG_PATH', None)
        logger.debug('Saving scores locally')

        save_filepath = os.path.join(debug_path, task.id, f'{item.id}.json')
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
        scores_json = list()
        for score in all_scores:
            scores_json.append(score.to_json())
        with open(save_filepath, 'w', encoding='utf-8') as f:
            json.dump(scores_json, f, ensure_ascii=False, indent=4)

        logger.debug(f'SAVED score to: {save_filepath}')

    return item


@scorer.add_function(display_name='Create scores for model predictions on a dataset per annotation')
def create_model_score(dataset: dl.Dataset = None,
                       model: dl.Model = None,
                       filters: dl.Filters = None,
                       ignore_labels=False,
                       match_threshold=0.01,
                       compare_types=None) -> dl.Model:
    """
    Measures scores for a set of model predictions compared against ground truth annotations.

    :param dataset: Dataset associated with the ground truth annotations
    :param filters: DQL Filter for retrieving the test items
    :param model: Model for evaluating predictions
    :param ignore_labels: bool, True means every annotation will be cross-compared regardless of label
    :param match_threshold: float, threshold for matching annotations
    :param compare_types: annotation types to compare
    :return:
    """

    if dataset is None:
        raise KeyError('No dataset provided, please provide a dataset.')
    if model is None:
        raise KeyError('No model provided, please provide a model.')
    if filters is None:
        items_list = list(dataset.items.list().all())
    else:
        items_list = list(dataset.items.list(filters=filters).all())
    if compare_types is None:
        compare_types = all_compare_types
    if not isinstance(compare_types, list):
        if compare_types not in model.output_type:  # TODO check this validation logic
            raise ValueError(
                f'Annotation type {compare_types} does not match model output type {model.output_type}')
        compare_types = [compare_types]

    annot_set_1 = []
    annot_set_2 = []

    if model.name is None:
        raise KeyError('No model name found for the second set of annotations, please provide model name.')
    if not items_list:
        raise KeyError('No items found in the dataset, please check the dataset and filters.')

    ########################################
    # Create list of item annotation lists #
    ########################################
    pbar = tqdm(items_list)
    for item in pbar:
        pbar.set_description(f'Loading annotations from items... ')
        item_annots_1 = []
        item_annots_2 = []
        for annotation in item.annotations.list():
            if annotation.metadata.get('user', {}).get('model') is None:
                item_annots_1.append(annotation)
            elif annotation.metadata['user']['model']['name'] == model.name:
                item_annots_2.append(annotation)
        annot_set_1.append(item_annots_1)
        annot_set_2.append(item_annots_2)

    #########################################################
    # Compare annotations and return concatenated dataframe #
    #########################################################
    all_results = pd.DataFrame()
    for i in range(len(items_list)):
        # compare annotations for each item
        if len(annot_set_1[i]) == 0 and len(annot_set_2[i]) == 0:
            continue
        else:
            results = measure_annotations(annotations_set_one=annot_set_1[i],
                                          annotations_set_two=annot_set_2[i],
                                          match_threshold=match_threshold,  # default 0.01 to get all possible matches
                                          ignore_labels=ignore_labels,
                                          compare_types=compare_types)
            for compare_type in compare_types:
                try:
                    results_df = results[compare_type].to_df()
                except KeyError:
                    continue
                results_df['item_id'] = [items_list[i].id] * results_df.shape[0]
                results_df['annotation_type'] = [compare_type] * results_df.shape[0]
                all_results = pd.concat([all_results, results_df],
                                        ignore_index=True)

    ###############################################
    # Save results to csv for IOU/label/attribute #
    ###############################################
    # TODO save via feature vectors when ready
    # file format "/.modelscores/modelId.csv"
    all_results['model_id'] = [model.id] * all_results.shape[0]
    all_results['dataset_id'] = [dataset.id] * all_results.shape[0]

    if not os.path.isdir(os.path.join(os.getcwd(), '.dataloop')):
        os.mkdir(os.path.join(os.getcwd(), '.dataloop'))
    scores_filepath = os.path.join(os.getcwd(), '.dataloop', f'{model.id}.csv')

    all_results.to_csv(scores_filepath, index=False)
    item = dataset.items.upload(local_path=scores_filepath,
                                remote_path=f'/.modelscores',
                                overwrite=True)
    logger.info(f'Successfully created model scores and saved as item {item.id}.')
    return model


def calculate_model_item_score(model_scores: pd.DataFrame):
    """
    Calculate scores for each item that a model predicts on

    @return:
    """

    pass


@scorer.add_function(display_name='Compare two annotation sets for scoring')
def calculate_annotation_score(annot_collection_1: Union[dl.AnnotationCollection, List[dl.Annotation]],
                               annot_collection_2: Union[dl.AnnotationCollection, List[dl.Annotation]],
                               ignore_labels=False,
                               include_confusion=True,
                               match_threshold=0.5,
                               compare_types=None,
                               score_types=None) -> List[Score]:
    """
    Creates Scores from comparing two annotation lists.

    The first annotation collection is considered the reference, and the second collection is the set for comparing.
    If we switch the order of the annotation collections, the scores remain the same but the user id context changes.

    :param annot_collection_1: dl.AnnotationCollection or list of annotations
    :param annot_collection_2: dl.AnnotationCollection or list of annotations
    :param ignore_labels: bool, True means every annotation will be cross-compared regardless of label classification
    :param match_threshold: float, threshold for considering two annotations a "match"
    :param compare_types: dl.AnnotationType entity or string for the annotation types to be compared
    :param score_types: dl.ScoreType entity or string for the score types to be calculated (e.g. "annotation_iou")
    :return: list of Score entities
    """
    if score_types is None:
        score_types = [ScoreType.ANNOTATION_LABEL, ScoreType.ANNOTATION_IOU, ScoreType.ANNOTATION_ATTRIBUTE]
    if compare_types is None:
        compare_types = all_compare_types
    if not isinstance(score_types, list):
        score_types = [score_types]

    #######################
    # compare annotations #
    #######################
    results = measure_annotations(
        annotations_set_one=annot_collection_1,
        annotations_set_two=annot_collection_2,
        compare_types=compare_types,
        ignore_labels=ignore_labels,
        match_threshold=match_threshold)

    all_results = pd.DataFrame()
    for compare_type in compare_types:
        try:
            results_df = results[compare_type].to_df()
            all_results = pd.concat([all_results, results_df])
        except KeyError:
            continue

    #########################
    # create score entities #
    #########################
    logger.info(f'Creating scores for types: {score_types}')
    annotation_scores = []
    for i, row in all_results.iterrows():
        for score_type in score_types:
            if row['second_id'] is None:
                continue

            annot_score = Score(type=score_type,
                                value=row[results_columns[score_type.value.lower()]],
                                entity_id=row['second_id'],
                                relative=row['first_id'])
            annotation_scores.append(annot_score)

    ##############################################
    # create label confusion scores for this set #
    ##############################################
    if include_confusion is True:
        if all_results.shape[0] > 0:

            label_confusion_set = all_results[['first_label', 'second_label']]
            label_confusion_set = label_confusion_set.fillna('unlabeled')

            label_confusion_summary = label_confusion_set.groupby(['first_label', 'second_label']).size().reset_index(
                name='counts')
            print(label_confusion_summary)
            for i, row in label_confusion_summary.iterrows():
                confusion_score = Score(type=ScoreType.LABEL_CONFUSION,
                                        value=row['counts'],
                                        entity_id=row['second_label'],  # assignee label
                                        relative=row['first_label'])  # ground truth label
                annotation_scores.append(confusion_score)

    return annotation_scores


# @scorer.add_function(display_name='Create label confusion matrix')
def calculate_confusion_matrix_item(item: dl.Item,
                                    scores: List[Score],
                                    save_plot=True) -> pd.DataFrame:
    """
    Calculate confusion matrix from a set of label confusion scores

    :return:
    """
    scores_dl = []
    for score in scores:
        scores_dl.append(Score.from_json(score))

    # ###############################
    # # create table of comparisons #
    # ###############################
    label_names = []
    for score in scores_dl:
        if score.type == ScoreType.LABEL_CONFUSION:
            if score.entity_id not in label_names:
                label_names.append(score.entity_id)
            if score.relative not in label_names:
                label_names.append(score.relative)

    conf_matrix = pd.DataFrame(index=label_names, columns=label_names)

    for score in scores_dl:
        if score.type == ScoreType.LABEL_CONFUSION:
            conf_matrix.loc[score.entity_id] = score.value
            conf_matrix.loc[score.entity_id, score.relative] = score.value

    conf_matrix.fillna(0, inplace=True)
    conf_matrix.rename(columns={None: 'unlabeled'}, inplace=True)
    conf_matrix.rename(index={None: 'unlabeled'}, inplace=True)
    label_names = ['unlabeled' if label is None else label for label in label_names]

    if save_plot is True:
        if os.environ.get('SCORES_DEBUG_PATH', None) is not None:
            debug_path = os.environ.get('SCORES_DEBUG_PATH', None)

            plot_matrix(item_title=f'label confusion matrix {item.id}',
                        filename=os.path.join(debug_path, 'label_confusion', f'label_confusion_matrix_{item.id}.png'),
                        matrix_to_plot=conf_matrix,
                        axis_labels=label_names)

        else:
            plot_matrix(item_title=f'label confusion matrix {item.id}',
                        filename=os.path.join('.dataloop', 'label_confusion', f'label_confusion_matrix_{item.id}.png'),
                        matrix_to_plot=conf_matrix,
                        axis_labels=label_names)

    return conf_matrix


def plot_matrix(item_title, filename, matrix_to_plot, axis_labels):
    # annotators matrix plot, per item
    mask = np.zeros_like(matrix_to_plot, dtype=bool)
    # mask[np.triu_indices_from(mask)] = True

    sns.set(rc={'figure.figsize': (20, 10),
                'axes.facecolor': 'white'},
            font_scale=2)
    sns_plot = sns.heatmap(matrix_to_plot,
                           annot=True,
                           mask=mask,
                           cmap='Blues',
                           xticklabels=axis_labels,
                           yticklabels=axis_labels,
                           vmin=0,
                           vmax=1)
    sns_plot.set(title=item_title)
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation=270)

    fig = sns_plot.get_figure()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename)
    plt.close()

    return filename


@scorer.add_function(display_name='Get model annotation scores dataframe from scores csv')
def get_model_scores_df(dataset: dl.Dataset, model: dl.Model) -> pd.DataFrame:
    """
    Retrieves the dataframe for all the scores for a given model on a dataset via a hidden csv file.
    :param dataset: Dataset where the model was evaluated
    :param model: Model entity
    :return: matched_annots_df: dataframe of all annotations in ground truth and model predictions
    """
    file_name = f'{model.id}.csv'
    local_path = os.path.join(os.getcwd(), '.dataloop', file_name)
    filters = dl.Filters(field='name', values=file_name)
    filters.add(field='hidden', values=True)
    pages = dataset.items.list(filters=filters)

    if pages.items_count > 0:
        for item in pages.all():
            item.download(local_path=local_path)
    else:
        raise ValueError(
            f'No matched annotations file found for model {model.id} on dataset {dataset.id}. Please evaluate model on the dataset first.')

    model_scores_df = pd.read_csv(local_path)
    return model_scores_df
