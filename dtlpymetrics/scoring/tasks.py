import logging
import json
import os

import dtlpy as dl
import numpy as np

from typing import List
from ..dtlpy_scores import Score, ScoreType, Scores
from ..utils import mean_or_default, add_score_context, check_if_video, calculate_annotation_score

logger = logging.getLogger('scoring-and-metrics')


def calc_task_score(task: dl.Task,
                    score_types=None) -> dl.Task:
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
                calc_task_item_score(item=item, task=task, score_types=score_types)
            else:
                logger.info(f'Item {item.id} is not complete, skipping scoring')
                continue

    return task


def calc_task_item_score(item: dl.Item,
                         task: dl.Task,
                         score_types=None,
                         upload=True) -> dl.Item:
    """
    Create scores for items in a task. This is the main function for creating score entities

    In the case of qualification and honeypot, the first set of annotations is considered the reference set.
    In the case of consensus, annotations are compared twice-- once as a reference set, and once as a test set.
    :param item: dl.Item entity (optional)
    :param task: dl.Task entity (optional)
    :param score_types: list of ScoreTypes to calculate (e.g. [ScoreType.ANNOTATION_IOU, ScoreType.ANNOTATION_LABEL]) (optional)
    :param upload: bool, default True means scores will be uploaded to the platform (optional)
    :return: item
    """
    logger.info(f'Starting scoring for item: {item.id} and task: {task.id}')
    if task.metadata['system'].get('consensusTaskType') == 'consensus':
        task_type = 'consensus'
    elif task.metadata['system'].get('consensusTaskType') in ['qualification', 'honeypot']:
        task_type = 'testing'
    else:
        raise ValueError(f'Task type is not suitable for scoring.')

    ###################################
    # collect assignments for sorting #
    ###################################
    assignments = task.assignments.list()

    is_quality_task = _check_task_type(task_type=task_type,
                                       task=task,
                                       item=item)
    if is_quality_task is False:
        logging.info('Item was not annotated via quality task. No scores were created.')
    else:
        [assignments_by_id, assignments_by_annotator] = _sort_assignments(task_type=task_type,
                                                                          item=item,
                                                                          assignments=assignments)
        # if no assignments are associated with this item
        if len(assignments_by_id) == 0:
            raise ValueError(
                f'No assignments found for task {task.id} and item {item.id}. Please check that the task was properly '
                f'configured and completed.')

        #########################################
        # sort annotations and calculate scores #
        #########################################
        annotations = item.annotations.list()
        annotators_list = [assignment.annotator for assignment in assignments_by_id.values()]
        logger.info(f'Starting scoring for assignments: {annotators_list}')

        is_video = check_if_video(item=item)
        if is_video is True:  # video items
            annotations_by_frame = _split_video_to_frames(annotations=annotations,
                                                          item=item,
                                                          task=task,
                                                          task_type=task_type,
                                                          assignments_by_id=assignments_by_id)
            all_scores = get_video_scores(annotations_by_frame=annotations_by_frame,
                                          assignments_by_annotator=assignments_by_annotator,
                                          task=task,
                                          item=item,
                                          score_types=score_types,
                                          task_type=task_type)
        else:  # image items
            annots_by_assignment = _sort_annotations(task=task,
                                                     item=item,
                                                     task_type=task_type,
                                                     annotations=annotations,
                                                     assignments_by_id=assignments_by_id)
            all_scores = get_image_scores(annots_by_assignment=annots_by_assignment,
                                          assignments_by_annotator=assignments_by_annotator,
                                          task=task,
                                          item=item,
                                          score_types=score_types,
                                          task_type=task_type)

        # overall item score is an average of all overall annotation scores
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
        debug_path = os.environ.get('SCORES_DEBUG_PATH', None)
        if debug_path is not None:
            upload = False
            save_dir = debug_path
        else:
            save_dir = os.path.join(os.getcwd(), '../.dataloop')

        if upload is True:
            logger.info(f'About to delete all scores with context item ID: {item.id} and task ID: {task.id}')
            dl_scores = Scores(client_api=dl.client_api)
            dl_scores.delete(context={'itemId': item.id,
                                      'taskId': task.id})
            dl_scores = dl_scores.create(all_scores)
            logger.info(f'Uploaded {len(dl_scores)} scores to platform.')
        else:
            logger.info(f'Saving scores locally, {save_dir}')
            save_filepath = os.path.join(save_dir, task.id, f'{item.id}.json')
            os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
            scores_json = list()
            for score in all_scores:
                scores_json.append(score.to_json())
            with open(save_filepath, 'w', encoding='utf-8') as f:
                json.dump(scores_json, f, ensure_ascii=False, indent=4)

            logger.info(f'SAVED score to: {save_filepath}')
    return item


def _check_task_type(task_type,
                     task: dl.Task,
                     item: dl.Item) -> bool:
    run_fxn = False
    if task_type == 'consensus':
        consensus_assignment = task.metadata['system']['consensusAssignmentId']
        all_item_refs = item.metadata['system']['refs']
        if consensus_assignment is not None:
            for ref_dict in all_item_refs:
                if ref_dict['id'] == consensus_assignment:
                    run_fxn = True
    else:
        if item.dir.startswith('/.consensus/'):
            run_fxn = True

    return run_fxn


def _sort_assignments(task_type: str,
                      item: dl.Item,
                      assignments: list) -> list:
    if task_type == 'consensus':
        # get only assignments that had a consensus for this item:
        filters = dl.Filters(use_defaults=False)
        filters.add('spec.parentDatasetItemId', item.id)
        filters.add('dir', '/.consensus/*')
        asg_pages = item.dataset.items.list(filters=filters)
        consensus_assignments = list()
        for asg_items in asg_pages.all():
            refs = asg_items.metadata['system']['refs']
            for ref in refs:
                if ref.get('type') == 'assignment':
                    consensus_assignments.append(ref['id'])
    else:
        consensus_assignments = [asg.id for asg in assignments]

    # create lookup dictionaries getting assignments by id or annotator
    assignments_by_id = {}
    assignments_by_annotator = {}
    for assignment in assignments:
        if assignment.id in consensus_assignments:
            assignments_by_id[assignment.id] = assignment
            assignments_by_annotator[assignment.annotator] = assignment

    return [assignments_by_id, assignments_by_annotator]


def _sort_annotations(task: dl.Task,
                      item: dl.Item,
                      task_type: str,
                      annotations: List[dl.Annotation],
                      assignments_by_id: dict) -> dict:
    # group by some field (e.g. 'creator' or 'assignment id'), here we use assignment id
    annots_by_assignment = {assignment.annotator: [] for assignment in assignments_by_id.values()}
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

    # add in reference annotations if testing task
    if task_type == 'testing':
        # get all ref from the src item
        src_item = dl.items.get(item_id=item._src_item)
        annots_by_assignment['ref'] = src_item.annotations.list()
    return annots_by_assignment


def _split_video_to_frames(annotations: dl.AnnotationCollection,
                           item: dl.Item,
                           task: dl.Task,
                           task_type: str,
                           assignments_by_id: dict) -> dict:
    """
    Hidden function to split video annotations frame by frame and sort by assignment
    :param annotations:
    :param item:
    :param task:
    :param task_type:
    :param assignments_by_id:
    :return:
    """
    # sort all annotations by frame
    num_frames = item.metadata['system']['nb_frames']
    all_annotation_slices = dict()
    for f in range(num_frames):
        all_annotation_slices[f] = annotations.get_frame(frame_num=f)
    annotations_by_frame = {}

    # within each frame, sort all annotation slices to their corresponding assignment/annotator
    for frame, annotation_slices in all_annotation_slices.items():
        frame_annots_by_assignment = {assignment.annotator: [] for assignment in assignments_by_id.values()}
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
        # ref_annots_by_frame = get_annotations_from_frames(src_item.annotations.list())
        ref_annotations = src_item.annotations.list()
        num_frames = src_item.metadata['system']['nb_frames']
        for f in range(num_frames):
            annotations_by_frame[f]['ref'] = ref_annotations.get_frame(frame_num=f)
            # annotations_by_frame[frame]['ref'] = annotation_slices
    return annotations_by_frame


def get_image_scores(annots_by_assignment: dict,
                     assignments_by_annotator: dict,
                     task: dl.Task,
                     item: dl.Item,
                     score_types: list,
                     task_type: str) -> list:
    """
    Calculate scores for an image item

    :param annots_by_assignment: dict of annotations grouped by assignment
    :param assignments_by_annotator: dict of assignments groups by annotator
    :param task: dl.Task
    :param item: dl.Item
    :param score_types: list of score types to be calculated
    :param task_type: str, 'testing' or 'consensus'
    :return all_scores: list of all annotation and item scores

    """
    ####################
    # calculate scores #
    ####################
    labels = dl.recipes.get(task.recipe_id).ontologies.list()[0].labels_flat_dict.keys()

    # compare between each assignment and create Score entities
    all_scores = list()

    # do pairwise comparisons of each assignment for all annotations on the item
    for i_assignment, assignment_annotator_i in enumerate(annots_by_assignment):
        if task_type == "testing" and assignment_annotator_i != 'ref':
            # if "testing", compare only to ref
            continue
        for j_assignment, assignment_annotator_j in enumerate(annots_by_assignment):
            # don't compare a set to itself
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
                                                         ignore_labels=False,
                                                         ignore_attributes=True,
                                                         ignore_geometry=True,
                                                         match_threshold=0.01,
                                                         score_types=score_types)
            for score in pairwise_scores:
                if score.type == ScoreType.USER_CONFUSION:
                    updated_score = add_score_context(
                        score=score,
                        user_id=assignment_annotator_j,
                        task_id=task.id,
                        entity_id=assignment_annotator_j,
                        relative=assignment_annotator_i,
                        assignment_id=assignments_by_annotator[assignment_annotator_j].id,
                        item_id=item.id)
                else:
                    updated_score = add_score_context(
                        score=score,
                        user_id=assignment_annotator_j,
                        task_id=task.id,
                        assignment_id=assignments_by_annotator[assignment_annotator_j].id,
                        item_id=item.id)
                all_scores.append(updated_score)
    # accumulate label confusion for all compares
    confusion_scores = list()
    for i_score, score in reversed(list(enumerate(all_scores))):
        if score.type == ScoreType.LABEL_CONFUSION:
            confusion_scores.append(score)
            all_scores.pop(i_score)
    confusion_dict = dict()
    for score in confusion_scores:
        if score.entity_id not in confusion_dict:
            confusion_dict[score.entity_id] = dict()
        if score.relative not in confusion_dict[score.entity_id]:
            confusion_dict[score.entity_id][score.relative] = 0
        confusion_dict[score.entity_id][score.relative] += 1
    for entity_id, v in confusion_dict.items():
        for relative, count in v.items():
            all_scores.append(Score(type=ScoreType.LABEL_CONFUSION,
                                    value=count,
                                    entity_id=entity_id,  # assignee label
                                    relative=relative,
                                    task_id=task.id,
                                    item_id=item.id
                                    ))

    # mean over all ANNOTATION_OVERALL for each annotation id
    annotation_overalls = list()
    for i_score, score in reversed(list(enumerate(all_scores))):
        if score.type == ScoreType.ANNOTATION_OVERALL:
            annotation_overalls.append(score)
            all_scores.pop(i_score)

    unique_annotation_ids = np.unique([score.entity_id for score in annotation_overalls])
    for annotation_id in unique_annotation_ids:
        overalls = [score for score in annotation_overalls if score.entity_id == annotation_id]
        # this is a matching score between annotations to make it a probability we will add the current self match as 1
        # for instance, if we had [A,A,B], and the current is A, the overall probability is 2/3
        overalls_values = [s.value for s in overalls]
        overalls_values.append(1)  # the match to the current annotation, this will it the probability
        user_id = overalls[0].user_id
        assignment_id = overalls[0].assignment_id
        # add joint overall (single one for each annotation
        all_scores.append(Score(type=ScoreType.ANNOTATION_OVERALL,
                                value=mean_or_default(arr=overalls_values, default=0),
                                entity_id=annotation_id,
                                user_id=user_id,
                                task_id=task.id,
                                assignment_id=assignment_id,
                                item_id=item.id
                                ))
    return all_scores


def get_video_scores(annotations_by_frame: dict,
                     assignments_by_annotator: dict,
                     task: dl.Task,
                     item: dl.Item,
                     score_types: list,
                     task_type: str = 'testing'):
    """
    Create scores for a video item

    :param annotations_by_frame: dict
    :param assignments_by_annotator: dict
    :param task: dl.Task
    :param item: dl.Item
    :param score_types: list of scores
    :param task_type: str
    :return all_scores: list of all annotation and item scores
    """
    ####################
    # calculate scores #
    ####################
    all_scores_by_annotation = dict()

    for frame, annots_by_assignment in annotations_by_frame.items():
        # compare between each assignment and create Score entities
        frame_scores = list()

        # do pairwise comparisons of each assignment for all annotations on the item
        for i_assignment, assignment_annotator_i in enumerate(annots_by_assignment):
            if task_type == "testing" and assignment_annotator_i != 'ref':
                # if "testing", compare only to ref
                continue
            for j_assignment, assignment_annotator_j in enumerate(annots_by_assignment):
                # don't compare a set to itself
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

                updated_scores = []
                # update scores with context
                for score in pairwise_scores:
                    updated_score = add_score_context(score=score,
                                                      user_id=assignment_annotator_j,
                                                      task_id=task.id,
                                                      assignment_id=assignments_by_annotator[assignment_annotator_j].id,
                                                      item_id=item.id,
                                                      dataset_id=item.dataset.id)
                    updated_scores.append(updated_score)

                raw_annotation_scores = [score for score in updated_scores if score.type != ScoreType.LABEL_CONFUSION]
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
                                                      default=1),
                                entity_id=annotation_id,
                                task_id=task.id,
                                item_id=item.id,
                                dataset_id=item.dataset.id))
        all_scores.append(Score(type=ScoreType.ANNOTATION_LABEL,
                                value=mean_or_default(arr=[score.value for score in annotation_frame_scores if
                                                           score.type == ScoreType.ANNOTATION_LABEL.value],
                                                      default=1),
                                entity_id=annotation_id,
                                task_id=task.id,
                                item_id=item.id,
                                dataset_id=item.dataset.id))
        all_scores.append(Score(type=ScoreType.ANNOTATION_IOU,
                                value=mean_or_default(arr=[score.value for score in annotation_frame_scores if
                                                           score.type == ScoreType.ANNOTATION_IOU.value],
                                                      default=1),
                                entity_id=annotation_id,
                                task_id=task.id,
                                item_id=item.id,
                                dataset_id=item.dataset.id))
        all_scores.append(Score(type=ScoreType.ANNOTATION_ATTRIBUTE,
                                value=mean_or_default(arr=[score.value for score in annotation_frame_scores if
                                                           score.type == ScoreType.ANNOTATION_ATTRIBUTE.value],
                                                      default=1),
                                entity_id=annotation_id,
                                task_id=task.id,
                                item_id=item.id,
                                dataset_id=item.dataset.id))

    return all_scores
