import logging
import os
import json
import dtlpy as dl
import pandas as pd
from tqdm import tqdm
from typing import List
from dtlpymetrics.dtlpy_scores import Score, Scores, ScoreType
from dtlpymetrics import get_image_scores, get_video_scores
from dtlpymetrics.utils import check_if_video, plot_matrix, measure_annotations, all_compare_types, mean_or_default

dl.use_attributes_2()

scorer = dl.AppModule(name='Scoring and metrics function',
                      description='Functions for calculating scores between annotations.'
                      )
logger = logging.getLogger('scoring-and-metrics')

scores_debug = True


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
                create_task_item_score(item=item, task=task, score_types=score_types)
            else:
                logger.info(f'Item {item.id} is not complete, skipping scoring')
                continue

    return task


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

    #########################################
    # sort annotations and calculate scores #
    #########################################
    annotations = item.annotations.list()
    annotators_list = [assignment.annotator for assignment in assignments]
    logger.info(f'Starting scoring for assignments: {annotators_list}')

    is_video = check_if_video(item=item)
    if is_video is True:  # video items
        # sort all annotations by frame
        num_frames = item.metadata['system']['nb_frames']
        all_annotation_slices = dict()
        for f in range(num_frames):
            all_annotation_slices[f] = annotations.get_frame(frame_num=f)
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
            # ref_annots_by_frame = get_annotations_from_frames(src_item.annotations.list())
            ref_annotations = src_item.annotations.list()
            num_frames = src_item.metadata['system']['nb_frames']
            for f in range(num_frames):
                annotations_by_frame[f]['ref'] = ref_annotations.get_frame(frame_num=f)
                # annotations_by_frame[frame]['ref'] = annotation_slices

        # calculate scores
        all_scores = get_video_scores(annotations_by_frame=annotations_by_frame,
                                      assignments_by_annotator=assignments_by_annotator,
                                      task=task,
                                      item=item,
                                      score_types=score_types,
                                      task_type=task_type,
                                      logger=logger)
    else:  # image items
        # group by some field (e.g. 'creator' or 'assignment id'), here we use assignment id
        annots_by_assignment = {assignment.annotator: [] for assignment in assignments}
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

        # calculate scores
        all_scores = get_image_scores(annots_by_assignment=annots_by_assignment,
                                      assignments_by_annotator=assignments_by_annotator,
                                      task=task,
                                      item=item,
                                      score_types=score_types,
                                      task_type=task_type,
                                      logger=logger)

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
                                          match_threshold=match_threshold,
                                          # default 0.01 to get all possible matches
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
