import logging
import os
import datetime
from typing import List

import dtlpy as dl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtlpymetrics.metrics_utils import measure_annotations, all_compare_types
from dtlpymetrics.dtlpy_scores import Score, Scores, ScoreType

score_names = ['IOU', 'label', 'attribute']
results_columns = {'annotation_iou': 'geometry_score', 'annotation_label': 'label_score',
                   'attribute': 'attribute_score'}

scorer = dl.AppModule(name='Scoring and metrics function',
                      description='Functions for calculating scores between annotations.'
                      )
logger = logging.getLogger('scoring-and-metrics')


@scorer.add_function(display_name='Calculate the consensus task score',
                     inputs={"consensus_task": dl.Task},
                     outputs={"score_summary": dict,
                              "consensus_task": dl.Task}
                     )
def calculate_consensus_task_score(consensus_task: dl.Task):
    """
    Calculate scores for all items in a consensus task, based on the item scores from each assignment.

    :param consensus_task: dl.Task entity
    :return: consensus_task
    """
    # default filter for consensus tasks is hidden = True, so this hides the clones and returns only original items
    filters = dl.Filters()
    filters.add(field='hidden', values=False)
    # filters.add(field='metadata.status', values='consensus_done'
    # filters.add(field='metadata.system.refs', values='consensus_done', operator=dl.FILTERS_OPERATIONS_EXISTS))
    # filters.add(field='task', values=consensus_task.id)
    pages = consensus_task.get_items(filters=filters, get_consensus_items=True)

    for item in pages.all():
        all_item_tasks = item.metadata['system']['refs']
        for item_task in all_item_tasks:
            if item_task['id'] != consensus_task.id:
                continue
            if item_task.get('metadata', None) is None:
                continue
            elif item_task.get('metadata', None).get('status', None) == 'consensus_done':
                logging.info('Calculating score for item {}'.format(item.id))
                create_consensus_item_score(item=item, task=consensus_task)

    return consensus_task


@scorer.add_function(display_name='Create item consensus score',
                     inputs={"item": "Item",
                             "context": dl.Context},
                     outputs={"item": "Item"}
                     )
def create_consensus_item_score(item: dl.Item,
                                task: dl.Task = None,
                                context: dl.Context = None) -> dl.Item:
    """
    Create a consensus score for an item in a consensus task.

    The first set of annotations is considered the reference set.
    :param item: dl.Item entity
    :param context: dl.Context entity that includes references to associated entities
    :return: item
    """
    ####################################
    # collect assignments for grouping #
    ####################################
    if task is None:
        task = context.task
    assignments = task.assignments.list()

    ################################
    # sort annotations by assignee #
    ################################
    annotations = item.annotations.list()
    annots_by_assignment = {assignment.id: [] for assignment in assignments}
    logger.info(f'Starting scoring for assignments: {list(annots_by_assignment.keys())}')

    # group by some field (e.g. 'creator' or 'assignment id'), here we use assignment id
    for annotation in annotations:
        assignment_id = annotation.metadata['system'].get('assignmentId')
        if assignment_id is None:
            continue
        annots_by_assignment[assignment_id].append(annotation)

    # do pairwise comparisons of each assignment for all annotations on the item
    n_assignments = len(annots_by_assignment)
    scores_list = []
    for i_assignment in range(n_assignments):
        for j_assignment in range(n_assignments):
            if i_assignment == j_assignment:
                continue

            logger.info(
                f'Comparing assignee: {assignments[i_assignment].annotator!r} with assignee: {assignments[j_assignment].annotator!r}')
            annot_collection_1 = annots_by_assignment[assignments[i_assignment].id]
            annot_collection_2 = annots_by_assignment[assignments[j_assignment].id]

            pairwise_scores = calculate_annotation_scores(annot_collection_1=annot_collection_1,
                                                          annot_collection_2=annot_collection_2)
            # update scores with context
            for score in pairwise_scores:
                annotation = dl.annotations.get(annotation_id=score.entity_id)
                score.user_id = annotation.creator
                score.task_id = task.id
                score.assignment_id = assignments[j_assignment].id
                score.item_id = item.id

            scores_list.extend(pairwise_scores)

    #############################
    # upload scores to platform #
    #############################
    if len(scores_list) == 0:
        item_score_value = 0
        logger.info(f'No annotation scores to upload. Check that you have annotations from each assignee of the same type.')
    else:
        item_score_value = np.sum([score.value for score in scores_list]) / len(scores_list)

    item_score = Score(type=ScoreType.ITEM_OVERALL.value,
                       value=item_score_value,
                       entity_id=item.id,
                       task_id=task.id,
                       item_id=item.id,
                       dataset_id=item.dataset.id)
    scores_list.append(item_score)

    # clean previous scores before creating
    logger.info(f'About to delete all scores with context itemID{item.id} and taskID {task.id}')
    dl_scores = Scores(client_api=dl.client_api)
    dl_scores.delete(context={'itemId': item.id,
                              'taskId': task.id})

    dl_scores = dl_scores.create(scores_list)
    logger.info(f'Uploaded {len(dl_scores)} scores to platform.')

    return item


@scorer.add_function(display_name='Create item consensus score')
def create_model_score(dataset: dl.Dataset = None,
                       filters: dl.Filters = None,
                       model: dl.Model = None,
                       ignore_labels=False,
                       compare_types=None) -> (bool, str):
    """
    Measures scores for a set of model predictions compared against ground truth annotations.

    :param dataset: Dataset associated with the ground truth annotations
    :param filters: DQL Filter for retrieving the test items
    :param model: Model for evaluating predictions
    :param ignore_labels: bool, True means every annotation will be cross-compared regardless of label
    :param compare_types: annotation types to compare
    :return:
    """

    if dataset is None:
        return False, 'No dataset provided, please provide a dataset.'
    if model is None:
        return False, 'No model provided, please provide a model.'
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
        return False, 'No model name found for the second set of annotations, please provide model name.'
    if not items_list:
        return False, 'No items found in the dataset, please check the dataset and filters.'

    ########################################
    # Create list of item annotation lists #
    ########################################
    for item in items_list:
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
        # logger.info(f'item {i}: GT annots {len(annot_set_1[i])}, model annots {len(annot_set_2[i])}')
        if len(annot_set_1[i]) == 0 and len(annot_set_2[i]) == 0:
            continue
        else:
            results = measure_annotations(annotations_set_one=annot_set_1[i],
                                          annotations_set_two=annot_set_2[i],
                                          match_threshold=0.01,  # to get all possible matches
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

    return True, f'Successfully created model scores and saved as item {item.id}.'


@scorer.add_function(display_name='Compare two sets of annotations for scoring')
def calculate_annotation_scores(annot_collection_1,
                                annot_collection_2,
                                compare_types=None,
                                score_types=ScoreType.ANNOTATION_LABEL,
                                ignore_labels=False) -> List[Score]:
    """
    Creates scores for comparing two annotation lists.

    The first annotation collection is considered the reference, and the second collection is the set for comparing.
    If we switch the order of the annotation collections, the scores remain the same but the user id context changes..

    :param annot_collection_1: dl.AnnotationCollection or list of annotations
    :param annot_collection_2: dl.AnnotationCollection or list of annotations
    :param compare_types: dl.AnnotationType entity or string for the annotation types to be compared
    :param score_types: dl.ScoreType entity or string for the score types to be calculated (e.g. "annotation_iou")
    :return: dict of feature sets, indexed by the type of score (e.g. IOU)
    """
    if compare_types is None:
        compare_types = all_compare_types
    if not isinstance(score_types, list):
        score_types = [score_types]

    # compare bounding box annotations
    results = measure_annotations(
        annotations_set_one=annot_collection_1,
        annotations_set_two=annot_collection_2,
        compare_types=compare_types,
        ignore_labels=ignore_labels)

    all_results = pd.DataFrame()
    for compare_type in compare_types:
        try:
            results_df = results[compare_type].to_df()
            all_results = pd.concat([all_results, results_df])
        except KeyError:
            continue

    annotation_scores = []

    for i, row in all_results.iterrows():
        for score_type in score_types:
            if row['second_id'] is None:
                continue

            annot_score = Score(type=score_type.value,
                                value=row[results_columns[score_type.value.lower()]],
                                entity_id=row['second_id'],
                                relative=row['first_id'])
            annotation_scores.append(annot_score)

    return annotation_scores


@scorer.add_function(display_name='Get model annotation scores dataframe from scores csv')
def get_scores_df(model: dl.Model, dataset: dl.Dataset):
    """
    Retrieves the dataframe for all the scores for a given model on a dataset via a hidden csv file.
    :param model: Model entity
    :param dataset: Dataset where the model was evaluated
    :return:
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
            f'No scores file found for model {model.id} on dataset {dataset.id}. Please evaluate model on the dataset first.')

    scores_df = pd.read_csv(local_path)
    return scores_df
