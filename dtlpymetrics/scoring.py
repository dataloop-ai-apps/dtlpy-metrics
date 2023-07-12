import logging
import os
import dtlpy as dl
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from dtlpymetrics.metrics_utils import measure_annotations, all_compare_types
from dtlpymetrics.dtlpy_scores import Score, Scores, ScoreType

results_columns = {'annotation_iou': 'geometry_score',
                   'annotation_label': 'label_score',
                   'annotation_attribute': 'attribute_score',
                   'annotation_overall': 'annotation_score'
                   }

scorer = dl.AppModule(name='Scoring and metrics function',
                      description='Functions for calculating scores between annotations.'
                      )
logger = logging.getLogger('scoring-and-metrics')


@scorer.add_function(display_name='Calculate task scores for quality tasks')
def calculate_task_score(task: dl.Task) -> dl.Task:
    """
    Calculate scores for all items in a quality task, based on the item scores from each assignment.

    :param task: dl.Task entity
    :return: dl.Task entity
    """
    # # DEBUG
    # task = dl.tasks.get(task_id='64abe4ff24d52d7748b8f0dd')  # qualification
    # task = dl.tasks.get(task_id='64abe6af24d52d73d3b8f0fb')  # honeypot
    # task = dl.tasks.get(task_id='64aa776da54a2acafd368370')  # consensus

    # determine task type
    if task.metadata['system'].get('consensusTaskType') not in ['qualification', 'honeypot', 'consensus']:
        raise ValueError(f'Task type is not suitable for scoring')

    # default filter for consensus tasks is hidden = True, so this hides the clones and returns only original items
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
            elif item_task_dict.get('metadata').get('status', None) in ['complete', 'consensus_done']:
                create_task_item_score(item=item, task=task)

    return task


@scorer.add_function(display_name='Create scores for items in a quality task')
def create_task_item_score(item: dl.Item,
                           task: dl.Task = None,
                           context: dl.Context = None) -> dl.Task:
    """
    Create scores for items in a task.

    In the case of qualification and honeypot, the first set of annotations is considered the reference set.
    In the case of consensus, annotations are compared twice-- once as a reference set, and once as a test set.
    :param item: dl.Item entity
    :param task: dl.Task entity (optional)
    :param context: dl.Context entity that includes references to associated entities
    :return: item
    """
    ####################################
    # collect assignments for grouping #
    ####################################
    if task is None:
        if context is None:
            raise ValueError('Must provide either task or context.')
        else:
            task = context.task
    assignments = task.assignments.list()

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
    annots_by_assignment = {assignment.id: [] for assignment in assignments}
    annotations_ref = []

    logger.info(f'Starting scoring for assignments: {list(annots_by_assignment.keys())}')

    # group by some field (e.g. 'creator' or 'assignment id'), here we use assignment id
    for annotation in annotations:
        assignment_id = annotation.metadata['system'].get('assignmentId')
        if assignment_id is None:
            annotations_ref.append(annotation)
        else:
            annots_by_assignment[assignment_id].append(annotation)

    if task_type == 'consensus':
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
                                                              annot_collection_2=annot_collection_2,
                                                              ignore_labels=True,
                                                              match_threshold=0.01)
                # update scores with context
                for score in pairwise_scores:
                    if score.type != 'label_confusion':  # ScoreType.LABEL_CONFUSION:
                        annotation = dl.annotations.get(annotation_id=score.entity_id)
                        score.user_id = annotation.creator
                    score.task_id = task.id
                    score.assignment_id = assignments[j_assignment].id
                    score.item_id = item.id

                scores_list.extend(pairwise_scores)
    else:
        # compare annotation refs to assignee annotations
        logger.info(
            f'Comparing assignee: {annots_by_assignment[0].annotator!r} with ground truth annotations.')
        scores_list = []
        annot_collection_1 = annotations_ref
        annot_collection_2 = annots_by_assignment[assignments[0].id]

        pairwise_scores = calculate_annotation_scores(annot_collection_1=annot_collection_1,
                                                      annot_collection_2=annot_collection_2,
                                                      ignore_labels=True,
                                                      match_threshold=0.01)
        # update scores with context
        for score in pairwise_scores:
            annotation = dl.annotations.get(annotation_id=score.entity_id)
            score.user_id = annotation.creator
            score.task_id = task.id
            score.assignment_id = assignments[0].id
            score.item_id = item.id
            # print(score.print())  # DEBUG

        scores_list.extend(pairwise_scores)

    #############################
    # upload scores to platform #
    #############################
    if len(scores_list) == 0:
        item_score_value = 1
        logger.info(
            f'No annotation scores to upload. All assignees had no annotations.')
    else:
        filtered_scores = [score for score in scores_list if score.type is not ScoreType.ANNOTATION_OVERALL]
        item_score_value = np.sum([score.value for score in filtered_scores]) / len(filtered_scores)

    item_score = Score(type=ScoreType.ITEM_OVERALL.value,
                       value=item_score_value,
                       entity_id=item.id,
                       task_id=task.id,
                       item_id=item.id,
                       dataset_id=item.dataset.id)
    scores_list.append(item_score)

    if task_type == 'testing':
        user_score = Score(type=ScoreType.ITEM_OVERALL.value,
                           value=item_score_value,
                           entity_id=item.id,
                           task_id=task.id,
                           item_id=item.id,
                           user_id=assignments[0].annotator,
                           dataset_id=item.dataset.id)
        scores_list.append(user_score)

    # clean previous scores before creating new ones
    logger.info(f'About to delete all scores with context item ID: {item.id} and task ID: {task.id}')
    dl_scores = Scores(client_api=dl.client_api)
    dl_scores.delete(context={'itemId': item.id,
                              'taskId': task.id})

    dl_scores = dl_scores.create(scores_list)
    logger.info(f'Uploaded {len(dl_scores)} scores to platform.')

    return task


@scorer.add_function(display_name='Create model score')
def create_model_score(dataset: dl.Dataset = None,
                       filters: dl.Filters = None,
                       model: dl.Model = None,
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
    for item in (pbar := tqdm(items_list)):
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


@scorer.add_function(display_name='Compare two annotation sets for scoring')
def calculate_annotation_scores(annot_collection_1,
                                annot_collection_2,
                                compare_types=None,
                                score_types=None,
                                ignore_labels=False,
                                match_threshold=0.5) -> List[Score]:
    """
    Creates scores for comparing two annotation lists.

    The first annotation collection is considered the reference, and the second collection is the set for comparing.
    If we switch the order of the annotation collections, the scores remain the same but the user id context changes.

    :param annot_collection_1: dl.AnnotationCollection or list of annotations
    :param annot_collection_2: dl.AnnotationCollection or list of annotations
    :param compare_types: dl.AnnotationType entity or string for the annotation types to be compared
    :param score_types: dl.ScoreType entity or string for the score types to be calculated (e.g. "annotation_iou")
    :param ignore_labels: bool, True means every annotation will be cross-compared regardless of label classification
    :param match_threshold: float, threshold for matching annotations
    :return: list of Score entities
    """
    if score_types is None:
        score_types = [ScoreType.ANNOTATION_LABEL, ScoreType.ANNOTATION_IOU, ScoreType.ANNOTATION_OVERALL]
    if compare_types is None:
        compare_types = all_compare_types
    if not isinstance(score_types, list):
        score_types = [score_types]

    # compare bounding box annotations
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
    # all_results.to_csv('all_results.csv') # DEBUG

    #########################
    # create score entities #
    #########################
    logger.info(f'Creating scores for types: {score_types}')
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
            # print(f'creating score type{score_type}') # DEBUG

    ##############################################
    # create label confusion scores for this set #
    ##############################################
    # TODO check that this makes sense
    label_confusion_set = all_results[['first_label', 'second_label']]
    label_confusion_set = label_confusion_set.fillna('unlabeled')

    label_confusion_summary = label_confusion_set.groupby(['first_label', 'second_label']).size().reset_index(
        name='counts')

    for i, row in label_confusion_summary.iterrows():
        confusion_score = Score(type=ScoreType.LABEL_CONFUSION.value,
                                value=row['counts'],
                                entity_id=row['first_label'],
                                relative=row['second_label'])
        annotation_scores.append(confusion_score)

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
