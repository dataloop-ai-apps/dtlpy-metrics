import logging
import os
from typing import List

import dtlpy as dl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtlpymetrics.metrics_utils import measure_annotations
import datetime
from dtlpymetrics.dtlpy_scores import Score, Scores, ScoreType

score_names = ['IOU', 'label', 'attribute']
results_columns = {'annotation_iou': 'geometry_score', 'annotation_label': 'label_score',
                   'attribute': 'attribute_score'}
all_compare_types = list(dl.AnnotationType)

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
                create_consensus_item_score(item=item, task=None)

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
        consensus_task = context.task
    assignments = consensus_task.assignments.list()

    #################################
    # sort annotations by annotator #
    #################################
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
    annotation_scores = []  # TODO change this var to "assignment_scores"
    for i_assignment in range(n_assignments):
        for j_assignment in range(0, i_assignment + 1):
            annotation_scores = []  # TODO change this var to "assignment_scores"
            logger.info(
                f'Comparing annotator: {assignments[i_assignment].annotator!r} with annotator: {assignments[j_assignment].id!r}')
            annot_collection_1 = annots_by_assignment[assignments[i_assignment].id]
            annot_collection_2 = annots_by_assignment[assignments[j_assignment].id]

            pairwise_scores = calculate_annotation_scores(annot_collection_1=annot_collection_1,
                                                          annot_collection_2=annot_collection_2)
            annotation_scores.extend(pairwise_scores)

            # upload annotation scores
            if annotation_scores is None:
                logger.info(f'No scores to upload.')
            else:
                upload_task_annotation_scores(annotations=annotation_scores,
                                              scores=annotation_scores,
                                              assignee1_id=assignments[i_assignment].id,
                                              assignee2_id=assignments[j_assignment].id,
                                              task_id=consensus_task.id)

    # calculate overall item score as the average of all scores
    item_score_total = 0
    for annotation_score in annotation_scores:
        item_score_total += annotation_score.value

    #############################
    # upload scores to platform #
    #############################
    # clean previous scores first
    dl_scores = Scores(client_api=dl.client_api)
    dl_scores.delete(context={'itemId': item.id,
                              'taskId': consensus_task.id})
    if len(annotation_scores) == 0:
        item_score = 0
        logger.info(f'No scores.')
    else:
        dl_annot_scores = dl_scores.create(annotation_scores)
        item_score = item_score_total / len(annotation_scores)
        logger.info(f'Uploaded {len(dl_annot_scores)} annotation scores to platform.')

    item_score = Score(type=ScoreType.ITEM_OVERALL.value,
                       value=item_score,
                       entity_id=item.id,
                       task_id=consensus_task.id,
                       item_id=item.id,
                       dataset_id=item.dataset.id)
    dl_scores.create([item_score])
    logger.info(f'Uploaded overall score for item {item.id} to platform.')

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
            annot_score = Score(type=score_type.value,
                                value=row[results_columns[score_type.value.lower()]],
                                entity_id=row['second_id'])
            annotation_scores.append(annot_score)

    return annotation_scores


def upload_task_annotation_scores(annotations: List[dl.Annotation],
                                  scores: List[Score],
                                  task_id: str,
                                  assignee1_id: str,
                                  assignee2_id: str) -> List[Score]:
    """
    Uploads annotation scores to the platform for tasks. This includes two sets of scores with references for each annotator.

    :param annotations: list of annotations
    :param scores: list of scores
    :param task_id: task id
    :param assignee1_id: first annotator id
    :param assignee2_id: second annotator id
    :return: list of scores
    """
    if scores is None:
        logging.info('No scores to upload.')
    else:
        ############################
        # upload scores to platform #
        ############################

        dl_scores = Scores(client_api=dl.client_api)
        annotation_to_item_map = {}
        for annotation in annotations:
            annotation_to_item_map[annotation.id] = annotation.item_id
            dl_scores.delete(context={
                # 'annotationId': annotation.id,
                'itemId': annotation.item.id,
                'taskId': task_id}
            )

        # update scores with context
        scores_2 = scores.copy()

        for score in scores:
            score.task_id = task_id
            score.user_id = assignee1_id
            score.item_id = annotation_to_item_map[score.entity_id]
            score.relative = assignee2_id

        # a second set of scores associated with the first annotator/assignee
        for score2 in scores_2:
            score2.task_id = task_id
            score2.user_id = assignee2_id
            score2.item_id = annotation_to_item_map[score2.entity_id]
            score2.relative = assignee1_id

        dl_scores.create(scores)
        dl_scores.create(scores_2)

        logger.info(f'Uploaded {len(scores) + len(scores_2)} scores to platform.')

    return scores


@scorer.add_function(display_name='Calculate precision and recall')
def calc_precision_recall(dataset_id: str,
                          model_id: str,
                          # filters: dl.Filters = None,
                          iou_threshold=0.01) -> pd.DataFrame:
    # method_type='every_point'):
    """
    Plot precision recall curve for a given metric threshold

    :param dataset_id: str dataset ID
    :param model_id: str model ID
    :param iou_threshold: float Threshold for accepting matched annotations as a true positive
    :return: dataframe with all the points to plot for the dataset and individual labels
    """
    ################################
    # get matched annotations data #
    ################################
    # TODO use scoring once available
    model_filename = f'{model_id}.csv'
    items_filters = dl.Filters(field='hidden', values=True)
    items_filters.add(field='name', values=model_filename)
    dataset = dl.datasets.get(dataset_id=dataset_id)
    items = list(dataset.items.list(filters=items_filters).all())
    if len(items) == 0:
        raise ValueError(f'No scores found for model ID {model_id}.')
    elif len(items) > 1:
        raise ValueError(f'Found {len(items)} items with name {model_id}.')
    else:
        scores_file = items[0].download()

    scores = pd.read_csv(scores_file)
    labels = dataset.labels
    label_names = [label.tag for label in labels]
    if len(label_names) == 0:
        label_names = list(pd.concat([scores.first_label, scores.second_label]).dropna().drop_duplicates())

    ##############################
    # calculate precision/recall #
    ##############################
    dataset_points = {'iou_threshold': {},
                      'data': {},  # "dataset" or "label"
                      'label_name': {},  # label name or NA,
                      'precision': {},
                      'recall': {},
                      'confidence': {}
                      }

    num_gts = sum(scores.first_id.notna())
    detections = scores[scores.second_id.notna()].copy()

    detections.sort_values('second_confidence', inplace=True, ascending=False, ignore_index=True)
    detections['true_positives'] = detections['geometry_score'] >= iou_threshold
    detections['false_positives'] = detections['geometry_score'] < iou_threshold

    # get dataset-level precision/recall
    dataset_fps = np.cumsum(detections['false_positives'])
    dataset_tps = np.cumsum(detections['true_positives'])
    dataset_recall = dataset_tps / num_gts
    dataset_precision = np.divide(dataset_tps, (dataset_fps + dataset_tps))

    [_,
     dataset_plot_precision,
     dataset_plot_recall,
     dataset_plot_confidence] = \
        every_point_curve(recall=list(dataset_recall),
                          precision=list(dataset_precision),
                          confidence=list(detections['second_confidence']))

    dataset_points['iou_threshold'] = [iou_threshold] * len(dataset_plot_precision)
    dataset_points['data'] = ['dataset'] * len(dataset_plot_precision)
    dataset_points['label_name'] = ['NA'] * len(dataset_plot_precision)
    dataset_points['precision'] = dataset_plot_precision
    dataset_points['recall'] = dataset_plot_recall
    dataset_points['confidence'] = dataset_plot_confidence

    dataset_df = pd.DataFrame(dataset_points).drop_duplicates()

    ##########################################
    # calculate label-level precision/recall #
    ##########################################
    all_labels = pd.DataFrame(columns=dataset_df.columns)

    label_points = {key: {} for key in dataset_points}

    for label_name in list(set(label_names)):
        label_detections = detections.loc[
            (detections.first_label == label_name) | (detections.second_label == label_name)].copy()
        if label_detections.shape[0] == 0:
            label_plot_precision = [0]
            label_plot_recall = [0]
            label_plot_confidence = [0]
        else:
            label_detections.sort_values('second_confidence', inplace=True, ascending=False, ignore_index=True)

            label_fps = np.cumsum(label_detections['false_positives'])
            label_tps = np.cumsum(label_detections['true_positives'])
            label_recall = label_tps / num_gts
            label_precision = np.divide(label_tps, (label_fps + label_tps))

            # [_,
            #  label_plot_precision,
            #  label_plot_recall,
            #  label_plot_confidence] = \
            #     every_point_curve(recall=list(label_recall),
            #                              precision=list(label_precision),
            #                              confidence=list(label_detections['second_confidence']))
            label_plot_precision = label_precision
            label_plot_recall = label_recall
            label_plot_confidence = label_detections['second_confidence']

        label_points['iou_threshold'] = [iou_threshold] * len(label_plot_precision)
        label_points['data'] = ['label'] * len(label_plot_precision)
        label_points['label_name'] = [label_name] * len(label_plot_precision)
        label_points['precision'] = label_plot_precision
        label_points['recall'] = label_plot_recall
        label_points['confidence'] = label_plot_confidence
        label_points['dataset_name'] = [dataset.name] * len(label_plot_precision)

        label_df = pd.DataFrame(label_points).drop_duplicates()
        all_labels = pd.concat([all_labels, label_df])

    ####################
    # combine all data #
    ####################
    plot_points = pd.concat([dataset_df, all_labels])

    return plot_points


@scorer.add_function(display_name='Plot precision recall graph')
def plot_precision_recall(plot_points: pd.DataFrame,
                          label_names=None,
                          local_path=None):
    """
    Plot precision recall curve for a given metric threshold

    :param plot_points: dict generated from calculate_precision_recall with all the points to plot by label and
     the entire dataset. keys include: confidence threshold, iou threshold, dataset levels precision, recall, and
     confidence, and label-level precision, recall and confidence
    :param label_names: list of label names to plot
    :param local_path: path to save plot
    :return:
    """
    if local_path is None:
        root_dir = os.getcwd().split('dtlpymetrics')[0]
        save_dir = os.path.join(root_dir, 'dtlpymetrics', '.dataloop')
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))
    else:
        save_dir = os.path.join(local_path)

    ###################
    # plot by dataset #
    ###################
    plt.figure()
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    # plot each label separately
    dataset_points = plot_points[plot_points['data'] == 'dataset']
    dataset_legend = f"{dataset_points['dataset_name'].iloc[0]}"

    plt.plot(dataset_points['recall'],
             dataset_points['precision'],
             label=dataset_legend)

    plt.legend(loc='upper right')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.grid()

    # plot the dataset level
    plot_filename = f"dataset_precision_recall_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
    save_path = os.path.join(save_dir, plot_filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f'Saved dataset precision recall plot to {save_path}')

    #################
    # plot by label #
    #################
    all_labels = plot_points[plot_points['data'] == 'label']

    if (label_names is None) or (bool(label_names) is False):
        label_names = all_labels['label_name'].copy().drop_duplicates()

    plt.figure()
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    # plot each label separately
    for label_name in label_names:
        label_points = all_labels[all_labels['label_name'] == label_name].copy()

        plt.plot(label_points['recall'],
                 label_points['precision'],
                 label=label_name)

    plt.legend(loc='upper right')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.grid()

    # plot the dataset level
    plot_filename = f"label_precision_recall_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
    save_path = os.path.join(save_dir, plot_filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f'Saved labels precision recall plot to {save_path}')

    return save_dir


@scorer.add_function(display_name='Calculate precision-recall values for every point curve')
def every_point_curve(recall: list, precision: list, confidence: list):
    """
    Calculate precision-recall curve from a list of precision & recall values
    :param recall: list of recall values
    :param precision: list of precision values
    :param confidence: list of confidence values
    :return: list of average precision all values, precision points, recall points, confidence points
    """
    recall_points = np.concatenate([[0], recall, [1]])
    precision_points = np.concatenate([[0], precision, [0]])
    confidence_points = np.concatenate([[confidence[0]], confidence, [confidence[-1]]])

    # find the maximum precision between each recall value, backwards
    for i in range(len(precision_points) - 1, 0, -1):
        precision_points[i - 1] = max(precision_points[i - 1], precision_points[i])

    # build the simplified recall list, removing values when the precision doesnt change
    recall_intervals = []
    for i in range(len(recall_points) - 1):
        if recall_points[1 + i] != recall_points[i]:
            recall_intervals.append(i + 1)

    avg_precis = 0
    for i in recall_intervals:
        avg_precis = avg_precis + np.sum((recall_points[i] - recall_points[i - 1]) * precision_points[i])
    return [avg_precis,
            precision_points[0:len(precision_points) - 1],
            recall_points[0:len(precision_points) - 1],
            confidence_points[0:len(precision_points) - 1]]


@scorer.add_function(display_name='Calculate precision-recall values for eleven point curves')
def eleven_point_curve(recall: list, precision: list, confidence: list):
    # DEBUG
    recall = np.linspace(0, 1, 30)
    precision = np.linspace(1, 0, 30)
    confidence = np.linspace(0.2, 0.9, 30)

    recall_values = recall
    precision_values = precision
    confidence_values = confidence

    recall_intervals = np.linspace(0, 1, 11)
    recall_intervals = list(reversed(recall_intervals))

    rho_interpol = []
    recall_valid = []

    for recall_interval in recall_intervals:
        larger_recall = np.argwhere(recall_values[:] >= recall_interval)
        precision_max = 0

        if larger_recall.size != 0:
            precision_max = max(precision_values[larger_recall.min():])
            recall_valid.append(recall_interval)
            rho_interpol.append(precision_max)

    avg_precis = sum(rho_interpol) / 11

    # make points plot-ready
    recall_points = np.concatenate([[recall_valid[0]], recall_valid, [0]])
    precision_points = np.concatenate([[0], rho_interpol, [rho_interpol[-1]]])

    cc = []
    for i in range(len(recall_points)):
        point_1 = (recall_points[i], precision_points[i - 1])
        if point_1 not in cc:
            cc.append(point_1)
        point_2 = (recall_values[i], precision_values[i])
        if point_2 not in cc:
            cc.append(point_2)

        recall_intervals = [i[0] for i in cc]
        rho_interpol = [i[1] for i in cc]

    return [avg_precis, rho_interpol, recall_intervals, confidence_values]


@scorer.add_function(display_name='Create confusion matrix')
def calc_confusion_matrix(dataset_id: str,
                          model_id: str,
                          metric: str,
                          show_unmatched=True):
    """
    Calculate confusion matrix for a given model and metric

    :param dataset_id: str ID of test dataset
    :param model_id: str ID of model
    :param metric: name of the metric for comparing
    :param show_unmatched: display extra column showing which GT annotations were not matched
    :return:
    """
    if metric.lower() == 'iou':
        metric = 'geometry_score'
    elif metric.lower() == 'accuracy':
        metric = 'label_score'

    model_filename = f'{model_id}.csv'
    filters = dl.Filters(field='hidden', values=True)
    filters.add(field='name', values=model_filename)
    dataset = dl.datasets.get(dataset_id=dataset_id)
    items = list(dataset.items.list(filters=filters).all())
    if len(items) == 0:
        raise ValueError(f'No scores found for model ID {model_id}.')
    elif len(items) > 1:
        raise ValueError(f'Found {len(items)} items with name {model_id}.')
    else:
        scores_file = items[0].download()

    scores = pd.read_csv(scores_file)
    labels = dataset.labels
    label_names = [label.tag for label in labels]

    if metric not in scores.columns:
        raise ValueError(f'{metric} metric not included in scores.')

    #########################
    # plot precision/recall #
    #########################
    # calc
    if labels is None:
        label_names = pd.concat([scores.first_label, scores.second_label]).dropna()

    scores_cleaned = scores.dropna().reset_index(drop=True)
    scores_labels = scores_cleaned[['first_label', 'second_label']]
    grouped_labels = scores_labels.groupby(['first_label', 'second_label']).size()

    conf_matrix = pd.DataFrame(index=label_names, columns=label_names, data=0)
    for label1, label2 in grouped_labels.index:
        # index/rows are the ground truth, cols are the predictions
        conf_matrix.loc[label1, label2] = grouped_labels.get((label1, label2), 0)

    return conf_matrix


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


@scorer.add_function(display_name='Get list of model annotation false negatives from scores csv')
def get_false_negatives(model: dl.Model, dataset: dl.Dataset) -> pd.DataFrame:
    """
    Retrieves the dataframe for all the scores for a given model on a dataset via a hidden csv file,
    and returns a dataframe with the properties of all the false negatives.
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

    ########################
    # list false negatives #
    ########################
    model_fns = dict()
    annotation_to_item_map = {ann_id: item_id for ann_id, item_id in
                              zip(scores_df.first_id, scores_df.itemId)}
    fn_annotation_ids = scores_df[scores_df.second_id.isna()].first_id
    print(f'model: {model.name} with {len(fn_annotation_ids)} false negative')
    fn_items_ids = np.unique([annotation_to_item_map[ann_id] for ann_id in fn_annotation_ids])
    for i_id in fn_items_ids:
        if i_id not in model_fns:
            i_id: dl.Item
            url = dl.client_api._get_resource_url(
                "projects/{}/datasets/{}/items/{}".format(dataset.project.id, dataset.id, i_id))
            model_fns[i_id] = {'itemId': i_id,
                               'url': url}
        model_fns[i_id].update({model.name: True})

    model_fn_df = pd.DataFrame(model_fns.values()).fillna(False)
    model_fn_df.to_csv(os.path.join(os.getcwd(), f'{model.name}_false_negatives.csv'))

    return model_fn_df
