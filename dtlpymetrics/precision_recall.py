import logging
import os
import datetime
from typing import List

import dtlpy as dl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

prec_rec = dl.AppModule(name='Scoring and metrics function',
                        description='Functions for calculating scores between annotations.'
                        )
logger = logging.getLogger('scoring-and-metrics')


@prec_rec.add_function(display_name='Calculate precision and recall')
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
        scores_file = items[0].download(overwrite=True)
        logger.info(f'Downloaded scores file to {scores_file}')

    scores = pd.read_csv(scores_file)
    labels = dataset.labels
    label_names = [label.tag for label in labels]
    if len(label_names) == 0:
        label_names = list(pd.concat([scores.first_label, scores.second_label]).dropna().drop_duplicates())

    ##############################
    # calculate precision/recall #
    ##############################
    logger.info('Calculating precision/recall')

    dataset_points = {'iou_threshold': {},
                      'data': {},  # "dataset" or "label"
                      'label_name': {},  # label name or NA,
                      'precision': {},
                      'recall': {},
                      'confidence': {}
                      }

    # first set is GT
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

    # detections.to_csv(index=False)  # DEBUG

    [_,
     dataset_plot_precision,
     dataset_plot_recall,
     dataset_plot_confidence] = \
        every_point_curve(recall=list(dataset_recall),
                          precision=list(dataset_precision),
                          confidence=list(detections['second_confidence']))
    # dataset_plot_precision = dataset_precision
    # dataset_plot_recall = dataset_recall
    # dataset_plot_confidence = detections['second_confidence']

    dataset_points['iou_threshold'] = [iou_threshold] * len(dataset_plot_precision)
    dataset_points['data'] = ['dataset'] * len(dataset_plot_precision)
    dataset_points['label_name'] = ['NA'] * len(dataset_plot_precision)
    dataset_points['precision'] = dataset_plot_precision
    dataset_points['recall'] = dataset_plot_recall
    dataset_points['confidence'] = dataset_plot_confidence
    dataset_points['dataset_name'] = [dataset.name] * len(dataset_plot_precision)

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

            [_,
             label_plot_precision,
             label_plot_recall,
             label_plot_confidence] = \
                every_point_curve(recall=list(label_recall),
                                  precision=list(label_precision),
                                  confidence=list(label_detections['second_confidence']))
            # label_plot_precision = label_precision
            # label_plot_recall = label_recall
            # label_plot_confidence = label_detections['second_confidence']

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
    logger.info('Saving precision recall plot data')
    plot_points = pd.concat([dataset_df, all_labels])
    # plot_points.to_csv(os.path.join(os.getcwd(), 'plot_points.csv'), index=False)     # DEBUG

    return plot_points


@prec_rec.add_function(display_name='Plot precision recall graph')
def plot_precision_recall(plot_points: pd.DataFrame,
                          dataset_name=None,
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
    logger.info('Plotting precision recall')

    plt.figure()
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    # plot each label separately
    dataset_points = plot_points[plot_points['data'] == 'dataset']
    dataset_legend = f"{dataset_points['dataset_id'].iloc[0]}" if dataset_name is None else dataset_name

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
    # plt.close()
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
    # plt.close()
    logger.info(f'Saved labels precision recall plot to {save_path}')

    return save_dir


@prec_rec.add_function(display_name='Calculate precision-recall values for every point curve')
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


@prec_rec.add_function(display_name='Calculate precision-recall values for eleven point curves')
def eleven_point_curve(recall: list, precision: list, confidence: list):
    # TODO - implement
    # recall = np.linspace(0, 1, 30)  # DEBUG
    # precision = np.linspace(1, 0, 30)  # DEBUG
    # confidence = np.linspace(0.2, 0.9, 30)  # DEBUG

    recall_all = recall
    precision_all = precision
    confidence_all = confidence

    recall_intervals = np.linspace(1, 0, 11)
    # recall_intervals = list(reversed(recall_intervals))

    rho_interpol = []  # the interpolated precision values for each interval range
    recall_valid = []
    conf_indices = []

    for recall_interval in recall_intervals:
        larger_recall = np.argwhere(recall_all[:] >= recall_interval)
        precision_max = 0

        if larger_recall.size != 0:
            precision_max = max(precision_all[larger_recall.min():])
            conf_indices.append(list(precision_all).index(precision_max))
        else:
            conf_indices.append(0)  # TODO check

        recall_valid.append(recall_interval)
        rho_interpol.append(precision_max)

    avg_precis = sum(rho_interpol) / 11

    # make points plot-ready
    recall_points = np.concatenate([[recall_valid[0]], recall_valid, [0]])
    precision_points = np.concatenate([[0], rho_interpol, [rho_interpol[-1]]])
    confidence_valid = [confidence_all[i] for i in conf_indices]
    confidence_points = np.concatenate([[confidence_valid[0]], confidence_valid, [confidence_valid[-1]]])

    cc = []
    for i in range(len(recall_points)):
        point_1 = (recall_points[i], precision_points[i - 1], confidence_points[i - 1])
        if point_1 not in cc:
            cc.append(point_1)
        point_2 = (recall_all[i], precision_all[i], confidence_all[i])
        if point_2 not in cc:
            cc.append(point_2)

    recall_plot = [i[0] for i in cc]
    precision_plot = [i[1] for i in cc]
    confidence_plot = [i[2] for i in cc]

    print(len(rho_interpol), len(recall_intervals), len(confidence_points))  # DEBUG
    print(len(recall_plot), len(precision_plot), len(confidence_plot))  # DEBUG

    return [avg_precis, recall_plot, precision_plot, confidence_plot]


@prec_rec.add_function(display_name='Create confusion matrix')
def calc_confusion_matrix(dataset_id: str,
                          model_id: str,
                          metric: str,
                          show_unmatched=True) -> pd.DataFrame:
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

    ###############################
    # create table of comparisons #
    ###############################
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


@prec_rec.add_function(display_name='Get list of model annotation false negatives from scores csv')
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
