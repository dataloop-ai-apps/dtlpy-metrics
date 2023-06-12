import os
from typing import Dict, Union, List, Any

import numpy as np
import pandas as pd
import dtlpy as dl
import matplotlib.pyplot as plt


def calc_precision_recall(dataset_id: str,
                          model_id: str,
                          conf_threshold=0.5,
                          iou_threshold=0.5):
    # method_type='every_point'):
    """
    Plot precision recall curve for a given metric threshold

    :param dataset_id: str dataset ID
    :param model_id: str model ID
    :param conf_threshold: 
    :param iou_threshold:
    :return:
    """

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

    ##############################
    # calculate precision/recall #
    #############################
    # calc
    if labels is None:
        labels = pd.concat([scores.first_label, scores.second_label]).dropna()

    plot_points = {'conf_threshold': conf_threshold,
                   'iou_threshold': iou_threshold,
                   'labels': {},
                   'dataset_precision': [],
                   'dataset_recall': []
                   }

    num_gts = sum(scores.first_id.notna())

    scores_positives = scores[scores['geometry_score'] > iou_threshold].copy()

    scores_positives.sort_values('second_confidence', inplace=True, ascending=True, ignore_index=True)
    scores_positives['true_positives'] = scores_positives['second_confidence'] >= conf_threshold
    scores_positives['false_positives'] = scores_positives['second_confidence'] < conf_threshold

    # get dataset-level precision/recall
    dataset_fps = np.cumsum(scores_positives['false_positives'])
    dataset_tps = np.cumsum(scores_positives['true_positives'])
    dataset_recall = dataset_tps / num_gts
    dataset_precision = np.divide(dataset_tps, (dataset_fps + dataset_tps))

    [_, dataset_plot_precision, dataset_plot_recall] = every_point_curve(dataset_recall, dataset_precision)
    plot_points['dataset_precision'] = dataset_plot_precision
    plot_points['dataset_recall'] = dataset_plot_recall

    # get label-level precision/recall
    for label in list(set(label_names)):
        label_positives = scores_positives[scores_positives.first_label == label].copy()
        label_positives.sort_values('second_confidence', inplace=True, ascending=True, ignore_index=True)

        label_fps = np.cumsum(label_positives['false_positives'])
        label_tps = np.cumsum(label_positives['true_positives'])
        label_recall = label_tps / num_gts
        label_precision = np.divide(label_tps, (label_fps + label_tps))

        [_, label_plot_precision, label_plot_recall] = every_point_curve(label_recall, label_precision)

        plot_points['labels'].update({label: {
            'precision': label_plot_precision,
            'recall': label_plot_recall}})

    return plot_points


def every_point_curve(recall: list, precision: list):
    """
    Calculate precision-recall curve from a list of precision & recall values
    :param recall: list of recall values
    :param precision: list of precision values
    :return:
    """
    recall_points = np.concatenate([[0], recall, [1]])
    precision_points = np.concatenate([[0], precision, [1]])

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
            recall_points[0:len(precision_points) - 1]]


def calc_confusion_matrix(dataset_id: str,
                          model_id: str,
                          metric: str,
                          show_unmatched=True):
    """
    Calculate confusion matrix for a given model and metric
    :param dataset_id:
    :param model_id:
    :param metric:
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
        labels = pd.concat([scores.first_label, scores.second_label]).dropna()

    scores_cleaned = scores.dropna().reset_index(drop=True)
    scores_labels = scores_cleaned[['first_label', 'second_label']]
    grouped_labels = scores_labels.groupby(['first_label', 'second_label']).size()

    conf_matrix = pd.DataFrame(index=label_names, columns=label_names, data=0)
    for label1, label2 in grouped_labels.index:
        # index/rows are the ground truth, cols are the predictions
        conf_matrix.loc[label1, label2] = grouped_labels.get((label1, label2), 0)

    return conf_matrix


def plot_precision_recall(plot_points, local_path=None):
    labels = list(plot_points['labels'].keys())

    plt.figure()
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    for label in labels:
        plt.plot(plot_points['labels'][label]['recall'],
                 plot_points['labels'][label]['precision'],
                 label=[label])
    plt.legend()

    # plot_filename = f'precision_recall_{dataset_id}_{model_id}_{plot_points[metric]}_{metric_threshold}.png'
    plot_filename = f'precision_recall.png'
    if local_path is None:
        save_path = os.path.join(os.getcwd(), '.dataloop', plot_filename)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
    else:
        save_path = os.path.join(local_path, plot_filename)
        plt.savefig(save_path)

    plt.close()

    return save_path


if __name__ == '__main__':
    dl.setenv('rc')

    dataset_id = '646e2c13a8386f8b38d5efb5'  # big cats GT
    # model_id = '6473185c93bd97c6a30a47b9'  # resnet
    model_id = '64803fcc9e5ee9b3b5716832'  # resnet with unmatched predictions
    # model_id = '' # yolov8

    plot_points = calc_precision_recall(dataset_id=dataset_id,
                                        model_id=model_id,
                                        conf_threshold=0.5)
    plot_precision_recall(plot_points)

    metric = 'accuracy'
    conf_table = calc_confusion_matrix(dataset_id=dataset_id,
                                       model_id=model_id,
                                       metric=metric)
    print("columns are model predictions, rows are ground truth labels")
    print(conf_table)
    print()
